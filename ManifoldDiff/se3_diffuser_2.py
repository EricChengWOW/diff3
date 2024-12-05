import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.optim import Adam
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import wandb
import math
from utils import *

# Diffuser Class
class SE3Diffusion_2:
    def __init__(self, score_model, beta_start=0.1, beta_end=1.0, trans_scale=1.0):
        self.score_model = score_model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trans_scale = trans_scale

    def beta(self, t):
        """Continuous beta(t) as a linear schedule."""
        return self.beta_start + t * (self.beta_end - self.beta_start)
  
    def alpha(self, t):
        return torch.exp(-0.5 * self.beta(t) * t)

    def alpha_bar(self, t):
        """Continuous cumulative product of alpha_bar(t)."""
        beta_t_integral = (
            -0.5 * (self.beta_end - self.beta_start) * t**2 - self.beta_start * t
        )
        return torch.exp(beta_t_integral)
    
    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.beta(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.beta(t) * x

    def marginal_b_t(self, t):
        return t*self.beta_start + (1/2)*(t**2)*(self.beta_end-self.beta_start)

    def igso3_sample(self, shape, device):
        """
        Sample from IGSO(3), which we'll assume to be random rotation vectors in 3D space.
        
        Args:
            shape (torch.Size): Desired output shape, must include 3 as the last dimension.
            device (torch.device): The device to create the tensor on.
        
        Returns:
            torch.Tensor: Sampled rotation vectors with shape [..., 3].
        """
        # Sample random unit vectors in 3D space (can be viewed as sampling rotation axes)
        random_angles = torch.rand(*shape[:-1], 1, device=device) * 2 * math.pi  # Random angles between 0 and 2pi
        random_axis = torch.randn(*shape[:-1], 3, device=device)  # Random vectors in 3D
        random_axis = random_axis / torch.norm(random_axis, dim=-1, keepdim=True)  # Normalize to get unit vectors
        
        # Rotation vector is axis * angle (using random_angle as the angle for each axis)
        rotation_vectors = random_axis * random_angles  # Shape: [..., 3]
        return rotation_vectors

    def forward_process(self, x1, x2, t):
        """Forward diffusion process with different noise sources for x1 and x2."""
        # Sample noise for x1 from normal distribution
        epsilon1 = torch.randn_like(x1)

        # Sample noise for x2 from IGSO(3)
        epsilon2 = self.igso3_sample(x2.shape, x2.device)

        alpha_t = self.alpha(t)
        alpha_t = alpha_t.view(-1, 1, 1)
        alpha_t = alpha_t.expand_as(x1)

        coef1 = torch.exp(-1/2*self.marginal_b_t(t)).view(-1, 1, 1).expand_as(x1)
        coef2 = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t))).view(-1, 1, 1).expand_as(x1)
        
        x1_t = coef1 * x1 + coef2 * epsilon1

        # print(sqrt_one_minus_alpha_t.shape, epsilon2.shape )
        x2_scaled = coef1 * x2  # Scaling input rotation vector
        epsilon2_scaled = coef2 * epsilon2  # Scaling noise

        # Apply Rodrigues' rotation formula to create rotation matrices
        x2_rot_mat = rotvec_to_matrix(x2_scaled)
        epsilon2_rot_mat = rotvec_to_matrix(epsilon2_scaled)

        # Perform matrix multiplication for diffusion
        combined_rot_mat = torch.matmul(x2_rot_mat, epsilon2_rot_mat)

        # Convert back to rotation vectors
        x2_t = matrix_to_rotvec(combined_rot_mat)

        return (x1_t, epsilon1), (x2_t, epsilon2)

    def compute_loss(self, x1, x2, t):
        """Compute the diffusion loss for x1 and x2."""
        (x1_t, epsilon1), (x2_t, epsilon2) = self.forward_process(x1 * self.trans_scale, x2, t)
        

        if self.score_model.name == "Unet":
            x1_t = x1_t.transpose(1,2)
            x2_t = x2_t.transpose(1,2)

        # Predict scores using the score model
        predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t, t)

        if self.score_model.name == "Unet":
            predicted_score1 = predicted_score1.transpose(1,2)
            predicted_score2 = predicted_score2.transpose(1,2)
        
        # Loss for each stream
        loss1 = F.l1_loss(predicted_score1, epsilon1)
        loss2 = F.l1_loss(predicted_score2, epsilon2)

        # Average the two losses
        return (loss1 + loss2) / 2

    def sample(self, shape, device, num_steps=1000):
        """
        Sample both x1 and x2 from the reverse diffusion process.

        Args:
            shape (torch.Size): Desired shape of the output tensors.
            device (torch.device): The device to create the tensor on.
            num_steps (int): The number of diffusion steps to reverse.

        Returns:
            torch.Tensor, torch.Tensor: Two tensors containing the sampled values for x1 and x2.
        """
        with torch.no_grad():
            # Initialize both x1 and x2 with random noise
            x1_t = torch.randn(*shape, device=device)  # Random noise for x1 (from normal distribution)
            x2_t = self.igso3_sample(shape, device)  # Random noise for x2 (from IGSO(3))

            # Reverse the diffusion process for both x1 and x2
            for t in reversed(range(num_steps)):

                t /= num_steps - 1
                t = torch.full((shape[0],), t, device=device)
                dt = 1 / (num_steps - 1)

                # Predict scores using the score model
                predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t, t)
                print(predicted_score1.mean(), predicted_score2.mean())

                x2_t = rotvec_to_matrix(x2_t)
                
                # Get the current diffusion coefficients
                alpha_t = self.alpha(t).view(-1, 1, 1)
                alpha_t_bar = self.alpha_bar(t).view(-1, 1, 1)
                beta_t = self.beta(t).view(-1, 1, 1)

                # g_t = self.diffusion_coef(t).view(-1, 1, 1).expand_as(x1_t)
                # f_t = self.drift_coef(x1_t, t)
                # z = torch.randn(x1_t.shape, device=device) if t > 0 else torch.zeros(x1_t.shape, device=device)
                # perturb_1 = (f_t - g_t**2 * predicted_score1) * dt + g_t * math.sqrt(dt) * z

                # x1_t -= perturb_1
                # x1_t = x1_t / self.trans_scale
                beta_t = self.beta(t)
                drift1 = self.drift_coef(x1_t, t)
                diffusion_t = self.diffusion_coef(t)
                noise1 = torch.randn_like(x1_t)
                x1_t = x1_t + (drift1 - beta_t * predicted_score1) * dt + diffusion_t * np.sqrt(dt) * noise1

                # Reverse the diffusion step for x2
                predicted_score2 = beta_t * predicted_score2
                epsilon2_rot_mat = rotvec_to_matrix(beta_t * predicted_score2)
                
                # We want to reverse the matrix multiplication: x2_t = R_noise * x2
                # So, we apply the inverse of the rotation matrix
                epsilon2_rot_mat_inv = epsilon2_rot_mat.transpose(-2, -1)  # Inverse of rotation matrix = transpose
                x2_t = torch.matmul(x2_t, epsilon2_rot_mat_inv) # Apply inverse rotation to reverse the noise

                rot_noise = rotvec_to_matrix(torch.sqrt(beta_t) * self.igso3_sample(matrix_to_rotvec(x2_t).shape, x2_t.device))

                x2_t = matrix_to_rotvec(torch.matmul(x2_t, rot_noise))

            x2_t = rotvec_to_matrix(x2_t)

            return compose_se3(x2_t, x1_t / self.trans_scale)  # Return the sampled tensors

    def train(self, data_loader, optimizer, device, epochs=10, num_timesteps=30, project_name="dual_input_diffusion", run_name="Diffusion"):
        wandb.init(project=project_name, name=run_name, config={"epochs": epochs, "learning_rate": optimizer.param_groups[0]['lr']})

        for epoch in range(epochs):
            epoch_loss = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for i, batch in enumerate(pbar):
                    batch = batch.to(device)
                    rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
                    rotations = matrix_to_rotvec(rotations)  # [B, n, 3]
                    translations = batch[:, :, :3, 3]  # [B, n, 3]

                    optimizer.zero_grad()
                    t = torch.randint(0, num_timesteps, (rotations.shape[0],), device=rotations.device) * (1 / (num_timesteps - 1))
                
                    # Compute the loss for x1 and x2
                    loss = self.compute_loss(translations, rotations, t)
                
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    wandb.log({"batch_loss": loss.item()})

                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(data_loader)
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        wandb.finish()
