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

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: A tensor contains the values of all time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)

# Diffuser Class
class DDPM_Diff:
    def __init__(self, score_model, beta_start=0.1, beta_end=1.0, trans_scale=1.0, device="cuda", timesteps=30):
        self.score_model = score_model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trans_scale = trans_scale
        self.device = device
        self.num_timesteps = timesteps

        def f(t):
            s = 0.008
            return torch.square(torch.cos(torch.Tensor([((t/self.num_timesteps + s) /(1+s)) * (torch.pi / 2)])))

        a_bar_0 = torch.Tensor([1]).to(self.device)
        self.a = torch.cat((a_bar_0, cosine_schedule(self.num_timesteps).to(self.device)))

        self.a_bars = torch.cumprod(self.a, dim=0)
        self.a[0] = f(0) / f(self.num_timesteps)

        self.x0_param1 = 1 / torch.sqrt(self.a_bars[1:])
        self.x0_param2 = torch.sqrt(1 - self.a_bars[1:])

        self.sigma = torch.sqrt(torch.Tensor([(1 - self.a_bars[t-1]) * (1-self.a[t]) / (1 - self.a_bars[t]) for t in range(1,self.num_timesteps + 1)]).to(self.device))

        self.mean_param1 = torch.Tensor([torch.sqrt(self.a[t]) * (1-self.a_bars[t-1]) / (1-self.a_bars[t]) for t in range(1,self.num_timesteps + 1)]).to(self.device)
        self.mean_param2 = torch.Tensor([torch.sqrt(self.a_bars[t-1]) * (1-self.a[t]) / (1-self.a_bars[t]) for t in range(1,self.num_timesteps + 1)]).to(self.device)
        self.a_bars = self.a_bars[1:]
        self.a = self.a[1:]

        self.q_param1 = torch.sqrt(self.a_bars)
        self.q_param2 = torch.sqrt(1 - self.a_bars)

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

    def forward_process(self, x1, x2, t, trans_init=None, rot_init=None):
        """Forward diffusion process with different noise sources for x1 and x2."""
        # Sample noise for x1 from normal distribution
        epsilon1 = torch.randn_like(x1) if trans_init is None else trans_init

        # Sample noise for x2 from IGSO(3)
        epsilon2 = self.igso3_sample(x2.shape, x2.device) if rot_init is None else rot_init

        x1_t = extract(self.q_param1, t, x1.shape) * x1 + extract(self.q_param2, t, x1.shape) * epsilon1

        x2_scaled = extract(self.q_param1, t, x1.shape) * x2  # Scaling input rotation vector
        epsilon2_scaled = extract(self.q_param1, t, x1.shape) * epsilon2  # Scaling noise

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
            x1_t = x1_t.transpose(1,2)
            x2_t = x2_t.transpose(1,2)

        x0_1 = extract(self.x0_param1, t, x1_t.shape) * (x1_t - extract(self.x0_param2, t, x1_t.shape) * predicted_score1)
        
        # Loss for each stream
        loss1 = F.l1_loss(predicted_score1, epsilon1)
        loss_origin1 = F.l1_loss(x0_1, x1)
        loss2 = F.l1_loss(predicted_score2, epsilon2)
        loss_origin2 = 0

        # Average the two losses
        return loss1 + loss_origin1 + loss2, loss1, loss2, loss_origin1, loss_origin2

    def sample(self, shape, device, num_steps=30, trans_init=None, rot_init=None):
        """
        Sample both x1 and x2 from the reverse diffusion process.

        Args:
            shape (torch.Size): Desired shape of the output tensors (Batch, seq, dim).
            device (torch.device): The device to create the tensor on.
            num_steps (int): The number of diffusion steps to reverse.

        Returns:
            torch.Tensor, torch.Tensor: Two tensors containing the sampled values for x1 and x2.
        """
        with torch.no_grad():
            # Initialize both x1 and x2 with random noise
            x1_t = torch.randn(*shape, device=device) if trans_init is None else trans_init # Random noise for x1 (from normal distribution)
            x2_t = self.igso3_sample(shape, device) if rot_init is None else rot_init # Random noise for x2 (from IGSO(3))

            # Reverse the diffusion process for both x1 and x2
            for t in range(num_steps-1, -1, -1):

                t_tensor = torch.full((shape[0],), t, device=device)

                if self.score_model.name == "Unet":
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.transpose(1,2)

                # Predict scores using the score model
                predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t, t_tensor)

                if self.score_model.name == "Unet":
                    predicted_score1 = predicted_score1.transpose(1,2)
                    predicted_score2 = predicted_score2.transpose(1,2)
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.transpose(1,2)

                if t > 0:
                    noise = torch.randn_like(x1_t).to(device)
                else:
                    noise = torch.zeros_like(x1_t).to(device)

                x0 = extract(self.x0_param1, t_tensor, x1_t.shape) * (x1_t - extract(self.x0_param2, t_tensor, x1_t.shape) * predicted_score1)
                x0 = torch.clamp(x0, min=-1, max=1)

                mean = extract(self.mean_param1, t_tensor, x1_t.shape) * x1_t + extract(self.mean_param2, t_tensor, x1_t.shape) * x0
                sigma = extract(self.sigma, t_tensor, x1_t.shape)

                x1_t = mean + sigma * noise

            x2_t = rotvec_to_matrix(x2_t)
            x1_t = torch.clamp(x1_t, min=-1, max=1)

            return compose_se3(x2_t, x1_t / self.trans_scale)  # Return the sampled tensors

    def train(self, data_loader, optimizer, device, epochs=10, num_timesteps=30, log_wandb=False, project_name="dual_input_diffusion", run_name="Diffusion"):
        if log_wandb:
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
                    t = torch.randint(0, num_timesteps-1, (rotations.shape[0],), device=rotations.device)
                
                    # Compute the loss for x1 and x2
                    loss, trans_loss, rot_loss, trans_x0_loss, rot_x0_loss = self.compute_loss(translations * self.trans_scale, rotations, t)
                
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    if log_wandb:
                        wandb.log({"batch_loss": loss.item(), "Translation eps loss": trans_loss.item(), \
                                   "Rotation eps loss": rot_loss.item(), "Translation x0 loss": trans_x0_loss.item()})

                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(data_loader)
            if log_wandb:
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        wandb.finish()
