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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

# Diffuser Class
class DDPM_Continuous_Diff:
    def __init__(self, score_model, beta_start=0.1, beta_end=1.0, trans_scale=1.0, device="cuda", timesteps=30, seq_len=128):
        self.score_model = score_model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trans_scale = trans_scale
        self.device = device
        self.num_timesteps = timesteps
        self.seq_len = seq_len

    def beta(self, t):
        """Continuous beta(t) as a linear schedule."""
        return self.beta_start + t * (self.beta_end - self.beta_start)
    
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

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def conditional_var(self, t, use_torch=True):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def trans_score(self, x_t, x_0, t, use_torch=True):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp

        return (x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)

    def forward_process(self, x1, x2, t):
        """Forward diffusion process with different noise sources for x1 and x2."""
        t = t.unsqueeze(-1).unsqueeze(-1)

        # Sample noise for x1 from normal distribution
        epsilon1 = torch.randn_like(x1)

        # Sample noise for x2 from IGSO(3)
        epsilon2 = self.igso3_sample(x2.shape, x2.device)

        # print(x1.shape, t.shape)
        x1_t = torch.exp(-1/2*self.marginal_b_t(t)) * x1 + \
              torch.randn(t.shape, device=t.device) * torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))
        score_t = self.trans_score(x1_t, x1, t)
        # x1_t = self._unscale(x_t)

        return (x1_t, score_t), (rotvec_to_matrix(x1_t), epsilon2)

    def compute_loss(self, x1, x2, t):
        """Compute the diffusion loss for x1 and x2."""
        B, L, _ = x1.shape
        (x1_t, epsilon1), (x2_t, epsilon2) = self.forward_process(x1 * self.trans_scale, x2, t)

        x2_t_flatten = x2_t.reshape(B, L, 9)

        if self.score_model.name == "Unet":
            x1_t = x1_t.transpose(1,2)
            x2_t_flatten = x2_t_flatten.transpose(1,2)

        # Predict scores using the score model
        predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t_flatten, t)

        if self.score_model.name == "Unet":
            predicted_score1 = predicted_score1.transpose(1,2)
            predicted_score2 = predicted_score2.transpose(1,2)
            x1_t = x1_t.transpose(1,2)
            x2_t_flatten = x2_t_flatten.transpose(1,2)

        x0_1 = self.calc_trans_0(predicted_score1, x1_t, t)
        
        # Loss for each stream
        loss1 = F.l1_loss(predicted_score1, epsilon1)
        loss_origin1 = F.l1_loss(x0_1, x1)

        # print(predicted_score2.shape, epsilon2.shape)
        predicted_score2 = predicted_score2.reshape(B, L, 3, 3)

        # Compute quaternion loss
        # loss2 = rotation_distance_loss(predicted_score2, epsilon2)
        loss2 = 0
        loss_origin2 = 0

        # Average the two losses
        return loss1 + loss_origin1 + loss2 + loss_origin2, loss1, loss2, loss_origin1, loss_origin2

    def sample(self, shape, device, num_steps=30, trans_init=None, rot_init=None):
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
            B, L, _ = shape

            # Initialize both x1 and x2 with random noise
            x1_t = torch.randn(*shape, device=device) if trans_init is None else trans_init  # Random noise for x1 (from normal distribution)
            x2_t = self.igso3_sample(shape, device)  # Random noise for x2 (from IGSO(3))
            x2_t = rotvec_to_matrix(x2_t)
            x2_t = x2_t.reshape(B, L, 9)

            # Reverse the diffusion process for both x1 and x2
            for t in reversed(range(num_steps)):

                t /= num_steps - 1
                t = torch.full((shape[0],), t, device=device)
                t_bc = t.unsqueeze(-1).unsqueeze(-1)
                dt = 1 / (num_steps - 1)

                if self.score_model.name == "Unet":
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.transpose(1,2)

                # Predict scores using the score model
                predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t, t)
                # print(predicted_score1.mean(), predicted_score2.mean())

                if self.score_model.name == "Unet":
                    predicted_score1 = predicted_score1.transpose(1,2)
                    predicted_score2 = predicted_score2.transpose(1,2)
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.transpose(1,2)

                g_t = self.diffusion_coef(t).view(-1, 1, 1).expand_as(x1_t)
                f_t = self.drift_coef(x1_t, t)
                z = torch.randn(x1_t.shape, device=device) if t > 0 else torch.zeros(x1_t.shape, device=device)
                perturb_1 = (f_t - g_t**2 * predicted_score1) * dt + g_t * math.sqrt(dt) * z

                x1_t -= perturb_1
                # x1_t = x1_t / self.trans_scale

            # x2_t = rotvec_to_matrix(x2_t)
            x2_t = x2_t.reshape(B, L, 3, 3)

            return compose_se3(x2_t, x1_t)  # Return the sampled tensors

    def train(self, data_loader, optimizer, device, epochs=10, num_timesteps=30, log_wandb=False, project_name="dual_input_diffusion", run_name="Diffusion"):
        if log_wandb:
            wandb.init(project=project_name, name=run_name, config={"epochs": epochs, "learning_rate": optimizer.param_groups[0]['lr']})
            all_R3_images = []  # To store all scatter plots across epochs
            all_se3_images = []      # To store all SE(3) trajectory plots across epochs
            sample_shape = (1, self.seq_len, 3)
            sample_noise_t = torch.randn(*sample_shape, device=device)
            sample_noise_r = self.igso3_sample(sample_shape, device)
        
        generate_loop = epochs // 5
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_translation_eps_loss = 0.0
            epoch_translation_x0_loss = 0.0
            epoch_rotation_eps_loss = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for i, batch in enumerate(pbar):
                    batch = batch.to(device)
                    rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
                    # rotations = matrix_to_rotvec(rotations)  # [B, n, 3]
                    translations = batch[:, :, :3, 3]  # [B, n, 3]

                    optimizer.zero_grad()
                    t = torch.randint(0, num_timesteps-1, (rotations.shape[0],), device=rotations.device) + 1
                    t = t.to(torch.float32) / num_timesteps
                
                    # Compute the loss for x1 and x2
                    loss, trans_loss, rot_loss, trans_x0_loss, rot_x0_loss = self.compute_loss(translations * self.trans_scale, rotations, t)
                
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_translation_eps_loss += trans_loss.item()
                    epoch_translation_x0_loss += trans_x0_loss.item()
                    # epoch_rotation_eps_loss += rot_loss.item()
                    if log_wandb:
                        wandb.log({"batch_loss": loss.item(), "Translation eps loss": trans_loss.item(), \
                                   "Translation x0 loss": trans_x0_loss.item()})

                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(data_loader)
            avg_translation_eps_loss = epoch_translation_eps_loss / len(data_loader)
            avg_translation_x0_loss = epoch_translation_x0_loss / len(data_loader)
            avg_rotation_eps_loss = epoch_rotation_eps_loss / len(data_loader)
            if log_wandb:
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1,
                           "epoch_translation_eps_loss": avg_translation_eps_loss,
                           "epoch_translation_x0_loss": avg_translation_x0_loss,
                           "epoch_rotation_eps_loss": avg_rotation_eps_loss})

                if (epoch + 1) % generate_loop == 0:
                    generated_se3 = self.sample((1, self.seq_len, 3), device, num_timesteps, trans_init=sample_noise_t, rot_init=sample_noise_r)
                    trajectory = np.array([se3[:, :3, 3].detach().cpu().numpy() for se3 in generated_se3])[0]

                    # Plot scatter
                    R3_fig = plt.figure()
                    ax = R3_fig.add_subplot(111, projection='3d')
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")
                    ax.set_title("Generated R3 Trajectory epoch="+str(epoch+1))
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    R3_img = wandb.Image(R3_fig)
                    plt.close(R3_fig)

                    # Plot SE(3) trajectory with orientations
                    se3_fig = plt.figure(figsize=(12, 8))
                    ax = se3_fig.add_subplot(111, projection='3d')
                    # trajectory = generated_se3[0, :, :3, 3].cpu().numpy()
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue", linewidth=2)
                    
                    # Add quivers
                    step = 5  # Reduce clutter by plotting quivers every 5 points
                    scale = 0.1
                    for i in range(0, len(trajectory) - 1, step):
                        direction = trajectory[i + 1] - trajectory[i]
                        direction = direction / np.linalg.norm(direction)
                        ax.quiver(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2], 
                                  direction[0], direction[1], direction[2], 
                                  length=scale, color='red', alpha=0.6)
                    ax.set_title("Generated SE(3) Trajectory epoch="+str(epoch+1))
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    ax.legend()
                    se3_img = wandb.Image(se3_fig)
                    plt.close(se3_fig)

                    all_R3_images.append(R3_img)
                    all_se3_images.append(se3_img)

            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if log_wandb:
            wandb.log({
                "All R3 Plots": all_R3_images,
                "All SE(3) Plots": all_se3_images
            })
            wandb.finish()
