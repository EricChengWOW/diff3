import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import math
import matplotlib.pyplot as plt

from torch.nn.modules import loss
from torch.optim import Adam
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm
from utils import *

# Diffuser Class
class DDPM_Diff:
    def __init__(self, score_model, beta_start=1e-4, beta_end=0.02, trans_scale=1.0, device="cuda", timesteps=30, seq_len=128):
        self.score_model = score_model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trans_scale = trans_scale
        self.device = device
        self.num_timesteps = timesteps
        self.seq_len = seq_len

        ### Translation Euclidean diffusion scheduler
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

        ### Rotation SO3 diffusion scheduler
        self.beta_t = torch.linspace(beta_start, beta_end, self.num_timesteps, device=device)
        self.alpha_t = 1.0 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)

    '''def forward_process(self, x1, R0, t, trans_init=None, rot_init=None):
        """Forward diffusion process with different noise sources for translation x1 and rotation R0."""
        B, L, _ = x1.shape

        x1 = self.trans_scale * x1

        # Euclidean Translation forward
        epsilon1 = torch.randn_like(x1) if trans_init is None else trans_init

        x1_t = extract(self.q_param1, t, x1.shape) * x1 + extract(self.q_param2, t, x1.shape) * epsilon1

        # SO3 Manifold forward with IGSO3 noise and rotation matrix multiply
        v0 = so3_log_map(R0)
        alpha_bar_t = extract(self.alpha_bar_t, t, v0.shape)
        alpha_bar_t_sqrt = torch.sqrt(alpha_bar_t)
        epsilon2 = torch.randn_like(v0) if trans_init is None else rot_init

        vt = alpha_bar_t_sqrt * v0 + torch.sqrt(1.0 - alpha_bar_t) * so3_log_map(epsilon2)
        Rt = so3_exp_map(vt)

        return (x1_t, epsilon1), (Rt, epsilon2)'''


    def forward_process(self, x1, v0, t, trans_init=None, rot_init=None):
        """
        Forward diffusion process with different noise sources for translation x1 and rotation tangent vector v0.
        
        Args:
            x1 (torch.Tensor): Translation vectors with timestamps [B, L, 4]
            v0 (torch.Tensor): SO(3) tangent vectors with timestamps [B, L, 4] (already log-mapped)
            t (torch.Tensor): Timestep indices
            trans_init (torch.Tensor, optional): Translation noise. Defaults to None.
            rot_init (torch.Tensor, optional): Rotation noise. Defaults to None.
            
        Returns:
            tuple: Tuple of noised translation data and rotation data with their respective noises
        """
        B, L, _ = x1.shape  # [B, L, 4]
        #print(v0.shape)
        
        # Split into spatial coordinates and timestamps
        spatial_coords = x1[:, :, :3]  # [B, L, 3]
        timestamps = x1[:, :, 3:4]     # [B, L, 1]
        
        #spatial_coords = self.trans_scale * spatial_coords

        # Euclidean Translation forward (only for spatial coordinates)
        epsilon1 = torch.randn_like(x1) if trans_init is None else trans_init
        
        spatial_coords_t = extract(self.q_param1, t, spatial_coords.shape) * spatial_coords + \
                           extract(self.q_param2, t, spatial_coords.shape) * epsilon1[:, :, :3]

        # Recombine spatial coordinates with timestamps
        x1_t = torch.cat([spatial_coords_t, timestamps], dim=2)

        # Split rotation tangent vectors and timestamps
        rot_vec = v0[:, :, :3]          # [B, L, 3]
        rot_timestamps = v0[:, :, 3:4]  # [B, L, 1]

        # SO3 Manifold forward (v0 is already a tangent vector)
        alpha_bar_t = extract(self.alpha_bar_t, t, rot_vec.shape)
        alpha_bar_t_sqrt = torch.sqrt(alpha_bar_t)
        
        epsilon2 = torch.randn_like(v0) if rot_init is None else rot_init

        # Apply diffusion directly to tangent vectors
        vt = alpha_bar_t_sqrt * rot_vec + torch.sqrt(1.0 - alpha_bar_t) * epsilon2[:, :, :3]
        
        # Convert to rotation matrix for the output
        # Rt_matrices = so3_exp_map(vt)
        
        # Keep the full tangent vector with timestamp for model input 
        vt_with_time = torch.cat([vt, rot_timestamps], dim=2)

        return (x1_t, epsilon1), (vt_with_time, epsilon2)

    '''def compute_loss(self, x1, x2, t):
        """Compute the diffusion loss for translation and rotation."""
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

        # Translation origin x0
        x0_1 = extract(self.x0_param1, t, x1_t.shape) * (x1_t - extract(self.x0_param2, t, x1_t.shape) * predicted_score1)

        # Loss for translation
        loss1 = F.l1_loss(predicted_score1, epsilon1)
        loss_origin1 = F.l1_loss(x0_1, x1)

        # Rotation origin R0
        v_t = so3_log_map(x2_t)

        # Reconstruct v_0:
        p = extract(self.alpha_bar_t, t, v_t.shape)
        alpha_bar_t_sqrt = torch.sqrt(p)
        v_0_pred = (v_t - torch.sqrt(1 - p) * predicted_score2) / alpha_bar_t_sqrt

        # Approximate x_0 from v_0_pred
        R_0_approx = so3_exp_map(v_0_pred)

        # Compute rotational loss
        R_pred = so3_exp_map(predicted_score2)   # (B,3,3)
        R_true = so3_exp_map(epsilon2)
        loss2 = rotation_distance_loss(R_pred, R_true)
        loss_origin2 = rotation_distance_loss(R_0_approx, x2)

        # Average the 4 losses
        return loss1 + loss_origin1 + loss2 + loss_origin2, loss1, loss2, loss_origin1, loss_origin2'''

    def compute_loss(self, x1, v0, t):
        """
        Compute the diffusion loss for translation and rotation.
        
        Args:
            x1 (torch.Tensor): Translation vectors with timestamps [B, L, 4]
            v0 (torch.Tensor): SO(3) tangent vectors with timestamps [B, L, 4]
            t (torch.Tensor): Diffusion timesteps
            
        Returns:
            tuple: Combined loss and individual loss components
        """
        B, L, _ = x1.shape  # [B, L, 4]
        #print(x1.shape, v0.shape)
        # Forward process with data already in the right format
        (x1_t, epsilon1), (vt_with_time, epsilon2) = self.forward_process(x1, v0, t)
        #print(vt_with_time.shape)
        #print(x1_t.shape, vt_with_time.shape)
        #print(epsilon1.shape)
        # Extract spatial and rotation components
        x1_t_spatial = x1_t[:, :, :3]  # [B, L, 3]
        vt = vt_with_time[:, :, :3]    # [B, L, 3]
        
        # The DoubleUnet expects inputs with shape [B, 4, L]
        # Prepare translation input with timestamps
        x1_t_model = x1_t.transpose(1, 2)  # [B, 4, L]
        
        # Prepare rotation input with timestamps
        vt_model = vt_with_time.transpose(1, 2)  # [B, 4, L]
        
        # Predict scores using the score model
        predicted_score1, predicted_score2 = self.score_model(x1_t_model, vt_model, t)
        # Convert back to [B, L, C] format for loss computation
        predicted_score1_spatial = predicted_score1.transpose(1, 2)  # [B, L, 4]
        predicted_score2_spatial = predicted_score2.transpose(1, 2)  # [B, L, 4]
        
        # Use only the spatial components for loss calculation
        '''predicted_score1_spatial = predicted_score1[:, :, :3]  # [B, L, 3]
        predicted_score2_spatial = predicted_score2[:, :, :3]  # [B, L, 3]'''

        # Translation origin x0
        x0_1 = extract(self.x0_param1, t, x1_t.shape) * (x1_t - extract(self.x0_param2, t, x1_t.shape) * predicted_score1_spatial)

        # Loss for translation (original x1 had timestamps, but we only compare spatial coords)
        loss1 = F.l1_loss(predicted_score1_spatial, epsilon1)
        loss_origin1 = F.l1_loss(x0_1, x1)

        # Rotation tangent vector reconstruction
        p = extract(self.alpha_bar_t, t, vt_with_time.shape)
        alpha_bar_t_sqrt = torch.sqrt(p)
        #print(p.shape, vt.shape, predicted_score2_spatial.shape)
        v_0_pred = (vt_with_time - torch.sqrt(1 - p) * predicted_score2_spatial) / alpha_bar_t_sqrt

        # For loss computation, convert tangent vectors to rotation matrices
        R_0_approx = so3_exp_map(v_0_pred)
        R_original = so3_exp_map(v0)  # Convert original tangent vectors to matrices
        R_pred = so3_exp_map(predicted_score2_spatial)
        R_true = so3_exp_map(epsilon2)
        
        # Compute rotational loss
        loss2 = rotation_distance_loss(R_pred, R_true)
        loss_origin2 = rotation_distance_loss(R_0_approx, R_original)

        # Average the 4 losses
        return loss1 + loss_origin1 + loss2 + loss_origin2, loss1, loss2, loss_origin1, loss_origin2

    def sample(self, shape, device, num_steps=30, trans_init=None, rot_init=None, trans_noise=None, rot_noise=None):
        """
        Sample both x1 and x2 from the reverse diffusion process.

        Args:
            shape (torch.Size): Desired shape of the output tensors (Batch, seq, dim).
            device (torch.device): The device to create the tensor on.
            num_steps (int): The number of diffusion steps to reverse.
            trans_init (torch.tensor): The initial translation noise
            rot_init (torch.tensor): The initial rotation noise
            trans_noise (torch.tensor): The translation noise added for forward process
            rot_noise (torch.tensor): The rotation noise added for forward process

        Returns:
            torch.Tensor: SE3 Tensor
        """
        with torch.no_grad():

            # Initialize both x1 and x2 with random noise
            B, L, _ = shape
            print(shape)
            x1_t = torch.randn(*shape, device=device) if trans_init is None else trans_init # Random noise for x1 (from normal distribution)
            if rot_init is not None:
                x2_t = rot_init
            else:
                v_T = torch.randn(B,L,3, device=device)
                x2_t = so3_exp_map(v_T)

            # Reverse the diffusion process for both x1 and x2
            for t in range(num_steps-1, -1, -1):

                t_tensor = torch.full((shape[0],), t, device=device)

                if self.score_model.name == "Unet":
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.reshape(B, L, 9)
                    x2_t = x2_t.transpose(1,2)
                print(x1_t.shape, x2_t.shape)
                # Predict scores using the score model
                predicted_score1, predicted_score2 = self.score_model(x1_t, x2_t, t_tensor)

                if self.score_model.name == "Unet":
                    predicted_score1 = predicted_score1.transpose(1,2)
                    predicted_score2 = predicted_score2.transpose(1,2)
                    x1_t = x1_t.transpose(1,2)
                    x2_t = x2_t.transpose(1,2)
                    x2_t = x2_t.reshape(B, L, 3, 3)

                if t > 0:
                    if trans_noise is not None:
                        noise = trans_noise[t]
                    else:
                        noise = torch.randn_like(x1_t).to(device)
                else:
                    noise = torch.zeros_like(x1_t).to(device)

                x0 = extract(self.x0_param1, t_tensor, x1_t.shape) * (x1_t - extract(self.x0_param2, t_tensor, x1_t.shape) * predicted_score1)
                x0 = torch.clamp(x0, min=-1, max=1)

                mean = extract(self.mean_param1, t_tensor, x1_t.shape) * x1_t + extract(self.mean_param2, t_tensor, x1_t.shape) * x0
                sigma = extract(self.sigma, t_tensor, x1_t.shape)

                x1_t = mean + sigma * noise

                ### SO3

                # Compute v_t = log(x_t)
                v_t = so3_log_map(x2_t)  # (B,3)

                # Reconstruct v_0:
                alpha_bar_t_sqrt = torch.sqrt(self.alpha_bar_t[t])
                v_0_pred = (v_t - torch.sqrt(1 - self.alpha_bar_t[t]) * predicted_score2) / alpha_bar_t_sqrt

                # Approximate x_0 from v_0_pred
                x_0_approx = so3_exp_map(v_0_pred)

                # Compute mu_t(x_t, x_0):
                # µ_t = λ( (√α_{t-1} β_t / (1−α_bar_t)) , x_0_approx ) λ( (√(α_t(1−α_bar_{t-1})) / (1−α_bar_t)), x_t )
                # Handle t=0 case: x_{-1} doesn't exist
                if t > 0:
                    alpha_t_ = self.alpha_t[t]
                    beta_t_ = self.beta_t[t]
                    alpha_t_minus = self.alpha_t[t-1]
                    alpha_bar_t_minus = self.alpha_bar_t[t-1]

                    c1 = (torch.sqrt(alpha_bar_t_minus)*beta_t_) / (1 - self.alpha_bar_t[t])
                    c2 = (torch.sqrt(alpha_t_*(1 - alpha_bar_t_minus))) / (1 - self.alpha_bar_t[t])

                    def lambda_map(gamma, X):
                        vX = so3_log_map(X)
                        return so3_exp_map(gamma * vX)

                    part1 = lambda_map(c1, x_0_approx)
                    part2 = lambda_map(c2, x2_t)

                    mu_t = torch.matmul(part1, part2)  # (B,L,3,3)

                    if rot_noise is not None:
                        epsilon = rot_noise[t]
                    else:
                        epsilon = torch.randn_like(v_t)

                    v_mu = so3_log_map(mu_t)
                    v_t_minus = v_mu + torch.sqrt(beta_t_)*epsilon
                    x2_t = so3_exp_map(v_t_minus)
                else:
                    # t=0
                    x2_t = x_0_approx
                #all_samples.append(compose_se3(x2_t, torch.clamp(x1_t, min=-1, max=1) / self.trans_scale))

            x1_t = torch.clamp(x1_t, min=-1, max=1)

            #return all_samples  # Return the sampled tensors
            return compose_se3(x2_t, x1_t / self.trans_scale)

    def sample(self, shape, device, num_steps=30, trans_init=None, rot_init=None, trans_noise=None, rot_noise=None, timestamps=None):
        """
        Sample both x1 and x2 from the reverse diffusion process with timestamps.
        
        Args:
            shape (torch.Size): Desired shape of the output tensors (Batch, seq, dim).
            device (torch.device): The device to create the tensor on.
            num_steps (int): The number of diffusion steps to reverse.
            trans_init (torch.Tensor, optional): Initial translation noise. Defaults to None.
            rot_init (torch.Tensor, optional): Initial rotation noise. Defaults to None.
            trans_noise (torch.Tensor, optional): Translation noise for forward process. Defaults to None.
            rot_noise (torch.Tensor, optional): Rotation noise for forward process. Defaults to None.
            timestamps (torch.Tensor, optional): Custom timestamps. Defaults to None.
            
        Returns:
            list: List of SE(3) transformation matrices
        """
        with torch.no_grad():
            all_samples = []

            # Initialize both x1 and x2 with random noise
            B, L, D = shape  # D should be 3 for spatial dimensions
            
            # Generate sequential timestamps if not provided
            if timestamps is None:
                timestamps = torch.arange(1, L + 1, device=device).expand(B, L).float().unsqueeze(-1)  # [B, L, 1]
            elif timestamps.dim() == 2:
                timestamps = timestamps.unsqueeze(-1)  # Ensure [B, L, 1] shape
                
            # Initialize translation noise with timestamps
            x1_t_spatial = torch.randn(B, L, D, device=device) if trans_init is None else trans_init
            x1_t = torch.cat([x1_t_spatial, timestamps], dim=2)  # [B, L, 4]
            
            # Initialize rotation as tangent vectors with timestamps
            if rot_init is not None:
                v_t = rot_init[:, :, :3]  # Extract spatial part if timestamps included
            else:
                v_t = torch.randn(B, L, 3, device=device)
                
            # Create rotation matrices from tangent vectors
            x2_t = so3_exp_map(v_t)  # [B, L, 3, 3]
            
            # Create full tangent vectors with timestamps for model input
            v_t_with_time = torch.cat([v_t, timestamps], dim=2)  # [B, L, 4]

            # Reverse the diffusion process
            for t in range(self.num_timesteps-1, self.num_timesteps-num_steps-1, -1):
                t_tensor = torch.full((shape[0],), t, device=device)
                
                # Prepare model inputs for DoubleUnet which expects [B, 4, L] format
                # Convert translation with timestamps to channels-first format
                x1_t_model = x1_t.transpose(1, 2)  # [B, 4, L]
                
                # Convert rotation tangent vectors with timestamps to channels-first format
                v_t_model = v_t_with_time.transpose(1, 2)  # [B, 4, L]

                # Predict scores using the score model
                predicted_score1, predicted_score2 = self.score_model(x1_t_model, v_t_model, t_tensor)
                
                # Convert from channels-first to sequence-first format
                predicted_score1 = predicted_score1.transpose(1, 2)  # [B, L, 4]
                predicted_score2 = predicted_score2.transpose(1, 2)  # [B, L, 4]
                
                # Extract spatial components (first 3 dimensions) for diffusion
                predicted_score1_spatial = predicted_score1[:, :, :3]  # [B, L, 3]
                predicted_score2_spatial = predicted_score2[:, :, :3]  # [B, L, 3]

                # Translation update
                x1_t_spatial = x1_t[:, :, :3]  # Extract spatial part [B, L, 3]
                
                if t > 0:
                    if trans_noise is not None:
                        noise = trans_noise[t]
                    else:
                        noise = torch.randn_like(x1_t_spatial).to(device)
                else:
                    noise = torch.zeros_like(x1_t_spatial).to(device)

                x0 = extract(self.x0_param1, t_tensor, x1_t_spatial.shape) * (x1_t_spatial - extract(self.x0_param2, t_tensor, x1_t_spatial.shape) * predicted_score1_spatial)
                x0 = torch.clamp(x0, min=-1, max=1)

                mean = extract(self.mean_param1, t_tensor, x1_t_spatial.shape) * x1_t_spatial + extract(self.mean_param2, t_tensor, x1_t_spatial.shape) * x0
                sigma = extract(self.sigma, t_tensor, x1_t_spatial.shape)

                x1_t_spatial_new = mean + sigma * noise
                
                # Update translation with timestamps
                x1_t = torch.cat([x1_t_spatial_new, timestamps], dim=2)  # [B, L, 4]

                # Rotation update - first get current tangent vectors
                v_t = v_t_with_time[:, :, :3]  # Extract spatial part [B, L, 3]
                
                # Reconstruct v_0
                alpha_bar_t_sqrt = torch.sqrt(self.alpha_bar_t[t])
                v_0_pred = (v_t - torch.sqrt(1 - self.alpha_bar_t[t]) * predicted_score2_spatial) / alpha_bar_t_sqrt

                # Convert to rotation matrix
                x_0_approx = so3_exp_map(v_0_pred)  # [B, L, 3, 3]

                if t > 0:
                    alpha_t_ = self.alpha_t[t]
                    beta_t_ = self.beta_t[t]
                    alpha_t_minus = self.alpha_t[t-1]
                    alpha_bar_t_minus = self.alpha_bar_t[t-1]

                    c1 = (torch.sqrt(alpha_bar_t_minus)*beta_t_) / (1 - self.alpha_bar_t[t])
                    c2 = (torch.sqrt(alpha_t_*(1 - alpha_bar_t_minus))) / (1 - self.alpha_bar_t[t])

                    def lambda_map(gamma, X):
                        vX = so3_log_map(X)
                        return so3_exp_map(gamma * vX)

                    part1 = lambda_map(c1, x_0_approx)
                    part2 = lambda_map(c2, x2_t)

                    mu_t = torch.matmul(part1, part2)  # [B, L, 3, 3]

                    if rot_noise is not None:
                        epsilon = rot_noise[t]
                    else:
                        epsilon = torch.randn_like(v_t)

                    v_mu = so3_log_map(mu_t)
                    v_t_new = v_mu + torch.sqrt(beta_t_)*epsilon
                    x2_t = so3_exp_map(v_t_new)  # [B, L, 3, 3]
                else:
                    # t=0
                    x2_t = x_0_approx
                
                # Create and store SE(3) transformation
                se3_transform = compose_se3(
                    x2_t, 
                    torch.clamp(x1_t_spatial_new, min=-1, max=1) / self.trans_scale
                )
                all_samples.append(se3_transform)

            return all_samples

    def sample(self, shape, device, num_steps=30, trans_init=None, rot_init=None, trans_noise=None, rot_noise=None, timestamps=None):
      """
      Sample both x1 and x2 from the reverse diffusion process with timestamps.
      
      Args:
          shape (torch.Size): Desired shape of the output tensors (Batch, seq, dim).
                            Should be [B, L, 3] (will be augmented to [B, L, 4] with timestamps).
          device (torch.device): The device to create the tensor on.
          num_steps (int): The number of diffusion steps to reverse.
          trans_init (torch.Tensor, optional): Initial translation noise. Defaults to None.
          rot_init (torch.Tensor, optional): Initial rotation noise. Defaults to None.
          trans_noise (torch.Tensor, optional): Translation noise for forward process. Defaults to None.
          rot_noise (torch.Tensor, optional): Rotation noise for forward process. Defaults to None.
          timestamps (torch.Tensor, optional): Custom timestamps. Defaults to None.
          
      Returns:
          list: List of SE(3) transformation matrices
      """
      with torch.no_grad():
          all_samples = []

          # Initialize both x1 and x2 with random noise
          B, L, D = shape  # D should be 3 for spatial dimensions
          
          # Generate sequential timestamps if not provided
          if timestamps is None:
              timestamps = torch.arange(1, L + 1, device=device).expand(B, L).float().unsqueeze(-1)  # [B, L, 1]
          elif timestamps.dim() == 2:
              timestamps = timestamps.unsqueeze(-1)  # Ensure [B, L, 1] shape
              
          # Initialize translation noise with timestamps
          if trans_init is None:
              # Generate spatial noise only
              x1_t_spatial = torch.randn(B, L, 3, device=device)
          else:
              # If trans_init already has timestamps (shape [B, L, 4])
              if trans_init.shape[-1] == 4:
                  x1_t_spatial = trans_init[:, :, :3]  # Extract spatial components
                  # Use provided timestamps if available
                  if timestamps is None:
                      timestamps = trans_init[:, :, 3:4]
              else:
                  # Just spatial components
                  x1_t_spatial = trans_init
          
          # Combine spatial components with timestamps
          x1_t = torch.cat([x1_t_spatial, timestamps], dim=2)  # [B, L, 4]
          
          # Initialize rotation as tangent vectors with timestamps
          if rot_init is not None:
              # If rot_init already has timestamps (shape [B, L, 4])
              if rot_init.shape[-1] == 4:
                  v_t = rot_init[:, :, :3]  # Extract spatial part
                  # Use provided timestamps if not already set
                  if timestamps is None:
                      timestamps = rot_init[:, :, 3:4]
              else:
                  # Just spatial components
                  v_t = rot_init
          else:
              v_t = torch.randn(B, L, 3, device=device)
              
          # Create rotation matrices from tangent vectors for visualization
          x2_t = so3_exp_map(v_t)  # [B, L, 3, 3]
          
          # Create full tangent vectors with timestamps for model input
          v_t_with_time = torch.cat([v_t, timestamps], dim=2)  # [B, L, 4]

          # Reverse the diffusion process
          for t in range(num_steps-1, -1, -1):
              t_tensor = torch.full((shape[0],), t, device=device)
              
              # Prepare model inputs for DoubleUnet which expects [B, 4, L] format
              # Convert translation with timestamps to channels-first format
              x1_t_model = x1_t.transpose(1, 2)  # [B, 4, L]
              
              # Convert rotation tangent vectors with timestamps to channels-first format
              v_t_model = v_t_with_time.transpose(1, 2)  # [B, 4, L]

              # Predict scores using the score model
              predicted_score1, predicted_score2 = self.score_model(x1_t_model, v_t_model, t_tensor)
              
              # Convert from channels-first to sequence-first format
              predicted_score1 = predicted_score1.transpose(1, 2)  # [B, L, 4]
              predicted_score2 = predicted_score2.transpose(1, 2)  # [B, L, 4]
              
              # Extract spatial components (first 3 dimensions) for diffusion
              predicted_score1_spatial = predicted_score1[:, :, :3]  # [B, L, 3]
              predicted_score2_spatial = predicted_score2[:, :, :3]  # [B, L, 3]

              # Translation update
              x1_t_spatial = x1_t[:, :, :3]  # Extract spatial part [B, L, 3]
              
              if t > 0:
                  if trans_noise is not None:
                      noise = trans_noise[t]
                  else:
                      noise = torch.randn_like(x1_t_spatial).to(device)
              else:
                  noise = torch.zeros_like(x1_t_spatial).to(device)

              x0 = extract(self.x0_param1, t_tensor, x1_t_spatial.shape) * (x1_t_spatial - extract(self.x0_param2, t_tensor, x1_t_spatial.shape) * predicted_score1_spatial)
              x0 = torch.clamp(x0, min=-1, max=1)

              mean = extract(self.mean_param1, t_tensor, x1_t_spatial.shape) * x1_t_spatial + extract(self.mean_param2, t_tensor, x1_t_spatial.shape) * x0
              sigma = extract(self.sigma, t_tensor, x1_t_spatial.shape)

              x1_t_spatial_new = mean + sigma * noise
              
              # Update translation with timestamps - preserve the original timestamps
              x1_t = torch.cat([x1_t_spatial_new, timestamps], dim=2)  # [B, L, 4]

              # Rotation update - first get current tangent vectors
              v_t = v_t_with_time[:, :, :3]  # Extract spatial part [B, L, 3]
              
              # Reconstruct v_0
              alpha_bar_t_sqrt = torch.sqrt(self.alpha_bar_t[t])
              v_0_pred = (v_t - torch.sqrt(1 - self.alpha_bar_t[t]) * predicted_score2_spatial) / alpha_bar_t_sqrt

              # Convert to rotation matrix
              x_0_approx = so3_exp_map(v_0_pred)  # [B, L, 3, 3]

              if t > 0:
                  alpha_t_ = self.alpha_t[t]
                  beta_t_ = self.beta_t[t]
                  alpha_t_minus = self.alpha_t[t-1]
                  alpha_bar_t_minus = self.alpha_bar_t[t-1]

                  c1 = (torch.sqrt(alpha_bar_t_minus)*beta_t_) / (1 - self.alpha_bar_t[t])
                  c2 = (torch.sqrt(alpha_t_*(1 - alpha_bar_t_minus))) / (1 - self.alpha_bar_t[t])

                  def lambda_map(gamma, X):
                      vX = so3_log_map(X)
                      return so3_exp_map(gamma * vX)

                  part1 = lambda_map(c1, x_0_approx)
                  part2 = lambda_map(c2, x2_t)

                  mu_t = torch.matmul(part1, part2)  # [B, L, 3, 3]

                  if rot_noise is not None:
                      epsilon = rot_noise[t]
                  else:
                      epsilon = torch.randn_like(v_t)

                  v_mu = so3_log_map(mu_t)
                  v_t_minus = v_mu + torch.sqrt(beta_t_)*epsilon
                  v_t = v_t_minus  # Update tangent vector
                  x2_t = so3_exp_map(v_t_minus)  # Update rotation matrix [B, L, 3, 3]
              else:
                  # t=0
                  v_t = v_0_pred  # Update tangent vector
                  x2_t = x_0_approx  # Update rotation matrix
              
              # Update tangent vector with timestamps
              v_t_with_time = torch.cat([v_t, timestamps], dim=2)  # [B, L, 4]
              
              # Create and store SE(3) transformation
              se3_transform = compose_se3(
                  x2_t, 
                  x1_t_spatial_new / self.trans_scale
              )
              all_samples.append(se3_transform)
          return compose_se3(x2_t, x1_t_spatial_new / self.trans_scale)
          #return all_samples
    '''def train(self, data_loader, optimizer, device, epochs=10, num_timesteps=30, log_wandb=False, project_name="dual_input_diffusion", run_name="Diffusion"):
        if log_wandb:
            wandb.init(project=project_name, name=run_name, config={"epochs": epochs, "learning_rate": optimizer.param_groups[0]['lr']})
            all_R3_images = []  # To store all scatter plots across epochs
            all_se3_images = []      # To store all SE(3) trajectory plots across epochs
            sample_shape = (1, self.seq_len, 3)
            sample_noise_t = torch.randn(*sample_shape, device=device)

            v_T = torch.randn(*sample_shape, device=device)
            sample_noise_r = so3_exp_map(v_T)
        
        generate_loop = epochs // 5
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_translation_eps_loss = 0.0
            epoch_translation_x0_loss = 0.0
            epoch_rotation_eps_loss = 0.0
            epoch_rotation_R0_loss = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for i, batch in enumerate(pbar):
                    batch = batch.to(device)
                    rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
                    translations = batch[:, :, :3, 3]  # [B, n, 3]

                    optimizer.zero_grad()
                    t = torch.randint(0, num_timesteps-1, (rotations.shape[0],), device=rotations.device)
                
                    # Compute the loss for x1 and x2
                    loss, trans_loss, rot_loss, trans_x0_loss, rot_x0_loss = self.compute_loss(translations * self.trans_scale, rotations, t)
                
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_translation_eps_loss += trans_loss.item()
                    epoch_translation_x0_loss += trans_x0_loss.item()
                    epoch_rotation_eps_loss += rot_loss.item()
                    epoch_rotation_R0_loss += rot_x0_loss.item()
                    if log_wandb:
                        wandb.log({"batch_loss": loss.item(), "Translation eps loss": trans_loss.item(), \
                                   "Rotation eps loss": rot_loss.item(), 
                                   "Translation x0 loss": trans_x0_loss.item(),
                                   "Rotation R0 loss": rot_x0_loss.item()})

                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(data_loader)
            avg_translation_eps_loss = epoch_translation_eps_loss / len(data_loader)
            avg_translation_x0_loss = epoch_translation_x0_loss / len(data_loader)
            avg_rotation_eps_loss = epoch_rotation_eps_loss / len(data_loader)
            avg_rotation_R0_loss = epoch_rotation_R0_loss / len(data_loader)
            if log_wandb:
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1,
                           "epoch_translation_eps_loss": avg_translation_eps_loss,
                           "epoch_translation_x0_loss": avg_translation_x0_loss,
                           "epoch_rotation_eps_loss": avg_rotation_eps_loss,
                           "epoch_rotation_R0_loss": avg_rotation_R0_loss})

                # Sample in the middle of training
                if (epoch + 1) % generate_loop == 0:
                    generated_se3 = self.sample((1, self.seq_len, 3), device, num_timesteps, trans_init=sample_noise_t, rot_init=sample_noise_r)
                    
                    trajectory = np.array([se3[:, :3, 3].detach().cpu().numpy() for se3 in generated_se3])[0]
                    rotations = np.array([se3[:, :3, :3].detach().cpu().numpy() for se3 in generated_se3])[0]

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
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue", linewidth=2)
                    
                    # Add quivers
                    step = 5
                    scale = 0.05
                    for i in range(0, len(trajectory) - 1, step):
                        point = trajectory[i]

                        R = rotations

                        x_axis = R[:, 0]
                        y_axis = R[:, 1]
                        z_axis = R[:, 2]

                        ax.quiver(point[0], point[1], point[2],
                                  x_axis[0], x_axis[1], x_axis[2],
                                  length=scale, color='r', linewidth=1.5, alpha=0.6)

                        ax.quiver(point[0], point[1], point[2],
                                  y_axis[0], y_axis[1], y_axis[2],
                                  length=scale, color='g', linewidth=1.5, alpha=0.6)

                        ax.quiver(point[0], point[1], point[2],
                                  z_axis[0], z_axis[1], z_axis[2],
                                  length=scale, color='b', linewidth=1.5, alpha=0.6)

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
            wandb.finish()'''

    def train(self, data_loader, optimizer, device, epochs=10, num_timesteps=30, log_wandb=False, project_name="dual_input_diffusion", run_name="Diffusion"):
        if log_wandb:
            wandb.init(project=project_name, name=run_name, config={"epochs": epochs, "learning_rate": optimizer.param_groups[0]['lr']})
            all_R3_images = []
            all_se3_images = []
            
            # Create sample noise for visualization
            sample_shape = (1, self.seq_len, 4)
            sample_noise_t = torch.randn(*sample_shape, device=device)
            sample_noise_r = torch.randn(*sample_shape, device=device)
            
            # Generate sequential timestamps for visualization
            sample_timestamps = (torch.arange(1, self.seq_len + 1, device=device)/(self.seq_len+1)).float().unsqueeze(0)  # [1, seq_len]
        
        #generate_loop = epochs // 5
        generate_loop = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_translation_eps_loss = 0.0
            epoch_translation_x0_loss = 0.0
            epoch_rotation_eps_loss = 0.0
            epoch_rotation_R0_loss = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for i, batch_data in enumerate(pbar):
                    # Handle different input formats (tuple vs tensor)
                    if len(batch_data) == 2:
                        # Tuple input format (se3, timestamps)
                        batch, timestamps = batch_data
                        #print(batch)
                        batch = batch.to(device)
                        timestamps = timestamps.to(device)
                    else:
                        # Standard input format (just se3 with timestamps)
                        print(len(batch_data))
                        batch = batch_data.to(device)
                        # Extract or generate timestamps
                        timestamps = torch.arange(1, batch.size(1) + 1, device=device).expand(batch.size(0), -1)
                    
                    # Extract rotations and translations
                    rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
                    rotations = so3_log_map(rotations) # [B, n, 3]
                    translations = batch[:, :, :3, 3]  # [B, n, 3]
                    
                    # Reshape timestamps and concatenate with translations
                    timestamps_reshaped = timestamps.unsqueeze(-1)  # [B, n, 1]
                    translations_with_time = torch.cat([translations, timestamps_reshaped], dim=2)  # [B, n, 4]
                    rotations_with_time = torch.cat([rotations, timestamps_reshaped], dim=2)  # [B, n, 4]

                    optimizer.zero_grad()
                    t = torch.randint(0, num_timesteps-1, (rotations.shape[0],), device=rotations.device)
                
                    # Compute the loss
                    loss, trans_loss, rot_loss, trans_x0_loss, rot_x0_loss = self.compute_loss(
                        translations_with_time, 
                        rotations_with_time, 
                        t
                    )
                
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_translation_eps_loss += trans_loss.item()
                    epoch_translation_x0_loss += trans_x0_loss.item()
                    epoch_rotation_eps_loss += rot_loss.item()
                    epoch_rotation_R0_loss += rot_x0_loss.item()
                    
                    if log_wandb:
                        wandb.log({
                            "batch_loss": loss.item(), 
                            "Translation eps loss": trans_loss.item(),
                            "Rotation eps loss": rot_loss.item(), 
                            "Translation x0 loss": trans_x0_loss.item(),
                            "Rotation R0 loss": rot_x0_loss.item()
                        })

                    pbar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(data_loader)
            avg_translation_eps_loss = epoch_translation_eps_loss / len(data_loader)
            avg_translation_x0_loss = epoch_translation_x0_loss / len(data_loader)
            avg_rotation_eps_loss = epoch_rotation_eps_loss / len(data_loader)
            avg_rotation_R0_loss = epoch_rotation_R0_loss / len(data_loader)
            
            if log_wandb:
                wandb.log({
                    "epoch_loss": avg_loss, 
                    "epoch": epoch + 1,
                    "epoch_translation_eps_loss": avg_translation_eps_loss,
                    "epoch_translation_x0_loss": avg_translation_x0_loss,
                    "epoch_rotation_eps_loss": avg_rotation_eps_loss,
                    "epoch_rotation_R0_loss": avg_rotation_R0_loss
                })

                # Sample in the middle of training
                if (epoch + 1) % generate_loop == 0:
                    generated_se3 = self.sample((1, self.seq_len, 4), device, num_timesteps, trans_init=sample_noise_t, rot_init=sample_noise_r, timestamps=sample_timestamps)
                    #print("finish sampling")
                    print(generated_se3[0].shape)
                    #trajectory = np.array([generated_se3[0][:, :, :3, 3].detach().cpu().numpy()])
                    trajectory = np.array([se3[:, :3, 3].detach().cpu().numpy() for se3 in generated_se3])[0]
                    rotations = np.array([se3[:, :3, :3].detach().cpu().numpy() for se3 in generated_se3])[0]
                    #rotations = np.array([generated_se3[0][:, :, :3, :3].detach().cpu().numpy()])
                    print(trajectory.shape)
                    R3_fig = plt.figure()
                    ax = R3_fig.add_subplot(111, projection='3d')
                    #ax.plot(trajectory[:,:, :, 0], trajectory[:,:, :, 1], trajectory[:,:, :, 2], label="Trajectory", color="blue")
                    ax.plot(trajectory[:, 0], trajectory[ :, 1], trajectory[ :, 2], label="Trajectory", color="blue")
                    ax.set_title("Generated R3 Trajectory epoch="+str(epoch+1))
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")
                    R3_img = wandb.Image(R3_fig)
                    print(R3_img)
                    plt.close(R3_fig)

                    # Plot SE(3) trajectory with orientations
                    se3_fig = plt.figure(figsize=(12, 8))
                    ax = se3_fig.add_subplot(111, projection='3d')
                    ax.plot(trajectory[:, 0], trajectory[ :, 1], trajectory[ :, 2], label="Trajectory", color="blue")
                    
                    # Add quivers
                    step = 5
                    scale = 0.05
                    for i in range(0, len(trajectory) - 1, step):
                        point = trajectory[i]

                        R = rotations

                        x_axis = R[:, 0]
                        y_axis = R[:, 1]
                        z_axis = R[:, 2]

                        ax.quiver(point[0], point[1], point[2],
                                  x_axis[0], x_axis[1], x_axis[2],
                                  length=scale, color='r', linewidth=1.5, alpha=0.6)

                        ax.quiver(point[0], point[1], point[2],
                                  y_axis[0], y_axis[1], y_axis[2],
                                  length=scale, color='g', linewidth=1.5, alpha=0.6)

                        ax.quiver(point[0], point[1], point[2],
                                  z_axis[0], z_axis[1], z_axis[2],
                                  length=scale, color='b', linewidth=1.5, alpha=0.6)

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
