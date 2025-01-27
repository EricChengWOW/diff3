#from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader, random_split

class MLPDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLPDiffusionModel, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim + 1, hidden_dim))  # +1 for timestep embedding
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))  # Predict noise of the same dimension as input

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t = t.unsqueeze(-1)  # Expand timestep to match batch size (batch_size, 1)
        x_t = torch.cat([x, t], dim=-1)  # Concatenate latent vector and timestep
        return self.mlp(x_t)

def get_signature_size(D, N):
    if D == 1:
        return N + 1
    return (D**(N + 1) - 1) // (D - 1)

def extract_signature_levels(signature, D, N):
    levels = []
    start = 1  # Skip the zeroth term which is always 1
    for k in range(1, N + 1):
        num_terms = D**k
        end = start + num_terms
        level_k = signature[:, start:end]
        levels.append(level_k)
        start = end
    return levels

def normalize_signature(signatures):
    return F.normalize(signatures, p=2, dim=1)

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer-based Noise Estimation Model
class TransformerDDPMNoiseEstimator(nn.Module):
    def __init__(self, D, N, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        """
        Initializes the Transformer-based DDPM Noise Estimator for Path Signatures.
        
        Args:
            D (int): Dimension of the path.
            N (int): Truncation depth of the signature.
            d_model (int): Dimension of the transformer model.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network in transformer.
            dropout (float): Dropout rate.
        """
        super(TransformerDDPMNoiseEstimator, self).__init__()
        self.D = D
        self.N = N
        self.signature_size = get_signature_size(D, N)
        self.seq_length = N  # Each level is a token in the sequence

        # Create a ModuleList of Linear projections for each level
        self.input_proj_list = nn.ModuleList([
            nn.Linear(D**k, d_model) for k in range(1, N + 1)
        ])  # One projection per level

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=N)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Time Embedding (e.g., sinusoidal or learned embeddings)
        self.time_embedding = nn.Embedding(1000, d_model)  # Assuming T=1000

        # Final MLP to estimate noise
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.signature_size)
        )

    def forward(self, signature, t):
        """
        Forward pass of the Transformer DDPM Noise Estimator.
        
        Args:
            signature: Tensor of shape (batch_size, signature_size)
            t: Tensor of shape (batch_size,) containing time steps
        Returns:
            noise_estimate: Tensor of shape (batch_size, signature_size)
        """
        # Normalize signatures
        signature = normalize_signature(signature)

        # Extract levels
        levels = extract_signature_levels(signature, self.D, self.N)  # List of tensors [(batch, D), (batch, D^2), ...]

        # Project each level separately
        projected_levels = []
        for level, proj in zip(levels, self.input_proj_list):
            # level: (batch_size, D^k)
            proj_level = proj(level)  # (batch_size, d_model)
            projected_levels.append(proj_level.unsqueeze(1))  # (batch_size, 1, d_model)

        # Concatenate projected levels to form a sequence
        x = torch.cat(projected_levels, dim=1)  # (batch_size, seq_length=N, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, seq_length, d_model)

        # Prepare for Transformer (requires shape: (seq_length, batch_size, d_model))
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model)

        # Get time embeddings
        t = t.long()  # Ensure t is of type Long for embedding
        t_embed = self.time_embedding(t)  # (batch_size, d_model)
        t_embed = t_embed.unsqueeze(0).repeat(self.seq_length, 1, 1)  # (seq_length, batch_size, d_model)

        # Add time embeddings to each token
        x = x + t_embed  # (seq_length, batch_size, d_model)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)

        # Aggregate Transformer outputs (e.g., take mean over sequence)
        x = x.mean(dim=0)  # (batch_size, d_model)

        # MLP to estimate noise
        noise_estimate = self.mlp(x)  # (batch_size, signature_size)

        return noise_estimate


class LatentDiffusionModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers, noise_steps, depth, beta_start=1e-4, beta_end=0.02, device="cuda", timesteps=30, seq_len=128):
        """
        Latent Diffusion Model with an MLP diffusion network.
        
        Args:
            input_dim (int): Input dimensionality of the data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Number of units in MLP hidden layers.
            num_layers (int): Number of hidden layers in the MLP.
            noise_steps (int): Number of diffusion timesteps.
            beta_schedule (torch.Tensor): Noise schedule for diffusion process.
        """
        super(LatentDiffusionModel, self).__init__()

        # self.mlp = MLPDiffusionModel(input_dim, input_dim, num_layers)
        self.model_trans = TransformerDDPMNoiseEstimator(3, depth, d_model=hidden_dim, num_encoder_layers=num_layers)
        self.model_rot   = TransformerDDPMNoiseEstimator(3, depth, d_model=hidden_dim, num_encoder_layers=num_layers)
        self.noise_steps = noise_steps

        self.device = device
        self.num_timesteps = timesteps
        self.seq_len = seq_len
        self.depth = depth

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

        self.loss = WassersteinSignatureLoss()

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_process(self, x1, t):
        l = x1.size(1)

        trans_sig = x1[:, : l//2]
        rot_sig   = x1[:, l//2 :]
        # print(x1.shape, trans_sig.shape, rot_sig.shape)
        trans_lie = signature_log(trans_sig, 3, self.depth)
        rot_lie = signature_log(rot_sig, 3, self.depth)

        epsilon0 = torch.randn_like(trans_sig)
        epsilon1 = torch.randn_like(rot_sig)

        x1_t_0 = extract(self.q_param1, t, trans_lie.shape) * trans_sig + \
                 extract(self.q_param2, t, trans_lie.shape) * epsilon0

        x1_t_1 = extract(self.q_param1, t, trans_lie.shape) * rot_sig + \
                 extract(self.q_param2, t, trans_lie.shape) * epsilon1

        # x1_t_0 = signature_exp(x1_t_0, 3, self.depth)
        # x1_t_1 = signature_exp(x1_t_1, 3, self.depth)

        x1_t = torch.cat((x1_t_0, x1_t_1), dim=1)
        epsilon = torch.cat((epsilon0, epsilon1), dim=1)
        return (x1_t, epsilon)

    def compute_loss(self, x1, t):
        (x1_t, epsilon) = self.forward_process(x1, t)

        l = x1.size(1)

        trans_sig = x1_t[:, : l//2]
        rot_sig   = x1_t[:, l//2 :]
        epsilon_t = epsilon[:, : l//2]
        epsilon_r = epsilon[:, l//2 :]

        predicted_score_t = self.model_trans(trans_sig, t)
        predicted_score_r = self.model_rot(rot_sig, t)
        # x0_1 = extract(self.x0_param1, t, x1_t.shape) * (x1_t - extract(self.x0_param2, t, x1_t.shape) * predicted_score1)

        loss_t = self.loss(predicted_score_t, epsilon_t)
        loss_t_l2 = F.l1_loss(predicted_score_t, epsilon_t)
        loss_r = self.loss(predicted_score_r, epsilon_r)
        loss_r_l2 = F.l1_loss(predicted_score_r, epsilon_r)

        return loss_t_l2+loss_r_l2, loss_t, loss_r, loss_t_l2, loss_r_l2
