#from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *

import numpy as np
import torch
import torch.nn as nn
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

        self.mlp = MLPDiffusionModel(input_dim, input_dim, num_layers)
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

        x1_t_0 = extract(self.q_param1, t, trans_lie.shape) * trans_lie + \
                 extract(self.q_param2, t, trans_lie.shape) * epsilon0

        x1_t_1 = extract(self.q_param1, t, trans_lie.shape) * rot_lie + \
                 extract(self.q_param2, t, trans_lie.shape) * epsilon1

        x1_t_0 = signature_exp(x1_t_0, 3, self.depth)
        x1_t_1 = signature_exp(x1_t_1, 3, self.depth)

        x1_t = torch.cat((x1_t_0, x1_t_1), dim=1)
        epsilon = torch.cat((epsilon0, epsilon1), dim=1)
        return (x1_t, epsilon)

    def compute_loss(self, x1, t):
        (x1_t, epsilon1) = self.forward_process(x1, t)
        predicted_score1 = self.mlp(x1_t, t)
        # x0_1 = extract(self.x0_param1, t, x1_t.shape) * (x1_t - extract(self.x0_param2, t, x1_t.shape) * predicted_score1)

        loss1 = self.loss(predicted_score1, epsilon1)
        #loss_origin1 = F.l1_loss(x0_1, x1)

        return loss1
