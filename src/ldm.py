from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt

import argparse

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(128, latent_dim)       # Mean of the latent space
        self.logvar_layer = nn.Linear(128, latent_dim)  # Log variance of the latent space

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class LatentDiffusionModel(nn.Module):
    def __init__(self, input_dim, latent_dim, unet_dim, noise_steps, beta_schedule):
        """
        Latent Diffusion Model with encoder, latent space diffusion, and decoder.

        Args:
            input_dim (int): Input dimensionality of the data.
            latent_dim (int): Dimensionality of the latent space.
            unet_dim (int): Dimensionality of the U-Net latent space.
            noise_steps (int): Number of diffusion timesteps.
            beta_schedule (torch.Tensor): Noise schedule for diffusion process.
        """
        super(LatentDiffusionModel, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.unet = Unet(dim=unet_dim, channels=latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)
        self.noise_steps = noise_steps
        self.beta_schedule = beta_schedule
        self.alpha = 1 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_diffusion(self, z, t):
        noise = torch.randn_like(z)
        alpha_t = self.alpha_bar[t].view(-1, 1)
        z_noisy = alpha_t.sqrt() * z + (1 - alpha_t).sqrt() * noise
        return z_noisy, noise

    def reverse_diffusion(self, z_noisy, t):
        predicted_noise = self.unet(z_noisy, t)
        return predicted_noise

    def forward(self, x, t):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Forward diffusion
        z_noisy, noise = self.forward_diffusion(z, t)

        # Reverse diffusion
        predicted_noise = self.reverse_diffusion(z_noisy, t)
        z_denoised = (z_noisy - (1 - self.alpha_bar[t]).sqrt() * predicted_noise) / self.alpha_bar[t].sqrt()

        # Decode back to data space
        x_reconstructed = self.decoder(z_denoised)

        return x_reconstructed, mu, logvar, noise, predicted_noise
