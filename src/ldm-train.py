from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *
from unet import *
from transformer import *
#from DDPM_Diff import *
#from DDPM_Continuous_Diff import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from L_dataset import LDataset
from T_dataset import TDataset

import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ldm import *
import numpy as np

def vae_loss(x, x_reconstructed, mu, logvar):
    # Reconstruction loss
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

def diffusion_loss(noise, predicted_noise):
    return nn.MSELoss()(predicted_noise, noise)

# Training Encoder and Decoder (Stage 1)
def train_vae(ldm, dataloader, optimizer, num_epochs, device, save_dir):
    ldm.to(device)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)

            mu, logvar = ldm.encoder(batch)
            z = ldm.reparameterize(mu, logvar)
            x_reconstructed = ldm.decoder(z)

            loss = vae_loss(batch, x_reconstructed, mu, logvar)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"VAE Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
        torch.save(ldm.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))

def train_diffusion(ldm, dataloader, optimizer, num_epochs, device, noise_steps, save_dir, num_timesteps=30):
    for param in ldm.encoder.parameters():
        param.requires_grad = False
    for param in ldm.decoder.parameters():
        param.requires_grad = False

    ldm.to(device)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
      epoch_loss = 0.0
      
      with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
          for i, batch in enumerate(pbar):
              batch = batch.to(device)
              mu, logvar = ldm.encoder(batch)
              z = ldm.reparameterize(mu, logvar)
              optimizer.zero_grad()
              t = torch.randint(0, num_timesteps-1, (batch.shape[0],), device=device)
              # Compute the loss for x1 and x2
              loss = ldm.compute_loss(z, t)
              loss.backward()
              optimizer.step()
              epoch_loss += loss.item()

      print(f"Diffusion Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
      torch.save(ldm.state_dict(), os.path.join(save_dir, f"diffusion_epoch_{epoch + 1}.pth"))

# Main Function
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 259
    latent_dim = 64
    hidden_dim = 128  # MLP hidden dimension
    num_layers = 4    # MLP number of layers
    noise_steps = 1000
    batch_size = 32
    vae_epochs = 5
    diffusion_epochs = 50
    learning_rate = 1e-3
    save_dir = "/content/drive/MyDrive/diff3"

    dataset = LDataset(seq_len=128, use_path_signature=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ldm = LatentDiffusionModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        noise_steps=noise_steps,
    )

    vae_optimizer = optim.Adam(list(ldm.encoder.parameters()) + list(ldm.decoder.parameters()), lr=learning_rate)
    diffusion_optimizer = optim.Adam(ldm.mlp.parameters(), lr=learning_rate)

    # Stage 1: Train VAE
    train_vae(ldm, dataloader, vae_optimizer, vae_epochs, device, os.path.join(save_dir, "vae"))

    # Stage 2: Train Diffusion
    train_diffusion(ldm, dataloader, diffusion_optimizer, diffusion_epochs, device, noise_steps, os.path.join(save_dir, "diffusion"))

if __name__ == "__main__":
    main()