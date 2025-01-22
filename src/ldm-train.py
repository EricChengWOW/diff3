from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import *
from unet import *
from transformer import *
from transformer import *
from ldm import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from L_dataset import LDataset
from T_dataset import TDataset

import argparse
import os

def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        args: Parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training with KITTI dataset.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 16).")
    parser.add_argument("--n", type=int, default=128, help="Number of data points per sequence (default: 128).")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs (default: 200).")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers in the transformer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--latent_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--scale_trans", type=float, default=1.0, help="Scale Factor for R3 translation")
    parser.add_argument("--level", type=float, default=3, help="Path Signature Level")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation, e.g., 'cuda' or 'cpu' (default: 'cuda').")
    parser.add_argument("--num_timesteps", type=int, default=30, help="Number of timesteps for diffusion process (default: 100).")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, 'KITTI' or 'Oxford' ")
    parser.add_argument("--save_path", type=str, required=True, help="File to save the trained model")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling of data (default: False)')
    parser.add_argument('--center', action='store_true', help='Center each trajectory in data set')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Training optimizer learning rate")
    parser.add_argument("--wandb", action='store_true', help="Log training loss to wandb")
    parser.add_argument("--path_signature_depth", type=int, default=3, help="The depth of path signature transformation")

    return parser.parse_args()

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on KITTI for ", len(dataloader), " batches")
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on Oxford Robot car for ", len(dataloader), " batches")
    elif dataset == "IROS":
        dataset = IROS20Dataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, scale_trans = args.scale_trans, level = args.level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on IROS20 6D for ", len(dataloader), " batches")
    elif dataset == "L":
        dataset = LDataset(seq_len=args.n, use_path_signature = True, level = args.level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L shape for ", len(dataloader), " batches")
    elif dataset == "L-rand":
        dataset = LDataset(seq_len=args.n, rand_shuffle = True)
        dataset.visualize_trajectory(idx=0, save_folder = args.save_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L-rand shape for ", len(dataloader), " batches")
    elif dataset == "T":
        dataset = TDataset(seq_len=args.n, use_path_signature = True, level = args.level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on T shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    return dataset, dataloader

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
              # mu, logvar = ldm.encoder(batch)
              # z = ldm.reparameterize(mu, logvar)
              optimizer.zero_grad()
              t = torch.randint(0, num_timesteps-1, (batch.shape[0],), device=device)
              # Compute the loss for x1 and x2
              loss = ldm.compute_loss(batch, t)
              loss.backward()
              optimizer.step()
              epoch_loss += loss.item()

      print(f"Diffusion Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
      torch.save(ldm.state_dict(), os.path.join(save_dir, f"diffusion_epoch_{epoch + 1}.pth"))

# Main Function
def main():
    args = parse_arguments()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 0
    for i in range(args.path_signature_depth + 1):
      input_dim += 6 ** i

    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim  # MLP hidden dimension
    num_layers = args.n_layers    # MLP number of layers
    noise_steps = args.num_timesteps
    batch_size = args.batch_size
    vae_epochs = args.num_epochs
    diffusion_epochs = args.num_epochs
    learning_rate = args.learning_rate
    save_dir = args.save_path

    dataset, dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)

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
    # train_vae(ldm, dataloader, vae_optimizer, vae_epochs, device, os.path.join(save_dir, "vae"))

    # Stage 2: Train Diffusion
    train_diffusion(ldm, dataloader, diffusion_optimizer, diffusion_epochs, device, noise_steps, os.path.join(save_dir, "diffusion"))

if __name__ == "__main__":
    main()