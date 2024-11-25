from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('/content/drive/MyDrive/10623/ManifoldDiff/')

from utils import *
from r3_diffuser import R3Diffuser, R3Conf
from unet import *
from se3_diffuser import *
from KITTI_dataset import KITTIOdometryDataset

import argparse

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
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--scale_trans", type=float, default=1.0, help="Scale Factor for R3 translation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation, e.g., 'cuda' or 'cpu' (default: 'cuda').")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps for diffusion process (default: 100).")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--save_path", type=str, required=True, help="File to save the trained model")

    return parser.parse_args()

def train_se3_diffusion(model, diffusion, dataloader, optimizer, n=128, num_timesteps=20, num_epochs=100, project_name="SE3_Diffusion"):
    """
    Args:
        model: SE3ScoreNetwork instance.
        diffusion: SE3Diffusion instance.
        dataloader: dataloader of SE(3) sequences [B, n, 4, 4].
        optimizer: Optimizer for the score network.
        n: seq length
        num_epochs: Number of training epochs.
        project_name: Name of the wandb project.
    """
    # Initialize wandb
    wandb.init(project=project_name, config={"num_epochs": num_epochs, "learning_rate": optimizer.param_groups[0]["lr"]})

    for epoch in range(num_epochs):
        total_loss = 0

        # Use tqdm to show progress for the current epoch
        epoch_progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in epoch_progress:
            # Extract SE(3) components
            batch = batch.to(diffusion.device)
            rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
            rotations = matrix_to_so3(rotations)  # [B, n, 3]
            translations = batch[:, :, :3, 3]  # [B, n, 3]

            # Sample t and perturb SE(3)
            t = torch.randint(0, num_timesteps, (rotations.shape[0],), device=rotations.device) * (1 / (num_timesteps - 1))
            t_bc = t.reshape(t.size(0),1,1).broadcast_to(translations.shape)
            noise_rot, noise_trans = sample_noise((rotations.shape[0], n, 3), device=rotations.device)

            rot_xt, trans_xt = diffusion.perturb(rotations, translations, t_bc, noise_rot, noise_trans)

            # Get scores
            rot_score, trans_score = model(rot_xt.transpose(1, 2), trans_xt.transpose(1, 2), t)
            rot_score = rot_score.transpose(1, 2)
            trans_score = trans_score.transpose(1, 2)

            # Compute loss
            loss_rot = F.l1_loss(rot_score, noise_rot)
            # print(trans_xt.mean(), trans_score.mean(), trans_score_real.mean())
            loss_trans = F.l1_loss(trans_score, noise_trans)
            loss = loss_trans

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update tqdm progress bar with current loss
            epoch_progress.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Log to wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Finish wandb logging
    wandb.finish()

def main():
    args = parse_arguments()

    if args.dataset == "KITTI":
        dataset = KITTIOdometryDataset(args.data_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print("Training on KITTI for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    diffusion = SE3Diffusion(num_timesteps=args.num_timesteps, scale_trans=args.scale_trans, device=args.device)
    model = DoubleUnet(dim=args.hidden_dim)
    if args.device == "cuda":
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    train_se3_diffusion(model, diffusion, dataloader, optimizer, n=args.n, num_timesteps=args.num_timesteps, num_epochs=args.num_epochs)

    # Save the model's state_dict
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
