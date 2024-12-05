from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append('/content/drive/MyDrive/10623/ManifoldDiff/')

from utils import *
from unet import *
from transformer import *
from se3_diffuser import *
from se3_diffuser_2 import *
from se3_diffuser_3 import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from L_dataset import LDataset

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
    parser.add_argument("--model_type", type=str, default="Transformer", help="The score model architecture")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers in the transformer")
    parser.add_argument("--unet_layer", type=int, default=4, help="Layers of unet dim changes")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of head in the transformer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--scale_trans", type=float, default=1.0, help="Scale Factor for R3 translation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation, e.g., 'cuda' or 'cpu' (default: 'cuda').")
    parser.add_argument("--num_timesteps", type=int, default=30, help="Number of timesteps for diffusion process (default: 100).")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, 'KITTI' or 'Oxford' ")
    parser.add_argument("--save_path", type=str, required=True, help="File to save the trained model")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling of data (default: False)')
    parser.add_argument('--center', action='store_true', help='Center each trajectory in data set')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Training optimizer learning rate")

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
            loss = loss_trans + loss_rot

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
        dataset = KITTIOdometryDataset(args.data_folder, seq_len=args.n, stride=args.data_stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Training on KITTI for ", len(dataloader), " batches")
    elif args.dataset == "Oxford":
        dataset = RobotcarDataset(args.data_folder, seq_len=args.n, stride=args.data_stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Training on Oxford Robot car for ", len(dataloader), " batches")
    elif args.dataset == "L":
        dataset = LDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Training on L shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    if args.model_type == "Transformer":
        model = DoubleTransformerEncoderUnet(dim=args.hidden_dim, num_heads=args.n_heads).to(args.device)
        diffusion = SE3Diffusion_3(model, trans_scale=args.scale_trans, timesteps=args.num_timesteps)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.1)
        run_name = "Diffusion_" + args.model_type + "_" + args.dataset + "_Epoch" + str(args.num_epochs)
        diffusion.train(dataloader, optimizer, args.device, num_timesteps=args.num_timesteps, epochs=args.num_epochs, project_name="Diff3", run_name=run_name)
    else:
        model = DoubleUnet(dim=args.hidden_dim, unet_layer=args.unet_layer).to(args.device)
        diffusion = SE3Diffusion_3(model, trans_scale=args.scale_trans)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.1)
        run_name = "Diffusion_" + args.model_type + "_" + args.dataset + "_Epoch" + str(args.num_epochs)
        diffusion.train(dataloader, optimizer, args.device, epochs=args.num_epochs, project_name="Diff3", run_name=run_name)

    # Save the model's state_dict
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
