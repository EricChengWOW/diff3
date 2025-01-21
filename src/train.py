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
from DDPM_Diff import *
from DDPM_Continuous_Diff import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from L_dataset import LDataset
from T_dataset import TDataset

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
    parser.add_argument("--wandb", action='store_true', help="Log training loss to wandb")
    parser.add_argument("--diffusion_type", type=str, default="DDPM", help="The Diffusion algorithm and scheduler to use [DDPM, DDPM_Continuous]")
    parser.add_argument("--path_signature_depth", type=int, default=3, help="The depth of path signature transformation")

    return parser.parse_args()

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
    elif args.dataset == "T":
        dataset = TDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Training on T shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    if args.model_type == "Transformer":
        model = DoubleTransformerEncoderUnet(dim=args.hidden_dim, num_heads=args.n_heads, num_layers=args.n_layers, unet_layer=args.unet_layer).to(args.device)
    else:
        model = DoubleUnet(dim=args.hidden_dim, unet_layer=args.unet_layer).to(args.device)
    
    if args.diffusion_type == "DDPM":
        diffusion = DDPM_Diff(model, trans_scale=args.scale_trans, seq_len=args.n)
    else:
        diffusion = DDPM_Continuous_Diff(model, trans_scale=args.scale_trans, seq_len=args.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.1)
    run_name = "Diffusion_" + args.model_type + "_" + args.dataset + "_Epoch" + str(args.num_epochs)
    diffusion.train(dataloader, optimizer, args.device, num_timesteps=args.num_timesteps, epochs=args.num_epochs, log_wandb=args.wandb, project_name="Diff3", run_name=run_name)

    # Save the model's state_dict
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
