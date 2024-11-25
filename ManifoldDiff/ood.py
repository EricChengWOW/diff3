from utils import *
from r3_diffuser import R3Diffuser, R3Conf
from unet import *
from se3_diffuser import *
from KITTI_dataset import KITTIOdometryDataset
import numpy as np
import torch
import pandas as pd

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
    parser.add_argument("--in_data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--in_dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--out_data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--out_dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--model_path", type=str, required=True, help="Fileof the trained model")

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.in_dataset == "KITTI":
        in_dataset = KITTIOdometryDataset(args.in_data_folder)
        in_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print("Training on KITTI for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    # Initialize the model
    loaded_model = DoubleUnet(dim=args.hidden_dim).cuda()

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path))

    # Set the model to evaluation mode
    loaded_model.eval()

    # Perform inference
    diffusion = SE3Diffusion(num_timesteps=args.num_timesteps, seq_len=args.n, scale_trans=args.scale_trans)

    rot_score_arr = []
    trans_score_arr = []
    batch_cnt = 0

    # Gather the distribution result
    for batch in in_dataloader:
        print("Batch ", batch_cnt)
        batch_cnt += 1

        batch = batch.to(args.device)
        rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
        rotations = matrix_to_so3(rotations)  # [B, n, 3]
        translations = batch[:, :, :3, 3]  # [B, n, 3]

        noise_rot, noise_trans = sample_noise((rotations.shape[0], args.n, 3), device=rotations.device)

        t = torch.full(translations.shape, 1, device=args.device)
        rot_t, trans_t = diffusion.perturb(rotations, translations, t, noise_rot, noise_trans)

        _, rot_score, trans_score = diffusion.reverse_diffusion(loaded_model, n=args.n, batch_size=batch.shape[0], return_score=True,\
                                                                rot_init=rot_t.transpose(1,2), trans_init=trans_t.transpose(1,2))
        rot_score_arr.append(rot_score)
        trans_score_arr.append(trans_score)

    rot_scores = np.stack(rot_score_arr)
    trans_scores = np.stack(trans_score_arr) # [B, step, batch_size, 3, n]

if __name__ == "__main__":
    main()
