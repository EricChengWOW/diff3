from utils import *
from r3_diffuser import R3Diffuser, R3Conf
from unet import *
from se3_diffuser import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

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
    parser.add_argument("--in_data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--out_data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")
    parser.add_argument("--out_dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--out_data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--model_path", type=str, required=True, help="Fileof the trained model")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling of data (default: False)')

    return parser.parse_args()

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    else:
        raise "Dataset type not supported"

    return dataset, dataloader

def main():
    args = parse_arguments()

    in_dataset, in_dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)
    out_dataset, out_dataloader = get_data(args.out_dataset, args.out_data_folder, args.out_data_stride, args)

    # Initialize the model
    loaded_model = DoubleUnet(dim=args.hidden_dim).to(args.device)

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
        rot_score_arr.append(rot_score.transpose(1,0,2,3))
        trans_score_arr.append(trans_score.transpose(1,0,2,3))

    rot_scores = np.stack(rot_score_arr, axis=0)
    trans_scores = np.stack(trans_score_arr, axis=0) # [B, step, batch_size, 3, n]
    eps = trans_scores.sum(axis=(2,3,4)).flatten()
    deps = np.diff(eps)

    # Fit model

    # Get test_set
    rot_score_arr = []
    trans_score_arr = []
    batch_cnt = 0
    for batch in out_dataloader:
        print("Out Batch ", batch_cnt)
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
        rot_score_arr.append(rot_score.transpose(1,0,2,3))
        trans_score_arr.append(trans_score.transpose(1,0,2,3))
    
    out_trans_scores = np.stack(trans_score_arr, axis=0) # [B, step, batch_size, 3, n]

    samples = out_trans_scores.sum(axis=(2,3,4)).flatten()
    dsamples = np.diff(samples)

    # Create a bar plot
    plt.hist([eps, samples], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

    # Add labels and title
    plt.xlabel('Eps')
    plt.ylabel('Freq')
    plt.title('Hist of Eps')

    plt.savefig("Eps_comp.png", dpi=300, bbox_inches='tight')
    plt.close()

    ###abs
    plt.hist([np.abs(eps), np.abs(samples)], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

    # Add labels and title
    plt.xlabel('Eps')
    plt.ylabel('Freq')
    plt.title('Hist of Eps')

    plt.savefig("Eps_comp.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.hist([deps, dsamples], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

    # Add labels and title
    plt.xlabel('dEps')
    plt.ylabel('Freq')
    plt.title('Hist of dEps')

    plt.savefig("dEps_comp.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.hist([deps ** 2, dsamples ** 2], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

    # Add labels and title
    plt.xlabel('dEps2')
    plt.ylabel('Freq')
    plt.title('Hist of dEps')

    plt.savefig("dEps2_comp.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.hist([deps ** 3, dsamples ** 3], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

    # Add labels and title
    plt.xlabel('dEps3')
    plt.ylabel('Freq')
    plt.title('Hist of dEps')

    plt.savefig("dEps3_comp.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
