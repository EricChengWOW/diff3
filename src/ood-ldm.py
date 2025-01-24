#from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from IROS20_dataset import IROS20Dataset
from L_dataset import LDataset
from T_dataset import TDataset

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, average_precision_score

import argparse
from ldm import *

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
    parser.add_argument("--in_data_folder", type=str, required=False, help="Path to the data folder containing the dataset.")
    parser.add_argument("--in_dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--in_data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--out_data_folder", type=str, required=False, help="Path to the data folder containing the dataset.")
    parser.add_argument("--out_dataset", type=str, required=True, help="Dataset name, e.g., 'KITTI'.")
    parser.add_argument("--out_data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--model_path", type=str, required=True, help="Fileof the trained model")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling of data (default: False)')
    parser.add_argument('--center', action='store_true', help='Center each trajectory in data set')
    parser.add_argument("--unet_layer", type=int, default=4, help="Layers of unet dim changes")
    parser.add_argument("--model_type", type=str, default="Transformer", help="The score model architecture")
    parser.add_argument("--save_folder", type=str, default=".", help="The folder to save GMM model and statistics graphs")
    parser.add_argument("--ood_mode", type=str, default="SE3", help="R3 or SE3 OOD metric")
    parser.add_argument("--latent_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--path_signature_depth", type=int, default=3, help="The depth of path signature transformation")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers in the transformer")

    return parser.parse_args()

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.path_signature_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on KITTI for ", len(dataloader), " batches")
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.path_signature_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on Oxford Robot car for ", len(dataloader), " batches")
    elif dataset == "IROS":
        dataset = IROS20Dataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, scale_trans = args.scale_trans, level = args.path_signature_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on IROS20 6D for ", len(dataloader), " batches")
    elif dataset == "L":
        dataset = LDataset(seq_len=args.n, use_path_signature = True, level = args.path_signature_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L shape for ", len(dataloader), " batches")
    elif dataset == "L-rand":
        dataset = LDataset(seq_len=args.n, rand_shuffle = True)
        dataset.visualize_trajectory(idx=0, save_folder = args.save_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L-rand shape for ", len(dataloader), " batches")
    elif dataset == "T":
        dataset = TDataset(seq_len=args.n, use_path_signature = True, level = args.path_signature_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on T shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    return dataset, dataloader

def diffusion_metrics(ldm, dataloader, args):
    rot_score_arr = []
    rot_score_arr_2 = []
    rot_score_arr_3 = []
    drot_score_arr = []
    drot_score_arr_2 = []
    drot_score_arr_3 = []

    batch_cnt = 0

    # Gather the distribution result
    for batch in dataloader:
        batch_cnt += 1

        batch = batch.to(args.device)
        # mu, logvar = ldm.encoder(batch)
        # x = ldm.reparameterize(mu, logvar)
        x = batch

        score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r1 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r2 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

        prev_score_r = None

        x_init = torch.randn_like(x)

        for t in range(args.num_timesteps):
            t_tensor = torch.full((batch.size(0),), t, device=args.device)

            (x_t, _) = ldm.forward_process(x, t_tensor)

            # Predict scores using the score model
            with torch.no_grad():
                rot_score = ldm.mlp(x_t, t_tensor)

            score_norm_r1 += torch.sum(rot_score)
            score_norm_r2 += torch.sum(rot_score ** 2)
            score_norm_r3 += torch.sum(rot_score ** 3)

            if prev_score_r is not None:
                delta = rot_score - prev_score_r
                dscore_norm_r1 += torch.sum(delta)
                dscore_norm_r2 += torch.sum(delta ** 2)
                dscore_norm_r3 += torch.sum(delta ** 3)

            prev_score_r = rot_score

        rot_score_arr.append(score_norm_r1.detach().cpu().numpy())
        rot_score_arr_2.append(score_norm_r2.detach().cpu().numpy())
        rot_score_arr_3.append(score_norm_r3.detach().cpu().numpy())

        drot_score_arr.append(dscore_norm_r1.detach().cpu().numpy())
        drot_score_arr_2.append(dscore_norm_r2.detach().cpu().numpy())
        drot_score_arr_3.append(dscore_norm_r3.detach().cpu().numpy())

    rot_scores_1 = np.concatenate(rot_score_arr, axis=0)
    rot_scores_2 = np.concatenate(rot_score_arr_2, axis=0)
    rot_scores_3 = np.concatenate(rot_score_arr_3, axis=0)
    drot_scores_1 = np.concatenate(drot_score_arr, axis=0)
    drot_scores_2 = np.concatenate(drot_score_arr_2, axis=0)
    drot_scores_3 = np.concatenate(drot_score_arr_3, axis=0)

    eps_r1 = rot_scores_1
    eps_r2 = np.sqrt(rot_scores_2)
    eps_r3 = rot_scores_3

    deps_r1 = drot_scores_1
    deps_r2 = np.sqrt(drot_scores_2)
    deps_r3 = drot_scores_3

    return (eps_r1, eps_r2, eps_r3), (deps_r1, deps_r2, deps_r3)

def main():
    args = parse_arguments()
    
    in_dataset, in_dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)

    total_length = len(in_dataset)  
    train_length = int(0.9 * total_length) 
    val_length = total_length - train_length 
    train_dataset, val_dataset = random_split(in_dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    out_dataset, out_dataloader = get_data(args.out_dataset, args.out_data_folder, args.out_data_stride, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 0
    for i in range(args.path_signature_depth + 1):
      input_dim += 6 ** i

    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim  # MLP hidden dimension
    num_layers = args.n_layers    # MLP number of layers
    noise_steps = args.num_timesteps
    batch_size = args.batch_size
    save_dir = args.model_path
    # Initialize the model

    loaded_model = LatentDiffusionModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        noise_steps=noise_steps,
    )

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path, weights_only=True))
    loaded_model.to(args.device)

    # Set the model to evaluation mode
    loaded_model.eval()

    stats_first_order, stats_second_order = diffusion_metrics(loaded_model, in_dataloader, args)
    in_eps_r1, in_eps_r2, in_eps_r3 = stats_first_order
    in_deps_r1, in_deps_r2, in_deps_r3 = stats_second_order
    print("finish calculating stats for train")
    # val
    stats_first_order, stats_second_order = diffusion_metrics(loaded_model, val_dataloader, args)
    val_eps_r1, val_eps_r2, val_eps_r3 = stats_first_order
    val_deps_r1, val_deps_r2, val_deps_r3 = stats_second_order
    print("finish calculating stats for val")
    # Get test_set
    stats_first_order, stats_second_order = diffusion_metrics(loaded_model, out_dataloader, args)
    out_eps_r1, out_eps_r2, out_eps_r3 = stats_first_order
    out_deps_r1, out_deps_r2, out_deps_r3 = stats_second_order
    print("finish calculating stats for test")

    ### Save plots of distributions
    def save_plot(in_data, out_data, in_dataset, out_dataset, metric="eps", path = "temp.png"):
        plt.hist([in_data, out_data], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=[in_dataset, out_dataset])

        # Add labels and title
        plt.xlabel(metric)
        plt.ylabel('Freq')
        plt.title('Distribution of ' + metric)
        plt.legend()

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    save_plot(in_eps_r1, out_eps_r1, args.in_dataset, args.out_dataset, metric="eps_rot", path=args.save_folder + "/eps_rot.png")
    save_plot(in_eps_r2, out_eps_r2, args.in_dataset, args.out_dataset, metric="eps_rot2", path=args.save_folder + "/eps_rot2.png")
    save_plot(in_eps_r3, out_eps_r3, args.in_dataset, args.out_dataset, metric="eps_rot3", path=args.save_folder + "/eps_rot3.png")

    save_plot(in_deps_r1, out_deps_r1, args.in_dataset, args.out_dataset, metric="deps_rot", path=args.save_folder + "/deps_rot.png")
    save_plot(in_deps_r2, out_deps_r2, args.in_dataset, args.out_dataset, metric="deps_rot2", path=args.save_folder + "/deps_rot2.png")
    save_plot(in_deps_r3, out_deps_r3, args.in_dataset, args.out_dataset, metric="deps_rot3", path=args.save_folder + "/deps_rot3.png")
    print("finish saving distribution plots")

    # Fit GMM on in distribution
    data = np.column_stack([in_eps_r1, in_eps_r2, in_eps_r3,in_deps_r1, in_deps_r2, in_deps_r3])
  
    n_components = 1
    gmm = GaussianMixture(n_components=n_components, random_state=42, reg_covar=1e-5)
    gmm.fit(data)
    print("finish fitting gmm")
    train_probs = gmm.score_samples(data)
    lower_threshold, upper_threshold = np.percentile(train_probs, 5), np.percentile(train_probs, 95)

    val_points = np.column_stack([val_eps_r1, val_eps_r2, val_eps_r3, val_deps_r1, val_deps_r2, val_deps_r3])
    val_probs = gmm.score_samples(val_points)
    ood_flags = (val_probs < lower_threshold) | (val_probs > upper_threshold)
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Val: {num_ood}")
    print(val_points.shape)

    test_points = np.column_stack([out_eps_r1, out_eps_r2, out_eps_r3, out_deps_r1, out_deps_r2, out_deps_r3])
    test_probs = gmm.score_samples(test_points)
    ood_flags = (test_probs < lower_threshold) | (test_probs > upper_threshold)
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Test: {num_ood}")
    print(ood_flags.shape)

    val_labels = np.ones(len(val_probs)) 
    test_labels = np.zeros(len(test_probs))

    all_scores = np.concatenate([np.exp(val_probs), np.exp(test_probs)])
    all_labels = np.concatenate([val_labels, test_labels])

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"AUROC: {auroc}")

    aupr = average_precision_score(all_labels, all_scores)
    print(f"AUPR: {aupr}")

    val_ood_flags = (val_probs > lower_threshold) & (val_probs < upper_threshold)
    test_ood_flags = (test_probs > lower_threshold) & (test_probs < upper_threshold)

    val_preds = val_ood_flags.astype(int)
    test_preds = test_ood_flags.astype(int)
    all_preds = np.concatenate([val_preds, test_preds])

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")

if __name__ == "__main__":
    main()