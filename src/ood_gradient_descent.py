# from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from IROS20_dataset import IROS20Dataset
from L_dataset import LDataset
from T_dataset import TDataset
from torchGMM import TorchGMM

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader, random_split, Subset
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.mixture import GaussianMixture
from torchGMM import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, average_precision_score

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
    parser.add_argument("--scale_trans_in", type=float, default=1.0, help="Scale Factor for R3 translation for inlier distribution")
    parser.add_argument("--scale_trans_out", type=float, default=1.0, help="Scale Factor for R3 translation for outlier distribution")
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
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--likelihood_weight", type=float, default=1e-5, help="likelihood loss weight")
    parser.add_argument("--smoothness_weight", type=float, default=1e-3, help="smoothness loss weight")

    return parser.parse_args()

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

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps each element in the input tensor to the range [-π, π].

    Args:
        x (torch.Tensor): Input tensor containing angles in radians.

    Returns:
        torch.Tensor: Tensor with all elements wrapped to [-π, π].
    """
    return (x + torch.pi) % (2 * torch.pi) - torch.pi

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on KITTI for ", len(dataloader), " batches")
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on Oxford Robot car for ", len(dataloader), " batches")
    elif dataset == "IROS":
        dataset = IROS20Dataset(dataset_path, seq_len=args.n, stride=stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on IROS20 6D for ", len(dataloader), " batches")
    elif dataset == "L":
        dataset = LDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L shape for ", len(dataloader), " batches")
    elif dataset == "L-rand":
        dataset = LDataset(seq_len=args.n, rand_shuffle = True)
        dataset.visualize_trajectory(idx=0, save_folder = args.save_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L-rand shape for ", len(dataloader), " batches")
    elif dataset == "T":
        dataset = TDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on T shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    return dataset, dataloader

def diffusion_metrics_batch (diffusion, batch, args, loaded_model, trans_init=None, rot_init=None, label=""):
    rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
    translations = batch[:, :, :3, 3]  # [B, n, 3]

    B, L, _ = translations.shape

    score_norm_t1 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    # score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)

    score_norm_t2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    # score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)

    score_norm_t3 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    # score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

    score_norm_rx = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    score_norm_ry = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    score_norm_rz = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    score_norm_rx2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    score_norm_ry2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    score_norm_rz2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)

    dscore_norm_t1 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    dscore_norm_r1 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)

    dscore_norm_t2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    dscore_norm_r2 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)

    dscore_norm_t3 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)
    dscore_norm_r3 = torch.zeros((batch.size(0),), device=args.device, requires_grad=True)

    prev_score_t = None
    prev_score_r = None

    if trans_init is None:
        trans_init = torch.randn_like(translations)
        v_T = torch.randn(B,L,3, device=translations.device)
        rot_init = so3_exp_map(v_T)
    else:
        trans_init = trans_init.expand(B, -1, -1)
        rot_init = rot_init.expand(B, -1, -1, -1)

    for t in range(args.num_timesteps):
        t_tensor = torch.full((batch.size(0),), t, device=args.device)

        (trans_t, trans_noise), (rot_t, rot_noise) = diffusion.forward_process(translations, rotations, t_tensor, trans_init=trans_init, rot_init=rot_init)

        if loaded_model.name == "Unet":
            trans_t = trans_t.transpose(1,2)
            rot_t = rot_t.reshape(B, L, 9)
            rot_t = rot_t.transpose(1,2)

        # Predict scores using the score model
        trans_score, rot_score = loaded_model(trans_t, rot_t, t_tensor)

        if loaded_model.name == "Unet":
            trans_score = trans_score.transpose(1,2)
            rot_score = rot_score.transpose(1,2)
            trans_t = trans_t.transpose(1,2)
            rot_t = rot_t.transpose(1,2)
            rot_t = rot_t.reshape(B, L, 3, 3)
        
        rot_score = wrap_to_pi(rot_score)

        metric_x = rot_score[:, :, 0]
        score_norm_rx = score_norm_rx + torch.mean(metric_x, dim=1)
        score_norm_rx2 = score_norm_rx2 + torch.mean(metric_x ** 2, dim=1)

        metric_y = rot_score[:, :, 1]
        score_norm_ry = score_norm_ry + torch.mean(metric_y, dim=1)
        score_norm_ry2 = score_norm_ry2 + torch.mean(metric_y ** 2, dim=1)

        metric_z = rot_score[:, :, 2]
        score_norm_rz = score_norm_rz + torch.mean(metric_z, dim=1)
        score_norm_rz2 = score_norm_rz2 + torch.mean(metric_z ** 2, dim=1)

        score_norm_t1 = score_norm_t1 + torch.mean(trans_score, dim=(-2, -1))
        score_norm_t2 = score_norm_t2 + torch.mean(trans_score ** 2, dim=(-2, -1))
        score_norm_t3 = score_norm_t3 + torch.mean(trans_score ** 3, dim=(-2, -1))

        if prev_score_t is not None:
            delta = trans_score - prev_score_t
            dscore_norm_t1 = dscore_norm_t1 + torch.mean(delta, dim=(-2, -1))
            dscore_norm_t2 = dscore_norm_t2 + torch.mean(delta ** 2, dim=(-2, -1))
            dscore_norm_t3 = dscore_norm_t3 + torch.mean(delta ** 3, dim=(-2, -1))

            delta = rot_score - prev_score_r
            dscore_norm_r1 = dscore_norm_r1 + torch.mean(delta, dim=(-2, -1))
            dscore_norm_r2 = dscore_norm_r2 + torch.mean(delta ** 2, dim=(-2, -1))
            dscore_norm_r3 = dscore_norm_r3 + torch.mean(delta ** 3, dim=(-2, -1))


        prev_score_t = trans_score
        prev_score_r = rot_score

    eps_t1 = score_norm_t1
    eps_t2 = torch.sqrt(score_norm_t2)
    eps_t3 = score_norm_t3

    deps_t1 = dscore_norm_t1
    deps_t2 = torch.sqrt(dscore_norm_t2)
    deps_t3 = dscore_norm_t3

    rot_x_all = score_norm_rx
    rot_y_all = score_norm_ry
    rot_z_all = score_norm_rz
    rot_x_all2 = score_norm_rx2
    rot_y_all2 = score_norm_ry2
    rot_z_all2 = score_norm_rz2
    print("test metric", eps_t1[0].detach().cpu().numpy(), 
                         eps_t2[0].detach().cpu().numpy(),
                         eps_t3[0].detach().cpu().numpy(), 
                         rot_x_all[0].detach().cpu().numpy(), 
                         rot_y_all[0].detach().cpu().numpy(), 
                         rot_z_all[0].detach().cpu().numpy())

    return (eps_t1, eps_t2, eps_t3, rot_x_all, rot_y_all, rot_z_all), (deps_t1, deps_t2, deps_t3, rot_x_all2, rot_y_all2, rot_z_all2) 


def diffusion_metrics(diffusion, dataloader, args, loaded_model, trans_single=None, rot_single=None, label=""):
    rot_score_arr = []
    trans_score_arr = []

    rot_score_arr_2 = []
    trans_score_arr_2 = []

    rot_score_arr_3 = []
    trans_score_arr_3 = []

    drot_score_arr = []
    dtrans_score_arr = []

    drot_score_arr_2 = []
    dtrans_score_arr_2 = []

    drot_score_arr_3 = []
    dtrans_score_arr_3 = []

    rot_x = []
    rot_y = []
    rot_z = []
    rot_x2 = []
    rot_y2 = []
    rot_z2 = []

    trans_x = []
    trans_y = []
    trans_z = []

    batch_cnt = 0

    # Gather the distribution result
    for batch in dataloader:
        if batch_cnt == 0:
          print(batch[0][0])
        batch_cnt += 1

        batch = batch.to(args.device)
        batch.requires_grad_(True)
        rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
        translations = batch[:, :, :3, 3]  # [B, n, 3]

        B, L, _ = translations.shape

        score_norm_t1 = torch.zeros((batch.size(0),), device=args.device)
        # score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t2 = torch.zeros((batch.size(0),), device=args.device)
        # score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t3 = torch.zeros((batch.size(0),), device=args.device)
        # score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_rx = torch.zeros((batch.size(0),), device=args.device)
        score_norm_ry = torch.zeros((batch.size(0),), device=args.device)
        score_norm_rz = torch.zeros((batch.size(0),), device=args.device)
        score_norm_rx2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_ry2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_rz2 = torch.zeros((batch.size(0),), device=args.device)

        dscore_norm_t1 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r1 = torch.zeros((batch.size(0),), device=args.device)

        dscore_norm_t2 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r2 = torch.zeros((batch.size(0),), device=args.device)

        dscore_norm_t3 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

        prev_score_t = None
        prev_score_r = None

        if trans_single is None:
            trans_init = torch.randn_like(translations)
            v_T = torch.randn(B,L,3, device=translations.device)
            rot_init = so3_exp_map(v_T)
        else:
            trans_init = trans_single.expand(B, -1, -1)
            rot_init = rot_single.expand(B, -1, -1, -1)

        for t in range(args.num_timesteps):
            t_tensor = torch.full((batch.size(0),), t, device=args.device)

            (trans_t, trans_noise), (rot_t, rot_noise) = diffusion.forward_process(translations, rotations, t_tensor, trans_init=trans_init, rot_init=rot_init)

            if loaded_model.name == "Unet":
                trans_t = trans_t.transpose(1,2)
                rot_t = rot_t.reshape(B, L, 9)
                rot_t = rot_t.transpose(1,2)

            # Predict scores using the score model
            with torch.no_grad():
                trans_score, rot_score = loaded_model(trans_t, rot_t, t_tensor)

            if loaded_model.name == "Unet":
                trans_score = trans_score.transpose(1,2)
                rot_score = rot_score.transpose(1,2)
                trans_t = trans_t.transpose(1,2)
                rot_t = rot_t.transpose(1,2)
                rot_t = rot_t.reshape(B, L, 3, 3)
            
            rot_score = wrap_to_pi(rot_score)

            metric_x = rot_score[:, :, 0]
            score_norm_rx += torch.mean(metric_x, dim=1)
            score_norm_rx2 += torch.mean(metric_x ** 2, dim=1)

            metric_y = rot_score[:, :, 1]
            score_norm_ry += torch.mean(metric_y, dim=1)
            score_norm_ry2 += torch.mean(metric_y ** 2, dim=1)

            metric_z = rot_score[:, :, 2]
            score_norm_rz += torch.mean(metric_z, dim=1)
            score_norm_rz2 += torch.mean(metric_z ** 2, dim=1)

            score_norm_t1 += torch.mean(trans_score, dim=(-2, -1))
            score_norm_t2 += torch.mean(trans_score ** 2, dim=(-2, -1))
            score_norm_t3 += torch.mean(trans_score ** 3, dim=(-2, -1))

            arr1 = trans_score[:, :, 0].cpu().numpy().flatten()
            arr2 = trans_score[:, :, 1].cpu().numpy().flatten()
            arr3 = trans_score[:, :, 2].cpu().numpy().flatten()
            trans_x.append(arr1)
            trans_y.append(arr2)
            trans_z.append(arr3)

            if prev_score_t is not None:
                delta = trans_score - prev_score_t
                dscore_norm_t1 += torch.mean(delta, dim=(-2, -1))
                dscore_norm_t2 += torch.mean(delta ** 2, dim=(-2, -1))
                dscore_norm_t3 += torch.mean(delta ** 3, dim=(-2, -1))

                delta = rot_score - prev_score_r
                dscore_norm_r1 += torch.mean(delta, dim=(-2, -1))
                dscore_norm_r2 += torch.mean(delta ** 2, dim=(-2, -1))
                dscore_norm_r3 += torch.mean(delta ** 3, dim=(-2, -1))

            prev_score_t = trans_score
            prev_score_r = rot_score

        rot_x.append(score_norm_rx.detach().cpu().numpy())
        rot_y.append(score_norm_ry.detach().cpu().numpy())
        rot_z.append(score_norm_rz.detach().cpu().numpy())
        rot_x2.append(score_norm_rx2.detach().cpu().numpy())
        rot_y2.append(score_norm_ry2.detach().cpu().numpy())
        rot_z2.append(score_norm_rz2.detach().cpu().numpy())
        trans_score_arr.append(score_norm_t1.detach().cpu().numpy())
        trans_score_arr_2.append(score_norm_t2.detach().cpu().numpy())
        trans_score_arr_3.append(score_norm_t3.detach().cpu().numpy())

        drot_score_arr.append(dscore_norm_r1.detach().cpu().numpy())
        drot_score_arr_2.append(dscore_norm_r2.detach().cpu().numpy())
        drot_score_arr_3.append(dscore_norm_r3.detach().cpu().numpy())
        dtrans_score_arr.append(dscore_norm_t1.detach().cpu().numpy())
        dtrans_score_arr_2.append(score_norm_t2.detach().cpu().numpy())
        dtrans_score_arr_3.append(score_norm_t3.detach().cpu().numpy())

    trans_scores_1 = np.concatenate(trans_score_arr, axis=0)
    trans_scores_2 = np.concatenate(trans_score_arr_2, axis=0)
    trans_scores_3 = np.concatenate(trans_score_arr_3, axis=0)
    dtrans_scores_1 = np.concatenate(dtrans_score_arr, axis=0)
    dtrans_scores_2 = np.concatenate(dtrans_score_arr_2, axis=0)
    dtrans_scores_3 = np.concatenate(dtrans_score_arr_3, axis=0)

    arr1 = np.concatenate(trans_x, axis=0)
    arr2 = np.concatenate(trans_y, axis=0)
    arr3 = np.concatenate(trans_z, axis=0)
    plt.hist([arr1, arr2, arr3], bins=50, color=['skyblue', 'orange', 'green'], edgecolor='black', label=['x', 'y', 'z'])

    # Add labels and title
    plt.xlabel('metric')
    plt.ylabel('Freq')
    plt.title('Distribution')
    plt.legend()

    plt.savefig('./tranxyz_' + label + '.png', dpi=300, bbox_inches='tight')
    plt.close()

    rot_x_all = np.concatenate(rot_x, axis=0)
    rot_y_all = np.concatenate(rot_y, axis=0)
    rot_z_all = np.concatenate(rot_z, axis=0)
    rot_x_all2 = np.concatenate(rot_x2, axis=0)
    rot_y_all2 = np.concatenate(rot_y2, axis=0)
    rot_z_all2 = np.concatenate(rot_z2, axis=0)

    eps_t1 = trans_scores_1
    eps_t2 = np.sqrt(trans_scores_2)
    eps_t3 = trans_scores_3

    deps_t1 = dtrans_scores_1
    deps_t2 = np.sqrt(dtrans_scores_2)
    deps_t3 = dtrans_scores_3

    print("train metric", eps_t1[0], eps_t2[0], eps_t3[0], rot_x_all[0], rot_y_all[0], rot_z_all[0])

    return (eps_t1, eps_t2, eps_t3, rot_x_all, rot_y_all, rot_z_all), (deps_t1, deps_t2, deps_t3, rot_x_all2, rot_y_all2, rot_z_all2)

def sample_data(dataset, num_samples=5):
    """Randomly sample data points from the dataset."""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return torch.stack([dataset[i] for i in range(0,5)])

def optimize_samples(samples, diffusion, gmm_model, args, target_likelihood=1e-5, num_samples=5, num_epochs=10, lr=0.01, 
                     likelihood_weight=1e-5, smoothness_weight=1e-3, basis_type='rbf', num_basis=10,
                     trans_init=None, rot_init=None):
    """
    Optimize selected samples through gradient descent with smoothness penalty.
    
    Args:
        samples: Input samples to optimize
        diffusion: Diffusion model
        gmm_model: GMM model for likelihood evaluation
        args: Arguments
        num_samples: Number of samples to optimize
        num_epochs: Number of optimization epochs
        lr: Learning rate
        smoothness_weight: Weight for the smoothness penalty term
        basis_type: Type of basis functions ('fourier' or 'rbf')
        num_basis: Number of basis functions to use
    """
    samples.requires_grad_(True)

    optimizer = optim.Adam([samples], lr=lr)
    
    # Get translations for smoothness penalty
    B, L, _, _ = samples.shape
    
    # Create basis functions
    if basis_type == 'rbf':
        # Create RBF basis
        def compute_rbf_penalty(translations):
            """Compute smoothness penalty using RBF basis"""
            # Create RBF centers evenly spaced in the sequence
            centers = torch.linspace(0, L-1, num_basis, device=samples.device).long()
            sigma = L / (num_basis * 2)  # RBF width parameter
            
            # Compute second derivatives approximation
            diff1 = translations[:, 1:] - translations[:, :-1]
            diff2 = diff1[:, 1:] - diff1[:, :-1]  # Second derivative approximation
            
            # Apply RBF weighting - emphasize smoothness at center points
            smoothness_loss = 0
            for center in centers:
                if center < L-2:  # Ensure center is valid for diff2
                    # Compute RBF weights
                    t = torch.arange(L-2, device=samples.device)
                    rbf_weights = torch.exp(-((t - center)**2) / (2 * sigma**2))
                    
                    # Weight the second derivatives
                    weighted_diff = diff2 * rbf_weights.view(1, L-2, 1)
                    smoothness_loss += torch.sum(weighted_diff**2)
            
            return smoothness_loss / B  # Normalize by batch size
            
        smoothness_penalty_fn = compute_rbf_penalty
    
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    for epoch in range(num_epochs):
        # for param in diffusion.score_model.parameters():
        #   print(param)
        #   break
        # for param in gmm_model.parameters():
        #   print(param)
        #   break
        optimizer.zero_grad()
        # se3_samples = se3_from_vec(samples)
        se3_samples = samples
        if epoch == 0:
          print(se3_samples[0][0])
        
        # Compute GMM metrics
        stats_first_order, stats_second_order = diffusion_metrics_batch(
            diffusion, se3_samples, args, diffusion.score_model,
            trans_init=trans_init, rot_init=rot_init, label='GD'
        )
        
        if args.ood_mode == "R3":
            data = torch.stack([
                stats_first_order[0], stats_first_order[1], stats_first_order[2],
                stats_second_order[0], stats_second_order[1], stats_second_order[2]
            ], dim=1)
        else:
            data = torch.stack([
                *stats_first_order, *stats_second_order
            ], dim=1)
        
        # Compute GMM likelihood loss
        likelihood = gmm_model.score_samples(data)

        if likelihood < target_likelihood:
            print(f"Early Stop with likelihood {likelihood.item()} and target {target_likelihood}\n")
            # break

        gmm_loss = likelihood * likelihood_weight
        
        # Compute smoothness penalty
        smoothness_loss = smoothness_weight * smoothness_penalty_fn(samples.flatten(2,3))
        
        # Combine losses
        total_loss = gmm_loss + smoothness_loss
        
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - GMM Loss: {gmm_loss.item()}, " 
              f"Smoothness Loss: {smoothness_loss.item():.4f}, "
              f"Total Loss: {total_loss.item():.4f}")

    return samples.detach()

def plot_trajectories(original_samples, optimized_samples, save_folder="trajectory_comparisons", step=5, scale=0.5):
    """
    Saves 3D trajectory comparison plots for each sample.
    
    Args:
        original_samples (torch.Tensor): Initial trajectory samples (N, num_steps, 3, 3).
        optimized_samples (torch.Tensor): Optimized trajectory samples (N, num_steps, 3, 3).
        save_folder (str): Directory to save comparison plots.
        step (int): Step size for visualization.
        scale (float): Scaling factor for axis length.
    
    Returns:
        None (Saves individual trajectory plots).
    """
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

    num_samples = original_samples.shape[0]
    
    for i in range(num_samples):
        fig = plt.figure(figsize=(12, 6))

        # Original Sample Plot
        ax1 = fig.add_subplot(121, projection='3d')
        plot_se3_trajectory(ax1, original_samples[i], title=f"Original Sample {i+1}", step=step, scale=scale)

        # Optimized Sample Plot
        ax2 = fig.add_subplot(122, projection='3d')
        plot_se3_trajectory(ax2, optimized_samples[i], title=f"Optimized Sample {i+1}", step=step, scale=scale)

        save_path = os.path.join(save_folder, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved trajectory comparison plot: {save_path}")

def plot_se3_trajectory(ax, trajectory, title="Trajectory", step=5, scale=0.3):
    """
    Plots an SE3 trajectory on a given 3D axis.

    Args:
        ax (Axes3D): Matplotlib 3D axis.
        rotations (torch.Tensor): SE3 rotation matrices (num_steps, 3, 3).
        title (str): Title for the subplot.
        step (int): Step size for visualization.
        scale (float): Scale for axis length.
    """
    colors = ['r', 'g', 'b']  # X, Y, Z axes
    ax.set_title(title)

    translation = trajectory[:, :3, 3]
    rotations = trajectory[:, :3, :3]

    scale = translation.mean() / 3

    R = rotations

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    ax.plot(translation[:, 0], translation[:, 1], translation[:, 2], label="Trajectory", color="blue")

    for i, p in enumerate(trajectory):
        if i % step != 0:
            continue
        trans = translation[i]
        ax.quiver(trans[0], trans[1], trans[2], x_axis[0], x_axis[1], x_axis[2], length=scale, color='r', linewidth=1.5, alpha=0.6)
        ax.quiver(trans[0], trans[1], trans[2], y_axis[0], y_axis[1], y_axis[2], length=scale, color='g', linewidth=1.5, alpha=0.6)
        ax.quiver(trans[0], trans[1], trans[2], z_axis[0], z_axis[1], z_axis[2], length=scale, color='b', linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Step")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def main():
    args = parse_arguments()
    
    in_dataset, in_dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)

    total_length = len(in_dataset)  
    train_length = int(0.9 * total_length) 
    val_length = total_length - train_length 
    train_dataset, val_dataset = random_split(in_dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    #out_dataset, out_dataloader = get_data(args.out_dataset, args.out_data_folder, args.out_data_stride, args)

    # Initialize the model
    if args.model_type == "Transformer":
        loaded_model = DoubleTransformerEncoderUnet(dim=args.hidden_dim, num_heads=args.n_heads, num_layers=args.n_layers, unet_layer=args.unet_layer).to(args.device)
    else:
        loaded_model = DoubleUnet(dim=args.hidden_dim, unet_layer=args.unet_layer).cuda()

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # Set the model to evaluation mode
    loaded_model.eval()
    for param in loaded_model.parameters():
        param.requires_grad_(False)

    # Perform inference
    diffusion_in = DDPM_Diff(loaded_model, trans_scale=args.scale_trans_in)

    L = args.n 
    trans_init = torch.randn(1,L,3, device=args.device)
    v_T = torch.randn(1,L,3, device=args.device)
    rot_init = so3_exp_map(v_T)
    stats_first_order, stats_second_order = diffusion_metrics(diffusion_in, in_dataloader, args, 
              loaded_model, trans_single=trans_init, rot_single=rot_init, label='train')
    # _, _ = diffusion_metrics(diffusion_in, in_dataloader, args, 
    #           loaded_model, trans_single=trans_init, rot_single=rot_init, label='train')
    # _, _ = diffusion_metrics(diffusion_in, in_dataloader, args, loaded_model, 'train')
    # _, _ = diffusion_metrics(diffusion_in, in_dataloader, args, loaded_model, 'train')

    in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3 = stats_first_order
    in_deps_t1, in_deps_t2, in_deps_t3, in_deps_r1, in_deps_r2, in_deps_r3 = stats_second_order
    print("finish calculating stats for train")

    # Fit GMM on in distribution

    if args.ood_mode == "R3":
        data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_deps_t1, in_deps_t2, in_deps_t3])
    else:
        data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3,in_deps_t1, in_deps_t2, in_deps_t3, in_deps_r1, in_deps_r2, in_deps_r3])
  
    n_components = 1
    data = torch.tensor(data, dtype=torch.float32, device=args.device)
    gmm = GaussianMixture(n_components, data.shape[1], eps=1e-2, device=args.device)
    gmm.fit(data)

    diffusion_in.score_model.eval()
    gmm.eval()
    
    # Disable gradients for model parameters but keep forward pass active
    for param in gmm.parameters():
        param.requires_grad_(False)

    print("finish fitting gmm")
    train_probs = gmm.score_samples(data)
    lower_threshold = np.percentile(train_probs.detach().cpu().numpy(), 5)
    print(data[0], train_probs[0], lower_threshold)
    
    for batch in in_dataloader:
        samples = batch[0].unsqueeze(0).to(args.device)
        break
    # samples = sample_data(train_dataset, 1).to(args.device)

    original_samples = samples.clone().detach().cpu().numpy()
    new_samples = samples.clone()
    # new_samples = se3_to_vec(new_samples)

    # Optimize sampled data points
    optimized_samples = optimize_samples(new_samples, diffusion_in, gmm, args, target_likelihood=lower_threshold, num_samples=1, num_epochs=args.num_epochs, \
                                         lr=args.lr, likelihood_weight=args.likelihood_weight, smoothness_weight=args.smoothness_weight, \
                                         trans_init=trans_init, rot_init=rot_init)
    
    optimized_samples = se3_from_vec(optimized_samples)

    optimized_samples = optimized_samples.detach().cpu().numpy()

    plot_trajectories(original_samples, optimized_samples, save_folder=args.save_folder, step=5, scale=0.03)

if __name__ == "__main__":
    main()