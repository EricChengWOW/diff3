# from numpy.lib import nanprod
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
import scipy.stats as stats

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

    return parser.parse_args()

### Save plots of distributions
def save_plot(in_data, out_data, in_dataset, out_dataset, metric="eps", path = "temp.png"):
    plt.hist([in_data, out_data], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=[in_dataset, out_dataset])

    # Add labels and title
    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Freq', fontsize=14)
    plt.title('Distribution of ' + metric)
    plt.legend(fontsize=14)

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

def diffusion_metrics(diffusion, dataloader, args, loaded_model, label="", lambda_t_train=[], lambda_r_train=[]):
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
        batch_cnt += 1

        batch = batch.to(args.device)
        rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
        translations = batch[:, :, :3, 3]  # [B, n, 3]

        B, L, _ = translations.shape

        score_norm_t1 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t3 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

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

        trans_init = torch.randn_like(translations)
        v_T = torch.randn(B,L,3, device=translations.device)
        rot_init = so3_exp_map(v_T)

        for t in range(15):
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
            score_norm_r1 += torch.mean(rot_score, dim=(-2, -1))
            score_norm_r2 += torch.mean(rot_score ** 2, dim=(-2, -1))
            score_norm_r3 += torch.mean(rot_score ** 3, dim=(-2, -1))

            arr1 = rot_score[:, :, 0].cpu().numpy().flatten()
            arr2 = rot_score[:, :, 1].cpu().numpy().flatten()
            arr3 = rot_score[:, :, 2].cpu().numpy().flatten()
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
        rot_score_arr.append(score_norm_r1.detach().cpu().numpy())
        rot_score_arr_2.append(score_norm_r2.detach().cpu().numpy())
        rot_score_arr_3.append(score_norm_r3.detach().cpu().numpy())

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

    rot_scores_1 = np.concatenate(rot_score_arr, axis=0)
    rot_scores_2 = np.concatenate(rot_score_arr_2, axis=0)
    rot_scores_3 = np.concatenate(rot_score_arr_3, axis=0)
    drot_scores_1 = np.concatenate(drot_score_arr, axis=0)
    drot_scores_2 = np.concatenate(drot_score_arr_2, axis=0)
    drot_scores_3 = np.concatenate(drot_score_arr_3, axis=0)

    arr1 = np.concatenate(trans_x, axis=0)
    arr2 = np.concatenate(trans_y, axis=0)
    arr3 = np.concatenate(trans_z, axis=0)
    plt.hist([arr1, arr2, arr3], bins=50, color=['#4169E1', 'brown', 'orange'], label=['x', 'y', 'z'])

    # Add labels and title
    plt.xlabel(label+' Rotation metrics', fontsize=14)
    plt.ylabel('Freq', fontsize=14)
    # plt.title('Distribution')
    plt.legend(fontsize=14)

    plt.savefig('./tranxyz_' + label + '.png', dpi=300, bbox_inches='tight')
    plt.close()

    rot_x_all = np.concatenate(rot_x, axis=0)
    rot_y_all = np.concatenate(rot_y, axis=0)
    rot_z_all = np.concatenate(rot_z, axis=0)
    rot_x_all2 = np.concatenate(rot_x2, axis=0)
    rot_y_all2 = np.concatenate(rot_y2, axis=0)
    rot_z_all2 = np.concatenate(rot_z2, axis=0)

    # all_met = [rot_x_all2, rot_y_all2, rot_z_all2]

    # lambda_t_arr = []
    # lambda_r_arr = []
    # if label == 'train':
    #     for i in range(len(all_met)):
    #         all_met[i],   lambda_r = stats.boxcox(all_met[i] + 1)
    #         # all_met_t[i], lambda_t = stats.boxcox(all_met_t[i] + 1)
    #         # lambda_t_arr.append(lambda_t)
    #         lambda_r_arr.append(lambda_r)
    # else:
    #     for i in range(len(all_met)):
    #         lambda_r = lambda_r_train[i]
    #         # lambda_t = lambda_t_train[i]
    #         all_met[i] = (np.power(all_met[i]+1, lambda_r) - 1) / lambda_r if lambda_r != 0 else np.log(all_met[i]+1)
    #         # all_met_t[i] = (np.power(all_met_t[i]+1, lambda_t) - 1) / lambda_t if lambda_t != 0 else np.log(all_met_t[i]+1)

    # rot_x_all2, rot_y_all2, rot_z_all2 = all_met

    eps_t1 = trans_scores_1
    eps_t2 = np.sqrt(trans_scores_2)
    eps_t3 = trans_scores_3

    deps_t1 = dtrans_scores_1
    deps_t2 = np.sqrt(dtrans_scores_2)
    deps_t3 = dtrans_scores_3
    
    return (eps_t1, eps_t2, eps_t3, rot_scores_1, rot_scores_2, rot_scores_3), (deps_t1, deps_t2, deps_t3, drot_scores_1, drot_scores_2, drot_scores_3), []
    # return (eps_t1, eps_t2, eps_t3, rot_x_all, rot_y_all, rot_z_all), (deps_t1, deps_t2, deps_t3, rot_x_all2, rot_y_all2, rot_z_all2), []

def main():
    args = parse_arguments()
    
    in_dataset, in_dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)

    total_length = len(in_dataset)  
    train_length = int(0.9 * total_length) 
    val_length = total_length - train_length 
    train_dataset, val_dataset = random_split(in_dataset, [train_length, val_length], generator=torch.Generator().manual_seed(42))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    out_dataset, out_dataloader = get_data(args.out_dataset, args.out_data_folder, args.out_data_stride, args)

    # Initialize the model
    if args.model_type == "Transformer":
        loaded_model = DoubleTransformerEncoderUnet(dim=args.hidden_dim, num_heads=args.n_heads, num_layers=args.n_layers, unet_layer=args.unet_layer).to(args.device)
    else:
        loaded_model = DoubleUnet(dim=args.hidden_dim, unet_layer=args.unet_layer).cuda()

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # Set the model to evaluation mode
    loaded_model.eval()

    # Perform inference
    diffusion_in = DDPM_Diff(loaded_model, trans_scale=args.scale_trans_in)
    diffusion_out = DDPM_Diff(loaded_model, trans_scale=args.scale_trans_out)

    stats_first_order, stats_second_order, lambda_arr = diffusion_metrics(diffusion_in, in_dataloader, args, loaded_model, label='train')
    in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3 = stats_first_order
    in_deps_t1, in_deps_t2, in_deps_t3, in_deps_r1, in_deps_r2, in_deps_r3 = stats_second_order
    print("finish calculating stats for train")
    # val
    stats_first_order, stats_second_order,_ = diffusion_metrics(diffusion_in, val_dataloader, args, loaded_model, label='val', lambda_r_train=lambda_arr)
    val_eps_t1, val_eps_t2, val_eps_t3, val_eps_r1, val_eps_r2, val_eps_r3 = stats_first_order
    val_deps_t1, val_deps_t2, val_deps_t3, val_deps_r1, val_deps_r2, val_deps_r3 = stats_second_order
    print("finish calculating stats for val")
    # Get test_set
    stats_first_order, stats_second_order,_ = diffusion_metrics(diffusion_out, out_dataloader, args, loaded_model, label='test', lambda_r_train=lambda_arr)
    out_eps_t1, out_eps_t2, out_eps_t3, out_eps_r1, out_eps_r2, out_eps_r3 = stats_first_order
    out_deps_t1, out_deps_t2, out_deps_t3, out_deps_r1, out_deps_r2, out_deps_r3 = stats_second_order
    print("finish calculating stats for test")

    save_plot(in_eps_t1, out_eps_t1, args.in_dataset, args.out_dataset, metric="|eps_trans|", path=args.save_folder + "/eps_trans.png")
    save_plot(in_eps_t2, out_eps_t2, args.in_dataset, args.out_dataset, metric="|eps_trans^2|", path=args.save_folder + "/eps_trans2.png")
    save_plot(in_eps_t3, out_eps_t3, args.in_dataset, args.out_dataset, metric="eps_trans3", path=args.save_folder + "/eps_trans3.png")
    save_plot(in_eps_r1, out_eps_r1, args.in_dataset, args.out_dataset, metric="|eps_rot_x|", path=args.save_folder + "/eps_rot.png")
    save_plot(in_eps_r2, out_eps_r2, args.in_dataset, args.out_dataset, metric="|eps_rot_y|", path=args.save_folder + "/eps_rot2.png")
    save_plot(in_eps_r3, out_eps_r3, args.in_dataset, args.out_dataset, metric="|eps_rot_z|", path=args.save_folder + "/eps_rot3.png")

    save_plot(in_deps_t1, out_deps_t1, args.in_dataset, args.out_dataset, metric="deps_trans", path=args.save_folder + "/deps_trans.png")
    save_plot(in_deps_t2, out_deps_t2, args.in_dataset, args.out_dataset, metric="deps_trans2", path=args.save_folder + "/deps_trans2.png")
    save_plot(in_deps_t3, out_deps_t3, args.in_dataset, args.out_dataset, metric="deps_trans3", path=args.save_folder + "/deps_trans3.png")
    save_plot(in_deps_r1, out_deps_r1, args.in_dataset, args.out_dataset, metric="|eps_rot_x^2|", path=args.save_folder + "/deps_rot.png")
    save_plot(in_deps_r2, out_deps_r2, args.in_dataset, args.out_dataset, metric="|eps_rot_y^2|", path=args.save_folder + "/deps_rot2.png")
    save_plot(in_deps_r3, out_deps_r3, args.in_dataset, args.out_dataset, metric="|eps_rot_z^2|", path=args.save_folder + "/deps_rot3.png")
    print("finish saving distribution plots")

    # Fit GMM on in distribution

    if args.ood_mode == "R3":
        data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_deps_t1, in_deps_t2, in_deps_t3])
    else:
        data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3,in_deps_t1, in_deps_t2, in_deps_t3, in_deps_r1, in_deps_r2, in_deps_r3])
  
    n_components = 1
    gmm = GaussianMixture(n_components=n_components, random_state=42, reg_covar=1e-5)
    gmm.fit(data)
    print("finish fitting gmm")
    train_probs = np.exp(gmm.score_samples(data))
    #lower_threshold, upper_threshold = np.percentile(train_probs, 5), np.percentile(train_probs, 95)
    lower_threshold = np.percentile(train_probs, 5)

    if args.ood_mode == "R3":
        val_points = np.column_stack([val_eps_t1, val_eps_t2, val_eps_t3, val_deps_t1, val_deps_t2, val_deps_t3])
    else:
        val_points = np.column_stack([val_eps_t1, val_eps_t2, val_eps_t3, val_eps_r1, val_eps_r2, val_eps_r3, val_deps_t1, val_deps_t2, val_deps_t3, val_deps_r1, val_deps_r2, val_deps_r3])
    
    val_probs = np.exp(gmm.score_samples(val_points))
    ood_flags = (val_probs < lower_threshold)
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Val: {num_ood} / {len(val_points)}")
    # print(val_points.shape)

    if args.ood_mode == "R3":
        test_points = np.column_stack([out_eps_t1, out_eps_t2, out_eps_t3, out_deps_t1, out_deps_t2, out_deps_t3])
    else:
        test_points = np.column_stack([out_eps_t1, out_eps_t2, out_eps_t3, out_eps_r1, out_eps_r2, out_eps_r3, out_deps_t1, out_deps_t2, out_deps_t3, out_deps_r1, out_deps_r2, out_deps_r3])
    
    test_probs = np.exp(gmm.score_samples(test_points))

    ood_flags = (test_probs < lower_threshold)
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Test: {num_ood} / {len(ood_flags)}")
    # print(ood_flags.shape)

    val_labels = np.ones(len(val_probs)) 
    test_labels = np.zeros(len(test_probs))

    all_scores = np.concatenate([val_probs, test_probs])
    print(all_scores)
    all_labels = np.concatenate([val_labels, test_labels])

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"AUROC: {auroc}")

    aupr = average_precision_score(all_labels, all_scores)
    print(f"AUPR: {aupr}")

    val_ood_flags = (val_probs > lower_threshold)
    test_ood_flags = (test_probs > lower_threshold)

    val_preds = val_ood_flags.astype(int)
    test_preds = test_ood_flags.astype(int)
    all_preds = np.concatenate([val_preds, test_preds])
    print(all_labels)
    print(all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")

if __name__ == "__main__":
    main()