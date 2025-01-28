#from numpy.lib import nanprod
from re import I
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

import statistics
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
    parser.add_argument("--random_seq_len",action='store_true', help="Enable random sequence length mode")

    return parser.parse_args()

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.path_signature_depth, random_seq_len = args.random_seq_len)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on KITTI for ", len(dataloader), " batches")
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.path_signature_depth, random_seq_len = args.random_seq_len)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on Oxford Robot car for ", len(dataloader), " batches")
    elif dataset == "IROS":
        dataset = IROS20Dataset(dataset_path, seq_len=args.n, stride=stride, center=args.center, use_path_signature = True, scale_trans = args.scale_trans, level = args.path_signature_depth)
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

def diffusion_metrics(ldm, dataloader, args, lambda_t_train=[], lambda_r_train=[], label=''):
    rot_score_arr = []
    rot_score_arr_2 = []
    rot_score_arr_3 = []
    drot_score_arr = []
    drot_score_arr_2 = []
    drot_score_arr_3 = []
    trans_score_arr = []
    trans_score_arr_2 = []
    trans_score_arr_3 = []

    entry_arr = []
    lvl1_arr = []
    lvl2_arr = []
    lvl3_arr = []

    all_arr = []
    all_arr_t = []

    batch_cnt = 0

    # Gather the distribution result
    for batch in dataloader:
        batch_cnt += 1

        batch = batch.to(args.device)

        score_norm_t1 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_t2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_t3 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r1 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r2 = torch.zeros((batch.size(0),), device=args.device)
        dscore_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

        all_metric = torch.zeros((batch.size(0), batch.size(1) // 2), device=args.device)
        all_metric_t = torch.zeros((batch.size(0), batch.size(1) // 2), device=args.device)

        prev_score_r = None

        # x_init = torch.randn_like(x)

        for t in range(10):
            t_tensor = torch.full((batch.size(0),), t, device=args.device)

            (x_t, _) = ldm.forward_process(batch, t_tensor)

            # Predict scores using the score model
            with torch.no_grad():
                l = x_t.size(1)

                trans_sig = x_t[:, : l//2]
                rot_sig   = x_t[:, l//2 :]
                trans_score = ldm.model_trans(trans_sig, t_tensor)
                rot_score = ldm.model_rot(rot_sig, t_tensor)
                # trans_score = signature_exp(trans_score, 3, args.path_signature_depth)
                # rot_score = signature_exp(rot_score, 3, args.path_signature_depth)

            score_norm_t1 += torch.mean(trans_score, dim=1)
            score_norm_t2 += torch.mean(trans_score ** 2, dim=1)
            score_norm_t3 += torch.mean(trans_score ** 3, dim=1)
            score_norm_r1 += torch.mean(rot_score, dim=1)
            score_norm_r2 += torch.mean(rot_score ** 2, dim=1)
            score_norm_r3 += torch.mean(rot_score ** 3, dim=1)

            all_metric += rot_score ** 2
            all_metric_t += trans_score ** 2

            # all_arr.append((rot_score ** 2).transpose(0,1).cpu().numpy())
            # all_arr_t.append((trans_score ** 2).transpose(0,1).cpu().numpy())

            lvl1_arr.append(rot_score[1:4].cpu().numpy().flatten())
            lvl2_arr.append(rot_score[4:13].cpu().numpy().flatten())
            lvl3_arr.append(rot_score[13:40].cpu().numpy().flatten())

            if prev_score_r is not None:
                delta = rot_score - prev_score_r
                dscore_norm_r1 += torch.mean(delta, dim=1)
                dscore_norm_r2 += torch.mean(delta ** 2, dim=1)
                dscore_norm_r3 += torch.mean(delta ** 3, dim=1)

            prev_score_r = rot_score

        all_arr.append(all_metric.transpose(0,1).cpu().numpy())
        all_arr_t.append(all_metric_t.transpose(0,1).cpu().numpy())

        trans_score_arr.append(score_norm_t1.detach().cpu().numpy())
        trans_score_arr_2.append(score_norm_t2.detach().cpu().numpy())
        trans_score_arr_3.append(score_norm_t3.detach().cpu().numpy())
        rot_score_arr.append(score_norm_r1.detach().cpu().numpy())
        rot_score_arr_2.append(score_norm_r2.detach().cpu().numpy())
        rot_score_arr_3.append(score_norm_r3.detach().cpu().numpy())

        drot_score_arr.append(dscore_norm_r1.detach().cpu().numpy())
        drot_score_arr_2.append(dscore_norm_r2.detach().cpu().numpy())
        drot_score_arr_3.append(dscore_norm_r3.detach().cpu().numpy())

    trans_scores_1 = np.concatenate(trans_score_arr, axis=0)
    trans_scores_2 = np.concatenate(trans_score_arr_2, axis=0)
    trans_scores_3 = np.concatenate(trans_score_arr_3, axis=0)
    rot_scores_1 = np.concatenate(rot_score_arr, axis=0)
    rot_scores_2 = np.concatenate(rot_score_arr_2, axis=0)
    rot_scores_3 = np.concatenate(rot_score_arr_3, axis=0)
    drot_scores_1 = np.concatenate(drot_score_arr, axis=0)
    drot_scores_2 = np.concatenate(drot_score_arr_2, axis=0)
    drot_scores_3 = np.concatenate(drot_score_arr_3, axis=0)

    lvl1 = np.concatenate(lvl1_arr, axis=0)
    lvl2 = np.concatenate(lvl2_arr, axis=0)
    lvl3 = np.concatenate(lvl3_arr, axis=0)
    all_met = np.hstack(all_arr)
    all_met_t = np.hstack(all_arr_t)

    lambda_t_arr = []
    lambda_r_arr = []
    if label == 'train':
        for i in range(all_met.shape[0]):
            all_met[i],   lambda_r = stats.boxcox(all_met[i] + 1)
            all_met_t[i], lambda_t = stats.boxcox(all_met_t[i] + 1)
            lambda_t_arr.append(lambda_t)
            lambda_r_arr.append(lambda_r)
    else:
        for i in range(all_met.shape[0]):
            lambda_r = lambda_r_train[i]
            lambda_t = lambda_t_train[i]
            all_met[i] = (np.power(all_met[i]+1, lambda_r) - 1) / lambda_r if lambda_r != 0 else np.log(all_met[i]+1)
            all_met_t[i] = (np.power(all_met_t[i]+1, lambda_t) - 1) / lambda_t if lambda_t != 0 else np.log(all_met_t[i]+1)

    eps_r1 = rot_scores_1
    eps_r2 = np.sqrt(rot_scores_2)
    eps_r3 = rot_scores_3

    eps_t1 = trans_scores_1
    eps_t2 = np.sqrt(trans_scores_2)
    eps_t3 = trans_scores_3
    # eps_t1 = drot_scores_1
    # eps_t2 = np.sqrt(drot_scores_2)
    # eps_t3 = drot_scores_3

    return (eps_t1, eps_t2, eps_t3), (eps_r1, eps_r2, eps_r3), all_met, all_met_t, lambda_t_arr, lambda_r_arr

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
    # for i in range(args.path_signature_depth + 1):
    #   input_dim += 6 ** i
    for i in range(args.path_signature_depth + 1):
      input_dim += 3 ** i
    input_dim *= 2

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
        depth=args.path_signature_depth
    )

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path, weights_only=True))
    loaded_model.to(args.device)

    # Set the model to evaluation mode
    loaded_model.eval()

    stats_first_order, stats_second_order, met_in, met_in_t, lambda_t, lambda_r = diffusion_metrics(loaded_model, in_dataloader, args, label='train')
    in_eps_t1, in_eps_t2, in_eps_t3 = stats_first_order
    in_eps_r1, in_eps_r2, in_eps_r3 = stats_second_order
    print("finish calculating stats for train")
    # val
    stats_first_order, stats_second_order, met_val, met_val_t,_,_ = diffusion_metrics(loaded_model, val_dataloader, args, lambda_r_train=lambda_r, lambda_t_train=lambda_t, label='val')
    val_eps_t1, val_eps_t2, val_eps_t3 = stats_first_order
    val_eps_r1, val_eps_r2, val_eps_r3 = stats_second_order
    print("finish calculating stats for val")
    # Get test_set
    stats_first_order, stats_second_order, met_out, met_out_t,_,_ = diffusion_metrics(loaded_model, out_dataloader, args, lambda_r_train=lambda_r, lambda_t_train=lambda_t, label='test')
    out_eps_t1, out_eps_t2, out_eps_t3 = stats_first_order
    out_eps_r1, out_eps_r2, out_eps_r3 = stats_second_order
    print("finish calculating stats for test")

    for i in range(40):
        plt.hist([met_in[i], met_val[i], met_out[i]], bins=50, color=['skyblue', 'red', 'yellow'], edgecolor='black', label=['in', 'val', 'out'])

        # Add labels and title
        plt.xlabel('metric')
        plt.ylabel('Freq')
        plt.title('Distribution')
        plt.legend()

        plt.savefig(args.save_folder + '/rot_' + str(i) + '.png', dpi=300, bbox_inches='tight')
        plt.close()

    for i in range(40):
        plt.hist([met_in_t[i], met_val_t[i], met_out_t[i]], bins=50, color=['skyblue', 'red', 'yellow'], edgecolor='black', label=['in', 'val', 'out'])

        # Add labels and title
        plt.xlabel('metric')
        plt.ylabel('Freq')
        plt.title('Distribution')
        plt.legend()

        plt.savefig(args.save_folder + '/trans_' + str(i) + '.png', dpi=300, bbox_inches='tight')
        plt.close()

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

    save_plot(in_eps_r1, out_eps_r1, args.in_dataset, args.out_dataset, metric="eps_sigr", path=args.save_folder + "/eps_sigr.png")
    save_plot(in_eps_r2, out_eps_r2, args.in_dataset, args.out_dataset, metric="eps_sigr2", path=args.save_folder + "/eps_sigr2.png")
    save_plot(in_eps_r3, out_eps_r3, args.in_dataset, args.out_dataset, metric="eps_sigr3", path=args.save_folder + "/eps_sigr3.png")

    save_plot(in_eps_t1, out_eps_t1, args.in_dataset, args.out_dataset, metric="eps_sigt", path=args.save_folder + "/eps_sigt.png")
    save_plot(in_eps_t2, out_eps_t2, args.in_dataset, args.out_dataset, metric="eps_sigt2", path=args.save_folder + "/eps_sigt2.png")
    save_plot(in_eps_t3, out_eps_t3, args.in_dataset, args.out_dataset, metric="eps_sigt3", path=args.save_folder + "/eps_sigt3.png")
    print("finish saving distribution plots")

    # Fit GMM on in distribution
    # data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3])
    data = np.hstack([met_in.T, met_in_t.T])
  
    n_components = 1
    gmm = GaussianMixture(n_components=n_components, random_state=42, reg_covar=1e-5)
    gmm.fit(data)
    print("finish fitting gmm")
    train_probs = np.exp(gmm.score_samples(data))
    lower_threshold = np.percentile(train_probs, 5)

    val_points = np.column_stack([val_eps_t1, val_eps_t2, val_eps_t3, val_eps_r1, val_eps_r2, val_eps_r3])
    val_points = np.hstack([met_val.T, met_val_t.T])
    val_probs = np.exp(gmm.score_samples(val_points))
    ood_flags = (val_probs < lower_threshold) 
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Val: {num_ood}/{ood_flags.shape[0]}")

    test_points = np.column_stack([out_eps_t1, out_eps_t2, out_eps_t3, out_eps_r1, out_eps_r2, out_eps_r3])
    test_points = np.hstack([met_out.T, met_out_t.T])
    test_probs = np.exp(gmm.score_samples(test_points))
    ood_flags = (test_probs < lower_threshold) 
    num_ood = np.sum(ood_flags)
    print(f"Number of OOD samples in Test: {num_ood}/{ood_flags.shape[0]}")

    val_labels = np.ones(len(val_probs)) 
    test_labels = np.zeros(len(test_probs))

    all_scores = np.concatenate([np.exp(val_probs), np.exp(test_probs)])
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
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    return auroc

if __name__ == "__main__":
    np.random.seed(200120)
    auroc_collection = []
    for i in range(1):
      auroc = main()
      auroc_collection.append(auroc)
    # print(auroc_collection)
    mean = statistics.mean(auroc_collection)
    variance = statistics.variance(auroc_collection)

    print("10-fold Mean:", mean)
    print("10-fold Variance:", variance)