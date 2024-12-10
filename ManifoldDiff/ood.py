from numpy.lib import nanprod
from utils import *
from unet import *
from DDPM_Diff import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset
from L_dataset import LDataset
from T_dataset import TDataset

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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

    return parser.parse_args()

def get_data(dataset, dataset_path, stride, args):
    if dataset == "KITTI":
        dataset = KITTIOdometryDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on KITTI for ", len(dataloader), " batches")
    elif dataset == "Oxford":
        dataset = RobotcarDataset(dataset_path, seq_len=args.n, stride=stride, center=args.center)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on Oxford Robot car for ", len(dataloader), " batches")
    elif dataset == "L":
        dataset = LDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on L shape for ", len(dataloader), " batches")
    elif dataset == "T":
        dataset = TDataset(seq_len=args.n)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        print("Running on T shape for ", len(dataloader), " batches")
    else:
        raise "Dataset type not supported"

    return dataset, dataloader

def diffusion_metrics(diffusion, dataloader, args, loaded_model):
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

    batch_cnt = 0

    # Gather the distribution result
    for batch in dataloader:
        batch_cnt += 1

        batch = batch.to(args.device)
        rotations = batch[:, :, :3, :3]  # [B, n, 3, 3]
        # rotations = matrix_to_so3(rotations)  # [B, n, 3]
        translations = batch[:, :, :3, 3]  # [B, n, 3]

        B, L, _ = translations.shape

        score_norm_t1 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r1 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t2 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r2 = torch.zeros((batch.size(0),), device=args.device)

        score_norm_t3 = torch.zeros((batch.size(0),), device=args.device)
        score_norm_r3 = torch.zeros((batch.size(0),), device=args.device)

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

        for t in range(args.num_timesteps // 2):
            t_tensor = torch.full((batch.size(0),), t, device=args.device)

            # trans_init = torch.randn_like(translations)
            (trans_t, _), (rot_t, _) = diffusion.forward_process(translations, rotations, t_tensor, trans_init=trans_init, rot_init=rot_init)

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

            score_norm_t1 += torch.sum(trans_score, dim=(-2, -1))
            score_norm_r1 += torch.sum(rot_score, dim=(-2, -1))
            score_norm_t2 += torch.sum(trans_score ** 2, dim=(-2, -1))
            score_norm_r2 += torch.sum(rot_score ** 2, dim=(-2, -1))
            score_norm_t3 += torch.sum(trans_score ** 3, dim=(-2, -1))
            score_norm_r3 += torch.sum(rot_score ** 3, dim=(-2, -1))

            if prev_score_t is not None:
                delta = trans_score - prev_score_t
                dscore_norm_t1 += torch.sum(delta, dim=(-2, -1))
                dscore_norm_t2 += torch.sum(delta ** 2, dim=(-2, -1))
                dscore_norm_t3 += torch.sum(delta ** 3, dim=(-2, -1))

                delta = trans_score - prev_score_r
                dscore_norm_r1 += torch.sum(delta, dim=(-2, -1))
                dscore_norm_r2 += torch.sum(delta ** 2, dim=(-2, -1))
                dscore_norm_r3 += torch.sum(delta ** 3, dim=(-2, -1))

            prev_score_t = trans_score
            prev_score_r = rot_score

        rot_score_arr.append(score_norm_r1.detach().cpu().numpy())
        rot_score_arr_2.append(score_norm_r2.detach().cpu().numpy())
        rot_score_arr_3.append(score_norm_r3.detach().cpu().numpy())
        trans_score_arr.append(score_norm_t1.detach().cpu().numpy())
        trans_score_arr_2.append(score_norm_t2.detach().cpu().numpy())
        trans_score_arr_3.append(score_norm_t3.detach().cpu().numpy())

        drot_score_arr.append(dscore_norm_r1.detach().cpu().numpy())
        drot_score_arr_2.append(dscore_norm_r2.detach().cpu().numpy())
        drot_score_arr_3.append(dscore_norm_r3.detach().cpu().numpy())
        dtrans_score_arr.append(dscore_norm_t1.detach().cpu().numpy())
        dtrans_score_arr_2.append(score_norm_t2.detach().cpu().numpy())
        dtrans_score_arr_3.append(score_norm_t3.detach().cpu().numpy())

    rot_scores_1 = np.concatenate(rot_score_arr, axis=0)
    rot_scores_2 = np.concatenate(rot_score_arr_2, axis=0)
    rot_scores_3 = np.concatenate(rot_score_arr_3, axis=0)
    trans_scores_1 = np.concatenate(trans_score_arr, axis=0)
    trans_scores_2 = np.concatenate(trans_score_arr_2, axis=0)
    trans_scores_3 = np.concatenate(trans_score_arr_3, axis=0)

    drot_scores_1 = np.concatenate(drot_score_arr, axis=0)
    drot_scores_2 = np.concatenate(drot_score_arr_2, axis=0)
    drot_scores_3 = np.concatenate(drot_score_arr_3, axis=0)
    dtrans_scores_1 = np.concatenate(dtrans_score_arr, axis=0)
    dtrans_scores_2 = np.concatenate(dtrans_score_arr_2, axis=0)
    dtrans_scores_3 = np.concatenate(dtrans_score_arr_3, axis=0)

    eps_t1 = trans_scores_1
    eps_t2 = np.sqrt(trans_scores_2)
    eps_t3 = trans_scores_3
    eps_r1 = rot_scores_1
    eps_r2 = np.sqrt(rot_scores_2)
    eps_r3 = rot_scores_3

    deps_t1 = dtrans_scores_1
    deps_t2 = np.sqrt(dtrans_scores_2)
    deps_t3 = dtrans_scores_3
    deps_r1 = drot_scores_1
    deps_r2 = np.sqrt(drot_scores_2)
    deps_r3 = drot_scores_3

    return (eps_t1, eps_t2, eps_t3, eps_r1, eps_r2, eps_r3), (deps_t1, deps_t2, deps_t3, deps_r1, deps_r2, deps_r3)

def main():
    args = parse_arguments()

    in_dataset, in_dataloader = get_data(args.in_dataset, args.in_data_folder, args.in_data_stride, args)
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
    diffusion = DDPM_Diff(loaded_model, trans_scale=args.scale_trans)

    stats_first_order, stats_second_order = diffusion_metrics(diffusion, in_dataloader, args, loaded_model)
    in_eps_t1, in_eps_t2, in_eps_t3, in_eps_r1, in_eps_r2, in_eps_r3 = stats_first_order
    in_deps_t1, in_deps_t2, in_deps_t3, in_deps_r1, in_deps_r2, in_deps_r3 = stats_first_order

    # Get test_set
    stats_first_order, stats_second_order = diffusion_metrics(diffusion, out_dataloader, args, loaded_model)
    out_eps_t1, out_eps_t2, out_eps_t3, out_eps_r1, out_eps_r2, out_eps_r3 = stats_first_order
    out_deps_t1, out_deps_t2, out_deps_t3, out_deps_r1, out_deps_r2, out_deps_r3 = stats_first_order

    # Fit GMM on in distribution
    data = np.column_stack([in_eps_t1, in_eps_t2, in_eps_t3, in_deps_t1, in_deps_t2, in_deps_t3])

    n_components = 1
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    test_points = np.column_stack([out_eps_t1, out_eps_t2, out_eps_t3, out_deps_t1, out_deps_t2, out_deps_t3])

    probs = gmm.score_samples(test_points)
    # print(probs)

    def save_plot(in_data, out_data, metric="eps", path = "temp.png"):
        plt.hist([in_data, out_data], bins=50, color=['skyblue', 'orange'], edgecolor='black', label=['In', 'Out'])

        # Add labels and title
        plt.xlabel(metric)
        plt.ylabel('Freq')
        plt.title('Distribution of ' + metric)

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    save_plot(in_eps_t1, out_eps_t1, metric="eps_trans", path=args.save_folder + "/eps_trans.png")
    save_plot(in_eps_t2, out_eps_t2, metric="eps_trans2", path=args.save_folder + "/eps_trans2.png")
    save_plot(in_eps_t3, out_eps_t3, metric="eps_trans3", path=args.save_folder + "/eps_trans3.png")

    save_plot(in_deps_t1, out_deps_t1, metric="deps_trans", path=args.save_folder + "/deps_trans.png")
    save_plot(in_deps_t2, out_deps_t2, metric="deps_trans2", path=args.save_folder + "/deps_trans2.png")
    save_plot(in_deps_t3, out_deps_t3, metric="deps_trans3", path=args.save_folder + "/deps_trans3.png")

if __name__ == "__main__":
    main()
