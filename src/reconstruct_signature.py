import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import esig

from utils import *
from KITTI_dataset import KITTIOdometryDataset
from Oxford_Robotcar_dataset import RobotcarDataset

def compute_batch_path_signatures(batch, depth):
    signatures = [esig.stream2sig(path, depth) for path in batch]
    return torch.tensor(np.stack(signatures), dtype=torch.float32)

# Define the transformer-based reconstruction model
class TransformerReconstructor(nn.Module):
    def __init__(self, signature_dim, path_length):
        super(TransformerReconstructor, self).__init__()
        self.embedding = nn.Linear(signature_dim, 128)
        self.transformer = nn.Transformer(128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
        self.fc_out = nn.Linear(128, 3)  # Output dim 3 for each path point
        self.path_length = path_length

    def forward(self, signature):
        embedded = self.embedding(signature).unsqueeze(0)  # Add sequence dimension
        tgt = torch.zeros(self.path_length, embedded.size(1), 128, device=embedded.device)  # Target sequence placeholder
        transformer_output = self.transformer(embedded, tgt)
        reconstructed_path = self.fc_out(transformer_output).squeeze(0).transpose(0,1)
        return reconstructed_path

# Training function
def train_model(model_trans, model_rot, depth, dataloader, criterion, optimizer, device, epochs, save_path):
    model_trans.train()
    model_rot.train()
    # wandb.watch(model, log="all")

    for epoch in range(epochs):
        total_loss = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for paths in pbar:
                paths = paths.to(device)
                l = paths.size(1)

                trans_path = paths[:, :, :3, 3]
                rot_so3   = paths[:, :, :3, :3]
                rot_path   = so3_log_map(rot_so3)

                trans_np = trans_path.cpu().numpy()
                rot_np = rot_path.cpu().numpy()

                trans_sig = compute_batch_path_signatures(trans_np, depth).to(device)
                rot_sig = compute_batch_path_signatures(rot_np, depth).to(device)

                # trans_sig = torch.tensor(trans_sig, device=device)
                # rot_sig = torch.tensor(rot_sig, device=device)
                
                optimizer.zero_grad()

                reconstructed_paths_trans = model_trans(trans_sig)
                reconstructed_paths_rot = model_rot(rot_sig)

                # print(trans_sig.shape, trans_path.shape, reconstructed_paths_trans.shape)
                loss = criterion(reconstructed_paths_trans, trans_path) + \
                       criterion(reconstructed_paths_rot, rot_path)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(dataloader))

        wandb.log({"epoch": epoch + 1, "loss": total_loss / len(dataloader)})

    # Save the model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model_trans.state_dict(), os.path.join(save_path, "trans_reconstructor.pth"))
    torch.save(model_rot.state_dict(), os.path.join(save_path, "rot_reconstructor.pth"))

# Main function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="path-reconstruction", config=args)

    # Generate example data (replace with real data loading)
    dataset = RobotcarDataset(args.data_folder, seq_len=args.path_length, stride=args.data_stride, center=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    signature_dim = 0
    for i in range(args.depth + 1):
        signature_dim += 3 ** i
    model_trans = TransformerReconstructor(signature_dim, args.path_length).to(device)
    model_rot = TransformerReconstructor(signature_dim, args.path_length).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(list(model_trans.parameters()) + list(model_rot.parameters()), lr=args.lr)
    train_model(model_trans, model_rot, args.depth, dataloader, criterion, optimizer, device, args.epochs, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path Signature Reconstruction")
    parser.add_argument("--path_length", type=int, default=10, help="Length of the paths")
    parser.add_argument("--data_stride", type=int, default=1, help="stride for splitting data sequence to seq len")
    parser.add_argument("--depth", type=int, default=2, help="Depth of the path signature")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="models", help="Folder to save the trained model")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing the dataset.")

    args = parser.parse_args()
    main(args)
