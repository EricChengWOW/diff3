from utils import *
from unet import *
from DDPM_Diff import *
from transformer import *
import matplotlib.pyplot as plt
import numpy as np
import os

import argparse

def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        args: Parsed arguments as a namespace.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training with KITTI dataset.")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (default: 16).")
    parser.add_argument("--n", type=int, default=128, help="Number of data points per sequence (default: 128).")
    parser.add_argument("--model_type", type=str, default="Unet", help="The score model architecture")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers in the transformer")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of head in the transformer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128).")
    parser.add_argument("--num_timesteps", type=int, default=30, help="Number of timesteps for diffusion process (default: 100).")
    parser.add_argument("--scale_trans", type=float, default=1.0, help="Scale Factor for R3 translation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation, e.g., 'cuda' or 'cpu' (default: 'cuda').")
    parser.add_argument("--save_path", type=str, required=True, help="File to save the output image sample")
    parser.add_argument("--model_path", type=str, required=True, help="The model checkpoint path")
    parser.add_argument("--unet_layer", type=int, default=4, help="Layers of unet dim changes")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize the model
    if args.model_type == "Transformer":
        loaded_model = DoubleTransformerEncoderUnet(dim=args.hidden_dim, num_heads=args.n_heads, num_layers=args.n_layers, unet_layer=args.unet_layer).to(args.device)
    else:
        loaded_model = DoubleUnet(dim=args.hidden_dim, unet_layer=args.unet_layer).cuda()

    loaded_model.load_state_dict(torch.load(args.model_path, weights_only=True))

    loaded_model.eval()

    diffusion = DDPM_Diff(loaded_model, trans_scale=args.scale_trans)
    generated_se3 = diffusion.sample((args.batch_size, args.n, 3), args.device, num_steps=args.num_timesteps)
    
    trans = np.array([se3[0, :, :3, 3].detach().cpu().numpy() for se3 in generated_se3])
    rot = np.array([se3[0, :, :3, :3].detach().cpu().numpy() for se3 in generated_se3])

    for diff_step in range(len(generated_se3)):
        trajectory = trans[diff_step]
        rotations = rot[diff_step]
        # print(trajectory, rotations.shape)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(121, projection='3d')

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")

        # Set labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title("3D Trajectory of Robot")
        # ax.legend()

        ax = fig.add_subplot(122, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")

        # Plot orientation arrows (quivers)
        step = 5
        scale = 0.03
        for i in range(0, len(trajectory) - 1, step):
            t = trajectory[i]
            R = rotations

            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]

            ax.quiver(t[0], t[1], t[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      length=scale, color='r', linewidth=1.5, alpha=0.6)

            ax.quiver(t[0], t[1], t[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      length=scale, color='g', linewidth=1.5, alpha=0.6)

            ax.quiver(t[0], t[1], t[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      length=scale, color='b', linewidth=1.5, alpha=0.6)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.set_title("3D Trajectory of Robot")
        # ax.legend()

        plt.savefig(os.path.join(args.save_path, "path_" + str(diff_step) + ".png"), dpi=300, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define initial basis vectors
        origin = np.zeros((3,))
        colors = ['r', 'g', 'b']  # X, Y, Z axes
        
        for i, R in enumerate(rotations):
            if i % step != 0:
              continue

            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]

            ax.quiver(i // step, 0, 0,
                      x_axis[0], x_axis[1], x_axis[2],
                      length=scale, color='r', linewidth=1.5, alpha=0.6)

            ax.quiver(i // step, 0, 0,
                      y_axis[0], y_axis[1], y_axis[2],
                      length=scale, color='g', linewidth=1.5, alpha=0.6)

            ax.quiver(i // step, 0, 0,
                      z_axis[0], z_axis[1], z_axis[2],
                      length=scale, color='b', linewidth=1.5, alpha=0.6)

            # origin = np.array([i * 2, 0, 0])  # Shift each SO(3) visualization along x-axis
            
            # for j in range(3):  # Each column is a transformed basis vector
            #     ax.quiver(i*2, 0, 0, *R[:, j], color=colors[j], lw=2)
            
            # Add a marker at the origin of each frame
            # ax.scatter(*origin, color='k', s=30)

        # Set plot limits
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # plt.title("Sequence of SO(3) Rotations")
        plt.savefig(os.path.join(args.save_path, "rot_" + str(diff_step) + ".png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
