from utils import *
from r3_diffuser import R3Diffuser, R3Conf
from unet import *
from se3_diffuser import *
from se3_diffuser_2 import *
from se3_diffuser_3 import *
from transformer import *
import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument("--model_type", type=str, default="Transformer", help="The score model architecture")
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

    # Load the saved weights
    loaded_model.load_state_dict(torch.load(args.model_path))

    # Set the model to evaluation mode
    loaded_model.eval()

    diffusion = SE3Diffusion_3(loaded_model, trans_scale=args.scale_trans)
    generated_se3 = diffusion.sample((args.batch_size, args.n, 3), args.device, num_steps=args.num_timesteps)

    # print("Generated SE(3) sequence:")
    # print(generated_se3)

    trajectory = np.array([se3[:, :3, 3].detach().cpu().numpy() for se3 in generated_se3])[0]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection='3d')

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")

    # Set labels
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    ax.set_zlabel("Altitude (meters)")
    ax.set_title("3D Trajectory of Robot")
    ax.legend()

    ax = fig.add_subplot(122, projection='3d')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    
    # Plot trajectory
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")

    # Set labels
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    ax.set_zlabel("Altitude (meters)")
    ax.set_title("3D Trajectory of Robot")
    ax.legend()

    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
