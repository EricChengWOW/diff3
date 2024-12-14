# Diff3 : Out-of-Distribution Detection for SE3 trajectory sequence using single Diffusion model

# Abstract
Out-of-distribution (OOD) detection is a machine learning that seeks to identify abnormal samples. Traditionally, OOD detection  requires retraining for different inliner distribution and are subject to poor out-sample performance. Recent work have shown that a single diffusion model can perform OOD detection on image spaces. Hence, we are motivated to explore if a single diffusion model can effectively perform OOD detection on trajectory sequences. For this purpose, we introduce $\mathbf{Diff^3}$, which uses a single diffusion model trained on trajectory data to perform unconditional generation for OOD detection on trajectory spaces with two candidate data format -- 3D Coordinate ($\mathbf{R^3}$) and Special Euclidean Group in 3D (SE(3)).

# Installation
We recommend using Conda for environment control and the codebase requires minimal extra packages.

```shell
# Clone the repository
git clone https://github.com/EricChengWOW/diff3.git
cd diff3

conda create --name diff3
conda activate diff3
pip install -r requirements.txt
```

# Dataset
The codebase provides access to OxfordRobot Car and KITTI dataset. The L and T shape dataset could be directly utilized by importing the class and creating the dataset object. The iros20 dataset is located [here](https://github.com/Mlahoud/wenBowen20-6d-pose-tracking/tree/master).

# Training
The codebase provides with extensive customizability to neural network and model configurations. The following are the ones supported for train.py, and Infereance and OOD task shares similar arguments.

| **Argument**             | **Type**    | **Default**      | **Description**                                                       |
|--------------------------|-------------|------------------|-----------------------------------------------------------------------|
| `--batch_size`           | `int`       | `32`             | Batch size for training (default: 32).                               |
| `--n`                    | `int`       | `128`            | Number of data points per sequence (default: 128).                   |
| `--num_epochs`           | `int`       | `200`            | Number of training epochs (default: 200).                            |
| `--model_type`           | `str`       | `"Transformer"`  | The score model architecture.                                        |
| `--n_layers`             | `int`       | `3`              | Number of layers in the transformer.                                 |
| `--unet_layer`           | `int`       | `4`              | Layers of UNet dimension.                                            |
| `--n_heads`              | `int`       | `8`              | Number of heads in the transformer.                                  |
| `--hidden_dim`           | `int`       | `128`            | Hidden dimension size (default: 128).                                |
| `--data_stride`          | `int`       | `1`              | Stride for splitting data sequence to sequence length.               |
| `--scale_trans`          | `float`     | `1.0`            | Scale factor for R3 translation.                                     |
| `--device`               | `str`       | `"cuda"`         | Device to use for computation (default: `'cuda'`).                   |
| `--num_timesteps`        | `int`       | `30`             | Number of timesteps for diffusion process (default: 30).             |
| `--data_folder`          | `str`       | **Required**     | Path to the data folder containing the dataset.                      |
| `--dataset`              | `str`       | **Required**     | Dataset name, `'KITTI'` or `'Oxford'`.                               |
| `--save_path`            | `str`       | **Required**     | File path to save the trained model.                                 |
| `--shuffle`              | `Flag`      | `False`          | Enable shuffling of data (default: False).                           |
| `--center`               | `Flag`      | `False`          | Center each trajectory in the dataset.                               |
| `--learning_rate`        | `float`     | `1e-5`           | Training optimizer learning rate.                                    |
| `--wandb`                | `Flag`      | `False`          | Log training loss to Weights & Biases (wandb).                       |
| `--diffusion_type`       | `str`       | `"DDPM"`         | The diffusion algorithm to use (`[DDPM, DDPM_Continuous]`). |

## Sample Cmd

```shell
!python3 src/train.py --hidden_dim 128 \
--data_folder ./OxfordRobotcar/ \
--dataset Oxford --save_path model.pth --batch_size 32 \
--num_epochs 1500 --model_type Unet --data_stride 32 \
--scale_trans 1 --learning_rate 1e-4\
--num_timesteps 30 --unet_layer 4 --center --diffusion DDPM \
--wandb
```

# Inference

## Sample Cmd

```shell
!python3 src/inference.py --hidden_dim 128 \
--save_path output.png --model_path model.pth \
--scale_trans 1 --num_timesteps 30 --model_type Unet \
--batch_size 1 --unet_layer 4
```

# Out-of-Distribution Task

## Sample Cmd

```shell
!python3 src/ood.py --hidden_dim 128 \
--model_path model.pth --model_type Unet \
--in_data_folder ./OxfordRobotcar --in_dataset Oxford \
--out_data_folder ./KITTI --out_dataset KITTI \
--num_timesteps 30 --in_data_stride 64 --out_data_stride 16 --batch_size 128 --unet_layer 4 \
--save_folder ./stats --scale_trans 0.034 --center --ood_mode SE3
```
