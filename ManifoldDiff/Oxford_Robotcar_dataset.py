import os
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
        quat = [qw, qx, qy, qz]
        return R.from_quat(quat).as_matrix()

def compute_se3(row):
    # Extract translation components
    translation = np.array([row['easting'], row['northing'], -row['down']])

    # Extract rotation components (assuming they are in degrees; convert to radians if needed)
    roll, pitch, yaw = np.radians([row['roll'], row['pitch'], row['yaw']])

    # Create a rotation matrix from roll, pitch, yaw (Z-Y-X intrinsic rotations)
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # Construct SE(3) matrix
    se3 = np.eye(4)  # Start with an identity matrix
    se3[:3, :3] = rotation_matrix  # Top-left 3x3 block is the rotation matrix
    se3[:3, 3] = translation  # Top-right 3x1 block is the translation vector

    return se3

class RobotcarDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Oxford Robot Car dataset SE(3) data.
    """

    def __init__(self, folder_path, seq_len=128, stride=1):
        """
        Args:
            folder_path (str): Path to the folder containing Robot car .csv files.
        """
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.files.sort()  # Ensure consistent ordering of files

        if not self.files:
            raise ValueError(f"No .csv files found in folder: {folder_path}")

        self.traj = []
        for file in self.files:
            self.traj.extend(self._load_poses_from_file(file))

    def _load_poses_from_file(self, file_path):
        """
        Load SE(3) transformation matrices from a KITTI odometry pose file.

        Args:
            file_path (str): Path to the .csv file.

        Returns:
            list: List of SE(3) matrices (as NumPy arrays) from the file.
        """
        ins_data = pd.read_csv(file_path)
        poses = ins_data.apply(compute_se3, axis=1)
        self.points = poses
        
        ### 
        trajectories = []
        for i in range((len(poses) - self.seq_len) // self.stride):
            start = i * self.stride
            trajectories.append(np.stack(poses[start : start+self.seq_len]))

        return trajectories

    def __len__(self):
        """
        Returns the total number of poses in the dataset.
        """
        return len(self.traj)

    def __getitem__(self, idx):
        """
        Get the SE(3) transformation matrix at a specific index.

        Args:
            idx (int): Index of the desired pose.

        Returns:
            torch.Tensor: SE(3) matrix as a torch tensor.
        """
        traj = self.traj[idx]
        return torch.tensor(traj, dtype=torch.float32)
    
    def get_point(self, idx):
        return self.points[idx]
