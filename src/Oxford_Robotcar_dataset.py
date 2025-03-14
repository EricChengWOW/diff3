import os
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from utils import *

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
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3] = translation

    return se3

class RobotcarDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Oxford Robot Car dataset SE(3) data.
    """

    def __init__(self, folder_path, seq_len=128, stride=1, center=True, use_path_signature = False, random_seq_len = False, scale_trans = 1.0, level = 3):
        """
        Args:
            folder_path (str): Path to the folder containing Robot car .csv files.
        """
        self.use_path_signature = use_path_signature
        self.random_seq_len = random_seq_len
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.center = center
        self.scale_trans = scale_trans
        self.level = level
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.files.sort()  # Ensure consistent ordering of files

        if not self.files:
            raise ValueError(f"No .csv files found in folder: {folder_path}")

        self.traj = []
        for file in self.files:
            self.traj.extend(self._load_poses_from_file(file))

        max_entry = 0
        for traj in self.traj:
            max_entry = max(max_entry, np.max(np.abs(traj[:, :3, 3])))
        
        for traj in self.traj:
            traj[:, :3, 3] /= max_entry
        
        print("Oxford max ", max_entry)

        if self.use_path_signature: 
            for i in range(len(self.traj)):
                self.traj[i] = se3_to_path_signature(self.traj[i], level=self.level)

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
        if not self.random_seq_len: 
          for i in range((len(poses) - self.seq_len) // self.stride):
              start = i * self.stride
              traj = np.stack(poses[start : start+self.seq_len])
              
              if self.center:
                traj[:, :3, 3] -= traj[0, :3, 3]

              trajectories.append(traj)

        else: 
          for i in range((len(poses) - self.seq_len) // self.stride):
              random_seq_len = np.random.randint(
                  int(self.seq_len * 0.75), 
                  int(self.seq_len * 1.25) + 1
              )
              while random_seq_len in [99, 104,108]: #wtf?
                random_seq_len = np.random.randint(
                  int(self.seq_len * 0.75), 
                  int(self.seq_len * 1.25) + 1
                )
              start = i * self.stride
              end = start + random_seq_len
              if end > len(poses): 
                  end = len(poses)
              traj = np.stack(poses[start:end])
              
              if self.center:
                  traj[:, :3, 3] -= traj[0, :3, 3]

              trajectories.append(traj)
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
        return torch.tensor(self.traj[idx], dtype=torch.float32)
    
    def get_point(self, idx):
        return self.points[idx]
