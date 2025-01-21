import os
import torch
import numpy as np
from utils import *

class IROS20Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for KITTI Odometry dataset SE(3) data.
    """

    def __init__(self, folder_path, seq_len=128, stride=1, center=True, use_path_signature = False):
        """
        Args:
            folder_path (str): Path to the folder containing KITTI odometry .txt pose files.
        """
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.center = center
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.files.sort()  # Ensure consistent ordering of files

        if not self.files:
            raise ValueError(f"No .npy files found in folder: {folder_path}")

        self.use_path_signature = use_path_signature
        self.traj = []
        for file in self.files:
            self.traj.extend(self._load_poses_from_file(file))

    def _load_poses_from_file(self, file_path):
        """
        Load SE(3) transformation matrices from a KITTI odometry pose file.

        Args:
            file_path (str): Path to the .npy file.

        Returns:
            list: List of SE(3) matrices (as NumPy arrays) from the file.
        """
        poses = []
        try:
            with open(file_path, 'rb') as f:
                while True:
                    poses.append(np.load(f))
        except EOFError:
            pass  # End of file reached

        self.points = poses
        
        ### 
        trajectories = []
        for i in range((len(poses) - self.seq_len) // self.stride):
            start = i * self.stride
            traj = np.stack(poses[start : start+self.seq_len])
            if self.center:
                traj[:, :3, 3] -= traj[0, :3, 3]
            trajectories.append(traj)

        # for i in range(len(trajectories)):
        #     trajectories[i] /= np.max(trajectories[i])

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
        if self.use_path_signature: 
          #print(self.traj[idx].shape)
          sig = se3_to_path_signature(self.traj[idx], level=3)
          #print(sig.shape)
          return torch.tensor(sig, dtype=torch.float32)
        else:
          return torch.tensor(self.traj[idx], dtype=torch.float32)

    def get_point(self, idx):
        return self.points[idx]
