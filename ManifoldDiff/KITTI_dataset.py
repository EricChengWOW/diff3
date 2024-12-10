import os
import torch
import numpy as np

class KITTIOdometryDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for KITTI Odometry dataset SE(3) data.
    """

    def __init__(self, folder_path, seq_len=128, stride=1, center=True):
        """
        Args:
            folder_path (str): Path to the folder containing KITTI odometry .txt pose files.
        """
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.center = center
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        self.files.sort()  # Ensure consistent ordering of files

        if not self.files:
            raise ValueError(f"No .txt files found in folder: {folder_path}")

        self.traj = []
        for file in self.files:
            self.traj.extend(self._load_poses_from_file(file))

    def _load_poses_from_file(self, file_path):
        """
        Load SE(3) transformation matrices from a KITTI odometry pose file.

        Args:
            file_path (str): Path to the .txt file.

        Returns:
            list: List of SE(3) matrices (as NumPy arrays) from the file.
        """
        poses = []
        with open(file_path, 'r') as f:
            for line in f:
                pose = np.fromstring(line.strip(), sep=' ').reshape(3, 4)
                se3 = np.eye(4)
                se3[:3, :4] = pose
                poses.append(se3)

        self.points = poses
        
        ### 
        trajectories = []
        for i in range((len(poses) - self.seq_len) // self.stride):
            start = i * self.stride
            traj = np.stack(poses[start : start+self.seq_len])
            if self.center:
                traj -= traj[0]
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
        traj = self.traj[idx]
        return torch.tensor(traj, dtype=torch.float32)

    def get_point(self, idx):
        return self.points[idx]
