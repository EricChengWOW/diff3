import os
import torch
import numpy as np

class TDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Oxford Robot Car dataset SE(3) data.
    """

    def __init__(self, seq_len=128, size=500):
        self.T = []
        for i in range(seq_len // 2):
            self.T.append(np.array([i, 0, 0]))

        for i in range(seq_len // 2):
            self.T.append(np.array([seq_len // 4, i + 1, 0]))

        self.traj = np.zeros((size, seq_len, 4, 4))
        self.T = np.array(self.T) * 2 / seq_len
        self.traj[:, :, :3, 3] = self.T

        self.traj[:, :, :3, 3] = self.rotate_and_scale(self.traj[:, :, :3, 3])
        self.add_rotations()

    def random_rotation_matrix(self):
        """Generate a random 3D rotation matrix using quaternions."""
        u1, u2, u3 = np.random.rand(3)
        q = [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
        # Quaternion to rotation matrix
        q0, q1, q2, q3 = q
        R = np.array([
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ])
        return R

    def rotate_and_scale(self, array):
        """
        Rotate each L,3,1 group in the array of shape [N, L, 3, 1]
        by the same random rotation for each N.
        
        Parameters:
            array (np.ndarray): Input array of shape [N, L, 3, 1].
            
        Returns:
            np.ndarray: Transformed array of the same shape.
        """
        N, L, _ = array.shape
        array = array.reshape(N, L, 3, 1)
        transformed = np.zeros_like(array)
        
        for n in range(N):
            scale = np.random.rand(1)
            R = self.random_rotation_matrix()  # Generate a random rotation matrix for this N
            for l in range(L):
                point = array[n, l]  # Shape [3, 1]
                transformed[n, l] = scale * R @ point  # Apply rotation
        
        transformed = transformed.reshape(N, L, 3)
        return transformed

    def direction_to_rotation_matrix(self, direction):
        """Convert a direction vector to a rotation matrix."""
        z = direction / np.linalg.norm(direction)
        x = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(x, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack((x, y, z))

    def add_rotations(self):
        """
        Add rotation matrices based on directions between consecutive points.
        """
        for n in range(self.traj.shape[0]):
            for i in range(self.traj.shape[1] - 1):
                # Compute direction to the next point
                direction = self.traj[n, i + 1, :3, 3] - self.traj[n, i, :3, 3]
                R = self.direction_to_rotation_matrix(direction)
                self.traj[n, i, :3, :3] = R

            # Set the rotation of the last point to be the same as the previous
            self.traj[n, -1, :3, :3] = self.traj[n, -2, :3, :3]

    def __len__(self):
        """
        Returns the total number of poses in the dataset.
        """
        return 500

    def __getitem__(self, idx):
        """
        Get the SE(3) transformation matrix at a specific index.

        Args:
            idx (int): Index of the desired pose.

        Returns:
            torch.Tensor: SE(3) matrix as a torch tensor.
        """
        return torch.tensor(self.traj[idx], dtype=torch.float32)
