import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

class LDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for synthetic L-shaped SE(3) data.
    """

    def __init__(self, seq_len=128, size=500, rand_shuffle = False, use_path_signature = False, level = 3):
        self.L = []
        self.level = level
        self.use_path_signature = use_path_signature
        for i in range(seq_len // 2):
            self.L.append(np.array([i, 0, 0]))

        for i in range(seq_len // 2):
            self.L.append(np.array([seq_len // 2, i, 0]))

        self.traj = np.zeros((size, seq_len, 4, 4))
        self.L = np.array(self.L) * 2 / seq_len
        self.traj[:, :, :3, 3] = self.L

        self.traj[:, :, :3, 3] = self.rotate_and_scale(self.traj[:, :, :3, 3])
        self.add_rotations()
        if rand_shuffle: 
          for i in range(self.traj.shape[0]):  
            np.random.shuffle(self.traj[i])

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
        Rotate and scale trajectory.
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

    def visualize_trajectory(self, idx, save_folder):
        """
        Visualize the SE(3) trajectory for a given index.
        
        Args:
            idx (int): Index of the trajectory to visualize.
        """
        trajectory = self.traj[idx]
        
        # Extract the translation (position) components
        positions = trajectory[:, :3, 3]
        
        # Extract rotation components (optional for visualization)
        rotations = trajectory[:, :3, :3]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory in 3D space
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory Path", linewidth=2)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='r', label="Points")
        
        # Optionally plot rotation indicators (arrows or quivers)
        for i in range(0, len(positions), max(1, len(positions) // 20)):  # Sample points to avoid clutter
            pos = positions[i]
            rot = rotations[i]
            ax.quiver(
                pos[0], pos[1], pos[2], 
                rot[0, 0], rot[1, 0], rot[2, 0], color='g', length=0.1, normalize=True
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"SE(3) Trajectory Visualization (Index: {idx})")
        ax.legend()
        file_name = f"trajectory_{idx}.png"
        save_path = os.path.join(save_folder, file_name)
        plt.savefig(save_path, dpi=300)

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
        if self.use_path_signature: 
          #print(self.traj[idx].shape)
          sig = se3_to_path_signature(self.traj[idx], level=self.level)
          #print(sig.shape)
          return torch.tensor(sig, dtype=torch.float32)
        else:
          return torch.tensor(self.traj[idx], dtype=torch.float32)