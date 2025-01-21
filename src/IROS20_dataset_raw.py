import os
import torch
import numpy as np
from torch.utils.data import Dataset

class IROS20Dataset(Dataset):
    """
    PyTorch Dataset for IROS20 YCB video dataset SE(3) data.

    Args:
        folder_path (str): Root directory containing data subfolders.
        seq_len (int): Number of SE(3) points in each sequence.
        stride (int): Stride for sampling sequences.
        center (bool): Whether to center the SE(3) data (translation).
    """

    def __init__(self, folder_path, seq_len=128, stride=1, center=False):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.center = center

        # List to hold all sequences
        self.sequences = []
        self.trajs = []

        # Traverse the directory structure and collect sequences
        self._collect_sequences()

        # Optionally, center the data
        if self.center:
            self._center_sequences()

    def _collect_sequences(self):
        """
        Traverse the folder structure to locate and load all SE(3) sequences.
        """
        # Iterate over data subfolders
        for data_subfolder in os.listdir(self.folder_path):
            data_subfolder_path = os.path.join(self.folder_path, data_subfolder)
            if not os.path.isdir(data_subfolder_path):
                continue  # Skip if not a directory

            # Each data_subfolder contains only one subfolder
            only_subfolders = [d for d in os.listdir(data_subfolder_path)
                               if os.path.isdir(os.path.join(data_subfolder_path, d))]
            if not only_subfolders:
                continue  # No subfolders found
            only_subfolder_path = os.path.join(data_subfolder_path, only_subfolders[0])

            # Each only_subfolder contains multiple SE3 sequences
            for se3_sequence in os.listdir(only_subfolder_path):
                se3_sequence_path = os.path.join(only_subfolder_path, se3_sequence)
                if not os.path.isdir(se3_sequence_path):
                    continue  # Skip if not a directory

                # Load all SE3 points in the sequence
                se3_points = self._load_se3_sequence(se3_sequence_path)
                self.trajs.append(se3_points)

                # Split the sequence into smaller sequences based on seq_len and stride
                split_sequences = self._split_sequence(se3_points)
                self.sequences.extend(split_sequences)

    def _load_se3_sequence(self, se3_sequence_path):
        """
        Load SE(3) points from a sequence directory.

        Args:
            se3_sequence_path (str): Path to the SE3 sequence directory.

        Returns:
            list of np.ndarray: List of SE(3) matrices.
        """
        se3_files = sorted([f for f in os.listdir(se3_sequence_path)
                            if f.endswith('.txt')],
                           key=lambda x: int(os.path.splitext(x)[0]))  # Sort by sequence number

        se3_points = []
        for file in se3_files:
            file_path = os.path.join(se3_sequence_path, file)
            se3_matrix = self._read_se3_from_txt(file_path)
            if se3_matrix is not None:
                se3_points.append(se3_matrix)

        return se3_points

    def _read_se3_from_txt(self, file_path):
        """
        Read a single SE(3) matrix from a txt file.

        Args:
            file_path (str): Path to the txt file.

        Returns:
            np.ndarray: 4x4 SE(3) matrix, or None if failed to read.
        """
        try:
            with open(file_path, 'r') as f:
                data = f.read().strip().split()
                if len(data) != 16:
                    print(f"Warning: File {file_path} does not contain 16 elements.")
                    return None
                se3 = np.array(data, dtype=np.float32).reshape(4, 4)
                return se3
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _split_sequence(self, se3_points):
        """
        Split a long SE3 sequence into multiple shorter sequences based on seq_len and stride.

        Args:
            se3_points (list of np.ndarray): List of SE(3) matrices.

        Returns:
            list of list of np.ndarray: List of split sequences.
        """
        sequences = []
        total_points = len(se3_points)
        if total_points < self.seq_len:
            # Optionally, pad the sequence or skip
            # Here, we skip sequences shorter than seq_len
            return sequences

        for start in range(0, total_points - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            seq = se3_points[start:end]
            sequences.append(seq)

        return sequences

    def _center_sequences(self):
        """
        Center the SE3 sequences by subtracting the mean translation.
        """
        for seq in self.sequences:
            # Extract translations
            seq[:, :3, 3] -= seq[0, :3, 3]

    def __len__(self):
        """
        Return the total number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a single SE3 sequence as a tensor.

        Args:
            idx (int): Index of the sequence.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, 4, 4).
        """
        sequence = self.sequences[idx]
        # Convert list of np.ndarray to a single np.ndarray
        sequence_array = np.stack(sequence)  # Shape: (seq_len, 4, 4)
        # Convert to torch.Tensor
        sequence_tensor = torch.from_numpy(sequence_array).float()
        return sequence_tensor
    
    def get_traj(self):
        return self.trajs

