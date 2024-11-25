import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def chop_into_chunks(se3_sequence, chunk_size):
    """
    Chop a sequence of SE(3) matrices into chunks of a given size.

    Args:
        se3_sequence: Tensor of shape [L, 4, 4], where L is the sequence length.
        chunk_size: Number of SE(3) transformations per chunk.

    Returns:
        List of tensors, each of shape [chunk_size, 4, 4].
    """
    L = se3_sequence.shape[0]

    # Calculate the number of chunks
    num_chunks = L // chunk_size
    if L % chunk_size != 0:
        print(f"Warning: Sequence length {L} is not evenly divisible by {chunk_size}.")
    
    truncated_length = num_chunks * chunk_size
    se3_sequence = se3_sequence[:truncated_length]

    # Reshape into chunks
    chunks_tensor = se3_sequence.view(num_chunks, chunk_size, 4, 4)

    return chunks_tensor

def so3_to_matrix(so3):
    """Converts a Lie algebra element (so3 vector) to a rotation matrix."""
    theta = torch.norm(so3, dim=-1, keepdim=True)
    A = torch.sin(theta) / theta
    B = (1 - torch.cos(theta)) / (theta ** 2)
    skew_matrix = skew(so3)
    I = torch.eye(3, device=so3.device).expand(so3.shape[:-1] + (3, 3))
    return I + A[..., None] * skew_matrix + B[..., None] * torch.matmul(skew_matrix, skew_matrix)

def matrix_to_so3(rot_matrix):
    """Convert a batch of rotation matrices to rotation vectors."""
    batch_size, n, _, _ = rot_matrix.shape
    trace = torch.diagonal(rot_matrix, dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos((trace - 1) / 2.0)

    # Calculate the rotation vector components
    rot_vec = 0.5 * torch.stack([
        rot_matrix[..., 2, 1] - rot_matrix[..., 1, 2],  # R[2,1] - R[1,2]
        rot_matrix[..., 0, 2] - rot_matrix[..., 2, 0],  # R[0,2] - R[2,0]
        rot_matrix[..., 1, 0] - rot_matrix[..., 0, 1]   # R[1,0] - R[0,1]
    ], dim=-1)  # Stack along the last dimension to form the vector

    # Normalize the rotation vector
    rot_vec = rot_vec / (torch.sin(theta)[:, :, None] + 1e-8)

    return rot_vec

def skew(v):
    """Converts a vector into a skew-symmetric matrix."""
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1)
    ], dim=-2)

def compose_se3(rot, trans):
    """Combines rotation (3x3) and translation (3,) into SE(3) matrix."""
    batch_shape = rot.shape[:-2]
    se3 = torch.zeros(*batch_shape, 4, 4, device=rot.device)
    se3[..., :3, :3] = rot
    se3[..., :3, 3] = trans
    se3[..., 3, 3] = 1
    return se3

def sample_noise(shape, device):
    """Sample isotropic Gaussian noise for SE(3)."""
    so3_noise = torch.randn(shape, device=device)  # SO(3) noise
    transl_noise = torch.randn(shape, device=device)  # R^3 noise
    return so3_noise, transl_noise
