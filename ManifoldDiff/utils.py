import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation

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

def igso3_expansion(omega, eps, L=1000, use_torch=False):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        L: Truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.
    """

    ls = torch.arange(L)
    ls = ls.to(omega.device)

    if len(omega.shape) == 2:
        # Used during predicted score calculation.
        ls = ls[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [num_batch, num_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_batch, 1]
    else:
        raise ValueError("Omega must be 1D or 2D.")
    p = (2*ls + 1) * torch.exp(-ls*(ls+1)*eps**2/2) * torch.sin(omega*(ls+1/2)) / torch.sin(omega/2)

    return p.sum(dim=-1)

def compose_rotvec(r1, r2):
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum('...ij,...jk->...ik', R1, R2)
    return matrix_to_rotvec(cR)

# Convert from rotation vector to rotation matrix (using PyTorch tensors)
def rotvec_to_matrix(rotvec):
    # Ensure the input is a PyTorch tensor with the last dimension as 3 (rotation vector shape)
    assert rotvec.ndimension() >= 2 and rotvec.shape[-1] == 3, "Input must have the last dimension of size 3."
    
    # Reshape the input tensor to [N, 3] where N is the product of all dimensions except the last one
    shape = rotvec.shape[:-1]  # All dimensions except the last (3)
    rotvec_reshaped = rotvec.view(-1, 3)  # Flatten everything except the last dimension

    # Convert the PyTorch tensor to NumPy for using scipy
    rotvec_numpy = rotvec_reshaped.cpu().numpy()  # Ensure the tensor is on CPU for conversion
    
    # Apply the rotation conversion using scipy
    mat_numpy = Rotation.from_rotvec(rotvec_numpy).as_matrix()
    
    # Convert back to a PyTorch tensor with the original shape
    mat_tensor = torch.tensor(mat_numpy, dtype=rotvec.dtype, device=rotvec.device)
    
    # Reshape the result back to the original batch shape with [*, 3, 3]
    mat_tensor = mat_tensor.view(*shape, 3, 3)
    
    return mat_tensor

# Convert from rotation matrix to rotation vector (using PyTorch tensors)
def matrix_to_rotvec(mat):
    # Ensure the input is a PyTorch tensor with the last two dimensions as 3x3 (rotation matrix shape)
    assert mat.ndimension() >= 3 and mat.shape[-2:] == (3, 3), "Input must have shape [..., 3, 3]."
    
    # Reshape the input tensor to [N, 3, 3] where N is the product of all dimensions except the last two
    shape = mat.shape[:-2]  # All dimensions except the last two (3, 3)
    mat_reshaped = mat.view(-1, 3, 3)  # Flatten everything except the last two dimensions

    # Convert the PyTorch tensor to NumPy for using scipy
    mat_numpy = mat_reshaped.cpu().numpy()  # Ensure the tensor is on CPU for conversion
    
    # Apply the matrix to rotation vector conversion using scipy
    rotvec_numpy = Rotation.from_matrix(mat_numpy).as_rotvec()
    
    # Convert back to a PyTorch tensor with the original shape
    rotvec_tensor = torch.tensor(rotvec_numpy, dtype=mat.dtype, device=mat.device)
    
    # Reshape the result back to the original batch shape with [*, 3]
    rotvec_tensor = rotvec_tensor.view(*shape, 3)
    
    return rotvec_tensor

def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()
