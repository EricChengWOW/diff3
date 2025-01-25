import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import esig

def compose_se3(rot, trans):
    """Combines rotation (3x3) and translation (3,) into SE(3) matrix."""
    batch_shape = rot.shape[:-2]
    se3 = torch.zeros(*batch_shape, 4, 4, device=rot.device)
    se3[..., :3, :3] = rot
    se3[..., :3, 3] = trans
    se3[..., 3, 3] = 1
    return se3

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

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: A tensor contains the values of all time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)

def skew_symmetric(v):
    """Construct the skew-symmetric matrix S(v) from a vector v = (x,y,z)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1)
    ], dim=-2)

def so3_exp_map(v):
    """
    Exponential map from so(3) to SO(3).

    v: (..., 3) batch of vectors in R^3 (tangent space).
    Returns: (..., 3, 3) batch of rotation matrices in SO(3).
    """
    theta = torch.norm(v, dim=-1, keepdim=True)
    theta_clamped = theta.clamp(min=1e-8)
    V = skew_symmetric(v)  # (B,L,3,3)
    B, L = v.shape[0], v.shape[1]
    I = torch.eye(3, device=v.device).expand(B,L,3,3)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_div = (sin_theta / theta_clamped).unsqueeze(-1)
    term2 = sin_div * V
    one_minus_cos_div = ((1 - cos_theta) / (theta_clamped**2)).unsqueeze(-1)
    term3 = one_minus_cos_div * (V @ V)
    R = I + term2 + term3
    return R

def so3_log_map(R):
    """
    Logarithm map from SO(3) to so(3).

    R: (..., 3, 3) batch of rotation matrices in SO(3).
    Returns: (..., 3) batch of tangent vectors.
    """
    trace_R = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cos_theta = (trace_R - 1.) / 2.
    cos_theta = cos_theta.clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(cos_theta)
    theta_clamped = theta.clamp(min=1e-8)

    R_T = R.transpose(-1,-2)
    Q = (R - R_T) / 2.0
    x = (Q[...,2,1] - Q[...,1,2]) / 2
    y = (Q[...,0,2] - Q[...,2,0]) / 2
    z = (Q[...,1,0] - Q[...,0,1]) / 2
    v = torch.stack([x, y, z], dim=-1)  # (B,L,3)

    sin_theta = torch.sin(theta_clamped)
    scale = (theta / sin_theta).unsqueeze(-1)
    v = v * scale
    return v

def so3_interpolate(x, y, gamma):
    """
    Geodesic interpolation on SO(3): 
    lambda(gamma, x) = exp(gamma log(x))
    
    Here we just scale the tangent vector of x by gamma.
    
    x: (..., 3, 3) rotation matrix
    y: not needed here if we just scale from identity to x.
    gamma: scalar or (...,) shape broadcastable factor
    """
    v = so3_log_map(x)  # (...,3)
    new_v = gamma * v
    return so3_exp_map(new_v)

def rotation_distance_loss(R_pred, R_true):
    """
    Compute the rotation distance loss between two batches of rotation matrices.

    Args:
        P: Predicted rotation matrices of shape (B, L, 3, 3).
        Q: True rotation matrices of shape (B, L, 3, 3).

    Returns:
        Loss value (mean rotation distance in radians).
    """
    Rt = R_pred.transpose(-1,-2)
    M = Rt @ R_true
    trace_val = M[...,0,0] + M[...,1,1] + M[...,2,2]
    # Clip for numerical stability
    cos_theta = (trace_val - 1.0)/2.0
    cos_theta = cos_theta.clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(cos_theta)  # (B,)
    return (theta**2).mean()

def se3_to_path_signature(se3, level=2):
    """
    Convert an SE(3) trajectory to a path signature representation.

    Args:
        se3 (torch.Tensor): SE(3) trajectory of shape (L, 4, 4), 
                            where B is the batch size, L is the sequence length.
        level (int): Level of the path signature (higher levels capture more details).

    Returns:
        torch.Tensor: Path signature representation of shape (signature_dim,6).
    """
    se3 = torch.tensor(se3)
    L, _, _ = se3.shape
    translation = se3[:, :3, 3]  # (L, 3)
    rotation_matrices = se3[:, :3, :3]  # (L, 3, 3)
    so3_vec = so3_log_map(rotation_matrices)  # (L, 3)
    trajectory = torch.cat([translation, so3_vec], dim=-1)  # (L, 6)
    # Convert to numpy for esig compatibility
    trajectory_np = trajectory.cpu().numpy()  # (L, 6)

    # Compute path signatures for each trajectory in the batch
    signatures =  esig.stream2sig(trajectory_np, level) 

    signature_tensor = torch.tensor(signatures, dtype=se3.dtype, device=se3.device)
    return signature_tensor
    