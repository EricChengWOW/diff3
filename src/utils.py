import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esig

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
from geomloss import SamplesLoss

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
    trans_np = translation.cpu().numpy()
    rot_np = so3_vec.cpu().numpy()

    # Compute path signatures for each trajectory in the batch
    # signatures = esig.stream2sig(trajectory_np, level) 
    # signature_tensor = torch.tensor(signatures, dtype=se3.dtype, device=se3.device)
    # print(signature_tensor.shape)
    trans_sig = esig.stream2sig(trans_np, level)
    rot_sig = esig.stream2sig(rot_np, level)
    signatures = np.concatenate([trans_sig, rot_sig])
    signature_tensor = torch.tensor(signatures, dtype=se3.dtype, device=se3.device)
    # print(signature_tensor.shape, trans_sig.shape, rot_sig.shape, signatures.shape)

    return signature_tensor

def path_signature_scalar_mult(sig, k, dim, depth):
    start = 0
    for i in range(depth):
        end = start + dim ** i
        sig[start:end] *= (k ** i)
        start = end

    return sig

def get_signature_size(D, N):
    """
    Computes the size of the truncated signature up to depth N for dimension D.
    
    Args:
        D (int): Dimension of the path.
        N (int): Truncation depth.
        
    Returns:
        int: Size of the signature vector.
    """
    if D == 1:
        return N + 1  # For 1D, each level adds one term
    return (D**(N + 1) - 1) // (D - 1)

def extract_signature_levels(signature, D, N):
    """
    Extracts signature terms for each level up to depth N.
    
    Args:
        signature (torch.Tensor): Tensor of shape (batch_size, signature_size).
        D (int): Dimension of the path.
        N (int): Truncation depth.
        
    Returns:
        List[torch.Tensor]: List containing tensors for each level from 1 to N.
    """
    levels = []
    start = 1  # Skip the zeroth term which is always 1
    for k in range(1, N + 1):
        num_terms = D**k
        end = start + num_terms
        level_k = signature[:, start:end]
        levels.append(level_k)
        start = end
    return levels

def normalize_signature(signatures):
    """
    Normalizes each signature vector to have unit norm.
    
    Args:
        signatures (torch.Tensor): Tensor of shape (batch_size, signature_size).
        
    Returns:
        torch.Tensor: Normalized signatures.
    """
    return F.normalize(signatures, p=2, dim=1)

def signature_log(signature, D, N):
    """
    Approximates the logarithm of a truncated path signature using the Magnus Expansion.
    
    Args:
        signature (torch.Tensor): Truncated signature tensor of shape (batch_size, signature_size).
        D (int): Dimension of the path.
        N (int): Maximum depth of the signature (e.g., 4).
        
    Returns:
        torch.Tensor: Logarithm of the signature in the Lie algebra, same shape as signature.
    """
    batch_size = signature.size(0)
    signature_size = signature.size(1)
    
    log_sig = torch.zeros_like(signature)
    
    # Extract signature levels
    levels = extract_signature_levels(signature, D, N)
    
    # Level 1: log(S)^1 = S^1
    log_sig[:, 1:D+1] = levels[0]
    
    if N >= 2:
        # Level 2: log(S)^2 = S^2 - 0.5 * S^1 ⊗ S^1
        S1 = levels[0]  # (batch_size, D)
        S2 = levels[1]  # (batch_size, D^2)
        S1_outer = torch.einsum('bi,bj->bij', S1, S1).reshape(batch_size, -1)  # (batch_size, D^2)
        log_sig[:, D+1:D+1+D**2] = S2 - 0.5 * S1_outer
    
    if N >= 3:
        # Level 3: log(S)^3 = S^3 - S^1 ⊗ S^2 + (1/3) * S^1 ⊗ S^1 ⊗ S^1
        S1 = levels[0]  # (batch_size, D)
        S2 = levels[1]  # (batch_size, D^2)
        S3 = levels[2]  # (batch_size, D^3)
        
        # Compute S1 ⊗ S2
        # S1 has shape (batch_size, D)
        # S2 has shape (batch_size, D^2)
        # Reshape S2 to (batch_size, D, D)
        S2_reshaped = S2.view(batch_size, D, D)
        S1_S2 = torch.einsum('bi,bjk->bijk', S1, S2_reshaped).reshape(batch_size, -1)  # (batch_size, D^3)
        
        # Compute S1 ⊗ S1 ⊗ S1
        S1_S1_S1 = torch.einsum('bi,bj,bk->bijk', S1, S1, S1).reshape(batch_size, -1)  # (batch_size, D^3)
        
        # Combine terms
        log_sig[:, D+1+D**2:D+1+D**2+D**3] = S3 - S1_S2 + (1.0 / 3.0) * S1_S1_S1
    
    if N >= 4:
        # Level 4: log(S)^4 = S^4 - S^1 ⊗ S^3 - S^2 ⊗ S^2 + S^1 ⊗ S^1 ⊗ S^2 + S^1 ⊗ S^2 ⊗ S^1 + S^2 ⊗ S^1 ⊗ S^1 - (1/4) * S^1 ⊗ S^1 ⊗ S^1 ⊗ S^1
        S1 = levels[0]  # (batch_size, D)
        S2 = levels[1]  # (batch_size, D^2)
        S3 = levels[2]  # (batch_size, D^3)
        S4 = levels[3]  # (batch_size, D^4)
        
        # Compute S1 ⊗ S3
        # S3 reshaped to (batch_size, D, D^2)
        S3_reshaped = S3.view(batch_size, D, D**2)
        S1_S3 = torch.einsum('bi,bjk->bijk', S1, S3_reshaped).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute S2 ⊗ S2
        # S2 reshaped to (batch_size, D, D)
        S2_reshaped = S2.view(batch_size, D, D)
        # Compute outer product: (batch_size, D, D) x (batch_size, D, D) -> (batch_size, D^2, D^2)
        S2_outer = torch.einsum('bik,bjt->bikjt', S2_reshaped, S2_reshaped)
        # Sum over appropriate dimensions to get (batch_size, D^4)
        S2_S2 = S2_outer.reshape(batch_size, D**4)
        
        # Compute S1 ⊗ S1 ⊗ S2
        S1_S1_S2 = torch.einsum('bi,bj,bk->bijk', S1, S1, S2).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute S1 ⊗ S2 ⊗ S1
        S1_S2_S1 = torch.einsum('bi,bj,bk->bijk', S1, S2, S1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute S2 ⊗ S1 ⊗ S1
        S2_S1_S1 = torch.einsum('bi,bj,bk->bijk', S2, S1, S1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute S1 ⊗ S1 ⊗ S1 ⊗ S1
        S1_S1_S1_S1 = torch.einsum('bi,bj,bk,bl->bijkl', S1, S1, S1, S1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Combine all terms
        log_sig[:, D+1+D**2+D**3:D+1+D**2+D**3+D**4] = (
            S4
            - S1_S3
            - S2_S2
            + S1_S1_S2
            + S1_S2_S1
            + S2_S1_S1
            - (1.0 / 4.0) * S1_S1_S1_S1
        )
    
    return log_sig

def signature_exp(log_signature, D, N):
    """
    Approximates the exponential of a truncated log path signature using the Magnus Expansion.
    
    Args:
        log_signature (torch.Tensor): Logarithm of the signature in the Lie algebra, shape (batch_size, signature_size).
        D (int): Dimension of the path.
        N (int): Maximum depth of the signature (e.g., 4).
        
    Returns:
        torch.Tensor: Exponentiated signature tensor, same shape as log_signature.
    """
    batch_size = log_signature.size(0)
    signature_size = log_signature.size(1)
    
    exp_sig = torch.zeros_like(log_signature)
    
    # Zeroth level is always 1
    exp_sig[:,0] = 1.0
    
    # Extract log signature levels
    levels = extract_signature_levels(log_signature, D, N)
    
    # Level 1: exp(log(S))^1 = log(S)^1
    exp_sig[:, 1:D+1] = levels[0]
    
    if N >= 2:
        # Level 2: exp(log(S))^2 = log(S)^2 + 0.5 * log(S)^1 ⊗ log(S)^1
        L1 = levels[0]  # (batch_size, D)
        L2 = levels[1]  # (batch_size, D^2)
        L1_outer = torch.einsum('bi,bj->bij', L1, L1).reshape(batch_size, -1)  # (batch_size, D^2)
        exp_sig[:, D+1:D+1+D**2] = L2 + 0.5 * L1_outer
    
    if N >= 3:
        # Level 3: exp(log(S))^3 = log(S)^3 + log(S)^1 ⊗ log(S)^2 + (1/6) * log(S)^1 ⊗ log(S)^1 ⊗ log(S)^1
        L1 = levels[0]  # (batch_size, D)
        L2 = levels[1]  # (batch_size, D^2)
        L3 = levels[2]  # (batch_size, D^3)
        
        # Compute L1 ⊗ L2
        L1_L2 = torch.einsum('bi,bjk->bijk', L1, L2.view(batch_size, D, D)).reshape(batch_size, -1)  # (batch_size, D^3)
        
        # Compute L1 ⊗ L1 ⊗ L1
        L1_L1_L1 = torch.einsum('bi,bj,bk->bijk', L1, L1, L1).reshape(batch_size, -1)  # (batch_size, D^3)
        
        # Combine terms
        exp_sig[:, D+1+D**2:D+1+D**2+D**3] = L3 + L1_L2 + (1.0 / 6.0) * L1_L1_L1
    
    if N >= 4:
        # Level 4: exp(log(S))^4 = log(S)^4 + log(S)^1 ⊗ log(S)^3 + log(S)^2 ⊗ log(S)^2 +
        # log(S)^1 ⊗ log(S)^1 ⊗ log(S)^2 + log(S)^1 ⊗ log(S)^2 ⊗ log(S)^1 +
        # log(S)^2 ⊗ log(S)^1 ⊗ log(S)^1 + (1/24)*log(S)^1 ⊗ log(S)^1 ⊗ log(S)^1 ⊗ log(S)^1
        
        L1 = levels[0]  # (batch_size, D)
        L2 = levels[1]  # (batch_size, D^2)
        L3 = levels[2]  # (batch_size, D^3)
        L4 = levels[3]  # (batch_size, D^4)
        
        # Compute L1 ⊗ L3
        S1_S3 = torch.einsum('bi,bjkl->bijkl', L1, L3.view(batch_size, D, D, D)).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute L2 ⊗ L2
        S2_reshaped = L2.view(batch_size, D, D)
        S2_S2 = torch.einsum('bij,bkl->bijkl', S2_reshaped, S2_reshaped).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute L1 ⊗ L1 ⊗ L2
        S1_S1_S2 = torch.einsum('bi,bj,bk->bijk', L1, L1, L2).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute L1 ⊗ L2 ⊗ L1
        S1_S2_S1 = torch.einsum('bi,bj,bk->bijk', L1, L2, L1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute L2 ⊗ L1 ⊗ L1
        S2_S1_S1 = torch.einsum('bi,bj,bk->bijk', L2, L1, L1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Compute L1 ⊗ L1 ⊗ L1 ⊗ L1
        S1_S1_S1_S1 = torch.einsum('bi,bj,bk,bl->bijkl', L1, L1, L1, L1).reshape(batch_size, -1)  # (batch_size, D^4)
        
        # Combine all terms
        exp_sig[:, D+1+D**2+D**3:D+1+D**2+D**3+D**4] = (
            L4
            + S1_S3
            + S2_S2
            + S1_S1_S2
            + S1_S2_S1
            + S2_S1_S1
            + (1.0 / 24.0) * S1_S1_S1_S1
        )
    
    return exp_sig

class WassersteinSignatureLoss(nn.Module):
    def __init__(self, p=2, blur=0.05, scaling=0.5):
        """
        Initializes the WassersteinSignatureLoss using geomloss.
        
        Args:
            p (int): The order of the Wasserstein distance (usually 1 or 2).
            blur (float): The blur parameter for entropic regularization.
            scaling (float): The scaling parameter for geomloss.
        """
        super(WassersteinSignatureLoss, self).__init__()
        self.wasserstein = SamplesLoss(loss='sinkhorn', p=p, blur=blur, scaling=scaling)
    
    def forward(self, pred_signature, target_signature):
        """
        Computes the Wasserstein distance between predicted and target signatures.
        
        Args:
            pred_signature (torch.Tensor): Predicted signature tensor of shape (batch_size, signature_size).
            target_signature (torch.Tensor): Target signature tensor of shape (batch_size, signature_size).
        
        Returns:
            torch.Tensor: Scalar loss value representing the average Wasserstein distance.
        """
        # Ensure the inputs are float tensors
        pred_signature = pred_signature.float()
        target_signature = target_signature.float()
        
        # Compute Wasserstein distance for each pair in the batch
        # geomloss expects inputs of shape (batch_size, features)
        distances = self.wasserstein(pred_signature, target_signature)
        
        # Return the mean distance across the batch
        return distances.mean()
    