import torch
import numpy as np
from utils import *

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

class SE3Diffusion:
    """Score-based diffusion model for SE(3)."""

    def __init__(self, num_timesteps = 20, seq_len=128, scale_trans=1.0, scale_rot=1.0, min_b=0.1, max_b=20, device="cuda"):
        self.device = device
        self.num_timesteps = num_timesteps
        self.scale_t = scale_trans
        self.scale_r = scale_rot
        self.min_b = min_b
        self.max_b = max_b
        self.seq_len = seq_len

    def b_t(self, t):
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def sample_noise(self, batch_size, n):
        """Sample isotropic Gaussian noise for SE(3)."""
        so3_noise = torch.randn(batch_size, n, 3, device=self.device)  # SO(3) noise
        transl_noise = torch.randn(batch_size, n, 3, device=self.device)  # R^3 noise
        return so3_noise, transl_noise

    def scale_trans(self, trans):
        """Scale the translation component."""
        return self.scale_t * trans
    
    def scale_rot(self, rot):
        """Scale the rotation component."""
        return self.scale_r * rot

    def unscale_trans(self, trans):
        """Unscale the translation component."""
        return trans / self.scale_t

    def unscale_rot(self, rot):
        """Unscale the rotation component."""
        return rot / self.scale_r
    
    def sample_igso3(
            self,
            t: float,
            n_samples: float=1):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_samples: number of samples to draw.

        Returns:
            [n_samples] angles of rotation.
        """
        # if not torch.isscalar(t):
        #     raise ValueError(f'{t} must be a scalar.')
        x = torch.random.rand(n_samples)
        return torch.interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)

    def sample(
            self,
            t: float,
            n_samples: float=1):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def perturb(self, rotations, translations, t, noise_rot, noise_trans):
        """
        Forward diffusion process.
        Args:
            rotations: [B, n, 3] rotation matrices.
            translations: [B, n, 3] translations.

        Returns:
            Noisy SE(3) components.
        """
        rotations = self.scale_rot(rotations)
        translations = self.scale_trans(translations)

        perturbed_trans = torch.exp(-1/2*self.marginal_b_t(t)) * translations + \
                          noise_trans * torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))
        
        sampled_rots = self.sample(t, n_samples=rotations.size(0))

        # Right multiply.
        perturbed_rot = compose_rotvec(rotations, sampled_rots)
        # perturbed_rot   = torch.exp(-1/2*self.marginal_b_t(t)) * rotations + \
        #                   noise_rot * torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))

        return perturbed_rot, perturbed_trans
    
    def reverse_marginal(self, t, t_index, rot, trans, model, batch_size, noise_scale=1.0, return_score=False, eps=1e-6):
        if t_index == 0:
            z = (torch.zeros(rot.shape, device=rot.device), torch.zeros(trans.shape, device=trans.device))
        else:
            z = self.sample_noise(batch_size, self.seq_len)
            z = (z[0].transpose(1,2), z[1].transpose(1,2))

        rot = self.scale_rot(rot)
        trans = self.scale_trans(trans)
        rot_score, trans_score = model(rot, trans, t)

        dt= 1 / self.num_timesteps
        t = t.reshape(t.size(0),1,1).broadcast_to(trans.shape)
        g_t = self.diffusion_coef(t)
        f_t_trans = self.drift_coef(trans, t)
        z = noise_scale * z[1]
        perturb_trans = (f_t_trans - g_t**2 * trans_score) * dt + g_t * np.sqrt(dt) * z

        f_t_rot = self.drift_coef(rot, t)
        z = noise_scale * z[0]
        perturb_rot = (f_t_rot - g_t**2 * rot_score) * dt + g_t * np.sqrt(dt) * z

        trans -= perturb_trans

        rot = compose_rotvec(
            rot,
            perturb_rot
        )

        rot = self.unscale_rot(rot)
        trans = self.unscale_trans(trans)

        if return_score:
            return rot, trans, perturb_rot, perturb_trans

        return rot, trans

    def reverse_diffusion(self, model, n, batch_size=1, return_score=False, rot_init=None, trans_init=None):
        """
        Perform reverse diffusion to sample SE(3) sequences.

        Args:
            diffusion: SE3Diffusion instance.
            model: Trained SE3ScoreNetwork.
            n: Length of the SE(3) sequence.
            steps: Number of reverse diffusion steps.
            device: Device to run inference on ('cuda' or 'cpu').

        Returns:
            Generated SE(3) sequences [B, n, 4, 4].
        """
        with torch.no_grad():
          if rot_init is None:
              rot = torch.randn(batch_size, 3, n, device=self.device)
          else:
              rot = rot_init
          
          if trans_init is None:
              trans = torch.randn(batch_size, 3, n, device=self.device)
          else:
              trans = trans_init

          rot_scores = []
          trans_scores = []

          t_values = torch.linspace(1, 0, self.num_timesteps, device=self.device)

          for t_index in t_values:
              t = torch.full((batch_size,), t_index, device=self.device)  # Scaled time step

              if return_score:
                  rot, trans, rot_score, trans_score = self.reverse_marginal(t, t_index, rot, trans, model, batch_size, return_score=return_score)
                  rot_scores.append(rot_score.detach().cpu().numpy())
                  trans_scores.append(trans_score.detach().cpu().numpy())
                  del rot_score, trans_score
              else:
                  rot, trans = self.reverse_marginal(t, t_index, rot, trans, model, batch_size)

          # Convert the final outputs into SE(3) matrices
          final_rot = so3_to_matrix(rot.transpose(1,2))  # Convert SO(3) vectors to matrices
          final_se3 = compose_se3(final_rot, trans.transpose(1,2))  # Combine into SE(3)

          if return_score:
              rot_scores = np.stack(rot_scores)
              trans_scores = np.stack(trans_scores)
              return final_se3, rot_scores, trans_scores

          return final_se3
