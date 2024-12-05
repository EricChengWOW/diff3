import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
from tqdm import tqdm
from torch.optim import AdamW

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim=-1)

class Transformer1D(nn.Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        num_heads: int = 8, 
        num_layers: int = 2, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        max_seq_length: int = 5000  # Adjust based on expected sequence lengths
    ):
        super(Transformer1D, self).__init__()
        
        # Input projection to match dim_out
        self.input_proj = nn.Linear(dim, dim_out)
        
        # Positional Encoding to retain sequence information
        self.positional_encoding = PositionalEncoding(dim_out, dropout, max_seq_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_out, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (optional, can be used to refine the output)
        # self.output_proj = nn.Linear(dim_out, dim_out)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, dim, seq_length)
        
        Returns:
            Tensor of shape (batch_size, dim_out, seq_length)
        """
        # Transpose to (batch_size, seq_length, dim)
        x = x.permute(0, 2, 1)
        
        # Input projection
        x = self.input_proj(x)  # Shape: (batch_size, seq_length, dim_out)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transpose to (seq_length, batch_size, dim_out) for Transformer
        x = x.permute(1, 0, 2)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, dim_out)
        
        # Transpose back to (batch_size, dim_out, seq_length)
        x = x.permute(1, 2, 0)
        
        # Optional Output projection
        # x = self.output_proj(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional Encoding as described in "Attention is All You Need".
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor with positional encodings added
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)

# Helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, num_heads=8, num_layers=2):
        super().__init__()
        self.proj = Transformer1D(dim=dim, dim_out=dim_out, num_heads=num_heads, num_layers=num_layers)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, num_heads=8, num_layers=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups, num_heads=num_heads, num_layers=num_layers)
        self.block2 = Block(dim_out, dim_out, groups=groups, num_heads=num_heads, num_layers=num_layers)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(2)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, l), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum(q, k, 'b h d i, b h d j -> b h i j') * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, 'b h i j, b h d j -> b h i d')
        out = out.view(b, self.hidden_dim, l)
        return self.to_out(out)

# model
class TransformerEncoderUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        num_heads = 8, 
        num_layers = 2, 
        channels=3,
        resnet_block_groups=8,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_class(dim_in, dim_in, time_emb_dim=time_dim, num_heads=num_heads, num_layers=num_layers),
                block_class(dim_in, dim_in, time_emb_dim=time_dim, num_heads=num_heads, num_layers=num_layers),
                Residual(PreNorm(dim_in, Attention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim, num_heads=num_heads, num_layers=num_layers),
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim, num_heads=num_heads, num_layers=num_layers),
                Residual(PreNorm(dim_out, Attention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time):
        time_emb = self.time_mlp(time)

        x = self.init_conv(x)
        r = x.clone()
        down_feats = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            down_feats.append(x)
            x = block2(x, time_emb)
            down_feats.append(x)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            skip = down_feats.pop()
            x = torch.cat((x, skip), dim=1)
            x = block1(x, time_emb)

            skip = down_feats.pop()
            x = torch.cat((x, skip), dim=1)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, time_emb)
        return self.final_conv(x)

class DoubleTransformerEncoderUnet(nn.Module):
    def __init__(self, dim, time_dim=64, channels=3, num_heads = 8, num_layers = 2):
        super().__init__()
        self.name = "Unet"
        self.unet1 = TransformerEncoderUnet(dim=dim, channels=channels, num_heads=num_heads, num_layers=num_layers)
        self.unet2 = TransformerEncoderUnet(dim=dim, channels=channels, num_heads=num_heads, num_layers=num_layers)
        self.final_conv = nn.Conv1d(dim * 2, channels, 1)  # Combine both outputs from U-Nets

    def forward(self, data1, data2, time):
        # Forward pass through both U-Nets
        out1 = self.unet1(data1, time)
        out2 = self.unet2(data2, time)

        return out1, out2
