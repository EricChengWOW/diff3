import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
from tqdm import tqdm
from torch.optim import AdamW

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim=-1)

# helper modules
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

class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=-1, keepdim=True)
        var = weight.var(dim=-1, keepdim=True, unbiased=False)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
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
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
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
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
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
                block_class(dim_in, dim_in, time_emb_dim=time_dim),
                block_class(dim_in, dim_in, time_emb_dim=time_dim),
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
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
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

class DoubleUnet(nn.Module):
    def __init__(self, dim, time_dim=64, channels=3, unet_layer=4):
        super().__init__()
        self.name = "Unet"
        self.dims = [1]
        for i in range(unet_layer-1):
          self.dims.append(self.dims[-1] * 2)
        self.unet1 = Unet(dim=dim, channels=channels, dim_mults=tuple(self.dims))
        self.unet2 = Unet(dim=dim, channels=channels, dim_mults=tuple(self.dims))
        self.final_conv = nn.Conv1d(dim * 2, channels, 1)  # Combine both outputs from U-Nets

    def forward(self, data1, data2, time):
        # Forward pass through both U-Nets
        out1 = self.unet1(data1, time)
        out2 = self.unet2(data2, time)

        return out1, out2
