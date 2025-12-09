from torch.distributions import Normal, Independent
import torch
from train_utils import kingdon_flatten, logger
from torch_motor_utils import *
from data_loader import *
from typing import Any, Dict, Optional, List

TOTAL_DIFFUSION_STEPS = 10
GRAPH_PE = json.load(open("serialized/graph_pe.json", "r"))
GRAPH_PE = torch.tensor(GRAPH_PE, dtype=torch.float32, device=device)

def betas_cosine(T, s=0.008, dtype=torch.float32, device='cpu'):
    # Nichol & Dhariwal cosine schedule
    steps = torch.arange(0, T+1, dtype=dtype, device=device)
    t = steps / T
    f = torch.cos(((t + s) / (1 + s)) * (torch.pi / 2)) ** 2
    # normalize so f(0)=1
    f = f / f[0]
    alphas_cum = f  # this is \bar\alpha_t for t=0..T
    betas = []
    for i in range(1, T+1):
        a_t = alphas_cum[i]
        a_prev = alphas_cum[i-1]
        beta = max(0.0, min(0.999, 1.0 - a_t / a_prev))  # clamp for numerical safety
        betas.append(beta)
    return torch.tensor(betas, dtype=dtype, device=device)

betas = betas_cosine(TOTAL_DIFFUSION_STEPS)
alphas = 1.0 - betas
alphas_cum = torch.cumprod(alphas, dim=0)


def get_alphas_cum():
    betas = betas_cosine(TOTAL_DIFFUSION_STEPS)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def mean_std_from_stats(mean=0.0, M2=0.0, t_count=0, unbiased=True):
    denom = (t_count - 1 if unbiased else t_count) or 1
    var = torch.tensor(M2, dtype=torch.float32) / denom
    var = torch.clamp(var, min=0.0)
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.sqrt(var + 1e-6)
    return mean, std


def flat_normalize(flattened, stats_key):
    mean, std = mean_std_from_stats(**STATS_MAP[stats_key])
    return (flattened - mean[:, None, :, None].to(flattened.device)) / (std[:, None, :, None].to(flattened.device) + 1e-8)


depth = GRAPH_PE[:, 0]
depth_weight = 1.0 / (4.0 + depth)
depth_weight = depth_weight / depth_weight.mean()

def mv_to_flat_normalized(mv, stats_key):
    flattened = kingdon_flatten(mv)
    flattened = flat_normalize(flattened, stats_key)
    if 'body' in stats_key:
        flattened = flattened * depth_weight[None, None, :, None]
    return flattened.permute(0, 2, 1, 3).flatten(end_dim=1).permute(1, 2, 0)


def mv_dict_flatten(batch):
    keys = list(batch.keys())
    new_keys = []
    for k in keys:
        if not isinstance(batch[k], MultiVector):
            continue
        stats_key = k.replace('_0', '').replace('_1', '')
        if stats_key not in STATS_MAP:
            continue
        batch[k] = kingdon_flatten(batch[k])                      # [blades, nodes, T, B]
        batch[k] = flat_normalize(batch[k], stats_key)
        new_keys.append(k)
    return new_keys


def gaussian_from_stats(mean=0.0, M2=0.0, t_count=0, unbiased=True):
    mean, std = mean_std_from_stats(mean, M2, t_count, unbiased)
    return Independent(Normal(mean, std), 1)


def diffuse_lie_data(torch_data, noise_level, diffuse_keys, alpha_cum, pred_len=0):
    # 0 is the highest noise_level
    r = alpha_cum[noise_level]
    diffused_data = {}

    for k in diffuse_keys:
        # temperary fix for _0 keys
        if k.endswith('_0'):
            if pred_len > 0:
                continue
            torch_data[k] = torch_data[k][:, :1, ...]
        dist = gaussian_from_stats(**STATS_MAP[k])
        sample = dist.sample((torch_data[k].shape[1], torch_data[k].shape[-1])).permute(2, 0, 3, 1).to(torch_data[k].device)
        target_data = torch_data[k]
        blended = ((1 - r) * target_data + r * sample).to(device)
        diffused_data[k] = alg.multivector(keys=KEY_MAP[k.replace('_0', '')], values=blended)
    return diffused_data


def flat_to_mv(torch_data, diffuse_keys):
    diffused_data = {}
    for k in diffuse_keys:
        v = torch_data[k]
        # temperary fix for _0 keys
        if k.endswith('_0'):
            diffused_data[k] = alg.multivector(keys=KEY_MAP[k.replace('_0', '')], values=v[:, :1, ...].to(device))
        diffused_data[k] = alg.multivector(keys=KEY_MAP[k.replace('_0', '')], values=v.to(device))
    return diffused_data