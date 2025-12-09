import logging
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from abc import ABC, abstractmethod
import numpy as np
import torch
import os
from kingdon import MultiVector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional

def kingdon_flatten(x):
    if not isinstance(x, MultiVector):
        return x
    stacked = x.values()
    if isinstance(stacked, list):
        stacked = torch.stack(stacked)
    return stacked


def setup_logger(name=__name__, level=None, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = level or os.getenv("LOG_LEVEL", "INFO")
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(lvl)
    logger.addHandler(sh)
    lf = log_file or os("LOG_FILE")
    if lf:
        fh = logging.FileHandler(lf)
        fh.setFormatter(fmt)
        fh.setLevel(lvl)
        logger.addHandler(fh)
    return logger


logger = setup_logger(__name__, log_file="training.log")


def log_tensor_stats(name, t: torch.Tensor, level=logging.DEBUG):
    try:
        if t is None:
            return
        with torch.no_grad():
            nan = int(torch.isnan(t).sum().item()) if t.is_floating_point() else 0
            inf = int(torch.isinf(t).sum().item()) if t.is_floating_point() else 0
            safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) if t.is_floating_point() else t
            msg = {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "device": str(t.device),
                "min": float(safe.min().item()) if t.numel() else None,
                "max": float(safe.max().item()) if t.numel() else None,
                "mean": float(safe.mean().item()) if t.numel() and t.is_floating_point() else None,
                "std": float(safe.std(unbiased=False).item()) if t.numel() and t.is_floating_point() else None,
                "nan": nan,
                "inf": inf,
            }
            logger.log(level, f"{name}: {msg}")
    except Exception:
        logger.exception(f"Failed logging stats for {name}")

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float()
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps=50):
        self._weights = np.ones(num_timesteps)

    def weights(self):
        return self._weights
    

from copy import deepcopy
import numpy as _np
class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        dynamic_text_mode = type(kargs['model_kwargs']['y']['text'][0]) == list  # Text changes on the fly - prompt per prediction is provided as a list (instead of a single prompt)
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        # Autoregressive sampling
        for i in range(n_iterations):
            
            # Build the current kargs
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i], cur_kargs['model_kwargs']['y']['text_embed'][1][:, i])
                else:
                    raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
            
            # Sample the next prediction
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            # Buffer the sample
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # Update the prefix
            cur_prefix = sample.clone()[..., -self.args.context_len:]

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch


# def dictionary_flatten(flattened):
#     lie_template = flattened['log_velocity_root_rotor']
#     id_lie_tensor = torch.zeros_like(lie_template[:1])
#     flattened = {
#         'body_rotors': [flattened[k] for k in ['log_body_rotors_0', 'log_velocity_body_rotors']],
#         'root_translator': [id_lie_tensor] + [flattened[k] for k in ['log_velocity_root_translator']],
#         'root_rotor': [id_lie_tensor] + [flattened[k] for k in ['log_velocity_root_rotor']],
#     }

#     flattened = {k: torch.cat(v, dim=0) for k, v in flattened.items()}
#     return torch.cat(list(flattened.values()), dim=1).permute(0, 2, 1)


def dictionary_flatten(flattened):
    flattened = {
        # 'floor': [flattened[k] for k in ['floor_0', 'floor_1', 'floor']],
        # 'root_motor': [id_motor_tensor]+[flattened[k] for k in ['root_motor_1', 'root_motor']],
        # 'chained_body_rotors': [flattened[k] for k in ['chained_body_rotors_0', 'chained_body_rotors_1', 'chained_body_rotors']],
        'body_rotors': [flattened[k] for k in ['log_body_rotors_0', 'log_velocity_body_rotors_0', 'log_acceleration_body_rotors']],
        'root_translator': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_translator_0', 'log_acceleration_root_translator']],
        'root_rotor': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_rotor_0', 'log_acceleration_root_rotor']],
    }
    # print([[t.shape for t in l] for l in flattened.values()])
    flattened = {k: torch.cat(v, dim=0) for k, v in flattened.items()}
    return torch.cat(list(flattened.values()), dim=1).permute(0, 2, 1)

def pred_dictionary_flatten(flattened):
    return torch.cat(list(flattened.values()), dim=1).permute(0, 2, 1)

def prefix_dictionary_flatten(flattened):
    flattened0 = {
        # 'floor': [flattened[k] for k in ['floor_0', 'floor_1', 'floor']],
        'root_motor': [flattened[k] for k in ['id_motor_tensor', 'root_motor_1', 'root_motor']],
        'chained_body_rotors': [flattened[k] for k in ['chained_body_rotors_0', 'chained_body_rotors_1', 'chained_body_rotors']],
        # 'body_rotors': [flattened[k] for k in ['log_body_rotors_0', 'log_velocity_body_rotors_0', 'log_acceleration_body_rotors']],
        # 'root_translator': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_translator_0', 'log_acceleration_root_translator']],
        # 'root_rotor': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_rotor_0', 'log_acceleration_root_rotor']],
    }
    flattened0 = {k: torch.cat(v, dim=0) for k, v in flattened0.items()}
    flattened0 = torch.cat(list(flattened0.values()), dim=1).permute(0, 2, 1)
    flattened1 = {
        # 'floor': [flattened[k] for k in ['floor_0', 'floor_1', 'floor']],
        # 'root_motor': [flattened[k] for k in ['id_motor_tensor', 'root_motor_1', 'root_motor']],
        # 'chained_body_rotors': [flattened[k] for k in ['chained_body_rotors_0', 'chained_body_rotors_1', 'chained_body_rotors']],
        'body_rotors': [flattened[k] for k in ['log_body_rotors_0', 'log_velocity_body_rotors_0', 'log_acceleration_body_rotors']],
        'root_translator': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_translator_0', 'log_acceleration_root_translator']],
        'root_rotor': [flattened[k] for k in ['id_lie_tensor', 'log_velocity_root_rotor_0', 'log_acceleration_root_rotor']],
    }
    flattened1 = {k: torch.cat(v, dim=0) for k, v in flattened1.items()}
    flattened1 = torch.cat(list(flattened1.values()), dim=1).permute(0, 2, 1)
    
    return flattened0, flattened1
    

def create_mask(lengths, max_len=None, reserve_index0=0, pred_len=0):
    """
    Returns src_key_padding_mask with True only at PAD positions.
    Shape: [batch, S], where S == 1 + max_len (index 0 reserved for special token).
    """
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    max_len = max_len if max_len is not None else (max(lengths) if lengths else 0)

    mask = torch.ones((len(lengths), max_len + reserve_index0), dtype=torch.bool)
    for i, L in enumerate(lengths):
        L = int(L)
        # Unmask special token + valid timesteps [0..L]
        mask[i, : min(L + reserve_index0, max_len + reserve_index0)] = False
    return mask


# note that wrapper functions are used for Python closure
# so that we can pass arguments.

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        """Forward pass hook which attaches backward pass hooks to intermediate tensors"""
        output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        """Backward pass hook which appends gradients"""
        grads.append((module_name, grad))
    return hook

def _module_depths(model):
    """Compute a depth index for every module name based on its dotted path."""
    depths = {}
    for name, _ in model.named_modules():
        depths[name] = 0 if name == "" else len(name.split("."))
    return depths

def get_all_layers(model, hook_forward, hook_backward):
    """Register forward pass hook to leaf modules and collect meta to save memory."""
    layers = {}
    grads = []
    depths = _module_depths(model)

    for name, layer in model.named_modules():
        # only leaf modules (no children)
        if not any(layer.children()):
            try:
                num_params = sum(p.numel() for p in layer.parameters(recurse=False))
            except Exception:
                num_params = 0
            layers[name] = {
                "type": layer.__class__.__name__,
                "num_params": int(num_params),
                "depth": int(depths.get(name, 0)),
            }
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

def plot_depth_histogram(layers_meta):
    """Plot histogram of module depths to verify expected depth distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not layers_meta:
        return None
    depths = sorted([m["depth"] for m in layers_meta.values() if isinstance(m, dict) and "depth" in m])
    if not depths:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(depths, bins=min(20, max(5, len(set(depths)))), color="#3b82f6", alpha=0.8)
    ax.set_title(f"Module depth histogram (min={min(depths)}, max={max(depths)})")
    ax.set_xlabel("Depth (dotted path length)")
    ax.set_ylabel("# leaf modules")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

def plot_grad_flow_by_layer(grads):
    """Create a matplotlib Figure plotting mean gradient per leaf module in backward (flow) order."""
    if not grads:
        return None
    # preserve first-seen order from the backward pass
    layer_means = {}
    order = []
    seen = set()
    for name, g in grads:
        if g is None or not torch.is_tensor(g):
            continue
        if name not in seen:
            order.append(name)
            seen.add(name)
        with torch.no_grad():
            safe = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            mean = float(safe.abs().mean().item()) if safe.numel() else 0.0
        layer_means.setdefault(name, []).append(mean)

    if not layer_means:
        return None

    names_in_flow_order = [n for n in order if n in layer_means]
    means = [float(torch.tensor(layer_means[n]).mean().item()) for n in names_in_flow_order]

    fig_w = max(8, len(names_in_flow_order) * 0.22)
    fig_h = 4.5  # taller to fit labels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.plot(range(len(names_in_flow_order)), means, marker="o", linewidth=1.8)
    ax.set_title("Mean abs gradient by layer (flow order)")
    ax.set_xlabel("Layer (backward hook order)")
    ax.set_ylabel("Mean(|grad|)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(names_in_flow_order) <= 80:
        ax.set_xticks(range(len(names_in_flow_order)))
        ax.set_xticklabels(names_in_flow_order, rotation=90, fontsize=7)
    plt.tight_layout(pad=0.6)
    return fig

class GradientAverager:
    """Keeps running mean of gradient per layer name across training steps."""
    def __init__(self):
        self.sum = {}   # name -> sum of means
        self.count = {} # name -> number of updates
        self.order = [] # first-seen order from gradient flow across steps
        self._seen = set()

    def update(self, grads):
        # record names in first-seen backward order
        for name, g in grads:
            if name not in self._seen:
                self.order.append(name)
                self._seen.add(name)
            if g is None or not torch.is_tensor(g):
                continue
            with torch.no_grad():
                safe = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                mean = float(safe.abs().mean().item()) if safe.numel() else 0.0
            self.sum[name] = self.sum.get(name, 0.0) + mean
            self.count[name] = self.count.get(name, 0) + 1

    def running_mean(self):
        return {n: (self.sum[n] / max(self.count[n], 1)) for n in self.sum.keys()}

def plot_grad_running_mean_by_layer(running_means, names_order=None):
    """Plot running mean gradient per layer. Uses flow order if names_order is provided."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not running_means:
        return None
    if names_order is None:
        names = list(running_means.keys())  # preserve insertion if dict was built in order
    else:
        names = [n for n in names_order if n in running_means]

    means = [float(running_means[n]) for n in names]

    fig_w = max(8, len(names) * 0.22)
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.plot(range(len(names)), means, marker="o", linewidth=1.8)
    ax.set_title("Running mean abs gradient by layer (flow order across steps)")
    ax.set_xlabel("Layer (first-seen backward order)")
    ax.set_ylabel("Mean(|grad|) over steps")
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(names) <= 80:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
    plt.tight_layout(pad=0.6)
    return fig
def grads_summary(grads):
    """Compute lightweight stats for a list of (name, grad_tensor) tuples."""
    summary = []
    for name, g in grads:
        if g is None or not torch.is_tensor(g):
            continue
        try:
            with torch.no_grad():
                safe = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                numel = int(safe.numel())
                if numel == 0:
                    continue
                abs_safe = safe.abs()
                summary.append({
                    "name": name,
                    "numel": numel,
                    "mean_abs": float(abs_safe.mean().item()),            # changed: mean of abs
                    "std_abs": float(abs_safe.std(unbiased=False).item()),# changed: std of abs
                    "min_abs": float(abs_safe.min().item()),              # changed: min of abs
                    "max_abs": float(abs_safe.max().item()),              # changed: max of abs
                    "norm": float(safe.norm().item()),
                })
        except Exception:
            continue
    return summary

def log_grads_to_tensorboard(writer, grads, step, max_histograms=64, layers_meta=None, plot_every=10, grad_averager=None):
    """Log gradients: per-layer scalars, per-layer mean abs figure, optional running mean abs over steps."""
    if writer is None:
        return

    # Update running averager (also captures flow order across steps)
    if grad_averager is not None:
        grad_averager.update(grads)

    # Per-layer scalars (abs)
    try:
        for s in grads_summary(grads):
            writer.add_scalar(f"grads/{s['name']}/norm", s["norm"], step)
            writer.add_scalar(f"grads/{s['name']}/mean_abs", s["mean_abs"], step)  # changed
            writer.add_scalar(f"grads/{s['name']}/std_abs", s["std_abs"], step)    # changed
            writer.add_scalar(f"grads/{s['name']}/min_abs", s["min_abs"], step)    # changed
            writer.add_scalar(f"grads/{s['name']}/max_abs", s["max_abs"], step)    # changed
    except Exception:
        logger.exception("Failed writing per-layer grad scalars")

    # Plot mean abs grad vs layer for this step (flow order)
    try:
        if step % plot_every == 0:
            names_order = []
            seen = set()
            for n, _ in grads:
                if n not in seen:
                    names_order.append(n)
                    seen.add(n)
            means_map = {s["name"]: s["mean_abs"] for s in grads_summary(grads)}  # changed
            fig_w = max(8, len(names_order) * 0.22)
            fig_h = 4.5
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
            y = [means_map.get(n, 0.0) for n in names_order]
            ax.plot(range(len(names_order)), y, marker="o", linewidth=1.8)
            ax.set_title("Mean abs gradient by layer (this step, flow order)")
            ax.set_xlabel("Layer (backward hook order)")
            ax.set_ylabel("Mean(|grad|)")
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(names_order) <= 80:
                ax.set_xticks(range(len(names_order)))
                ax.set_xticklabels(names_order, rotation=90, fontsize=7)
            plt.tight_layout(pad=0.6)
            writer.add_figure("grads_layers/mean_vs_layer_step", fig, global_step=step, close=True)
    except Exception:
        logger.exception("Failed writing per-step per-layer grad figure")

    # Plot running mean abs across steps (flow order learned over time)
    try:
        if grad_averager is not None and (step % plot_every == 0):
            fig_avg = plot_grad_running_mean_by_layer(grad_averager.running_mean(), names_order=grad_averager.order)
            if fig_avg is not None:
                writer.add_figure("grads_layers/running_mean_vs_layer", fig_avg, global_step=step, close=True)
    except Exception:
        logger.exception("Failed writing running mean per-layer grad figure")

    # Limited histograms
    try:
        count = 0
        for name, g in grads:
            if count >= max_histograms:
                break
            if g is None or not torch.is_tensor(g):
                continue
            with torch.no_grad():
                writer.add_histogram(f"grads/{name}/hist", g.detach().cpu(), global_step=step)
            count += 1
    except Exception:
        logger.exception("Failed writing grad histograms")

def save_checkpoint(model, optimizer, lr_scheduler, epoch, loss, path="checkpoints", filename="checkpoint.pt"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, os.path.join(path, filename))
    logger.info(f"Saved checkpoint at epoch {epoch} to {path}/{filename}")


def collect_param_grad_stats(model):
    stats = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n = float(g.norm().item())
        m = float(g.abs().mean().item())
        stats[name] = (n, m)
    return stats


def _decoder_layer_index_from_name(name: str) -> Optional[int]:
    if "seqTransDecoder" not in name or ".layers." not in name:
        return None
    parts = name.split(".")
    try:
        i = parts.index("layers") + 1
        return int(parts[i])
    except Exception:
        return None

def classify_param_stage(name: str, num_layers: int) -> str:
    # Early: projections/embeddings/PE
    if name.startswith("mv_proj.") or name.startswith("mv_type_emb") or name.startswith("node_pe"):
        return "early"
    if name.startswith("poseHead"):
        return "late"
    # Decoder layers split into thirds -> early/mid/late
    li = _decoder_layer_index_from_name(name)
    if li is None:
        return "other"
    third = max(1, num_layers // 3)
    if li < third:
        return "early"
    if li >= 2 * third:
        return "late"
    return "mid"


@torch.no_grad()
def grad_flow_stats(model, num_layers: int, eps: float = 1e-12):
    per_stage = defaultdict(list)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        m = float(g.abs().mean().item())
        stage = classify_param_stage(name, num_layers)
        per_stage[stage].append(m)
    stats = {}
    for stage, vals in per_stage.items():
        t = torch.tensor(vals)
        if t.numel() == 0:
            continue
        p50 = torch.quantile(t, 0.50)
        p95 = torch.quantile(t, 0.95)
        p05 = torch.quantile(t, 0.05)
        stats[stage] = {
            "count": int(t.numel()),
            "frac_nonzero": float((t > eps).float().mean().item()),
            "mean_abs": float(t.mean().item()),
            "median_abs": float(p50.item()),
            "max_abs": float(t.max().item()),
            "min_abs": float(t.min().item()),
            "std_abs": float(t.std(unbiased=False).item()),
            "spread_ratio": float(t.max().item() / (p50.item() + eps)),  # max/median
            "log10_range": float(torch.log10(t.max() + eps) - torch.log10(t.min() + eps)),
            # robust spreads
            "p95_over_p50": float((p95 / (p50 + eps)).item()),
            "p95_over_p05": float((p95 / (p05 + eps)).item()),
            "log10_p95_p05": float((torch.log10(p95 + eps) - torch.log10(p05 + eps)).item()),
            "cv": float((t.std(unbiased=False) / (t.mean() + eps)).item()),
        }
    return stats

def assert_grad_flow(model, step: int, writer, num_layers: int):
    stats = grad_flow_stats(model, num_layers)
    if writer is not None:
        for stage, s in stats.items():
            writer.add_scalar(f"grad_flow/{stage}/frac_nonzero", s["frac_nonzero"], step)
            writer.add_scalar(f"grad_flow/{stage}/mean_abs", s["mean_abs"], step)
            writer.add_scalar(f"grad_flow/{stage}/median_abs", s["median_abs"], step)
            writer.add_scalar(f"grad_flow/{stage}/max_abs", s["max_abs"], step)
            writer.add_scalar(f"grad_flow/{stage}/min_abs", s["min_abs"], step)
            writer.add_scalar(f"grad_flow/{stage}/std_abs", s["std_abs"], step)
            writer.add_scalar(f"grad_flow/{stage}/spread_ratio", s["spread_ratio"], step)
            writer.add_scalar(f"grad_flow/{stage}/log10_range", s["log10_range"], step)
            writer.add_scalar(f"grad_flow/{stage}/p95_over_p50", s["p95_over_p50"], step)
            writer.add_scalar(f"grad_flow/{stage}/p95_over_p05", s["p95_over_p05"], step)
            writer.add_scalar(f"grad_flow/{stage}/log10_p95_p05", s["log10_p95_p05"], step)
            writer.add_scalar(f"grad_flow/{stage}/cv", s["cv"], step)
    if os.getenv("GRAD_FLOW_ASSERT", "0") == "1":
        min_frac = float(os.getenv("GF_MIN_FRAC", "0.7"))
        max_spread = float(os.getenv("GF_MAX_SPREAD", "1e3"))        # ~log10_range 3
        max_log_range = float(os.getenv("GF_MAX_LOG_RANGE", "6.0"))  # ~1e6 dynamic range
        max_p95_p05 = float(os.getenv("GF_MAX_P95_P05", "1e3"))      # robust spread
        warmup_steps = int(os.getenv("GF_WARMUP_STEPS", "50"))
        if step > warmup_steps:
            for stage_key in ("early", "mid", "late"):
                if stage_key in stats and stats[stage_key]["count"] > 0:
                    s = stats[stage_key]
                    if s["frac_nonzero"] < min_frac:
                        raise RuntimeError(f"Grad flow failed at step {step}: {stage_key} frac_nonzero={s['frac_nonzero']:.3f} < {min_frac}")
                    if s["spread_ratio"] > max_spread or s["p95_over_p05"] > max_p95_p05:
                        raise RuntimeError(f"Grad spread too wide at step {step}: {stage_key} "
                                           f"max/median={s['spread_ratio']:.2e}, log10_range={s['log10_range']:.2f}, p95/p05={s['p95_over_p05']:.2e}")


def adaptive_gradient_clipping(model, clip_factor: float = 0.01, eps: float = 1e-3):
    """
    Adaptive Gradient Clipping (AGC):
    Clip per-parameter gradients based on parameter norm: ||g|| <= clip_factor * (||p|| + eps)
    """
    total_scaled = 0
    for p in model.parameters():
        if p.grad is None or not p.requires_grad:
            continue
        g = p.grad
        param_norm = p.norm()
        grad_norm = g.norm()
        max_norm = clip_factor * (param_norm + eps)
        if grad_norm > max_norm and grad_norm > 0:
            scale = max_norm / (grad_norm + eps)
            g.mul_(scale)
            total_scaled += 1
    return total_scaled


import io
import torch.nn as nn

@torch.no_grad()
def grad_outlier_report(model, topk: int = 10):
    """Return top/bottom params by grad-to-param ratio and absolute grad."""
    entries = []
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        g = p.grad.detach()
        w = p.detach()
        g_abs_mean = float(g.abs().mean().item())
        w_abs_mean = float(w.abs().mean().item()) + 1e-12
        ratio = g_abs_mean / w_abs_mean  # scale-invariant indicator
        entries.append((name, p.shape, g_abs_mean, w_abs_mean, ratio))
    if not entries:
        return {"top_ratio": [], "bottom_ratio": [], "top_grad": []}
    # sort by ratio and by abs grad
    by_ratio = sorted(entries, key=lambda x: x[4], reverse=True)
    by_grad = sorted(entries, key=lambda x: x[2], reverse=True)
    return {
        "top_ratio": by_ratio[:topk],
        "bottom_ratio": by_ratio[-topk:],
        "top_grad": by_grad[:topk],
    }

def _tb_text_from_entries(title: str, rows):
    buf = io.StringIO()
    print(title, file=buf)
    print("name | shape | mean|grad| | mean|param| | ratio(|g|/|w|)", file=buf)
    for name, shape, g_abs_mean, w_abs_mean, ratio in rows:
        print(f"{name} | {tuple(shape)} | {g_abs_mean:.3e} | {w_abs_mean:.3e} | {ratio:.3e}", file=buf)
    return buf.getvalue()

def log_grad_outlier_report(writer, report, step, tag_prefix="grad_outliers"):
    if writer is None or not report:
        return
    if report.get("top_ratio"):
        writer.add_text(tag_prefix + "/top_ratio", _tb_text_from_entries("Top ratio |g|/|w|", report["top_ratio"]), step)
    if report.get("bottom_ratio"):
        writer.add_text(tag_prefix + "/bottom_ratio", _tb_text_from_entries("Bottom ratio |g|/|w|", report["bottom_ratio"]), step)
    if report.get("top_grad"):
        writer.add_text(tag_prefix + "/top_grad", _tb_text_from_entries("Top mean|grad|", report["top_grad"]), step)

def register_activation_monitors(model):
    """
    Register lightweight forward hooks on leaf modules to record activation mean/std.
    Returns a dict that is updated in-place by hooks: act_stats[name] = (mean, std).
    """
    act_stats = {}
    def _hook(name):
        def h(module, inp, out):
            try:
                x = out if torch.is_tensor(out) else out[0]
                x = x.detach()
                # reduce across batch/time/token dims
                m = float(x.mean().item())
                s = float(x.std(unbiased=False).item())
                act_stats[name] = (m, s)
            except Exception:
                pass
        return h

    for name, m in model.named_modules():
        # leaf modules only
        if any(m.children()):
            continue
        # focus on modules likely to cause scaling issues
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            m.register_forward_hook(_hook(name))
    return act_stats

def log_activation_stats(writer, act_stats, step, max_items=100):
    """Log per-layer activation mean/std (limited to max_items to save TB size)."""
    if writer is None or not act_stats:
        return
    count = 0
    for name, (m, s) in act_stats.items():
        writer.add_scalar(f"activations/{name}/mean", m, step)
        writer.add_scalar(f"activations/{name}/std", s, step)
        count += 1
        if count >= max_items:
            break