from torch_motor_utils import *
from data_loader import *
from train_utils import *

from typing import Any, Dict, Optional, List
import clip

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import os
import math
from ga_mdm import ga_mdm
from loss import compute_loss
# from text_encoder import bert_encode_text

TOTAL_DIFFUSION_STEPS = 10
DIFFUSE_KEYS = DATA_KEY_LISTS['acceleration_lie_rot_trans_diffuse_keys']
ANSWER_KEYS = DATA_KEY_LISTS['lie_rot_trans_answer_keys_2']
FULL_LENGTH_KEY = "chained_body_rotors"
GRAPH_PE = json.load(open("serialized/graph_pe.json", "r"))
GRAPH_PE = torch.tensor(GRAPH_PE, dtype=torch.float32, device=device)

LOSS_KEYS = [
    'log_root_rotor',
    'log_root_translator',
    'log_velocity_root_rotor',
    'log_velocity_root_translator',
    'log_acceleration_root_rotor',
    'log_acceleration_root_translator',
    'log_body_rotors',
    'log_velocity_body_rotors',
    'log_acceleration_body_rotors',
    'log_chained_body_rotors',
    ]


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader, RandomSampler, BatchSampler
    from torch.utils.tensorboard import SummaryWriter
    # Optional Anomaly Detection (enable via env)
    # if os.getenv("TORCH_ANOMALY_DETECT", "0") == "1":
    #     torch.autograd.set_detect_anomaly(True)
    monitor_grads = os.getenv("MONITOR_GRADS", "0") == "1"
    num_steps = 20000
    batch_size = 8 if monitor_grads else 32
    # data_path = "joints"
    data_path = "test_data"
    dataset = NPZDictDataset(data_path, keys=set(DIFFUSE_KEYS + ANSWER_KEYS), fixed_length=40)
    # ---for repeat sampling during debugging---
    steps_per_epoch = 10
    sampler = RandomSampler(dataset, replacement=True,
                num_samples=batch_size * steps_per_epoch)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    # ---for repeat sampling during debugging---

    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8, collate_fn=dict_array_collate_fn, pin_memory=True)
    num_epochs = num_steps // (batch_size*steps_per_epoch)
    save_interval = 5000 // batch_size
    model = ga_mdm()
    opt = AdamW(
        # with amp, we don't need to use the mp_trainer's master_params
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.9,
        patience=5,
        min_lr=1e-6,
    )
    schedule_sampler = UniformSampler(TOTAL_DIFFUSION_STEPS)
    step = 0
    fail_count = 0
    LOG_EVERY_N = int(os.getenv("LOG_EVERY_N", "50"))
    max_grad_norm = float(os.getenv("MAX_GRAD_NORM", "1.0"))
    logger.info(f"Train steps: {num_steps}")

    act_stats = None
    if monitor_grads:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.getenv("TB_LOGDIR", os.path.join(repo_root, "runs", "ga_mdm"))
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"TensorBoard log_dir: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)
        layers_monitoring, grads_monitoring = get_all_layers(model, hook_forward, hook_backward)
        grad_averager = GradientAverager()
        # Activation monitors to correlate grad spread with poor conditioning
        act_stats = register_activation_monitors(model)
        depth_lines = [f"{n}: depth={m['depth']}, type={m['type']}, params={m['num_params']}"
                       for n, m in sorted(layers_monitoring.items(), key=lambda kv: kv[1]['depth'])]
        logger.info("Leaf module depths:\n" + "\n".join(depth_lines))

    for epoch in range(num_epochs):
        logger.info(f'Starting epoch {epoch}')
        total_loss = 0.0
        for batch in tqdm(loader):
            step += 1
            t, weights = schedule_sampler.sample(batch['batch_size'])
            # TESTING: fixed noise level
            # t = torch.ones(batch_size, dtype=torch.int64) * 5
            batch, answers, diffuse_shapes = model(batch, t)

            loss = compute_loss(batch, answers, diffuse_shapes)
            with torch.no_grad():
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Skipping step {step} due to NaN/Inf loss")
                    del batch
                    opt.zero_grad()
                    torch.cuda.empty_cache()
                    fail_count += 1
                    if fail_count > 10:
                        logger.error("Too many consecutive failures due to loss NaN/Inf. Exiting.")
                        exit(1)
                    continue
            total_loss += loss.item()
            # Zero gradients before backprop to avoid accumulation
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Apply AGC before global clipping/step
            agc_factor = float(os.getenv("AGC_FACTOR", "0.01"))
            scaled_params = adaptive_gradient_clipping(model, clip_factor=agc_factor)

            # Optional global clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Log gradient flow and spread
            # Log gradient flow and spread
            assert_grad_flow(model, step, writer if monitor_grads else None, getattr(model, "num_layers", 1))
            if monitor_grads:
                # Outlier report every N steps
                if step % int(os.getenv("GRAD_OUTLIER_EVERY", "50")) == 0:
                    rep = grad_outlier_report(model, topk=int(os.getenv("GRAD_OUTLIER_TOPK", "10")))
                    log_grad_outlier_report(writer, rep, step)
                # Activation stats (correlate with flat middle)
                log_activation_stats(writer, act_stats, step, max_items=int(os.getenv("ACT_LOG_MAX", "100")))
                writer.add_scalar("grad_health/total_grad_norm", float(total_norm), step)
                writer.add_scalar("grad_health/agc_scaled_params", float(scaled_params), step)
                log_grads_to_tensorboard(
                    writer,
                    grads_monitoring,
                    step,
                    max_histograms=int(os.getenv("TB_MAX_HISTS", "64")),
                    layers_meta=layers_monitoring,
                    plot_every=int(os.getenv("TB_PLOT_EVERY", "10")),
                    grad_averager=grad_averager,
                )
                grads_monitoring.clear()
                writer.add_scalar("train/loss_step", loss.item(), step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                # Optional: log a few stage-level bars every N steps

            opt.step()
            if step % save_interval == 0:
                save_checkpoint(model, opt, lr_scheduler, epoch+1, loss, filename=f"checkpoint_step_{step}.pt")
        avg_train_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}, train_loss={avg_train_loss:.6f}")
        if monitor_grads:
            writer.add_scalar("loss/train", avg_train_loss, epoch + 1)
            writer.flush()  # ensure data is written to disk

        lr_scheduler.step(avg_train_loss)
    if monitor_grads:
        writer.close()