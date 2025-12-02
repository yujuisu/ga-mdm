from torch_motor_utils import *
from data_loader import *
from train_utils import *
from text_encoder import bert_encode_text
from diffusion import *
import json
import numpy as np

from typing import Any, Dict, Optional, List
import clip

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import os
import math


TOTAL_DIFFUSION_STEPS = 50
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    

class MultiBranchPoseHead(nn.Module):
    def __init__(self, latent_dim: int, use_merge_norm: bool = True, use_head_norm: bool = True):
        super().__init__()
        # Define output blocks (match DIFFUSE_KEYS ordering you flatten later)
        self.specs = [
            (k, len(STATS_MAP[k]['mean'])) for k in DIFFUSE_KEYS
        ]
        self.heads = nn.ModuleDict()
        HeadNorm = nn.LayerNorm if use_head_norm else (lambda d: nn.Identity())
        for k, blades in self.specs:
            nodes = 21 if 'body' in k else 1
            out_dim = blades * nodes
            self.heads[k] = nn.Sequential(
                HeadNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, out_dim),
            )
        self.merge_norm = nn.LayerNorm(latent_dim) if use_merge_norm else nn.Identity()

    def forward(self, x: torch.Tensor, diffused: Dict[str, Any]) -> Dict[str, Any]:
        """
        x: [T, B, latent] -> returns dict[str -> alg.multivector] with reverse-normalized values.
        """
        T, B, _ = x.shape
        x = self.merge_norm(x)
        out_dict: Dict[str, Any] = {}

        for k, blades in self.specs:
            nodes = 21 if 'body' in k else 1
            # predict flattened [T, B, blades*nodes]
            y = self.heads[k](x)  # [T,B,out_dim]
            stats_key = k.replace('_0', '').replace('_1', '')
            frame_start_idx = 0
            new_T = T
            if 'velocity' in k:
                frame_start_idx = 1
                new_T -= 1
            if 'acceleration' in k:
                frame_start_idx = 2
                new_T -= 2
            # reshape to [blades, new_T, nodes, B]
            y = y[frame_start_idx:, :, :].reshape(new_T, B, blades, nodes).permute(2, 0, 3, 1)

            # reverse normalization
            mean, std = mean_std_from_stats(**STATS_MAP[stats_key])
            y = std[:, None, :, None].to(y.device) * y + mean[:, None, :, None].to(y.device)

            # handle *_0 (keep only first frame)
            if k.endswith('_0'):
                values = y[:, :1, ...]
            else:
                values = y
            out_dict[k] = alg.multivector(keys=KEY_MAP[k.replace('_0', '')], values=values) + diffused[k]
        return out_dict
    

def simple_mlp(K, D):
    return nn.Sequential(
            nn.Linear(K, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )


def build_time_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build an additive causal mask for nn.TransformerDecoder.
    Shape: [seq_len, seq_len]; 0 where allowed, -inf where disallowed.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle -inf, diagonal and below 0
    return mask
  

class ga_mdm(nn.Module):
    def __init__(self):
        super(ga_mdm, self).__init__()
        
        self.mv_latent_dim = 16
        self.enable_time_causality = True
        self.latent_dim = self.mv_latent_dim * 23
        self.dropout = 0.1
        self.num_heads = 8
        self.num_layers = 8
        self.ff_size = self.latent_dim * 4
        self.cond_mask_prob = 0.1
        self.activation = 'gelu'
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=500).to(device)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder).to(device)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=True,  # pre-norm improves gradient flow in deep stacks
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer,
            num_layers=self.num_layers
        ).to(device)

        self.clip_dim = 768
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim).to(device)
        self.poseHead = MultiBranchPoseHead(self.latent_dim).to(device)
        # Node (graph) positional encoding once; reuse for all multivectors
        self.node_pe = simple_mlp(6, self.mv_latent_dim).to(device)
        self.node_pe_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.alphas_cum = get_alphas_cum()

        # Lazy per-multivector projection heads and type embeddings
        self.mv_proj = nn.ModuleDict()         # key -> WSLinear(F_k, latent_dim)
        self.mv_type_emb = nn.ParameterDict()  # key -> (latent_dim,)
        
        # Define other layers here
    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def _project_mv(self, k: str, x: torch.Tensor, graph_pe_latent: torch.Tensor) -> torch.Tensor:
        """
        x: [blades, nodes, T, B]
        returns tokens: [nodes*T, B, latent_dim]
        """
        blades, T, nodes, B = x.shape
        F_k = blades  # feature dim we project over
        if k not in self.mv_proj:
            self.mv_proj[k] = nn.Linear(F_k, self.mv_latent_dim).to(device)
        if k not in self.mv_type_emb:
            self.mv_type_emb[k] = nn.Parameter(torch.randn(self.mv_latent_dim, device=device) * 0.02)
        # normalize already done before call
        # project blades -> latent
        x_lin = self.mv_proj[k](x.permute(2, 1, 3, 0))  # [nodes, T, B, latent]
        # add node positional encoding
        if 'body' in k:
            x_lin = x_lin + self.node_pe_scale * graph_pe_latent[:nodes].view(nodes, 1, 1, -1)  
        x_lin = x_lin + self.mv_type_emb[k].view(1, 1, 1, -1)
        # flatten nodes & time into sequence tokens
        # keep time dimension (T) for timewise positional encoding:
        # x_lin: [nodes, T, B, latent] -> tokens: [T, nodes, B, latent]
        tokens = x_lin.permute(0, 3,1,2).flatten(start_dim=0, end_dim=1).permute(1,0,2)  # [nodes*T, B, latent]
        return tokens

    def forward(self, batch: Dict[str, Any], noise_level: list) -> Dict[str, Any]:
        with torch.no_grad():
            answers = {k: batch[k] for k in ANSWER_KEYS}
            caption = batch.pop('text_caption')
            lengths = batch.pop('lengths')
            bs = batch['batch_size']
            # Just a key of full lengths
            lengths = lengths[FULL_LENGTH_KEY]
            diffuse_shapes = [(batch[k].shape[0], batch[k].shape[2]) for k in DIFFUSE_KEYS]
            prefix, batch = diffuse_lie_data(batch, noise_level, DIFFUSE_KEYS, self.alphas_cum)
            diffused = batch.copy()
            # batch = accumulations(batch)
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
            batch['id_lie_tensor'] = torch.zeros_like(batch['log_velocity_root_translator_0'])
            new_keys.append('id_lie_tensor')

        # Precompute node graph PE in latent space and pool across nodes to a global graph token
        # GRAPH_PE: [nodes, 6] -> graph_pe_latent_nodes: [nodes, latent]
        graph_pe_latent = self.node_pe(GRAPH_PE)  # [total_nodes, latent_dim]
        for k in new_keys:
            batch[k] = self._project_mv(k, batch[k], graph_pe_latent)  # [nodes*T, B, latent]
        
        batch = dictionary_flatten(batch)
        # batch = torch.nan_to_num(batch, nan=0.0, posinf=10.0, neginf=-10.0)
        # batch = torch.clamp(batch, -10., 10.)
        log_tensor_stats("input/after_flatten", batch, level=logging.DEBUG)

        # Prepare text conditioning
        with torch.no_grad():
            # Run BERT on the same device as the model (GPU if available) for speed
            enc_text, text_mask = bert_encode_text(caption, device=device)
            if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                text_mask = torch.repeat_interleave(text_mask, bs, dim=0)

        level_emb = self.embed_timestep(noise_level.to(device))
        text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=False))  # casting mask for the single-prompt-for-all case
        emb = text_emb + level_emb

        # batch = torch.cat((emb, batch), axis=0)
        batch = self.sequence_pos_encoder(batch)

        seq_len = batch.shape[0]
        mask = create_mask(lengths).to(device)
        assert mask.shape[1] == seq_len, f"Mask S={mask.shape[1]} != input S={seq_len}"
        # Warn if any sequence is fully masked (degenerate attention)
        fully_masked = mask.all(dim=1)
        if bool(fully_masked.any()):
            bad_idx = torch.nonzero(fully_masked).flatten().tolist()
            logger.warning(f"Fully-masked rows in src_key_padding_mask at indices {bad_idx}; seq_len={seq_len}, lengths[bad]={[lengths[i] for i in bad_idx]}")
        
        # --- Time-causal attention mask ---
        tgt_mask = None
        if self.enable_time_causality:
            tgt_mask = build_time_causal_mask(seq_len, device)

        # Decode with or without causal mask
        if tgt_mask is not None:
            batch = self.seqTransDecoder(
                tgt=batch,
                memory=emb,
                memory_key_padding_mask=text_mask,
                tgt_key_padding_mask=mask,
                tgt_mask=tgt_mask
            )
        else:
            batch = self.seqTransDecoder(
                tgt=batch,
                memory=emb,
                memory_key_padding_mask=text_mask,
                tgt_key_padding_mask=mask
            )
        
        batch = self.poseHead(batch, diffused)
        return batch, answers, diffuse_shapes