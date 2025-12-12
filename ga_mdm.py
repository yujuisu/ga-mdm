from torch_motor_utils import *
from data_loader import *
from train_utils import *
from text_encoder import bert_encode_text
from diffusion import *
from loss import accumulations, extract_index_list
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
TRAJECTORY_KEYS = DATA_KEY_LISTS['trajectory']
LOSS_KEYS = DATA_KEY_LISTS["loss"]
FULL_LENGTH_KEY = "chained_body_rotors"
GRAPH_PE = json.load(open("serialized/graph_pe.json", "r"))
GRAPH_PE = torch.tensor(GRAPH_PE, dtype=torch.float32, device=device)


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

    def forward(self, x, start_pos=0):
        # not used in the final model
        # support scalar start_pos or per-batch list/tensor of start positions
        S, B, D = x.shape
        pe = self.pe.squeeze(1).to(x.device)  # [max_len, d_model]

        # normalize start_pos into tensor or int
        if isinstance(start_pos, (list, tuple, np.ndarray)):
            start_pos = torch.tensor(start_pos, device=x.device, dtype=torch.long)
        if isinstance(start_pos, torch.Tensor) and start_pos.numel() == 1:
            start_pos = int(start_pos.item())

        if isinstance(start_pos, torch.Tensor):
            if start_pos.shape[0] != B:
                raise ValueError(f"start_pos length {start_pos.shape[0]} != batch size {B}")
            # pos: [S, B] with absolute positions for each (time, batch)
            pos = torch.arange(S, device=x.device).unsqueeze(1) + start_pos.unsqueeze(0)
            x = x + pe[pos]  # pe indexed with [S,B] -> [S,B,D]
        else:
            # scalar start_pos (int)
            sp = int(start_pos)
            x = x + pe[sp:sp + S].unsqueeze(1)
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
        # zero-init final layer so time embed starts near 0 (reduces early spikes)
        nn.init.zeros_(self.time_embed[-1].weight)
        nn.init.zeros_(self.time_embed[-1].bias)

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    

class MultiBranchPoseHead(nn.Module):
    def __init__(self, latent_dim: int, use_merge_norm: bool = True, pred_len: int = 0):
        super().__init__()
        # Define output blocks (match DIFFUSE_KEYS ordering you flatten later)
        self.pred_len = pred_len
        if self.pred_len > 0:
            diffuse_keys = [k for k in DIFFUSE_KEYS if not k.endswith('_0')]
        else:
            diffuse_keys = DIFFUSE_KEYS
        self.specs = [
            (k, len(STATS_MAP[k]['mean'])) for k in diffuse_keys
        ]
        self.heads = nn.ModuleDict()
        for k, blades in self.specs:
            nodes = 21 if 'body' in k else 1
            out_dim = blades * nodes
            self.heads[k] = nn.Sequential(
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
            if self.pred_len == 0:
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
    if seq_len <= 1:
        return None
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle -inf, diagonal and below 0
    return mask
  

class ga_mdm(nn.Module):
    def __init__(self, fixed_length: Optional[int] = None, pred_len: int = 0):
        super(ga_mdm, self).__init__()
        self.pred_len = pred_len
        self.mv_latent_dim = 16
        self.enable_time_causality = False
        self.latent_dim = self.mv_latent_dim * 23
        self.fixed_length = fixed_length
        self.dropout = 0.1
        self.num_heads = 8
        self.num_layers = 4
        self.ff_size = self.latent_dim * 4
        self.cond_mask_prob = 0.1
        self.activation = 'gelu'
        self.max_length = self.fixed_length * 2 if self.fixed_length is not None else 200
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.max_length).to(device)
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
        self.poseHead = MultiBranchPoseHead(self.latent_dim, pred_len=self.pred_len).to(device)
        # Node (graph) positional encoding once; reuse for all multivectors
        self.node_pe = simple_mlp(6, self.mv_latent_dim).to(device)
        self.node_pe_scale = nn.Parameter(torch.tensor(0.1, device=device))
        self.alphas_cum = get_alphas_cum()

        # Add normalization and a learned scale for timestep embedding
        self.t_embed_norm = nn.LayerNorm(self.latent_dim).to(device)
        self.t_embed_scale = nn.Parameter(torch.tensor(0.1, device=device))  # small initial influence
        # Optional: also normalize text embedding to align scales
        self.text_embed_norm = nn.LayerNorm(self.latent_dim).to(device)

        # NEW: normalize and gate node positional encodings (per-node latent)
        self.node_pe_norm = nn.LayerNorm(self.mv_latent_dim).to(device)
        self.node_pe_gate = nn.Parameter(torch.tensor(0.1, device=device))  # small initial influence

        # Lazy per-multivector projection heads and type embeddings
        self.mv_proj = nn.ModuleDict()         # key -> WSLinear(F_k, latent_dim)
        self.mv_type_emb = nn.ParameterDict()  # key -> (latent_dim,)

        self.embed_prefix = nn.Linear(self.mv_latent_dim * 22, self.latent_dim).to(device)
        
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
            # graph_pe_latent: [total_nodes, mv_latent_dim] -> normalize per-node features
            pe_nodes = self.node_pe_norm(graph_pe_latent[:nodes])  # [nodes, mv_latent_dim]
            x_lin = x_lin + (self.node_pe_gate * self.node_pe_scale) * pe_nodes.view(nodes, 1, 1, -1)

        x_lin = x_lin + self.mv_type_emb[k].view(1, 1, 1, -1)
        # flatten nodes & time into sequence tokens
        # keep time dimension (T) for timewise positional encoding:
        # x_lin: [nodes, T, B, latent] -> tokens: [T, nodes, B, latent]
        tokens = x_lin.permute(0, 3,1,2).flatten(start_dim=0, end_dim=1).permute(1,0,2)  # [nodes*T, B, latent]
        return tokens

    def forward(self, batch: Dict[str, Any], pred: Optional[Dict[str, Any]], noise_level: list) -> Dict[str, Any]:
        with torch.no_grad():
            caption = batch.pop('text_caption')
            lengths = batch.pop('lengths')
            bs = batch['batch_size']
            # Just a key of full lengths
            lengths = lengths[FULL_LENGTH_KEY]

            if self.pred_len:
                answers = {k: pred[k] for k in ANSWER_KEYS}
            else:
                answers = {k: batch[k] for k in ANSWER_KEYS}
                pred = batch # let them be the same object, i.e. predict the whole batch
            
            diffuse_shapes = [(batch[k].shape[0], batch[k].shape[2]) for k in DIFFUSE_KEYS]
            pred = diffuse_lie_data(pred, noise_level, DIFFUSE_KEYS, self.alphas_cum, pred_len=self.pred_len)
            diffused = deepcopy(pred)
            keys_pred = mv_dict_flatten(pred)
            if self.pred_len:
                prefix = batch
                trajectory = {}
                for k in TRAJECTORY_KEYS:
                    offset = 0
                    if 'velocity' in k:
                        offset = 1
                    inds = torch.tensor(lengths) - 1 - offset  # [B]
                    trajectory = extract_index_list(prefix)
                prefix = flat_to_mv(prefix, DIFFUSE_KEYS)
                
                prefix = accumulations(prefix)
                keys_prefix = mv_dict_flatten(prefix)
                prefix['id_lie_tensor'] = torch.zeros_like(prefix['log_velocity_root_translator_0'])
                keys_prefix.append('id_lie_tensor')
                prefix['id_motor_tensor'] = torch.zeros_like(prefix['root_motor_1'])
                prefix['id_motor_tensor'][0, ...] = 1
                keys_prefix.append('id_motor_tensor')
            else:
                pred['id_lie_tensor'] = torch.zeros_like(pred['log_velocity_root_translator_0'])
                keys_pred.append('id_lie_tensor')
                trajectory = None

        # Precompute node graph PE in latent space and pool across nodes to a global graph token
        # GRAPH_PE: [nodes, 6] -> graph_pe_latent_nodes: [nodes, latent]
        graph_pe_latent = self.node_pe(GRAPH_PE)  # [total_nodes, latent_dim]
        for k in keys_pred:
            pred[k] = self._project_mv(k, pred[k], graph_pe_latent)  # [nodes*T, B, latent]
        
        

        if self.pred_len:
            pred = pred_dictionary_flatten(pred)
            for k in keys_prefix:
                prefix[k] = self._project_mv('prefix_'+k, prefix[k], graph_pe_latent)  # [nodes*T, B, latent]
            prefix0, prefix1 = prefix_dictionary_flatten(prefix)
            del prefix

        else:
            pred = dictionary_flatten(pred)


        # Prepare text conditioning
        with torch.no_grad():
            # Run BERT on the same device as the model (GPU if available) for speed
            enc_text, text_mask = bert_encode_text(caption, device=device)
            if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                text_mask = torch.repeat_interleave(text_mask, bs, dim=0)

        level_emb = self.embed_timestep(noise_level.to(device))
        text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=False))  # casting mask for the single-prompt-for-all case
        
        
        # Normalize and softly gate timestep embedding before summation
        level_emb = self.t_embed_norm(level_emb)
        text_emb = self.text_embed_norm(text_emb)
        if self.pred_len:
            prefix0 = self.embed_prefix(prefix0)
            prefix0 = self.sequence_pos_encoder(prefix0)
            prefix1 = self.sequence_pos_encoder(prefix1)
            emb = torch.cat((text_emb, self.t_embed_scale * level_emb, prefix0, prefix1), axis=0)
        else:
            emb = torch.cat((text_emb, self.t_embed_scale * level_emb), axis=0)
            
        seq_len = emb.shape[0]
        mask = create_mask(lengths).to(device)
        if self.pred_len:
            pred = self.sequence_pos_encoder(pred, start_pos=lengths)
            memory_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask, mask, mask], dim=1)
        else:
            pred = self.sequence_pos_encoder(pred)
            memory_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        assert memory_mask.shape[1] == seq_len, f"Mask S={memory_mask.shape[1]} != input S={seq_len}"
        # Warn if any sequence is fully masked (degenerate attention)
        fully_masked = memory_mask.all(dim=1)
        if bool(fully_masked.any()):
            bad_idx = torch.nonzero(fully_masked).flatten().tolist()
            logger.warning(f"Fully-masked rows in src_key_padding_mask at indices {bad_idx}; seq_len={seq_len}, lengths[bad]={[lengths[i] for i in bad_idx]}")

        # --- Time-causal attention mask ---
        tgt_mask = None
        if self.enable_time_causality:
            tgt_mask = build_time_causal_mask(pred.shape[0], device)

        if self.pred_len:
            tgt_key_padding_mask = None
        else:
            tgt_key_padding_mask = mask

        # Decode with or without causal mask
        if tgt_mask is not None:
            pred = self.seqTransDecoder(
                tgt=pred,
                memory=emb,
                memory_key_padding_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )
        else:
            pred = self.seqTransDecoder(
                tgt=pred,
                memory=emb,
                memory_key_padding_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        pred = self.poseHead(pred, diffused)
        return pred, answers, diffuse_shapes, trajectory, prefix_tensors, lengths