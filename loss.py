from torch_motor_utils import *
from data_loader import KEY_MAP, device
from typing import Any, Dict
from diffusion import mv_to_flat_normalized

def accumulations(batch: Dict[str, Any]) -> Dict[str, Any]:
    # Implement the forward pass logic here
    keys = list(batch.keys())
    for k in keys:
        if 'lengths' in k:
            continue
        if 'rotor' in k:
            batch[k.replace('log_', '')] = rot_exp(batch[k])
        elif 'translator' in k:
            batch[k.replace('log_', '')] = null_exp(batch[k])
        else:
            continue
    # root motor
    acceleration = batch['acceleration_root_translator'] * batch['acceleration_root_rotor']
    initial = batch['velocity_root_translator_0'] * batch['velocity_root_rotor_0']
    velocity = cumprod(initial, acceleration)
    # same initial because started in body frame--no translation and rotation in the first frame
    batch['root_motor_1'] = initial
    batch['root_motor'] = cumprod(batch['root_motor_1'], velocity)
    batch['velocity_root'] = velocity
    # batch['floor_1'] = ~batch['root_motor_1'] >> batch['floor_0']
    # batch['floor'] = ~batch['root_motor']>> batch['floor_0']

    # body rotors
    acceleration = batch['acceleration_body_rotors']
    initial = batch['velocity_body_rotors_0']
    velocity = cumprod(initial, acceleration)
    batch['body_rotors_1'] = batch['body_rotors_0'] * initial
    batch['body_rotors'] = cumprod(batch['body_rotors_1'], velocity)
    batch['velocity_body_rotors'] = velocity
    batch['chained_body_rotors'] = chain_accumulate(batch['body_rotors'])
    batch['chained_body_rotors_1'] = chain_accumulate(batch['body_rotors_1'])
    batch['chained_body_rotors_0'] = chain_accumulate(batch['body_rotors_0'])        
    return batch


def compute_loss(output, answers, diffuse_shapes, hierarchy_scale: float = 1000.0, use_ema=False):
    # output = flattened_to_mv_dict(output, DIFFUSE_KEYS, diffuse_shapes)
    output = accumulations(output)
    ans_root_motor = alg.multivector(keys=KEY_MAP['root_motor'], values=answers.pop('root_motor')[:, 2:].to(device))
    output['log_root_rotor'], output["log_root_translator"] = motor_origin_split_log(output.pop('root_motor') * ~ans_root_motor)
    del ans_root_motor
    ans_vel_root_motor = alg.multivector(keys=KEY_MAP['root_motor'], values=answers.pop('velocity_root_motor')[:, 1:].to(device))
    output['log_velocity_root_rotor'], output["log_velocity_root_translator"] = motor_origin_split_log(output.pop('velocity_root') * ~ans_vel_root_motor)
    del ans_vel_root_motor
    ans_acc_root_motor = alg.multivector(keys=KEY_MAP['root_motor'], values=answers.pop('acceleration_root_motor')[:, 0:].to(device))
    output['log_acceleration_root_rotor'], output["log_acceleration_root_translator"] = motor_origin_split_log(output.pop('acceleration_root_translator') * output.pop('acceleration_root_rotor') * ~ans_acc_root_motor)
    del ans_acc_root_motor
    ans_body_rotors = alg.multivector(keys=KEY_MAP['body_rotors'], values=answers.pop('body_rotors')[:, 2:].to(device))
    output['log_body_rotors'] = rotor_log(output['body_rotors'] * ~ans_body_rotors)
    del ans_body_rotors
    ans_vel_body_rotors = alg.multivector(keys=KEY_MAP['body_rotors'], values=answers.pop('velocity_body_rotors')[:, 1:].to(device))
    output['log_velocity_body_rotors'] = rotor_log(output['velocity_body_rotors'] * ~ans_vel_body_rotors)
    del ans_vel_body_rotors
    ans_acc_body_rotors = alg.multivector(keys=KEY_MAP['body_rotors'], values=answers.pop('acceleration_body_rotors')[:, 0:].to(device))
    output['log_acceleration_body_rotors'] = rotor_log(output['acceleration_body_rotors'] * ~ans_acc_body_rotors)
    del ans_acc_body_rotors
    ans_chained_body_rotors = alg.multivector(keys=KEY_MAP['chained_body_rotors'], values=answers.pop('chained_body_rotors')[:, 2:].to(device))
    b1, b0 = (motor_origin_split_log(output['chained_body_rotors'] * ~ans_chained_body_rotors))
    output["log_chained_body_rotors"] = b1 + b0
    del ans_chained_body_rotors
    del answers

    # Categorize keys
    chained_pos_keys = [
        'log_chained_body_rotors',
    ]
    pos_keys = [
        'log_root_rotor', 'log_root_translator', 'log_body_rotors',
    ]
    vel_keys = [
        'log_velocity_root_rotor', 'log_velocity_root_translator', 'log_velocity_body_rotors'
    ]
    acc_keys = [
        'log_acceleration_root_rotor', 'log_acceleration_root_translator', 
        'log_acceleration_body_rotors'
    ]
    # (Optional) you can include 'log_chained_body_rotors' under pos if needed.
    
    LOSS_KEYS = acc_keys + vel_keys #+ pos_keys + chained_pos_keys

    # Flatten & normalize
    flat_losses = {}
    for k in LOSS_KEYS:
        flat_losses[k] = mv_to_flat_normalized(output[k], k)  # [T,B,F_k]

    per_key_loss = {}
    for k, v in flat_losses.items():
        l2_t_f = (v ** 2).mean(dim=1)          # [F_k] (avg over batch)
        per_key_loss[k] = (l2_t_f).sum(dim=0)  # [F_k]

    # Concatenate all feature losses
    all_loss_vec = torch.cat([per_key_loss[k] for k in LOSS_KEYS], dim=0)  # [ΣF]
    all_loss_vec = torch.clamp(all_loss_vec, 0.0, 1e6)

    # Build category weights vector aligned with concatenated loss features
    cat_weights = []
    acc_w = hierarchy_scale ** 5
    vel_w = hierarchy_scale ** 4
    pos_w = hierarchy_scale ** 1
    chained_w = 1.0
    for k in LOSS_KEYS:
        w = chained_w if k in chained_pos_keys else pos_w if k in pos_keys else vel_w if k in vel_keys else acc_w
        cat_weights.append(torch.full((per_key_loss[k].shape[0],), w, dtype=all_loss_vec.dtype, device=all_loss_vec.device))
    cat_weights = torch.cat(cat_weights, dim=0)  # [ΣF]

    # Optional time-propagation weighting (earlier timesteps more impact):
    # Not applied since we already averaged over time; uncomment if needed:
    # T_weight = ...

    # EMA normalization (per feature)
    if use_ema:
        eps = 1e-8
        decay = 0.5
        ema_store = globals().setdefault("LOSS_EMA", [0.0] * all_loss_vec.shape[0])
        inv_mags = []
        for i, v in enumerate(all_loss_vec):
            val = float(v.detach().cpu().item())
            if ema_store[i] <= 0.0:
                ema_store[i] = val
            else:
                ema_store[i] = decay * ema_store[i] + (1.0 - decay) * val
            inv_mags.append(1.0 / (ema_store[i] + eps))
        inv_mags = torch.tensor(inv_mags, dtype=all_loss_vec.dtype, device=all_loss_vec.device)

        # Combine hierarchical category weight with inverse-magnitude reweighting
        combined_w = cat_weights * inv_mags
        combined_w = combined_w * (combined_w.numel() / (combined_w.sum() + eps))
        combined_w = torch.clamp(combined_w, min=1e-6, max=10.0)
    else:
        combined_w = cat_weights
        combined_w = combined_w * (combined_w.numel() / (combined_w.sum() + 1e-8))
        combined_w = torch.clamp(combined_w, min=1e-6, max=10.0)

    return torch.sum(combined_w * all_loss_vec)