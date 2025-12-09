import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from kingdon import Algebra
import numpy as np
import json
from amass_motor_utils import JOINT_TRANSLATORS
import os
import codecs as cs
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alg = Algebra(3, 0, 1)
locals().update(alg.blades)

STATS_MAP = json.load(open("serialized/stats.json", 'r'))
KEY_MAP = json.load(open('serialized/key_map.json', 'r'))
DATA_KEY_LISTS = json.load(open('serialized/data_key_list.json', 'r'))
UNIT_LENGTH = 4
for k in KEY_MAP.keys():
    KEY_MAP[k] = tuple(KEY_MAP[k])

JOINT_TRANSLATORS = torch.tensor(np.expand_dims(np.stack(JOINT_TRANSLATORS.values()),-1), dtype=torch.float32, device=device)
JOINT_TRANSLATORS = alg.multivector(keys=KEY_MAP['root_translator'], values=JOINT_TRANSLATORS)

K_CHAIN_PE = json.load(open('serialized/k_chain_pe.json', 'r'))
K_CHAIN_PE = torch.tensor(K_CHAIN_PE, dtype=torch.float32, device=device)
PRED_LENGTH = 20
FULL_LENGTH_KEY = "chained_body_rotors"

# Keep it on cpu for data loading efficiency
def np_map_to_torch(data, keys):
    return {k: torch.tensor(data[k], dtype=torch.float32) for k in keys}


def kingdon_to_full(kingdon_mv):
    full_tensor = []
    shape = kingdon_mv.shape[1:]
    for v in kingdon_mv.asfullmv().values():
        if isinstance(v, int):
            full_tensor.append(torch.zeros(*shape, dtype=torch.float32, device=device))
        else:
            full_tensor.append(v)
    return torch.stack(full_tensor, dim=-1)


# class NPZDictDataset(Dataset):
#     def __init__(self, data_dir, keys=None, transform=None):
#         self.files = sorted(Path(data_dir).glob("*.npz"))
#         self.keys = keys  # list of keys to keep, or None for all
#         # self.transform = transform

#     def __len__(self):
#         return len(self.files)


#     def __getitem__(self, idx):
#         path = self.files[idx]
#         data = np.load(path)

#         # Keep only selected keys (if specified)
        
#         return np_map_to_torch(data, self.keys)

        # if self.transform:
        #     data_dict = self.transform(data_dict)

        # return data_dict


# def dict_array_collate_fn(batch):
#     """
#     Pads and batches a list of dictionaries containing PyTorch tensors.
    
#     Args:
#         batch (list): A list of samples, where each sample is a dictionary 
#                       with tensor values.
    
#     Returns:
#         dict: A single dictionary with batched/padded tensors.
#     """
    
#     # Check if the batch is empty
#     if not batch:
#         return {}
    
#     batch = batch

#     # Get all keys from the first dictionary (assuming all samples have the same keys)
#     keys = batch[0].keys()
    
#     # Initialize the container for the batched result
#     batched_data = {"lengths": {}}
    
#     for key in keys:
#         # 1. Extract all values for the current key across all samples
#         data_list = [item[key] for item in batch]
        
#         # 2. Check the data type for the current key
#         if isinstance(data_list[0], torch.Tensor):
#             # Pad the sequences to the length of the longest one in this batch.
#             # batch_first=True makes the result shape (batch_size, max_length, ...)
#             permuted_tensors = []
#             lengths = []
#             for t in data_list:
#                 # Dynamic permutation for any rank > 1:
#                 dims = list(range(t.dim()))
#                 # Move dimension 1 to the front
#                 permute_order = [dims[1]] + dims[:1] + dims[2:] 
#                 permuted_tensors.append(t.permute(*permute_order))
#                 lengths.append(t.shape[1])
#             try:
#                 padded_batch_permuted = pad_sequence(permuted_tensors, batch_first=False, padding_value=0.0)
#                                 # We move the second dimension (Batch_Size) to the front (index 0).
#                 # The second dimension (index 1) becomes the padded length.
#                 target_permute_order = [2, 0, 3, 1]
#                 # batched_data[key] = padded_batch_permuted
#                 final_padded_batch = padded_batch_permuted.permute(*target_permute_order)
                
#                 # FINAL SHAPE: [Batch_Size, Max_L, 3, 1]
#                 batched_data[key] = final_padded_batch
#                 # Create mask
#                 if '_0' not in key:
#                     # max_len = padded_batch_permuted.shape[0]
#                     # mask_tuple = lengths, max_len
#                     # mask = torch.zeros(, dtype=torch.bool)
#                     # for idx, seq_len in enumerate(lengths):
#                     #     mask[idx, :seq_len] = True
#                     batched_data["lengths"][key] = lengths
#             except RuntimeError as e:
#                 # Catching issues if the Tensors don't have the same rank (e.g., mixing 1D and 2D)
#                 print(f"Error padding key '{key}': {e}")
#                 # Fallback to simple stacking for non-sequence data (like labels)
#                 batched_data[key] = torch.stack(data_list)
                
#         elif isinstance(data_list[0], (int, float, bool, torch.LongTensor, torch.FloatTensor)):
#             # --- SIMPLE STACKING (for scalar/label data) ---
#             # Combine scalars or simple tensors (like labels) into a single tensor
#             batched_data[key] = torch.as_tensor(data_list)
#         else:
#             # --- REMAINDER (e.g., strings, metadata) ---
#             # For data types that cannot be batched, just return the list
#             batched_data[key] = data_list
            
#     return batched_data


def construct_text_metadata(text_dir):
    """Parse motion text files in a directory into structured metadata dictionaries.

    Parameters
    ----------
    text_dir : str
        Directory containing motion text annotation files (`*.txt`).
        If a single file path is provided, only that file is parsed.
    w_vectorizer : Optional[word vectorizer]
        Vectorizer used to obtain word embeddings and POS one-hot encodings.

    Returns
    -------
    tuple(dict, dict)
        A pair `(entries_map, metadata_map)` where each map is keyed by the
        motion name (file stem) and contains the parsed line entries and the
        aggregated metadata respectively.
    """

    entries_map = {}

    if text_dir is None:
        return entries_map

    candidate_files = []
    if os.path.isdir(text_dir):
        candidate_files = [
            os.path.join(text_dir, fname)
            for fname in sorted(os.listdir(text_dir))
            if fname.endswith('.txt')
        ]
    elif text_dir.endswith('.txt') and os.path.exists(text_dir):
        candidate_files = [text_dir]

    for file_path in candidate_files:
        if not os.path.exists(file_path):
            continue

        motion_name = os.path.splitext(os.path.basename(file_path))[0]
        text_entries = []

        with cs.open(file_path) as f_text:
            for raw_line in f_text.readlines():
                line_split = raw_line.strip().split('#')
                if len(line_split) < 4:
                    continue

                caption = line_split[0]
                # tokens = line_split[1].split(' ')
                try:
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                except ValueError:
                    continue

                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_entries.append({
                    'caption': caption,
                    'f_tag': f_tag,
                    'to_tag': to_tag,
                })

        entries_map[motion_name] = text_entries

    return entries_map

TEXT_LIST_MAP = construct_text_metadata('texts')

class NPZDictDataset(Dataset):
    def __init__(self, data_dir, keys=None, transform=None, context_len=0, pred_len=0):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.keys = keys  # list of keys to keep, or None for all
        # self.transform = transform
        self.context_len = context_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, id):
        path = self.files[id]
        name = path.stem
        try:
            data = np.load(path)
            text = random.choice(TEXT_LIST_MAP[name])
            start_idx = int(text.get('f_tag', 0) * 20)
            end_idx = int(text.get('to_tag', 0) * 20)
            data_ = {}
            for k in self.keys:
                if start_idx != end_idx:
                    k_ = k.replace('_0', '')
                    motion = data[k_][:, start_idx:end_idx, :]
                else:
                    k_ = k.replace('_0', '')
                    motion = data[k_]
                data_[k] = motion

            #--random crop to pred_len
            max_total_len = self.pred_len + self.context_len
            if self.pred_len:
                m_length = data_[FULL_LENGTH_KEY].shape[1]
                assert m_length >= self.pred_len + 2, f"Motion length {m_length} is less than pred_len + 2 {self.pred_len + 2}"
                max_start = m_length - self.pred_len - 2
                random_start_idx = random.randint(0, max(0, max_start))
                # +2 because we want to predict acceleration
                random_end_idx = random.randint(random_start_idx + 2 + self.pred_len, random_start_idx + self.pred_len + max_total_len)
                
                for k in self.keys:
                    offset = 0
                    if 'velocity' in k:
                        offset = 1
                    elif 'acceleration' in k:
                        offset = 2
                    if '_0' in k:
                        data_[k] = data_[k][:, random_start_idx:random_start_idx + 1, :]
                    else:
                        data_[k] = data_[k][:, random_start_idx:random_end_idx - offset, :]
                        assert data_[k].shape[1] > 0, f"After cropping, motion length for key {k} is zero."
            #--random crop to pred_len
            
            data = np_map_to_torch(data_, self.keys)
            data['text_caption'] = text['caption']
            return data
        except Exception as e:
            print(f"Error loading {path}: {e}")


# Keys that must be identity-padded (exp-space MVs)
MOTOR_KEYS = [
    "root_motor",
    "velocity_root_motor",
    "acceleration_root_motor",
]
ROTOR_KEYS = [
    "body_rotors",
    "velocity_body_rotors",
    "acceleration_body_rotors",
    "chained_body_rotors",
    # Optional intermediates often present in answers/batch
    "body_rotors_0", "body_rotors_1",
    "chained_body_rotors_0", "chained_body_rotors_1",
]
IDENTITY_PAD_KEYS = set(MOTOR_KEYS + ROTOR_KEYS)

def _identity_pad_mv_inplace(t: torch.Tensor, lengths, key: str):
    """
    In-place set padded frames to identity for MV tensors shaped [K, T, C, B].
    Identity: scalar coeff (index 0) = 1, others = 0 for t >= length[b].
    """
    if not isinstance(t, torch.Tensor) or t.dim() != 4:
        return  # expect [K, T, C, B]
    K, T, C, B = t.shape
    if K < 1:
        return
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    for b in range(B):
        L = int(lengths[b]) if lengths is not None else T
        if L >= T:
            continue
        t[:, L:, :, b] = 0.0
        t[0, L:, :, b] = 1.0

def _apply_identity_padding(batch: dict):
    """
    Replace zero-padded frames with identity for tor keys (motors/rotors/translators),
    i.e., any key containing 'tor' and not containing 'log'.
    Uses batch['lengths'] if available, falling back to the common chained_body_rotors length.
    """
    lengths_dict = batch.get("lengths", {}) or {}
    fallback_len = (
        lengths_dict.get("chained_body_rotors")
        or lengths_dict.get("body_rotors")
        or lengths_dict.get("root_motor")
    )
    for key, val in list(batch.items()):
        if not isinstance(val, torch.Tensor):
            continue
        if val.dim() != 4:
            continue  # expect [K, T, C, B]
        if ("tor" in key) and ("log" not in key):
            lens = lengths_dict.get(key, fallback_len)
            _identity_pad_mv_inplace(val, lens, key)

def dict_array_collate_fn(batch, pred_len=0):
    """
    Pads and batches a list of dictionaries containing PyTorch tensors.
    
    Args:
        batch (list): A list of samples, where each sample is a dictionary 
                      with tensor values.
    
    Returns:
        dict: A single dictionary with batched/padded tensors.
    """
    # Filter out None samples to avoid DataLoader errors
    valid_batch = [item for item in batch if item is not None]
    dropped = len(batch) - len(valid_batch)
    if dropped > 0:
        print(f"dict_array_collate_fn: dropped {dropped} invalid sample(s) (None).")
    batch = valid_batch
    # Check if the batch is empty
        # Check if the batch is empty after filtering
    if not batch:
        print("dict_array_collate_fn: empty batch after filtering invalid samples.")
        return {}

    # Get all keys from the first dictionary (assuming all samples have the same keys)
    keys = batch[0].keys()
    
    # Initialize the container for the batched result
    batched_data = {"lengths": {}, 'batch_size': len(batch)}
    pred_data = {} if pred_len > 0 else None
    
    for key in keys:
        # 1. Extract all values for the current key across all samples
        data_list = [item[key] for item in batch]
        # 2. Check the data type for the current key
        if isinstance(data_list[0], torch.Tensor):
            # Pad the sequences to the length of the longest one in this batch.
            # batch_first=True makes the result shape (batch_size, max_length, ...)
            permuted_tensors = []
            pred_tensors = []
            lengths = []
            for t in data_list:
                # Dynamic permutation for any rank > 1:
                if '_0' not in key:
                    if pred_len > 0:
                        t, pred = t[:, :-pred_len, :], t[:, -pred_len:, :]
                        pred_tensors.append(pred)
                dims = list(range(t.dim()))
                # Move dimension 1 to the front
                permute_order = [dims[1]] + dims[:1] + dims[2:] 
                permuted_tensors.append(t.permute(*permute_order))
                lengths.append(t.shape[1])
            try:
                padded_batch_permuted = pad_sequence(permuted_tensors, batch_first=False, padding_value=0.0)
                                # We move the second dimension (Batch_Size) to the front (index 0).
                # The second dimension (index 1) becomes the padded length.
                target_permute_order = [2, 0, 3, 1]
                # batched_data[key] = padded_batch_permuted
                final_padded_batch = padded_batch_permuted.permute(*target_permute_order)
                batched_data[key] = final_padded_batch
                # Create mask
                if '_0' not in key:
                    batched_data["lengths"][key] = lengths
            except RuntimeError as e:
                # Catching issues if the Tensors don't have the same rank (e.g., mixing 1D and 2D)
                print(f"Error padding key '{key}': {e}")
                # Fallback to simple stacking for non-sequence data (like labels)
                batched_data[key] = torch.stack(data_list)

            if pred_len > 0 and '_0' not in key:
                try:
                    pred_data[key] = torch.stack(pred_tensors).permute(1,2,3,0)
                except RuntimeError as e:
            
                    print('stack', [(key, s.shape, t.shape) for s, t in zip(data_list, pred_tensors)])
                    raise e
                
        elif isinstance(data_list[0], (int, float, bool, torch.LongTensor, torch.FloatTensor)):
            # --- SIMPLE STACKING (for scalar/label data) ---
            # Combine scalars or simple tensors (like labels) into a single tensor
            batched_data[key] = torch.as_tensor(data_list)
        else:
            # --- REMAINDER (e.g., strings, metadata) ---
            # For data types that cannot be batched, just return the list
            batched_data[key] = data_list

            
    # Ensure the expected [K, T, C, B] layout for MV arrays before identity padding.
    # Apply identity padding for motors/rotors.
    _apply_identity_padding(batched_data)
    
    return batched_data, pred_data