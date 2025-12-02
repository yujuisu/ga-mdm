import clip
import torch.nn as nn
import os
import torch
from data_loader import device

def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    return clip_model


class BERT(nn.Module):
    def __init__(self, modelpath: str):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath).to(device)  # ensure model is on GPU/desired device

    def forward(self, texts):
        # Ensure a list of strings
        if isinstance(texts, str):
            texts = [texts]

        max_len = getattr(self.tokenizer, "model_max_length", 512)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids_list = []
        attention_mask_list = []

        for t in texts:
            tokens = self.tokenizer.tokenize(t)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            ids = self.tokenizer.build_inputs_with_special_tokens(ids)

            if len(ids) > max_len:
                sep_id = getattr(self.tokenizer, "sep_token_id", None)
                cls_id = getattr(self.tokenizer, "cls_token_id", None)
                if sep_id is not None and cls_id is not None and len(ids) >= 2:
                    ids = [cls_id] + ids[1:max_len-1] + [sep_id]
                else:
                    ids = ids[:max_len]

            attn = [1] * len(ids)
            if len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [pad_id] * pad_len
                attn = attn + [0] * pad_len

            # create tensors directly on target device
            input_ids_list.append(torch.tensor(ids, dtype=torch.long, device=device))
            attention_mask_list.append(torch.tensor(attn, dtype=torch.long, device=device))

        input_ids = torch.stack(input_ids_list, dim=0)       # [B, S]
        attention_mask = torch.stack(attention_mask_list, dim=0)  # [B, S]

        encoded_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            outputs = self.text_model(**encoded_inputs)
        # Transformers 2.x returns tuple; 4.x returns ModelOutput
        if hasattr(outputs, "last_hidden_state"):
            enc = outputs.last_hidden_state
        else:
            enc = outputs[0]  # [B, S, H]
        return enc, attention_mask


def load_bert(model_path):
    bert = BERT(model_path)
    bert.eval()
    bert.text_model.training = False
    bert.to(device)  # ensure module buffers are on device
    for p in bert.parameters():
        p.requires_grad = False
    return bert


# clip_model = load_and_freeze_clip('ViT-B/32')
  # Sorry for that, the naming is for backward compatibility
print("Loading BERT...")
# bert_model_path = 'model/BERT/distilbert-base-uncased'
bert_model_path = 'distilbert-base-uncased'
bert_model = load_bert(bert_model_path)


def bert_encode_text(raw_text, device):
    """Encode text prompts using the DistilBERT backbone associated with ``model``."""
    with torch.no_grad():
        enc_text, attention_mask = bert_model(raw_text)
    enc_text = enc_text.permute(1, 0, 2)  # [S, B, H]
    pad_mask = (attention_mask == 0)      # [B, S], True = pad
    return enc_text.to(device), pad_mask.to(device)


def clip_encode_text(raw_text):
    """Encode text prompts using the CLIP backbone associated with ``model``."""
    max_text_len = 20
    # if max_text_len is not None:
    default_context_length = 77
    context_length = max_text_len + 2  # start_token + 20 + end_token
    assert context_length < default_context_length
    texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
    zero_pad = torch.zeros(
        [texts.shape[0], default_context_length - context_length],
        dtype=texts.dtype,
        device=texts.device,
    )
    texts = torch.cat([texts, zero_pad], dim=1)
    # else:
    #     texts = clip.tokenize(raw_text, truncate=True).to(device)
    encoded = clip_model.encode_text(texts).float().unsqueeze(0)
    return encoded
