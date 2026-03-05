"""
Text encoders for SALAD denoiser.
  - FrozenCLIPTextEncoder     (English, ViT-B/32 or ViT-L/14)
  - XLMRTextEncoder           (Multilingual, xlm-roberta — supports frozen / full / LoRA)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# LoRA Implementation (no external dependency)
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""
    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Init: A ~ N(0, 1/r), B = 0 → initial LoRA output is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weight
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # W*x + (B @ A)*x * scaling
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model, rank=16, alpha=32.0, target_modules=None):
    """
    Replace target linear layers in model with LoRA-wrapped versions.
    Default targets: query & value projections in attention layers.
    Returns list of (name, LoRALinear) for tracking.
    """
    if target_modules is None:
        target_modules = ['query', 'value']  # XLM-R attention layer names

    lora_layers = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Navigate to parent and replace
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1]

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
                lora_layers.append((name, lora_layer))

    return lora_layers


def _apply_lora_to_xlmr(xlmr_model, rank=16, alpha=32.0):
    """Apply LoRA specifically to XLM-R self-attention Q/V projections."""
    lora_layers = []

    for layer_idx, layer in enumerate(xlmr_model.encoder.layer):
        attn = layer.attention.self

        # Q projection
        lora_q = LoRALinear(attn.query, rank=rank, alpha=alpha)
        attn.query = lora_q
        lora_layers.append((f'encoder.layer.{layer_idx}.attention.self.query', lora_q))

        # V projection
        lora_v = LoRALinear(attn.value, rank=rank, alpha=alpha)
        attn.value = lora_v
        lora_layers.append((f'encoder.layer.{layer_idx}.attention.self.value', lora_v))

    return lora_layers


# =============================================================================
# Utility
# =============================================================================

def move_cache():
    """Move HuggingFace cache to project-local dir."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir


# =============================================================================
# CLIP text encoder (English only — always frozen)
# =============================================================================

class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        move_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.opt = opt
        if opt.clip_version == "ViT-B/32":
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        elif opt.clip_version == "ViT-L/14":
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        else:
            raise ValueError(f"Invalid CLIP version: {opt.clip_version}")

        self.max_length = self.tokenizer.model_max_length
        self.freeze()
        print(f"Loaded CLIP text encoder version {opt.clip_version}")

    def freeze(self):
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device).bool()
        if text_input_ids.shape[-1] > self.max_length:
            text_input_ids = text_input_ids[:, :self.max_length]

        word_emb = self.model.text_model(text_input_ids).last_hidden_state

        return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)

    @torch.no_grad()
    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens)


# =============================================================================
# XLM-RoBERTa text encoder (multilingual — frozen / full / LoRA)
# =============================================================================

XLMR_MODELS = {
    "xlm-roberta-base":  {"hf_name": "xlm-roberta-base",  "dim": 768},
    "xlm-roberta-large": {"hf_name": "xlm-roberta-large", "dim": 1024},
}


class XLMRTextEncoder(nn.Module):
    """
    XLM-RoBERTa text encoder for multilingual sign language generation.

    finetune_mode:
      'none'  — fully frozen (default, backward-compatible)
      'full'  — all params trainable
      'lora'  — LoRA on Q/V projections only (~2M trainable params)

    Returns:
        word_emb:       [B, T, D]  token-level embeddings
        text_attn_mask: [B, T]     bool attention mask (True = valid)
        token_pos:      [B]        position of </s> token
    """
    def __init__(self, opt):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        xlmr_version = getattr(opt, 'xlmr_version', 'xlm-roberta-base')
        if xlmr_version not in XLMR_MODELS:
            raise ValueError(f"Invalid XLM-R version: {xlmr_version}. "
                             f"Choose from {list(XLMR_MODELS.keys())}")

        info = XLMR_MODELS[xlmr_version]
        self.tokenizer = AutoTokenizer.from_pretrained(info["hf_name"])
        self.model = AutoModel.from_pretrained(info["hf_name"])
        self.hidden_dim = info["dim"]
        self.max_length = min(self.tokenizer.model_max_length, 128)

        # Fine-tune mode
        self.finetune_mode = getattr(opt, 'finetune_text_encoder', 'none')
        self.lora_layers = []

        if self.finetune_mode == 'none':
            self._freeze_all()
        elif self.finetune_mode == 'full':
            self._unfreeze_all()
        elif self.finetune_mode == 'lora':
            self._freeze_all()
            lora_rank = getattr(opt, 'lora_rank', 16)
            lora_alpha = getattr(opt, 'lora_alpha', 32.0)
            self.lora_layers = _apply_lora_to_xlmr(self.model, rank=lora_rank, alpha=lora_alpha)
            n_lora_params = sum(
                p.numel() for _, layer in self.lora_layers
                for p in [layer.lora_A, layer.lora_B]
            )
            print(f"  LoRA applied: rank={lora_rank}, alpha={lora_alpha}, "
                  f"trainable params={n_lora_params/1e6:.2f}M")
        else:
            raise ValueError(f"Unknown finetune_text_encoder: {self.finetune_mode}")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"Loaded XLM-RoBERTa: {xlmr_version} (dim={self.hidden_dim}, "
              f"mode={self.finetune_mode}, trainable={n_train/1e6:.2f}M/{n_total/1e6:.1f}M)")

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        """Return only trainable parameters (for optimizer param groups)."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """Return state_dict with only trainable parameters (for checkpoint saving)."""
        if self.finetune_mode == 'none':
            return {}
        elif self.finetune_mode == 'lora':
            # Save only LoRA A/B matrices
            state = {}
            for name, layer in self.lora_layers:
                state[f'{name}.lora_A'] = layer.lora_A.data
                state[f'{name}.lora_B'] = layer.lora_B.data
            return state
        else:  # full
            return self.state_dict()

    def load_trainable_state_dict(self, state_dict):
        """Load trainable parameters from checkpoint."""
        if self.finetune_mode == 'none' or not state_dict:
            return
        elif self.finetune_mode == 'lora':
            for name, layer in self.lora_layers:
                a_key = f'{name}.lora_A'
                b_key = f'{name}.lora_B'
                if a_key in state_dict:
                    layer.lora_A.data.copy_(state_dict[a_key])
                if b_key in state_dict:
                    layer.lora_B.data.copy_(state_dict[b_key])
            print(f"  Loaded LoRA weights ({len(state_dict)} tensors)")
        else:  # full
            self.load_state_dict(state_dict)
            print(f"  Loaded full text encoder weights")

    @property
    def is_trainable(self):
        return self.finetune_mode != 'none'

    def encode_text(self, text):
        """
        Encode text to embeddings.
        NOTE: No @torch.no_grad() — gradient flows when fine-tuning.
        """
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        input_ids = tokens.input_ids.to(self.model.device)
        attn_mask = tokens.attention_mask.to(self.model.device).bool()

        word_emb = self.model(input_ids=input_ids,
                              attention_mask=attn_mask).last_hidden_state  # [B, T, D]

        eos_id = self.tokenizer.eos_token_id or 2
        token_pos = (input_ids == eos_id).float().argmax(dim=-1)  # [B]

        return word_emb, attn_mask, token_pos

    @torch.no_grad()
    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# Backward compatibility alias
FrozenXLMRTextEncoder = XLMRTextEncoder


# =============================================================================
# Factory
# =============================================================================

def build_text_encoder(opt):
    """Build text encoder based on opt.text_encoder."""
    text_enc = getattr(opt, 'text_encoder', 'clip')

    if text_enc == 'clip':
        return FrozenCLIPTextEncoder(opt)
    elif text_enc == 'xlm-roberta':
        return XLMRTextEncoder(opt)
    else:
        raise ValueError(f"Unknown text_encoder: {text_enc}. "
                         f"Choose 'clip' or 'xlm-roberta'.")


def get_text_encoder_dim(opt):
    """Return hidden dim of the chosen text encoder."""
    text_enc = getattr(opt, 'text_encoder', 'clip')

    if text_enc == 'clip':
        return 512 if opt.clip_version == "ViT-B/32" else 768
    elif text_enc == 'xlm-roberta':
        xlmr_version = getattr(opt, 'xlmr_version', 'xlm-roberta-base')
        return XLMR_MODELS[xlmr_version]["dim"]
    else:
        raise ValueError(f"Unknown text_encoder: {text_enc}")