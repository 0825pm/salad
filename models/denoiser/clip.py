import os
import torch
import torch.nn as nn
from os.path import join as pjoin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import move_cache


# =============================================================================
# Original CLIP text encoder (English only)
# =============================================================================

class FrozenCLIPTextEncoder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
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
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        return tokens

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens)


# =============================================================================
# XLM-RoBERTa text encoder (multilingual: EN/ZH/DE + 100 languages)
# =============================================================================

XLMR_MODELS = {
    "xlm-roberta-base":  {"hf_name": "xlm-roberta-base",  "dim": 768},
    "xlm-roberta-large": {"hf_name": "xlm-roberta-large", "dim": 1024},
}


class FrozenXLMRTextEncoder(nn.Module):
    """
    XLM-RoBERTa text encoder for multilingual sign language generation.
    Drop-in replacement for FrozenCLIPTextEncoder — same encode_text() interface.

    Returns:
        word_emb:      [B, T, D]  token-level embeddings
        text_attn_mask: [B, T]    bool attention mask (True = valid)
        token_pos:      [B]       position of </s> token (analogous to CLIP EOS)
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

        # XLM-R max_length = 512, use a reasonable cap
        self.max_length = min(self.tokenizer.model_max_length, 128)

        self.freeze()
        print(f"Loaded XLM-RoBERTa text encoder: {xlmr_version} (dim={self.hidden_dim})")

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
        input_ids = tokens.input_ids.to(self.model.device)
        attn_mask = tokens.attention_mask.to(self.model.device).bool()

        word_emb = self.model(input_ids=input_ids,
                              attention_mask=attn_mask).last_hidden_state  # [B, T, D]

        # Find </s> token position (EOS, id=2 in XLM-R) — analogous to CLIP EOS
        eos_id = self.tokenizer.eos_token_id or 2
        # For each sample, find the first occurrence of EOS
        token_pos = (input_ids == eos_id).float().argmax(dim=-1)  # [B]

        return word_emb, attn_mask, token_pos

    @torch.no_grad()
    def tokenize(self, text):
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        return tokens

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# =============================================================================
# Factory
# =============================================================================

def build_text_encoder(opt):
    """Build text encoder based on opt.text_encoder."""
    text_enc = getattr(opt, 'text_encoder', 'clip')

    if text_enc == 'clip':
        return FrozenCLIPTextEncoder(opt)
    elif text_enc == 'xlm-roberta':
        return FrozenXLMRTextEncoder(opt)
    else:
        raise ValueError(f"Unknown text_encoder: {text_enc}. Choose 'clip' or 'xlm-roberta'.")


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