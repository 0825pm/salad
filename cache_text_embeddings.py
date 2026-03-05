"""
cache_text_embeddings.py — Text embedding 사전 계산 및 개별 .pt 저장

생성 구조:
    How2Sign/{split}/text_emb/{encoder_name}/{sample_name}.pt
    CSL-Daily/text_emb/{encoder_name}/{sample_name}.pt
    Phoenix_2014T/text_emb/{encoder_name}/{sample_name}.pt
    + __empty__.pt  (CFG dropout용)

Usage:
    cd ~/Projects/research/salad

    # XLM-RoBERTa (기존)
    python cache_text_embeddings.py \
        --text_encoder xlm-roberta --xlmr_version xlm-roberta-base \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --sign_dataset how2sign_csl_phoenix \
        --splits train val test \
        --batch_size 128

    # mBART (SOKE pretrained)
    python cache_text_embeddings.py \
        --text_encoder mbart \
        --mbart_path deps/mbart-h2s-csl-phoenix \
        --data_root /home/user/Projects/research/SOKE/data/How2Sign \
        --csl_root /home/user/Projects/research/SOKE/data/CSL-Daily \
        --phoenix_root /home/user/Projects/research/SOKE/data/Phoenix_2014T \
        --sign_dataset how2sign_csl_phoenix \
        --splits train val test \
        --batch_size 64
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sign_dataset import _build_annotations
from data.load_sign_data import get_encoder_cache_name


def resolve_cache_dir(ann, split, data_root, csl_root, phoenix_root, encoder_name):
    """annotation 1개에 대한 cache 저장 디렉토리 반환"""
    src = ann['src']
    if src == 'how2sign':
        return os.path.join(data_root, split, 'text_emb', encoder_name)
    elif src == 'csl':
        return os.path.join(csl_root, 'text_emb', encoder_name)
    elif src == 'phoenix':
        return os.path.join(phoenix_root, 'text_emb', encoder_name)
    raise ValueError(f"Unknown src: {src}")


def main():
    parser = argparse.ArgumentParser(description='Precompute text embeddings for sign datasets')
    parser.add_argument('--text_encoder', default='xlm-roberta', choices=['clip', 'xlm-roberta', 'mbart'])
    parser.add_argument('--clip_version', default='ViT-B/32')
    parser.add_argument('--xlmr_version', default='xlm-roberta-base')
    parser.add_argument('--mbart_path', default='deps/mbart-h2s-csl-phoenix', help='mBART model path (SOKE)')
    parser.add_argument('--data_root', required=True, help='How2Sign root')
    parser.add_argument('--csl_root', default=None)
    parser.add_argument('--phoenix_root', default=None)
    parser.add_argument('--sign_dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoder_name = get_encoder_cache_name(args.text_encoder, args.clip_version, args.xlmr_version)

    print("=" * 60)
    print("Text Embedding Cache Generator")
    print(f"  encoder:      {args.text_encoder} ({encoder_name})")
    print(f"  device:       {device}")
    print(f"  splits:       {args.splits}")
    print("=" * 60)

    # ── Build text encoder ──
    print("\nLoading text encoder...")
    encoder_opt = Namespace(
        text_encoder=args.text_encoder,
        clip_version=args.clip_version,
        xlmr_version=args.xlmr_version,
        mbart_path=getattr(args, 'mbart_path', 'deps/mbart-h2s-csl-phoenix'),
        device=device,
    )
    from models.denoiser.clip import build_text_encoder
    encoder = build_text_encoder(encoder_opt)
    encoder.eval().to(device)
    print(f"  Loaded: {args.text_encoder}")

    # ── Cache empty string (for CFG dropout) ──
    print("\nCaching empty string embedding...")
    with torch.no_grad():
        empty_word, empty_mask, empty_pos = encoder.encode_text([""])
    empty_cache = {
        'word_emb': empty_word[0].cpu(),
        'attn_mask': empty_mask[0].cpu(),
        'token_pos': empty_pos[0].cpu(),
    }

    # Save __empty__.pt in each data root
    for root in [args.data_root, args.csl_root, args.phoenix_root]:
        if root is None:
            continue
        d = os.path.join(root, 'text_emb', encoder_name)
        os.makedirs(d, exist_ok=True)
        torch.save(empty_cache, os.path.join(d, '__empty__.pt'))
        print(f"  Saved: {d}/__empty__.pt")

    # ── Process each split ──
    total_cached = 0
    total_skipped = 0

    for split in args.splits:
        print(f"\n{'='*40} {split} {'='*40}")
        all_data = _build_annotations(
            split=split,
            dataset_name=args.sign_dataset,
            data_root=args.data_root,
            csl_root=args.csl_root,
            phoenix_root=args.phoenix_root,
        )

        # Create output dirs
        seen_dirs = set()
        for ann in all_data:
            d = resolve_cache_dir(ann, split, args.data_root, args.csl_root, args.phoenix_root, encoder_name)
            if d not in seen_dirs:
                os.makedirs(d, exist_ok=True)
                seen_dirs.add(d)

        # Batch encode
        for start in tqdm(range(0, len(all_data), args.batch_size), desc=f'Encoding {split}'):
            batch_anns = all_data[start:start + args.batch_size]
            texts = [ann.get('text', '') or '' for ann in batch_anns]

            # Skip if all already cached
            paths = []
            all_exist = True
            for ann in batch_anns:
                d = resolve_cache_dir(ann, split, args.data_root, args.csl_root, args.phoenix_root, encoder_name)
                p = os.path.join(d, f"{ann['name']}.pt")
                paths.append(p)
                if not os.path.exists(p):
                    all_exist = False

            if all_exist:
                total_skipped += len(batch_anns)
                continue

            # Encode batch
            with torch.no_grad():
                word_embs, attn_masks, token_poses = encoder.encode_text(texts)

            # Save per-sample
            for j, (ann, path) in enumerate(zip(batch_anns, paths)):
                if os.path.exists(path):
                    total_skipped += 1
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({
                    'word_emb': word_embs[j].cpu(),
                    'attn_mask': attn_masks[j].cpu(),
                    'token_pos': token_poses[j].cpu(),
                }, path)
                total_cached += 1

        print(f"  [{split}] cached: {total_cached}, skipped (exists): {total_skipped}")

    print(f"\n{'='*60}")
    print(f"Done. Total cached: {total_cached}, skipped: {total_skipped}")
    print(f"Encoder: {encoder_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()