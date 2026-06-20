"""
Load and clean human-written paragraphs from Wikipedia.

Samples paragraphs from the Wikipedia 20231101.en dataset, filters them by
length, deduplicates, and writes a train/validation split to disk.

Usage:
    python src/data/load_human_data.py \
        --max_paragraphs 12000 \
        --fraction 1% \
        --out_dir data
"""

import argparse
import os

from datasets import load_dataset


def _normalize_text_field(row):
    t = row.get("text", "")
    if isinstance(t, dict):
        t = t.get("text", "")
    if t is None:
        t = ""
    return str(t)


def load_human_paragraphs(max_paragraphs=12000, fraction="1%", min_words=10, max_words=500):
    """Stream Wikipedia paragraphs, falling back to wikitext-103 if unavailable."""
    paras = []
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{fraction}]")
        for ex in ds:
            txt = _normalize_text_field(ex)
            if not txt:
                continue
            for p in txt.split("\n"):
                p = p.strip()
                w = p.split()
                if min_words <= len(w) <= max_words:
                    paras.append(p)
                    if len(paras) >= max_paragraphs:
                        return paras
        return paras
    except Exception as e:
        print("wikimedia/wikipedia failed:", type(e).__name__, "- falling back to wikitext-103")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")
        for ex in ds:
            p = _normalize_text_field(ex).strip()
            if p and not p.startswith(" =") and len(p.split()) >= min_words:
                paras.append(p)
                if len(paras) >= max_paragraphs:
                    return paras
        return paras


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_paragraphs", type=int, default=12000)
    ap.add_argument("--fraction", type=str, default="1%", help="Slice of the Wikipedia train split to stream, e.g. '1%'")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    paras = load_human_paragraphs(max_paragraphs=args.max_paragraphs, fraction=args.fraction)
    print("total paragraphs collected:", len(paras))

    os.makedirs(args.out_dir, exist_ok=True)
    split = int((1 - args.val_ratio) * len(paras))
    train_paras = paras[:split]
    val_paras = paras[split:]

    with open(os.path.join(args.out_dir, "human_train.txt"), "w", encoding="utf-8") as f:
        for p in train_paras:
            f.write(p + "\n")

    with open(os.path.join(args.out_dir, "human_val.txt"), "w", encoding="utf-8") as f:
        for p in val_paras:
            f.write(p + "\n")

    print(f"saved train: {len(train_paras)}  val: {len(val_paras)}  -> {args.out_dir}/")


if __name__ == "__main__":
    main()
