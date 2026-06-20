"""
Build a hybrid training set by mixing human-written and synthetic paragraphs.

The paper's best-performing hybrid configuration uses 20% human-written
text and 80% Gemma-2B-rewritten synthetic text.

Usage:
    python src/data/mix_datasets.py \
        --human_file data/human_train.txt \
        --synthetic_file data/synthetic_train_gemma2b.txt \
        --human_fraction 0.2 \
        --out_file data/mix.txt
"""

import argparse
import random


def mix_datasets(human_file, synthetic_file, human_fraction, seed, out_file):
    with open(human_file, encoding="utf-8") as f:
        human = [l.strip() for l in f if l.strip()]
    with open(synthetic_file, encoding="utf-8") as f:
        synth = [l.strip() for l in f if l.strip()]

    n = min(len(human), len(synth))
    k = int(n * human_fraction)

    mixed = human[:k] + synth[k:n]

    random.seed(seed)
    random.shuffle(mixed)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(mixed) + "\n")

    print(f"human lines used:    {k} ({human_fraction:.0%})")
    print(f"synthetic lines used: {n - k} ({1 - human_fraction:.0%})")
    print(f"wrote {len(mixed)} lines -> {out_file}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human_file", type=str, default="data/human_train.txt")
    ap.add_argument("--synthetic_file", type=str, default="data/synthetic_train_gemma2b.txt")
    ap.add_argument("--human_fraction", type=float, default=0.2, help="Fraction of the mix drawn from human data (paper uses 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", type=str, default="data/mix.txt")
    args = ap.parse_args()

    mix_datasets(args.human_file, args.synthetic_file, args.human_fraction, args.seed, args.out_file)


if __name__ == "__main__":
    main()
