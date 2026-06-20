"""
Generate side-by-side text continuations from all four models for the same
Wikipedia prompt, for qualitative inspection of fluency and factuality.

Usage:
    python src/evaluate/eval_qualitative.py --num_examples 5
"""

import argparse

from common import DEFAULT_MODEL_DIRS, load_local_model, generate_answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val_file", type=str, default="data/human_val.txt")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--human", type=str, default=DEFAULT_MODEL_DIRS["human"])
    ap.add_argument("--flant5", type=str, default=DEFAULT_MODEL_DIRS["flant5"])
    ap.add_argument("--gemma", type=str, default=DEFAULT_MODEL_DIRS["gemma"])
    ap.add_argument("--mixed", type=str, default=DEFAULT_MODEL_DIRS["mixed"])
    args = ap.parse_args()

    with open(args.val_file) as f:
        val_lines = [l.strip() for l in f if l.strip()]

    tok_h, model_h = load_local_model(args.human)
    tok_g, model_g = load_local_model(args.gemma)
    tok_f, model_f = load_local_model(args.flant5)
    tok_m, model_m = load_local_model(args.mixed)

    for i in range(args.num_examples):
        text = val_lines[i]
        sentences = text.split(". ")
        context = ". ".join(sentences[:3]) + "."
        gold_cont = ". ".join(sentences[3:6]) + "."

        gen_h = generate_answer(tok_h, model_h, context, args.max_new_tokens)
        gen_g = generate_answer(tok_g, model_g, context, args.max_new_tokens)
        gen_f = generate_answer(tok_f, model_f, context, args.max_new_tokens)
        gen_m = generate_answer(tok_m, model_m, context, args.max_new_tokens)

        print(f"\n====== Wikipedia Continuation Example {i + 1} ======")
        print("CONTEXT:\n", context)
        print("\nGOLD CONTINUATION:\n", gold_cont)
        print("\nHuman-trained continuation:\n", gen_h)
        print("\nSynthetic-trained continuation (Gemma):\n", gen_g)
        print("\nSynthetic-trained continuation (Flan-T5):\n", gen_f)
        print("\nMixed continuation:\n", gen_m)


if __name__ == "__main__":
    main()
