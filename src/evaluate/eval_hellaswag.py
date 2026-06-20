"""
Evaluate commonsense-reasoning accuracy on a sample of HellaSwag.

Each model is shown a real-world situation and four candidate continuations
and must pick the most plausible one by letter. This is the benchmark
reported in Table 3 of the paper.

Usage:
    python src/evaluate/eval_hellaswag.py --sample_size 50
"""

import argparse

from datasets import load_dataset

from common import DEFAULT_MODEL_DIRS, load_local_model, generate_answer, extract_choice_letter


def build_hs_prompt(ex):
    ctx = ex["ctx"].strip()
    endings = ex["endings"]
    labels = ["A", "B", "C", "D"]
    options_str = "\n".join(f"{lab}) {end.strip()}" for lab, end in zip(labels, endings))
    prompt = (
        "You are a multiple-choice commonsense reasoning system.\n"
        "Choose the most plausible continuation of the given situation.\n"
        "Answer with only the letter (A, B, C, or D).\n\n"
        f"Context: {ctx}\n\nOptions:\n{options_str}\n\nAnswer:"
    )
    return prompt


def eval_hellaswag(model_name, tokenizer, model, examples):
    correct = 0
    for ex in examples:
        prompt = build_hs_prompt(ex)
        gold = ["A", "B", "C", "D"][int(ex["label"])]
        pred_raw = generate_answer(tokenizer, model, prompt, max_new_tokens=8)
        pred_letter = extract_choice_letter(pred_raw)
        correct += int(pred_letter == gold)
    acc = correct / len(examples)
    print(f"{model_name:<25} HellaSwag accuracy: {correct}/{len(examples)} = {acc:.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human", type=str, default=DEFAULT_MODEL_DIRS["human"])
    ap.add_argument("--flant5", type=str, default=DEFAULT_MODEL_DIRS["flant5"])
    ap.add_argument("--gemma", type=str, default=DEFAULT_MODEL_DIRS["gemma"])
    ap.add_argument("--mixed", type=str, default=DEFAULT_MODEL_DIRS["mixed"])
    args = ap.parse_args()

    hs = load_dataset("hellaswag", split="validation")
    examples = hs.shuffle(seed=args.seed).select(range(args.sample_size))

    runs = {
        "Human (Wikipedia)": args.human,
        "Synthetic (Flan-T5)": args.flant5,
        "Synthetic (Gemma-2B)": args.gemma,
        "20% Human / 80% Gemma": args.mixed,
    }

    print("=== HellaSwag Evaluation ===")
    for name, path in runs.items():
        tok, model = load_local_model(path)
        eval_hellaswag(name, tok, model, examples)


if __name__ == "__main__":
    main()
