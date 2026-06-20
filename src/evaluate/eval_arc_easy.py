"""
Evaluate multiple-choice science QA accuracy on a sample of ARC-Easy.

Each model is prompted with the question and lettered options and must
answer with a single letter (A-D). This is the reasoning benchmark
reported in Table 2 of the paper.

Usage:
    python src/evaluate/eval_arc_easy.py --sample_size 50
"""

import argparse

from datasets import load_dataset

from common import DEFAULT_MODEL_DIRS, load_local_model, generate_answer, extract_choice_letter


def build_arc_prompt(ex):
    q = ex["question"]
    stem = q["stem"] if isinstance(q, dict) else q

    c = ex["choices"]
    if isinstance(c, dict):
        texts, labels = c["text"], c["label"]
    else:
        labels = [x["label"] for x in c]
        texts = [x["text"] for x in c]

    options_str = "\n".join(f"{lab}) {txt}" for lab, txt in zip(labels, texts))
    prompt = (
        "You are a multiple-choice question answering system.\n"
        "Read the question and options, then answer with only the letter (A, B, C, or D).\n\n"
        f"Question: {stem}\n\nOptions:\n{options_str}\n\nAnswer:"
    )
    return prompt


def eval_arc(model_name, tokenizer, model, examples):
    correct = 0
    for ex in examples:
        prompt = build_arc_prompt(ex)
        gold = ex["answerKey"]
        pred_raw = generate_answer(tokenizer, model, prompt, max_new_tokens=8)
        pred_letter = extract_choice_letter(pred_raw)
        correct += int(pred_letter == gold)
    acc = correct / len(examples)
    print(f"{model_name:<25} ARC-Easy accuracy: {correct}/{len(examples)} = {acc:.3f}")
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

    arc = load_dataset("ai2_arc", "ARC-Easy", split="test")
    examples = arc.shuffle(seed=args.seed).select(range(args.sample_size))

    runs = {
        "Human (Wikipedia)": args.human,
        "Synthetic (Flan-T5)": args.flant5,
        "Synthetic (Gemma-2B)": args.gemma,
        "20% Human / 80% Gemma": args.mixed,
    }

    print("=== ARC-Easy Evaluation ===")
    for name, path in runs.items():
        tok, model = load_local_model(path)
        eval_arc(name, tok, model, examples)


if __name__ == "__main__":
    main()
