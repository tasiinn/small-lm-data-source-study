"""
Evaluate held-out validation perplexity for each fine-tuned checkpoint.

This is the intrinsic language-modeling metric reported in Table 1 of the
paper: lower perplexity means the model's predicted distribution over
tokens is closer to the held-out human-written Wikipedia text.

Usage:
    python src/evaluate/eval_perplexity.py \
        --eval_file data/human_val.txt \
        --human runs/human-70m \
        --flant5 runs/synthetic-70m-flan \
        --gemma runs/synthetic-70m \
        --mixed runs/mix-70m
"""

import argparse
import math

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

from common import DEFAULT_MODEL_DIRS


def eval_model(model_dir, eval_file, seq_len=512, batch_size=8):
    print(f"Evaluating {model_dir} on {eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)

    eval_ds = load_dataset("text", data_files={"validation": eval_file})["validation"]

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len)

    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, eval_dataset=eval_tok, data_collator=collator)

    metrics = trainer.evaluate()
    loss = metrics["eval_loss"]
    ppl = math.exp(loss) if loss < 30 else float("inf")
    print(f"  eval_loss = {loss:.4f}, perplexity = {ppl:.2f}\n")
    return loss, ppl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_file", type=str, default="data/human_val.txt")
    ap.add_argument("--human", type=str, default=DEFAULT_MODEL_DIRS["human"])
    ap.add_argument("--flant5", type=str, default=DEFAULT_MODEL_DIRS["flant5"])
    ap.add_argument("--gemma", type=str, default=DEFAULT_MODEL_DIRS["gemma"])
    ap.add_argument("--mixed", type=str, default=DEFAULT_MODEL_DIRS["mixed"])
    args = ap.parse_args()

    runs = {
        "Human (Wikipedia)": args.human,
        "Synthetic (Flan-T5)": args.flant5,
        "Synthetic (Gemma-2B)": args.gemma,
        "20% Human / 80% Gemma": args.mixed,
    }

    print(f"{'Training Data':<25} {'Eval Loss':>10} {'Perplexity':>12}")
    for name, path in runs.items():
        loss, ppl = eval_model(path, args.eval_file)
        print(f"{name:<25} {loss:>10.4f} {ppl:>12.2f}")


if __name__ == "__main__":
    main()
