"""
Fine-tune a small causal language model (default: Pythia-70M) on a given
text file using the standard causal LM objective.

Used identically for all four training regimes in the study (human-only,
Flan-T5 synthetic, Gemma synthetic, and the human/synthetic hybrid) -- only
--train_file and --out_dir change between runs, so that any performance
difference is attributable to the training data rather than the recipe.

Usage:
    python src/train.py \
        --train_file data/human_train.txt \
        --eval_file data/human_val.txt \
        --out_dir runs/human-70m
"""

import argparse
import math
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer,
    TrainingArguments, set_seed,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="EleutherAI/pythia-70m", help="Use 'gpt2' if VRAM is tight")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--eval_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/exp")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_epochs", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    train_ds = load_dataset("text", data_files={"train": args.train_file})["train"]
    eval_ds = load_dataset("text", data_files={"validation": args.eval_file})["validation"]

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        fp16=True,
        report_to="none",
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 30 else float("inf")
    print(f"Final eval_loss={metrics['eval_loss']:.4f}  perplexity={ppl:.2f}")

    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"perplexity: {ppl}\n")

    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
