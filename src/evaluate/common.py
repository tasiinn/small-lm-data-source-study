"""
Shared helpers for loading fine-tuned checkpoints and running greedy
generation, used by every script in src/evaluate/.
"""

import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_local_model(model_dir):
    """Load a fine-tuned checkpoint saved by src/train.py."""
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return tok, model


def generate_answer(tokenizer, model, prompt, max_new_tokens=32):
    """Greedy-decode a continuation and return only the newly generated text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def extract_choice_letter(text):
    """Pull the first standalone A/B/C/D token out of a model's free-text answer."""
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m = re.match(r"\s*([ABCD])", text.upper())
    if m:
        return m.group(1)
    return None


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


# The four runs compared throughout the study, and where src/train.py
# saves each by default. Eval scripts accept --model_dirs to override this.
DEFAULT_MODEL_DIRS = {
    "human": "runs/human-70m",
    "flant5": "runs/synthetic-70m-flan",
    "gemma": "runs/synthetic-70m",
    "mixed": "runs/mix-70m",
}
