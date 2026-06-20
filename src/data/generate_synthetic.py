"""
Generate synthetic rewrites of human-written paragraphs using an LLM.

Supports two generator styles used in the study:
  - "flan-t5"  : seq2seq summarization/rewrite prompt for Flan-T5-Large
  - "gemma"    : chat-formatted elaboration prompt for Gemma-2B-it

Both produce one synthetic paragraph per input line, checkpointing to disk
every 100 examples so a long generation run can be resumed/inspected.

Usage:
    # Flan-T5 rewrite
    python src/data/generate_synthetic.py \
        --style flan-t5 --model google/flan-t5-large \
        --in_file data/human_train.txt --out_file data/synthetic_train_flant5.txt

    # Gemma rewrite (requires `huggingface-cli login` first for gated weights)
    python src/data/generate_synthetic.py \
        --style gemma --model google/gemma-2b-it \
        --in_file data/human_train.txt --out_file data/synthetic_train_gemma2b.txt
"""

import argparse
import itertools
import os

from transformers import pipeline


def read_lines(path, limit=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines if limit is None else lines[:limit]


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x.replace("\n", " ") + "\n")


def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def clean_gemma_output(text):
    """Strip conversational preambles ("Sure, here is...") some chat models prepend."""
    bad_prefixes = (
        "sure", "here is", "here's", "below is",
        "the rewritten paragraph", "rewritten paragraph",
        "here is the paragraph", "here's the paragraph",
    )
    t = text.strip()
    lower = t.lower()
    for bp in bad_prefixes:
        if lower.startswith(bp):
            t = t.split(".", 1)[-1].strip()
            break
    return t


def build_flan_t5_prompts(chunk):
    return [
        "Summarize the paragraph below into 2-3 sentences using different wording. "
        "Make it more information-dense and do not copy full sentences verbatim. "
        "Preserve all key facts:\n\n" + p
        for p in chunk
    ]


def build_gemma_prompts(chunk):
    return [
        "<start_of_turn>user\n"
        "Rewrite the paragraph below in a neutral, encyclopedic style similar to Wikipedia.\n\n"
        "Requirements:\n"
        "- Output ONLY the rewritten paragraph text.\n"
        "- Do NOT include introductions or phrases like \u201cSure\u201d, \u201cHere is\u201d, or \u201cBelow is\u201d.\n"
        "- Preserve all factual content exactly.\n"
        "- Use concise, information-dense sentences.\n"
        "- Do NOT add explanations, examples, or commentary.\n"
        "- Aim for similar length and sentence structure as the original.\n\n"
        "Paragraph:\n"
        f"{p}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        for p in chunk
    ]


def generate(style, model_name, in_file, out_file, max_new_tokens, batch_size, limit):
    src = read_lines(in_file, limit)
    print("loaded:", len(src), "source paragraphs")

    task = "text2text-generation" if style == "flan-t5" else "text-generation"
    gen = pipeline(task, model=model_name, device_map="auto")

    out = []
    processed = 0

    for chunk in batched(src, batch_size):
        prompts = build_flan_t5_prompts(chunk) if style == "flan-t5" else build_gemma_prompts(chunk)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            truncation=True,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        if style == "gemma":
            gen_kwargs["pad_token_id"] = gen.tokenizer.eos_token_id

        preds = gen(prompts, **gen_kwargs)

        for i, pred in enumerate(preds):
            if style == "flan-t5":
                text = pred.get("generated_text") if isinstance(pred, dict) else pred[0].get("generated_text")
                text = (text or "").strip()
            else:
                text = pred[0]["generated_text"].strip()
                if text.startswith(prompts[i]):
                    text = text[len(prompts[i]):].strip()
                text = clean_gemma_output(text)

            if not text or len(text.split()) < 3:
                text = chunk[i]  # fall back to the original paragraph
            out.append(text)

        processed += len(chunk)
        if processed % 100 == 0:
            print(f"generated {processed}/{len(src)}")
            write_lines(out_file, out)  # checkpoint

    write_lines(out_file, out)
    print("wrote:", out_file, "lines:", len(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["flan-t5", "gemma"], required=True)
    ap.add_argument("--model", type=str, required=True, help="e.g. google/flan-t5-large or google/gemma-2b-it")
    ap.add_argument("--in_file", type=str, default="data/human_train.txt")
    ap.add_argument("--out_file", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap for a quick smoke test")
    args = ap.parse_args()

    generate(
        style=args.style,
        model_name=args.model,
        in_file=args.in_file,
        out_file=args.out_file,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
