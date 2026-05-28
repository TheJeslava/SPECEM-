import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def normalize_dolly(row):
    instruction = row.get("instruction") or ""
    context = row.get("context") or ""
    if context.strip():
        instruction = instruction.strip() + "\n\n" + context.strip()
    return {
        "dataset": "fuseeval_lite_en_dolly",
        "instruction": instruction.strip(),
        "output": (row.get("response") or "").strip(),
    }


def normalize_alpaca(row):
    instruction = row.get("instruction") or ""
    input_text = row.get("input") or ""
    if input_text.strip():
        instruction = instruction.strip() + "\n\n" + input_text.strip()
    return {
        "dataset": "fuseeval_lite_en_alpaca",
        "instruction": instruction.strip(),
        "output": (row.get("output") or row.get("response") or "").strip(),
    }


def normalize_coig(row):
    instruction = (
        row.get("instruction")
        or row.get("question")
        or row.get("prompt")
        or row.get("query")
        or row.get("input")
        or row.get("text")
        or ""
    )
    output = row.get("output") or row.get("answer") or row.get("response") or row.get("target") or ""
    if isinstance(output, list):
        output = output[0] if output else ""
    return {
        "dataset": "fuseeval_lite_zh_coig",
        "instruction": str(instruction).strip(),
        "output": str(output).strip(),
    }


def take_valid(rows, n):
    out = []
    for row in rows:
        if row["instruction"] and row["output"]:
            out.append(row)
        if len(out) >= n:
            break
    return out


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_chinese_from_coig_cqia(n, seed):
    files = [
        ("m-a-p/COIG-CQIA", "human_value/coig_human_value_multi_choice.jsonl"),
        ("m-a-p/COIG-CQIA", "ruozhiba/ruozhiba_ruozhiba.jsonl"),
        ("BAAI/COIG", "human_value_alignment_instructions_part1.json"),
        ("BAAI/COIG", "human_value_alignment_instructions_part2.json"),
    ]
    rows = []
    rng = random.Random(seed)
    for repo_id, filename in files:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception:
            continue
        if filename.endswith(".jsonl"):
            raw = list(read_jsonl(local))
        else:
            payload = json.loads(Path(local).read_text(encoding="utf-8"))
            raw = payload if isinstance(payload, list) else payload.get("data", [])
        rng.shuffle(raw)
        rows.extend(take_valid((normalize_coig(x) for x in raw), max(0, n - len(rows))))
        if len(rows) >= n:
            break
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/fuseeval_lite.json")
    parser.add_argument("--samples-per-language", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    n = args.samples_per_language

    english = []
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_indices = list(range(len(dolly)))
    random.shuffle(dolly_indices)
    english.extend(take_valid((normalize_dolly(dolly[i]) for i in dolly_indices), n // 2))

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_indices = list(range(len(alpaca)))
    random.shuffle(alpaca_indices)
    english.extend(take_valid((normalize_alpaca(alpaca[i]) for i in alpaca_indices), n - len(english)))

    chinese = build_chinese_from_coig_cqia(n, args.seed)
    if len(chinese) < n:
        try:
            coig = load_dataset("BAAI/COIG", split="train", trust_remote_code=True)
            coig_indices = list(range(len(coig)))
            random.shuffle(coig_indices)
            chinese.extend(take_valid((normalize_coig(coig[i]) for i in coig_indices), n - len(chinese)))
        except Exception as exc:
            print(f"warning: fallback BAAI/COIG load failed: {type(exc).__name__}: {str(exc)[:300]}")

    rows = english[:n] + chinese[:n]
    if len(english) < n or len(chinese) < n:
        raise RuntimeError(f"Insufficient data: english={len(english)}, chinese={len(chinese)}, required={n}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out.resolve()}")


if __name__ == "__main__":
    main()
