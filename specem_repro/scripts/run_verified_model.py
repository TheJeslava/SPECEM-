import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", default="data/processed/fuseeval_lite.json")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--validation-per-language", type=int, default=10)
    parser.add_argument("--group-label", required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env["HF_ENDPOINT"] = args.hf_endpoint
    env["HF_HOME"] = args.hf_home

    validation_data = f"data/validation/{args.model_id}.json"
    validation_raw = f"results/validation/{args.model_id}.jsonl"
    validation_metrics = f"results/validation/{args.model_id}.metrics.json"
    final_raw = f"results/raw/fuseeval_lite_{args.model_id}.jsonl"
    final_metrics = f"results/metrics/fuseeval_lite_{args.model_id}.json"
    max_new_tokens = args.max_new_tokens
    if args.model_id.startswith("mimo_") and max_new_tokens > 128:
        max_new_tokens = 128

    run(
        [
            sys.executable,
            "scripts/build_validation_subset.py",
            "--input",
            args.data,
            "--out",
            validation_data,
            "--per-language",
            str(args.validation_per_language),
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/run_single_model.py",
            "--model-id",
            args.model_id,
            "--data",
            validation_data,
            "--out",
            validation_raw,
            "--metrics-out",
            validation_metrics,
            "--max-new-tokens",
            str(max_new_tokens),
            "--overwrite",
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/quality_gate_jsonl.py",
            "--input",
            validation_raw,
            "--expected-rows",
            str(args.validation_per_language * 2),
            "--require-languages",
            "--max-new-tokens",
            str(max_new_tokens),
            "--out",
            f"results/validation/{args.model_id}.quality_gate.json",
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/run_single_model.py",
            "--model-id",
            args.model_id,
            "--data",
            args.data,
            "--out",
            final_raw,
            "--metrics-out",
            final_metrics,
            "--max-new-tokens",
            str(max_new_tokens),
            "--delete-cache-after",
            "--group-label",
            args.group_label,
        ],
        env,
    )
    run(
        [
            sys.executable,
            "scripts/quality_gate_jsonl.py",
            "--input",
            final_raw,
            "--expected-rows",
            "400",
            "--require-languages",
            "--max-new-tokens",
            str(max_new_tokens),
            "--out",
            f"results/metrics/fuseeval_lite_{args.model_id}.quality_gate.json",
        ],
        env,
    )


if __name__ == "__main__":
    main()
