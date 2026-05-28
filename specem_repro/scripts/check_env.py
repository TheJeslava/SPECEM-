import argparse
import importlib.util
import os
from pathlib import Path

import yaml


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print(f"config: {config_path.resolve()}")
    print("packages:")
    for name in [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "evaluate",
        "sacrebleu",
        "rouge_score",
        "bert_score",
        "openai",
    ]:
        print(f"  {name}: {'ok' if has_module(name) else 'missing'}")

    if has_module("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"gpu_{idx}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

    cache_roots = [
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser(),
        Path(os.environ.get("MODEL_CACHE_ROOT", ".")).expanduser(),
    ]
    print("model_cache_hints:")
    for model in cfg["updated_base_models"]:
        hf_id = model["hf_id"]
        name = hf_id.split("/")[-1].lower()
        hits = []
        for root in cache_roots:
            if root.exists():
                hits.extend(str(p) for p in root.rglob("*") if p.is_dir() and name in p.name.lower())
        print(f"  {model['id']} ({hf_id}): {'cached' if hits else 'not_found'}")
        for hit in hits[:3]:
            print(f"    - {hit}")


if __name__ == "__main__":
    main()
