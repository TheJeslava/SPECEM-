import argparse
from pathlib import Path

import yaml


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--baselines", default="configs/reference_baselines.yaml")
    args = parser.parse_args()

    models = load(Path(args.models))
    experiments = load(Path(args.experiments))
    baselines = load(Path(args.baselines))

    model_ids = {item["id"] for item in models["updated_base_models"]}
    for method_name, method_cfg in experiments["methods"].items():
        if isinstance(method_cfg, dict):
            for key, value in method_cfg.items():
                if isinstance(value, list):
                    unknown = [item for item in value if item not in model_ids and not item.startswith(("MBR", "majority", "PairRank", "GenFuse", "MOA", "AlpacaEval"))]
                    if unknown:
                        raise ValueError(f"Unknown model/method ids in methods.{method_name}.{key}: {unknown}")

    required_baseline_keys = ["original_models_7b_9b", "table1_fuseeval_key_results", "table2_reasoning_key_results"]
    missing = [key for key in required_baseline_keys if key not in baselines]
    if missing:
        raise ValueError(f"Missing reference baseline keys: {missing}")

    if not experiments.get("approval_required", False):
        raise ValueError("experiments.yaml must keep approval_required: true until the user approves a scope.")

    print("configs: ok")
    print(f"updated_model_count: {len(model_ids)}")
    print(f"datasets: {', '.join(experiments['datasets'].keys())}")


if __name__ == "__main__":
    main()
