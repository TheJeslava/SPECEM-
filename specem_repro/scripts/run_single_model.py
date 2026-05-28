import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specem_repro.metrics import lexical_metrics
from specem_repro.progress import append_progress, mark_group_completed
from specem_repro.simple_runner import UnifiedTextGenerator, read_rows, remove_model_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--delete-cache-after", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group-label", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    models = {m["id"]: m for m in cfg["updated_base_models"]}
    model_cfg = models[args.model_id]
    rows = read_rows(Path(args.data), args.limit or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out.exists():
        out.unlink()
    completed = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            completed[item["instruction"]] = item
    if completed:
        append_progress(f"{args.model_id} 检测到断点输出 {len(completed)} 条，将跳过已完成样本。")

    append_progress(f"即将开始模型实验：{args.model_id}；此时才允许下载/加载 checkpoint：{model_cfg['hf_id']}")
    generator = UnifiedTextGenerator(args.model_id, model_cfg["hf_id"], quantized=True)
    generator.load()

    outputs = list(completed.values())
    with out.open("a", encoding="utf-8") as handle:
      for idx, row in enumerate(rows, start=1):
        if row["instruction"] in completed:
            continue
        result = generator.generate(
            row["instruction"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        entry = {
                **row,
                "model_id": args.model_id,
                "pred": result.text,
                "generation_config": {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                },
                "elapsed_s": result.elapsed_s,
                "new_tokens": result.new_tokens,
                "tokens_per_s": result.new_tokens / result.elapsed_s if result.elapsed_s > 0 else 0,
                "peak_vram_gb": result.peak_vram_gb,
        }
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        outputs.append(entry)
        if len(outputs) % 10 == 0:
            append_progress(f"{args.model_id} 已完成 {len(outputs)}/{len(rows)} 条样本。")

    metrics = lexical_metrics([x["pred"] for x in outputs], [x["output"] for x in outputs])
    metrics.update(
        {
            "model_id": args.model_id,
            "hf_id": model_cfg["hf_id"],
            "avg_elapsed_s": sum(x["elapsed_s"] for x in outputs) / len(outputs),
            "avg_tokens_per_s": sum(x["tokens_per_s"] for x in outputs) / len(outputs),
            "max_peak_vram_gb": max(x["peak_vram_gb"] for x in outputs),
            "hf_home": os.environ.get("HF_HOME", ""),
        }
    )
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    generator.unload()
    if args.delete_cache_after:
        remove_model_cache(model_cfg["hf_id"])
        append_progress(f"已删除模型缓存以释放空间：{model_cfg['hf_id']}")

    mark_group_completed(args.group_label or f"FuseEval-lite 单模型 {args.model_id}", str(out), str(metrics_out))


if __name__ == "__main__":
    main()
