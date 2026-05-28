from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PROGRESS_FILE = WORKSPACE_ROOT / "复现进度.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_progress(message: str) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- {utc_now()}：{message}\n")


def initialize_progress() -> None:
    if PROGRESS_FILE.exists():
        append_progress("继续复现实验；保留已有进度记录。")
        return
    PROGRESS_FILE.write_text(
        "# SPECEM 更新版复现进度\n\n"
        "- Scope A 资源检查：未开始\n"
        "- Scope A 数据准备：未开始\n"
        "- Smoke 冒烟实验：未开始\n"
        "- FuseEval-lite 单模型 qwen3_5_9b：未开始\n"
        "- FuseEval-lite 单模型 ministral3_8b_instruct_2512：未开始\n"
        "- FuseEval-lite 单模型 gemma3_12b_it：未开始\n"
        "- FuseEval-lite 单模型 mimo_7b_sft：未开始\n"
        "- FuseEval-lite SpecEM-4：未开始\n"
        "- Scope A 指标汇总：未开始\n",
        encoding="utf-8",
    )


def mark_group_completed(group_label: str, output_path: str, metrics_path: str | None = None) -> None:
    text = PROGRESS_FILE.read_text(encoding="utf-8") if PROGRESS_FILE.exists() else ""
    updated = text.replace(f"{group_label}：未开始", f"{group_label}：已完成")
    if updated == text:
        append_progress(f"实验组完成：{group_label}；输出：{output_path}" + (f"；指标：{metrics_path}" if metrics_path else ""))
        return
    PROGRESS_FILE.write_text(updated, encoding="utf-8")
    append_progress(f"实验组完成：{group_label}；输出：{output_path}" + (f"；指标：{metrics_path}" if metrics_path else ""))
