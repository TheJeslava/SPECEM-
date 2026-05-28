# SPECEM FuseEval-lite Reproduction with Updated Open Models

[中文说明](README-CN.md)

This repository contains a cleaned, self-contained reproduction package for an updated SPECEM-style FuseEval-lite experiment. The active model matrix replaces the lightweight Gemma3n candidate with Xiaomi MiMo-7B-SFT and evaluates four single-model baselines plus a local SpecEM-4 prefused ensemble.

## What is Included

- Reproducible code for data preparation, single-model generation, quality gates, metrics, and prefused SpecEM-4.
- The exact FuseEval-lite subset used in the experiment: 400 samples, 200 English and 200 Chinese.
- All final JSONL model outputs and metric files.
- A static project page in `site/index.html`.
- Bilingual LaTeX reports: `paper-en.tex` and `paper-zh.tex`.

No model checkpoints, Hugging Face tokens, local cache files, passwords, notebook checkpoints, or failed intermediate runs are included.

## Active Models

| Role | Model |
| --- | --- |
| Qwen-family baseline | `Qwen/Qwen3.5-9B` |
| Mistral-family baseline | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| Gemma-family baseline | `google/gemma-3-12b-it` |
| MiMo replacement baseline | `XiaomiMiMo/MiMo-7B-SFT` |
| Ensemble | `SpecEM-4-prefused` over the four outputs above |

## Main Results

| System | sacreBLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Quality gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma3-12B-it | 6.4067 | 24.4178 | 8.1359 | 21.6062 | 22.7684 | pass |
| MiMo-7B-SFT | 6.6759 | 30.2109 | 14.1462 | 24.7771 | 26.4840 | pass |
| Qwen3.5-9B | 10.0405 | 27.7300 | 9.9698 | 23.6687 | 24.4965 | pass |
| Ministral3-8B-Instruct-2512 | 1.9502 | 29.1905 | 12.2483 | 23.3590 | 26.4643 | pass |
| SpecEM-4-prefused | 5.8598 | 32.5354 | 15.3965 | 26.6490 | 27.6675 | pass |


The prefused SpecEM-4 system obtains the highest ROUGE scores in this reproduction (`ROUGE-L=26.6490`, `ROUGE-Lsum=27.6675`) while Qwen3.5-9B obtains the highest sacreBLEU (`10.0405`). This gap is methodologically important: BLEU rewards local lexical precision, while ROUGE better captures the consensus-style extractive overlap produced by the prefused ensemble.

## Reproduction Setup

The original run used a single NVIDIA RTX 4090 with sequential 4-bit model loading. Models are downloaded only immediately before a model-specific run and removed after the run to avoid exhausting local disk.

Recommended environment:

```bash
python -m pip install torch transformers accelerate bitsandbytes pyyaml sacrebleu rouge-score huggingface-hub
```

For gated checkpoints such as Gemma, authenticate outside this repository. Do not commit tokens:

```bash
huggingface-cli login
```

## Reproduce Metrics from Included Outputs

```bash
cd specem_repro
python scripts/quality_gate_jsonl.py   --input results/raw/fuseeval_lite_qwen3_5_9b.jsonl   --expected-rows 400 --require-languages   --max-new-tokens 256   --out results/metrics/check_qwen3_5_9b.quality_gate.json

python scripts/run_specem_prefused.py   --model-ids gemma3_12b_it mimo_7b_sft qwen3_5_9b ministral3_8b_instruct_2512   --out results/raw/fuseeval_lite_specem_4_prefused.jsonl   --metrics-out results/metrics/fuseeval_lite_specem_4_prefused.json
```

The summary file is `specem_repro/results/metrics/fuseeval_lite_active_summary.json`.

## Re-run Generation

```bash
cd specem_repro
HF_HOME=${HF_HOME:-~/.cache/huggingface} python scripts/run_verified_model.py   --model-id qwen3_5_9b   --group-label 'FuseEval-lite / Qwen3.5-9B rerun'
```

The verified runner first builds a bilingual validation subset, runs the quality gate, and only then launches the full 400-sample run. MiMo uses a special prompt adapter that pre-closes the reasoning channel to prevent `<think>` leakage.

## Important Methodological Note

`SpecEM-4-prefused` is not the source study's parallel multi-GPU SpecEM implementation. It is a local, single-GPU-compatible prefused variant that uses already generated outputs from the four verified models as candidate segments, applies unsupervised overlap scoring, and uses per-sample Hedge-style weight updates. It does not use reference answers for fusion. This limitation is intentionally documented rather than hidden.

## Repository Layout

```text
.github/workflows/     # GitHub Pages workflow for the static project page
.gitignore             # excludes local caches, secrets, checkpoints, and generated artifacts
DATA.md                # dataset and included-output documentation
LICENSE                # project license
README.md              # reproduction overview, results, and usage instructions
README-CN.md           # Chinese reproduction overview and usage instructions
paper-en.tex           # English academic-style reproduction report
paper-zh.tex           # Chinese academic-style reproduction report
requirements.txt       # minimal Python package requirements
site/                  # static project website
specem_repro/
  configs/              # model, experiment, and reference-baseline registry
  data/processed/       # FuseEval-lite subset used in the run
  results/raw/          # final predictions only
  results/metrics/      # metrics and quality gates
  scripts/              # reproduction and evaluation scripts
  src/specem_repro/     # generation, metrics, and utilities
```

## Privacy and Security

Before publication this directory was scanned for Hugging Face token patterns, the provided account name, password strings, absolute local cache paths, `.git` history from upstream clones, notebook checkpoints, and Python bytecode. None are intentionally included.
