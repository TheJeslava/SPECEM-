# SPECEM FuseEval-lite 更新开源模型复现

[English README](README.md)

本仓库是一个清理后的、可复现的 SPECEM 风格 FuseEval-lite 实验工程。实验用 Xiaomi MiMo-7B-SFT 替换了原先考虑的轻量 Gemma3n 候选，并评测四个单模型基线以及一个本地 `SpecEM-4-prefused` 集成系统。

## 仓库包含内容

- 数据准备、单模型生成、质量门控、指标计算和 prefused SpecEM-4 的可复现代码。
- 本次实验实际使用的 FuseEval-lite 子集：400 条样本，其中英文 200 条、中文 200 条。
- 最终 JSONL 模型输出与指标文件。
- 静态项目网站：`site/index.html`。
- 中英文 LaTeX 论文报告：`paper-zh.tex` 与 `paper-en.tex`。

仓库不包含模型权重、Hugging Face 缓存、本地认证凭据、notebook checkpoint 或失败的中间运行结果。

## 模型矩阵

| 角色 | 模型 |
| --- | --- |
| Qwen 系基线 | `Qwen/Qwen3.5-9B` |
| Mistral 系基线 | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| Gemma 系基线 | `google/gemma-3-12b-it` |
| MiMo 替换基线 | `XiaomiMiMo/MiMo-7B-SFT` |
| 集成系统 | 基于上述四个输出的 `SpecEM-4-prefused` |

## 主要结果

| 系统 | sacreBLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | 质量门控 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma3-12B-it | 6.4067 | 24.4178 | 8.1359 | 21.6062 | 22.7684 | pass |
| MiMo-7B-SFT | 6.6759 | 30.2109 | 14.1462 | 24.7771 | 26.4840 | pass |
| Qwen3.5-9B | 10.0405 | 27.7300 | 9.9698 | 23.6687 | 24.4965 | pass |
| Ministral3-8B-Instruct-2512 | 1.9502 | 29.1905 | 12.2483 | 23.3590 | 26.4643 | pass |
| SpecEM-4-prefused | 5.8598 | 32.5354 | 15.3965 | 26.6490 | 27.6675 | pass |

在本次复现中，`SpecEM-4-prefused` 取得最高 ROUGE 分数：`ROUGE-L=26.6490`，`ROUGE-Lsum=27.6675`。`Qwen3.5-9B` 取得最高 sacreBLEU：`10.0405`。这说明不同指标捕捉到的能力侧面并不相同：BLEU 更偏向局部词面精确度，ROUGE 更能反映 prefused 集成输出中的共识式重叠。

## 复现环境

原始运行使用单张 NVIDIA RTX 4090，并按模型顺序进行 4-bit 加载。为了避免本地磁盘被权重占满，实验流程只在即将运行某个模型时下载该模型，并在运行后清理。

推荐安装：

```bash
python -m pip install torch transformers accelerate bitsandbytes pyyaml sacrebleu rouge-score huggingface-hub
```

对于需要授权访问的模型，例如 Gemma，请在仓库外完成认证，不要把任何认证信息提交进仓库：

```bash
huggingface-cli login
```

## 用已包含输出复算指标

```bash
cd specem_repro
python scripts/quality_gate_jsonl.py   --input results/raw/fuseeval_lite_qwen3_5_9b.jsonl   --expected-rows 400 --require-languages   --max-new-tokens 256   --out results/metrics/check_qwen3_5_9b.quality_gate.json

python scripts/run_specem_prefused.py   --model-ids gemma3_12b_it mimo_7b_sft qwen3_5_9b ministral3_8b_instruct_2512   --out results/raw/fuseeval_lite_specem_4_prefused.jsonl   --metrics-out results/metrics/fuseeval_lite_specem_4_prefused.json
```

汇总指标文件位于：

```text
specem_repro/results/metrics/fuseeval_lite_active_summary.json
```

## 重新生成模型输出

```bash
cd specem_repro
HF_HOME=${HF_HOME:-~/.cache/huggingface} python scripts/run_verified_model.py   --model-id qwen3_5_9b   --group-label 'FuseEval-lite / Qwen3.5-9B rerun'
```

`run_verified_model.py` 会先构造双语验证子集，运行质量门控，再启动完整 400 样本生成。MiMo 使用专门的 prompt adapter，提前关闭 reasoning 通道，避免 `<think>` 泄漏。

## 方法说明

`SpecEM-4-prefused` 不是源研究中的多 GPU 并行 SPECEM 实现。它是一个适配单卡本地复现的 prefused 变体：先读取四个已验证模型的生成结果，将它们作为候选片段，再用无监督重叠打分和逐样本 Hedge 风格权重更新完成融合。融合过程不使用参考答案。

这个限制是实验结论的一部分：本仓库证明了在单卡资源下可以复现 SPECEM 风格的集成趋势，但不能把该实现等同于源研究中的完整并行系统。

## 一级目录和文件说明

```text
.github/workflows/     # GitHub Pages 自动部署流程
.gitignore             # 排除本地缓存、认证凭据、模型权重和生成产物
DATA.md                # 数据子集和输出文件说明
LICENSE                # 项目许可证
README.md              # 英文复现说明
README-CN.md           # 中文复现说明
paper-en.tex           # 英文学术风格复现实验报告
paper-zh.tex           # 中文学术风格复现实验报告
requirements.txt       # Python 依赖列表
site/                  # 静态项目网站
specem_repro/
  configs/              # 模型、实验和参考基线配置
  data/processed/       # 本次使用的 FuseEval-lite 子集
  results/raw/          # 最终模型预测
  results/metrics/      # 指标和质量门控结果
  scripts/              # 数据构建、评测、生成和融合脚本
  src/specem_repro/     # 生成、指标和通用工具代码
```

## 隐私和发布边界

发布前已检查仓库内容，排除了认证凭据、本地机器路径、模型权重、缓存目录、Python 字节码、notebook checkpoint 和失败的中间实验结果。中英文论文 LaTeX 报告随仓库发布，用于记录实验流程、结果与局限性。
