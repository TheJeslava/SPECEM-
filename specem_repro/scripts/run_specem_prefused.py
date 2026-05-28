import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specem_repro.metrics import lexical_metrics
from specem_repro.progress import append_progress, mark_group_completed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())


def split_segments(text: str, max_tokens: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    return [" ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def overlap_score(candidate: str, peers: list[str]) -> float:
    cand = Counter(tokenize(candidate))
    if not cand:
        return -1e9
    scores = []
    cand_total = sum(cand.values())
    for peer in peers:
        other = Counter(tokenize(peer))
        if not other:
            continue
        inter = sum((cand & other).values())
        precision = inter / cand_total
        recall = inter / sum(other.values())
        if precision + recall:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def brevity_penalty(text: str, peer_texts: list[str]) -> float:
    lengths = [max(1, len(tokenize(item))) for item in peer_texts if item.strip()]
    if not lengths:
        return 0.0
    median = sorted(lengths)[len(lengths) // 2]
    current = max(1, len(tokenize(text)))
    ratio = current / median
    if 0.45 <= ratio <= 1.8:
        return 0.0
    return -0.08 * abs(math.log(max(ratio, 1e-6)))


def quality_penalty(text: str) -> float:
    lowered = text.lower()
    penalty = 0.0
    if any(marker in lowered for marker in ["<think>", "thinking process", "analyze the request"]):
        penalty -= 2.0
    if lowered.count("answer:") > 1 or lowered.count("the answer") > 3:
        penalty -= 0.2
    if len(text.strip()) < 2:
        penalty -= 5.0
    return penalty


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def fuse_item(candidates: dict[str, str], weights: dict[str, float], segment_tokens: int, eta: float) -> tuple[str, str, dict]:
    model_ids = list(candidates)
    candidate_segments = {mid: split_segments(candidates[mid], segment_tokens) for mid in model_ids}
    max_steps = max((len(segments) for segments in candidate_segments.values()), default=0)
    output_segments = []
    hits = []
    step_records = []

    for step in range(max_steps):
        active = {mid: segments[step] for mid, segments in candidate_segments.items() if step < len(segments)}
        if not active:
            continue
        if len(active) == 1:
            continue
        scores = {}
        for mid, segment in active.items():
            peers = [text for peer_mid, text in active.items() if peer_mid != mid]
            scores[mid] = overlap_score(segment, peers) + brevity_penalty(segment, list(active.values())) + quality_penalty(segment)
        weighted = {mid: scores[mid] + math.log(max(weights[mid], 1e-12)) for mid in active}
        winner = max(weighted, key=weighted.get)
        output_segments.append(active[winner])
        hits.append(winner)

        rewards = softmax([scores[mid] for mid in model_ids if mid in active])
        for mid, reward in zip([mid for mid in model_ids if mid in active], rewards):
            weights[mid] *= math.exp(eta * reward)
        total_weight = sum(weights.values())
        for mid in weights:
            weights[mid] /= total_weight

        step_records.append({"step": step, "winner": winner, "scores": scores, "weights": dict(weights)})

    fused = " ".join(segment for segment in output_segments if segment.strip()).strip()
    if not fused:
        winner = max(
            candidates,
            key=lambda mid: overlap_score(candidates[mid], [text for other_mid, text in candidates.items() if other_mid != mid])
            + brevity_penalty(candidates[mid], list(candidates.values()))
            + quality_penalty(candidates[mid])
            + math.log(max(weights[mid], 1e-12)),
        )
        fused = candidates[winner]
        hits.append(winner)
    return fused, Counter(hits).most_common(1)[0][0], {"hits": hits, "steps": step_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ids", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--segment-tokens", type=int, default=48)
    parser.add_argument("--eta", type=float, default=0.35)
    args = parser.parse_args()

    model_rows = {}
    for model_id in args.model_ids:
        path = PROJECT_ROOT / f"results/raw/fuseeval_lite_{model_id}.jsonl"
        rows = read_jsonl(path)
        model_rows[model_id] = {row["instruction"]: row for row in rows}

    base_model = args.model_ids[0]
    instructions = list(model_rows[base_model])
    missing = {
        model_id: [instruction for instruction in instructions if instruction not in rows]
        for model_id, rows in model_rows.items()
    }
    missing = {model_id: items for model_id, items in missing.items() if items}
    if missing:
        raise ValueError(f"Model outputs are not instruction-aligned: {missing}")

    initial_weights = {model_id: 1.0 / len(args.model_ids) for model_id in args.model_ids}
    outputs = []
    hit_counts = defaultdict(int)
    for index, instruction in enumerate(instructions, start=1):
        row0 = model_rows[base_model][instruction]
        candidates = {model_id: model_rows[model_id][instruction].get("pred", "") for model_id in args.model_ids}
        weights = dict(initial_weights)
        spec_pred, dominant_model, trace = fuse_item(candidates, weights, args.segment_tokens, args.eta)
        hit_counts[dominant_model] += 1
        outputs.append(
            {
                "dataset": row0.get("dataset"),
                "instruction": instruction,
                "output": row0.get("output", ""),
                "spec_pred": spec_pred,
                "model_id": "specem_4_prefused",
                "source_models": args.model_ids,
                "dominant_model": dominant_model,
                "candidate_lengths": {model_id: len(tokenize(text)) for model_id, text in candidates.items()},
                "trace": trace,
            }
        )
        if index % 25 == 0:
            append_progress(f"SpecEM-4 prefused 已完成 {index}/{len(instructions)} 条样本。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = lexical_metrics([item["spec_pred"] for item in outputs], [item["output"] for item in outputs])
    metrics.update(
        {
            "model_id": "specem_4_prefused",
            "source_models": args.model_ids,
            "segment_tokens": args.segment_tokens,
            "eta": args.eta,
            "hit_counts": dict(hit_counts),
            "num_samples": len(outputs),
        }
    )
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mark_group_completed("FuseEval-lite / SpecEM-4", str(out), str(metrics_path))


if __name__ == "__main__":
    main()
