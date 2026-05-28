import argparse
import json
import sys
from pathlib import Path


BAD_MARKERS = [
    "thinking process",
    "analyze the request",
    "here's a thinking",
    "here is a thinking",
    "hidden thought",
    "<think>",
]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--max-thinking-rate", type=float, default=0.02)
    parser.add_argument("--max-empty-rate", type=float, default=0.0)
    parser.add_argument("--max-token-cap-rate", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prediction-key", default="pred")
    parser.add_argument("--require-languages", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    total = len(rows)
    if total == 0:
        raise SystemExit("quality gate failed: no rows")

    thinking = sum(any(marker in row.get(args.prediction_key, "").lower() for marker in BAD_MARKERS) for row in rows)
    empty = sum(not row.get(args.prediction_key, "").strip() for row in rows)
    capped = sum(int(row.get("new_tokens", 0)) >= args.max_new_tokens for row in rows)
    languages = {
        "en": sum("_en_" in row.get("dataset", "") for row in rows),
        "zh": sum("_zh_" in row.get("dataset", "") for row in rows),
    }
    report = {
        "input": str(Path(args.input).resolve()),
        "prediction_key": args.prediction_key,
        "total": total,
        "thinking_count": thinking,
        "thinking_rate": thinking / total,
        "empty_count": empty,
        "empty_rate": empty / total,
        "token_cap_count": capped,
        "token_cap_rate": capped / total,
        "languages": languages,
        "passed": True,
        "failures": [],
    }

    if args.expected_rows and total != args.expected_rows:
        report["failures"].append(f"expected {args.expected_rows} rows, got {total}")
    if report["thinking_rate"] > args.max_thinking_rate:
        report["failures"].append(f"thinking leakage rate {report['thinking_rate']:.4f} exceeds {args.max_thinking_rate:.4f}")
    if report["empty_rate"] > args.max_empty_rate:
        report["failures"].append(f"empty output rate {report['empty_rate']:.4f} exceeds {args.max_empty_rate:.4f}")
    if report["token_cap_rate"] > args.max_token_cap_rate:
        report["failures"].append(f"token cap rate {report['token_cap_rate']:.4f} exceeds {args.max_token_cap_rate:.4f}")
    if args.require_languages and not (languages["en"] and languages["zh"]):
        report["failures"].append(f"missing required language split: {languages}")

    report["passed"] = not report["failures"]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
