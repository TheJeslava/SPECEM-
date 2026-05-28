import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specem_repro.metrics import lexical_metrics


def read_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-key", default="spec_pred")
    parser.add_argument("--reference-key", default="output")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = read_json_or_jsonl(Path(args.input))
    predictions = [row.get(args.prediction_key, "") for row in rows]
    references = [row.get(args.reference_key, "") for row in rows]
    metrics = lexical_metrics(predictions, references)

    payload = {
        "input": str(Path(args.input).resolve()),
        "prediction_key": args.prediction_key,
        "reference_key": args.reference_key,
        "metrics": metrics,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

