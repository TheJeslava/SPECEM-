import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-language", type=int, default=10)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    en = [row for row in rows if "_en_" in row.get("dataset", "")]
    zh = [row for row in rows if "_zh_" in row.get("dataset", "")]
    if len(en) < args.per_language or len(zh) < args.per_language:
        raise SystemExit(
            f"not enough bilingual rows: en={len(en)}, zh={len(zh)}, requested={args.per_language}"
        )
    subset = en[: args.per_language] + zh[: args.per_language]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "en": args.per_language, "zh": args.per_language, "total": len(subset)}))


if __name__ == "__main__":
    main()
