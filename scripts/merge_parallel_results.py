#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_key(row: dict) -> tuple[str, str, float, int]:
    return (
        str(row.get("method")),
        str(row.get("ablation")),
        float(row.get("few_shot_ratio", 1.0)),
        int(row.get("seed")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Seed output directories or partial_results.jsonl files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    rows_by_key: dict[tuple[str, str, float, int], dict] = {}
    for item in args.inputs:
        path = Path(item)
        partial = path if path.is_file() else path / "partial_results.jsonl"
        if not partial.exists():
            continue
        for line in partial.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_key[row_key(row)] = row

    rows = sorted(rows_by_key.values(), key=lambda row: (int(row.get("seed", 0)), str(row.get("method")), str(row.get("ablation"))))
    if args.expected_rows is not None and len(rows) < args.expected_rows:
        raise SystemExit(f"only {len(rows)}/{args.expected_rows} rows are available")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "partial_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"merged {len(rows)} rows into {out_dir}")


if __name__ == "__main__":
    main()
