#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_line(line: str) -> tuple[str, str, str, str]:
    stripped = line.strip()
    parts = stripped.split("\t")
    if len(parts) < 4:
        parts = stripped.split(",")
    if len(parts) < 4:
        raise ValueError(f"expected at least 4 columns, got: {line!r}")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert temporal KG quadruple files to SSP-TGFM CSV format.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", default="train.txt")
    parser.add_argument("--valid", default="valid.txt")
    parser.add_argument("--test", default="test.txt")
    parser.add_argument(
        "--time-order",
        choices=["file", "lexical"],
        default="lexical",
        help="Use lexical timestamp ids or preserve first-seen file order.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    files = [input_dir / args.train, input_dir / args.valid, input_dir / args.test]
    rows: list[tuple[str, str, str, str, str]] = []
    for path, split in zip(files, ["train", "val", "test"]):
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    h, r, t, tm = parse_line(line)
                    rows.append((h, r, t, tm, split))

    entities = sorted({h for h, _, _, _, _ in rows}.union({t for _, _, t, _, _ in rows}))
    relations = sorted({r for _, r, _, _, _ in rows})
    if args.time_order == "lexical":
        times = sorted({tm for _, _, _, tm, _ in rows})
    else:
        times = []
        seen = set()
        for _, _, _, tm, _ in rows:
            if tm not in seen:
                seen.add(tm)
                times.append(tm)
    ent_map = {e: i for i, e in enumerate(entities)}
    rel_map = {r: i for i, r in enumerate(relations)}
    time_map = {tm: i for i, tm in enumerate(times)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "dst", "rel", "time", "split"])
        writer.writeheader()
        for h, r, t, tm, split in rows:
            writer.writerow({"src": ent_map[h], "dst": ent_map[t], "rel": rel_map[r], "time": time_map[tm], "split": split})
    with open(out / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for e, idx in ent_map.items():
            writer.writerow({"id": idx, "text": e.replace("_", " ")})
    with open(out / "relations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for r, idx in rel_map.items():
            writer.writerow({"id": idx, "text": r.replace("_", " ")})
    print(out)


if __name__ == "__main__":
    main()
