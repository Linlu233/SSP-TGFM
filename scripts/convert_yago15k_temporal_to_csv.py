#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


YEAR_RE = re.compile(r'"?(-?\d{1,4})-[#\d]{2}-[#\d]{2}"?')


def clean_text(value: str) -> str:
    return value.strip().strip("<>").strip('"').replace("_", " ")


def parse_year(value: str) -> int | None:
    match = YEAR_RE.fullmatch(value.strip())
    if match is None:
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert timed YAGO15K qualifier facts to SSP-TGFM CSV format.")
    parser.add_argument("--input-dir", default="data/raw/tkbc/src_data/yago15k")
    parser.add_argument("--out", default="data/raw/yago15k_temporal")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows: list[tuple[str, str, str, int]] = []
    for filename in ["train", "valid", "test"]:
        with (input_dir / filename).open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 5:
                    continue
                head, rel, tail, qualifier, time_raw = parts
                year = parse_year(time_raw)
                if year is None:
                    continue
                relation = f"{clean_text(rel)} {clean_text(qualifier)}"
                rows.append((clean_text(head), relation, clean_text(tail), year))

    if not rows:
        raise SystemExit("no timed YAGO15K rows found")

    entities = sorted({head for head, _, _, _ in rows}.union({tail for _, _, tail, _ in rows}))
    relations = sorted({rel for _, rel, _, _ in rows})
    ent_map = {entity: idx for idx, entity in enumerate(entities)}
    rel_map = {relation: idx for idx, relation in enumerate(relations)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "dst", "rel", "time"])
        writer.writeheader()
        for head, rel, tail, year in rows:
            writer.writerow({"src": ent_map[head], "dst": ent_map[tail], "rel": rel_map[rel], "time": year})
    with (out / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for entity, idx in ent_map.items():
            writer.writerow({"id": idx, "text": entity})
    with (out / "relations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for relation, idx in rel_map.items():
            writer.writerow({"id": idx, "text": relation})
    print({"rows": len(rows), "nodes": len(entities), "relations": len(relations), "out": str(out)})


if __name__ == "__main__":
    main()
