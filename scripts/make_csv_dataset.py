#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ssptgfm.data import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CSV dataset in the SSP-TGFM local format.")
    parser.add_argument("--out", default="data/raw/synthetic_csv")
    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--num-relations", type=int, default=4)
    parser.add_argument("--num-edges", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    dataset = generate_synthetic_dataset(
        num_nodes=args.num_nodes,
        num_relations=args.num_relations,
        num_edges=args.num_edges,
        seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset.edges.to_frame().to_csv(out / "edges.csv", index=False)
    with open(out / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for idx, text in enumerate(dataset.node_texts):
            writer.writerow({"id": idx, "text": text})
    with open(out / "relations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for idx, text in enumerate(dataset.relation_texts):
            writer.writerow({"id": idx, "text": text})
    print(out)


if __name__ == "__main__":
    main()
