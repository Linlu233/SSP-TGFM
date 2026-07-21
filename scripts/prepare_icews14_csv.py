#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["src", "rel", "dst", "date"], dtype=str)
    df["split"] = split
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TKBC ICEWS14 files to SSP-TGFM CSV format.")
    parser.add_argument("--src", default="data/raw/tkbc/src_data/ICEWS14")
    parser.add_argument("--out", default="data/raw/icews14")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    train = read_split(src_dir / "train", "train")
    valid = read_split(src_dir / "valid", "val")
    test = read_split(src_dir / "test", "test")
    df = pd.concat([train, valid, test], ignore_index=True)

    entities = sorted(set(df["src"]).union(set(df["dst"])))
    relations = sorted(df["rel"].unique().tolist())
    entity_to_id = {name: idx for idx, name in enumerate(entities)}
    relation_to_id = {name: idx for idx, name in enumerate(relations)}

    out_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.DataFrame(
        {
            "src": df["src"].map(entity_to_id).astype(int),
            "dst": df["dst"].map(entity_to_id).astype(int),
            "rel": df["rel"].map(relation_to_id).astype(int),
            "time": pd.to_datetime(df["date"], utc=True).astype("int64") // 10**9,
            "split": df["split"],
        }
    )
    edges.to_csv(out_dir / "edges.csv", index=False)
    pd.DataFrame({"id": range(len(entities)), "text": entities}).to_csv(out_dir / "nodes.csv", index=False)
    pd.DataFrame({"id": range(len(relations)), "text": relations}).to_csv(out_dir / "relations.csv", index=False)
    print(
        {
            "out": str(out_dir),
            "edges": len(edges),
            "nodes": len(entities),
            "relations": len(relations),
            "train": int((edges["split"] == "train").sum()),
            "val": int((edges["split"] == "val").sum()),
            "test": int((edges["split"] == "test").sum()),
        }
    )


if __name__ == "__main__":
    main()
