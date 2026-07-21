#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml


CONFIGS = [
    Path("configs/auto_target/ssptgfm_yago15k_temporal_unified_s15sf02_official.yaml"),
    Path("configs/auto_target/ssptgfm_tgb_smallpedia_unified_s15sf02_official.yaml"),
    Path("configs/auto_target/ssptgfm_tgb_yago_unified_s15sf02_official.yaml"),
    Path("configs/auto_target/ssptgfm_icews14_unified_s15sf02_official.yaml"),
    Path("configs/auto_target/ssptgfm_icews05_15_unified_s15sf02_official.yaml"),
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def row_key(row: dict) -> tuple[str, str, float, int]:
    return (
        str(row.get("method")),
        str(row.get("ablation")),
        float(row.get("few_shot_ratio", 1.0)),
        int(row.get("seed")),
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="configs/ablation_unified_s15sf02_official")
    parser.add_argument("--output-root", default="results/ablation_unified_s15sf02_official")
    args = parser.parse_args()

    config_root = Path(args.config_root)
    output_root = Path(args.output_root)
    mamba_available = importlib.util.find_spec("mamba_ssm") is not None
    exclude = [] if mamba_available else ["mamba_encoder"]
    blocked = {}
    if not mamba_available:
        blocked["mamba_encoder"] = (
            "mamba_ssm is unavailable in the verified base conda environment; "
            "installation was not completed to avoid changing torch/CUDA dependencies."
        )

    prepared = []
    for src in CONFIGS:
        cfg = load_yaml(src)
        original_output = Path(str(cfg["output_dir"]))
        out_dir = output_root / src.stem
        cfg["output_dir"] = str(out_dir)
        cfg["baselines"] = []
        cfg["strict_full_formula"] = True
        cfg["ablation"] = {"enabled": True}
        if exclude:
            cfg["ablation"]["exclude"] = exclude
            cfg["ablation"]["blocked"] = blocked

        dst_cfg = config_root / f"{src.stem}_ablation.yaml"
        dump_yaml(cfg, dst_cfg)

        source_rows = [
            row
            for row in read_jsonl(original_output / "partial_results.jsonl")
            if row.get("method") == "ssptgfm" and row.get("ablation") == "full"
        ]
        out_partial = out_dir / "partial_results.jsonl"
        existing_rows = read_jsonl(out_partial)
        existing_keys = {row_key(row) for row in existing_rows}
        merged_rows = existing_rows + [row for row in source_rows if row_key(row) not in existing_keys]
        write_jsonl(merged_rows, out_partial)

        prepared.append(
            {
                "dataset": cfg.get("data", {}).get("name"),
                "config": str(dst_cfg),
                "output_dir": str(out_dir),
                "prepopulated_full_rows": len(source_rows),
                "existing_rows_after_prepare": len(merged_rows),
                "excluded_ablations": exclude,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"prepared": prepared, "blocked_ablations": blocked}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if blocked:
        lines = ["# Blocked Ablations", ""]
        for name, reason in blocked.items():
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
        (output_root / "BLOCKED_ABLATIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"prepared": prepared, "blocked_ablations": blocked}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
