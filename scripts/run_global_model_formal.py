#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.auto_optimize_until_target import (  # noqa: E402
    BANKS,
    DATASETS,
    DatasetSpec,
    SEEDS,
    load_yaml,
    run_formal,
    write_compare_csv,
    write_yaml,
)
from scripts.report_global_candidate_search import deep_update, short_hash  # noqa: E402
from scripts.run_ssptgfm import validate_full_formula_config  # noqa: E402


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bank_name_from_search_dir(spec: DatasetSpec, search_dir: Path) -> str | None:
    prefix = f"search_{spec.label}_"
    if not search_dir.name.startswith(prefix):
        return None
    return search_dir.name[len(prefix) :]


def candidate_configs(search_results_path: Path) -> dict[str, dict[str, Any]]:
    config_path = Path("configs/auto_target") / f"ssptgfm_{search_results_path.parent.name}.yaml"
    if not config_path.exists():
        return {}
    cfg = load_yaml(config_path)
    base_model = cfg.get("model", {})
    out: dict[str, dict[str, Any]] = {}
    for candidate in cfg.get("search", {}).get("candidates", []):
        name = str(candidate.get("name", ""))
        out[name] = {
            "model": deep_update(base_model, candidate.get("model", {})),
            "train_patch": candidate.get("train", {}),
        }
    return out


def best_dataset_candidate(run_roots: list[Path], spec: DatasetSpec, model_signature: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for run_root in run_roots:
        for path in sorted(run_root.glob(f"search_{spec.label}_*/search_results.json")):
            bank_name = bank_name_from_search_dir(spec, path.parent)
            if bank_name is None:
                continue
            configs = candidate_configs(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for summary in payload.get("candidate_summaries", []):
                name = str(summary.get("candidate", ""))
                config = configs.get(name)
                if not config:
                    continue
                model = config["model"]
                if short_hash(model) != model_signature:
                    continue
                score = float(summary.get("score", float("nan")))
                if not finite(score):
                    continue
                if best is None or score > float(best["score"]):
                    best = {
                        "dataset": spec.label,
                        "bank": bank_name,
                        "candidate": name,
                        "score": score,
                        "model": model,
                        "train_patch": config["train_patch"],
                        "search_results": str(path),
                        "means": summary.get("means", {}),
                    }
    return best


def render_global_formal_config(
    spec: DatasetSpec,
    selected: dict[str, Any],
    bank: dict[str, Any],
    model_signature: str,
    out_root: Path,
) -> tuple[Path, Path]:
    cfg = load_yaml(spec.base_formal_config)
    formal_dir = out_root / f"ssptgfm_{spec.label}_global_{model_signature}"
    cfg["output_dir"] = str(formal_dir)
    cfg["seeds"] = list(SEEDS)
    cfg["baselines"] = []
    cfg["strict_full_formula"] = True
    cfg["model"] = {**cfg.get("model", {}), **selected["model"]}
    cfg["train"] = {**cfg.get("train", {}), **selected["train_patch"]}
    cfg["train"].update(
        {
            "epochs": int(bank.get("formal_epochs", spec.formal_epochs)),
            "patience": 5,
            "early_stop_metric": "val_composite",
            "early_stop_weights": bank["weights"],
            "val_rank_edges": spec.search_val_rank_edges,
            "progress_every_batches": 100,
            "progress_every_seconds": 120.0,
        }
    )
    if spec.batch_size is not None:
        cfg["train"]["batch_size"] = spec.batch_size
    cfg.setdefault("eval", {})
    cfg["eval"]["filtered_rank_edges"] = spec.formal_eval_rank_edges
    cfg["global_model_selection"] = {
        "model_signature": model_signature,
        "shared_model": "same model config across all datasets",
        "train_selection": "dataset-specific train hyperparameters selected on validation only",
        "candidate": selected["candidate"],
        "search_results": selected["search_results"],
        "validation_means": selected["means"],
        "test_usage": "not used for selection",
    }
    validate_full_formula_config(cfg, context=f"global model formal {spec.label} {model_signature}")
    path = Path("configs/auto_target") / f"ssptgfm_{spec.label}_global_{model_signature}.yaml"
    write_yaml(path, cfg)
    return path, formal_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal SSP-TGFM experiments with one shared model config across datasets.")
    parser.add_argument(
        "--search-run-root",
        action="append",
        required=True,
        help="Validation search root. Repeat to select one shared model from multiple validation-only search phases.",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-signature", required=True)
    parser.add_argument("--datasets", default=None, help="Comma-separated DatasetSpec labels.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    search_run_roots = [Path(path) for path in args.search_run_root]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_labels = set(args.datasets.split(",")) if args.datasets else None
    banks = {str(bank["name"]): bank for bank in BANKS}
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = "."

    selections: dict[str, dict[str, Any]] = {}
    model_payload: dict[str, Any] | None = None
    for spec in DATASETS:
        if selected_labels is not None and spec.label not in selected_labels:
            continue
        selected = best_dataset_candidate(search_run_roots, spec, args.model_signature)
        if selected is None:
            raise SystemExit(f"no validation candidate for dataset={spec.label} model_signature={args.model_signature}")
        if model_payload is None:
            model_payload = selected["model"]
        elif selected["model"] != model_payload:
            raise SystemExit(f"model signature collision or mismatch for dataset={spec.label}")
        if selected["bank"] not in banks:
            raise SystemExit(f"unknown bank for selected candidate: {selected['bank']}")
        selections[spec.label] = selected

    if not selections:
        raise SystemExit("no datasets selected")

    selection_path = out_root / f"global_model_{args.model_signature}_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "model_signature": args.model_signature,
                "seeds": list(SEEDS),
                "selection_uses": "validation metrics only",
                "test_usage": "not evaluated before selection",
                "search_run_roots": [str(path) for path in search_run_roots],
                "shared_model_config": model_payload,
                "datasets": selections,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print({"event": "global_model_selected", "signature": args.model_signature, "selection": str(selection_path)}, flush=True)

    for spec in DATASETS:
        if spec.label not in selections:
            continue
        selected = selections[spec.label]
        bank = banks[selected["bank"]]
        config_path, formal_dir = render_global_formal_config(spec, selected, bank, args.model_signature, out_root)
        print(
            {
                "event": "global_formal_ready",
                "dataset": spec.label,
                "model_signature": args.model_signature,
                "candidate": selected["candidate"],
                "config": str(config_path),
                "output": str(formal_dir),
                "dry_run": args.dry_run,
            },
            flush=True,
        )
        if args.dry_run:
            continue
        if not (formal_dir / "all_results.json").exists() or not args.resume:
            run_formal(spec, f"global_{args.model_signature}", config_path, formal_dir, out_root, env)
        compare_path = out_root / f"compare_{spec.label}_global_{args.model_signature}.csv"
        write_compare_csv(spec, formal_dir, compare_path)
        print({"event": "global_formal_done", "dataset": spec.label, "compare": str(compare_path)}, flush=True)


if __name__ == "__main__":
    main()
