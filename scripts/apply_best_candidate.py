#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import validate_full_formula_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply validation-selected hyperparameters to formal configs.")
    parser.add_argument("--search-results", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Refuse to update formal configs unless search_results.json has formal_allowed=true.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.search_results).read_text(encoding="utf-8"))
    if args.require_pass and payload.get("formal_allowed") is not True:
        raise SystemExit(f"{args.search_results} is not allowed for formal runs: formal_allowed is not true")
    candidate = payload.get("best_candidate")
    if not isinstance(candidate, dict):
        raise SystemExit("search_results.json has no best_candidate object")
    for target in args.targets:
        path = Path(target)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg["strict_full_formula"] = True
        cfg["model"] = {**cfg.get("model", {}), **candidate.get("model", {})}
        cfg["train"] = {**cfg.get("train", {}), **candidate.get("train", {})}
        validate_full_formula_config(cfg, context=str(path))
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"updated {path} from {args.search_results}: {candidate.get('name')}")


if __name__ == "__main__":
    main()
