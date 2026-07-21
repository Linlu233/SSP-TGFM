#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssptgfm.baseline_training import build_baseline, evaluate_baseline_binary, evaluate_baseline_ranking, train_baseline_one_seed
from ssptgfm.data import generate_synthetic_dataset, load_csv_dataset, load_tgb_dataset, split_by_labels, split_by_time, subsample_train_edges
from ssptgfm.experiment_splits import build_scenario, exact_k_shot_train
from ssptgfm.metrics import summarize_seed_metrics
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler
from ssptgfm.model import SSPTGFM
from ssptgfm.profiling import rough_forward_flops
from ssptgfm.text import encode_texts
from ssptgfm.training import TrainConfig, evaluate_test, train_one_seed
from ssptgfm.utils import Timer, count_parameters, cuda_memory_mb, env_report, ensure_dir, load_yaml, save_json, set_seed


FULL_FORMULA_MODEL_FLAGS = ("use_struct", "use_sem", "use_cross", "use_gate", "use_variational")
FULL_FORMULA_POSITIVE_MODEL_DIMS = ("prompt_tokens", "prompt_heads", "relation_rank", "adapter_rank", "temporal_layers")
FULL_FORMULA_POSITIVE_LOSSES = ("lambda_align", "lambda_kl", "lambda_meta", "lambda_ood")


def validate_full_formula_config(cfg: dict, *, context: str, strict_default: bool = True) -> None:
    """Fail fast if a non-ablation formal config disables SSP-TGFM formula terms."""
    if not bool(cfg.get("strict_full_formula", strict_default)):
        return
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    errors: list[str] = []
    for key in FULL_FORMULA_MODEL_FLAGS:
        if bool(model_cfg.get(key, True)) is not True:
            errors.append(f"model.{key} must be true")
    for key in FULL_FORMULA_POSITIVE_MODEL_DIMS:
        if int(model_cfg.get(key, 0)) <= 0:
            errors.append(f"model.{key} must be > 0")
    for key in FULL_FORMULA_POSITIVE_LOSSES:
        if float(train_cfg.get(key, 0.0)) <= 0.0:
            errors.append(f"train.{key} must be > 0")
    if float(train_cfg.get("meta_lr", 0.0)) <= 0.0:
        errors.append("train.meta_lr must be > 0")
    if int(train_cfg.get("meta_support_size", 0)) <= 0:
        errors.append("train.meta_support_size must be > 0")
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"{context} violates strict_full_formula: {joined}")


def build_dataset(cfg: dict) -> object:
    data_cfg = cfg.get("data", {})
    name = data_cfg.get("name", "synthetic")
    if name == "synthetic":
        return generate_synthetic_dataset(
            num_nodes=int(data_cfg.get("num_nodes", 256)),
            num_relations=int(data_cfg.get("num_relations", 4)),
            num_edges=int(data_cfg.get("num_edges", 1600)),
            num_topics=int(data_cfg.get("num_topics", 8)),
            seed=int(data_cfg.get("seed", 1)),
        )
    if data_cfg.get("format") == "csv":
        return load_csv_dataset(data_cfg["path"], name=name)
    if data_cfg.get("format") == "tgb":
        return load_tgb_dataset(name=name, root=data_cfg.get("root", "data/raw/tgb"), download=bool(data_cfg.get("download", True)))
    raise ValueError("data.name=synthetic or data.format=csv is currently supported")



def load_tgb_negative_sets(cfg: dict, dataset, split: str) -> dict[tuple[int, int, int], np.ndarray] | None:
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    if data_cfg.get("format") != "tgb" or not bool(eval_cfg.get("tgb_official", False)):
        return None
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported TGB negative split: {split}")
    name = str(data_cfg.get("name", dataset.name))
    root = Path(data_cfg.get("root", "data/raw/tgb"))
    path = root / "_".join(name.split("-")) / f"{name}_{split}_ns.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing TGB official {split} negatives: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    return {tuple(int(x) for x in key): np.asarray(value, dtype=np.int64) for key, value in payload.items()}

def make_train_config(cfg: dict, overrides: dict | None = None) -> TrainConfig:
    train = cfg.get("train", {})
    if overrides:
        train = {**train, **overrides}
    return TrainConfig(
        epochs=int(train.get("epochs", 50)),
        batch_size=int(train.get("batch_size", 256)),
        lr=float(train.get("lr", 0.001)),
        weight_decay=float(train.get("weight_decay", 0.0001)),
        num_neg_train=int(train.get("num_neg_train", 1)),
        num_neg_eval=int(train.get("num_neg_eval", 50)),
        negative_mode_train=str(train.get("negative_mode_train", "filtered")),
        negative_mode_eval=str(train.get("negative_mode_eval", "filtered")),
        filter_scope=str(train.get("filter_scope", "exact")),
        lambda_align=float(train.get("lambda_align", 0.1)),
        lambda_kl=float(train.get("lambda_kl", 0.001)),
        lambda_meta=float(train.get("lambda_meta", 0.0)),
        meta_lr=float(train.get("meta_lr", 0.01)),
        meta_support_size=int(train.get("meta_support_size", 16)),
        meta_query_size=int(train.get("meta_query_size", 16)),
        lambda_ood=float(train.get("lambda_ood", 0.0)),
        lambda_rank=float(train.get("lambda_rank", 0.0)),
        rank_margin=float(train.get("rank_margin", 1.0)),
        rank_loss_type=str(train.get("rank_loss_type", "hinge")),
        lambda_candidate_rank=float(train.get("lambda_candidate_rank", 0.0)),
        candidate_rank_size=int(train.get("candidate_rank_size", 32)),
        candidate_rank_sides=str(train.get("candidate_rank_sides", "both")),
        candidate_rank_queries=int(train.get("candidate_rank_queries", 0)),
        candidate_rank_tail_pool=str(train.get("candidate_rank_tail_pool", "sampler")),
        lambda_struct_aux=float(train.get("lambda_struct_aux", 0.0)),
        align_temperature=float(train.get("align_temperature", 0.2)),
        patience=int(train.get("patience", 10)),
        early_stop_metric=str(train.get("early_stop_metric", "val_auc")),
        early_stop_minima={str(k): float(v) for k, v in train.get("early_stop_minima", {}).items()},
        early_stop_weights={str(k): float(v) for k, v in train.get("early_stop_weights", {}).items()},
        val_binary_edges=(
            None if train.get("val_binary_edges", None) is None else int(train.get("val_binary_edges"))
        ),
        val_rank_edges=(
            None if train.get("val_rank_edges", None) is None else int(train.get("val_rank_edges"))
        ),
        val_eval_sample=str(train.get("val_eval_sample", "prefix")),
        eval_batch_size_candidates=(
            None
            if train.get("eval_batch_size_candidates", None) is None
            else int(train.get("eval_batch_size_candidates"))
        ),
        grad_clip=float(train.get("grad_clip", 1.0)),
        amp=bool(train.get("amp", False)),
        progress_every_batches=int(train.get("progress_every_batches", 100)),
        progress_every_seconds=float(train.get("progress_every_seconds", 60.0)),
    )


def make_model(cfg: dict, dataset, text_dim: int, ablation: dict | None = None) -> SSPTGFM:
    model_cfg = copy.deepcopy(cfg.get("model", {}))
    if ablation:
        model_cfg.update(ablation)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=text_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        time_dim=int(model_cfg.get("time_dim", 32)),
        prompt_tokens=int(model_cfg.get("prompt_tokens", 4)),
        prompt_heads=int(model_cfg.get("prompt_heads", 4)),
        relation_rank=int(model_cfg.get("relation_rank", 16)),
        adapter_rank=int(model_cfg.get("adapter_rank", 16)),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        temporal_encoder=str(model_cfg.get("temporal_encoder", "mlp")),
        time_encoder=str(model_cfg.get("time_encoder", "fourier")),
        use_struct=bool(model_cfg.get("use_struct", True)),
        use_sem=bool(model_cfg.get("use_sem", True)),
        use_cross=bool(model_cfg.get("use_cross", True)),
        use_gate=bool(model_cfg.get("use_gate", True)),
        use_variational=bool(model_cfg.get("use_variational", True)),
        use_history_prior=bool(model_cfg.get("use_history_prior", False)),
        history_prior_dim=int(model_cfg.get("history_prior_dim", 12)),
        history_prior_hidden_dim=(
            None
            if model_cfg.get("history_prior_hidden_dim", None) is None
            else int(model_cfg.get("history_prior_hidden_dim"))
        ),
        history_prior_init_scale=float(model_cfg.get("history_prior_init_scale", 0.25)),
        history_prior_mode=str(model_cfg.get("history_prior_mode", "mlp")),
        history_prior_weights=model_cfg.get("history_prior_weights", None),
        freeze_history_prior=bool(model_cfg.get("freeze_history_prior", False)),
        history_prior_layer_norm=bool(model_cfg.get("history_prior_layer_norm", False)),
        use_history_prior_gate=bool(model_cfg.get("use_history_prior_gate", False)),
        history_prior_gate_hidden_dim=(
            None
            if model_cfg.get("history_prior_gate_hidden_dim", None) is None
            else int(model_cfg.get("history_prior_gate_hidden_dim"))
        ),
        history_prior_gate_init_bias=float(model_cfg.get("history_prior_gate_init_bias", 0.0)),
        use_struct_feature_residual=bool(model_cfg.get("use_struct_feature_residual", False)),
        struct_feature_hidden_dim=(
            None
            if model_cfg.get("struct_feature_hidden_dim", None) is None
            else int(model_cfg.get("struct_feature_hidden_dim"))
        ),
        struct_feature_init_scale=float(model_cfg.get("struct_feature_init_scale", 0.1)),
        use_relation_entity_prior=bool(model_cfg.get("use_relation_entity_prior", False)),
        relation_entity_prior_rank=int(model_cfg.get("relation_entity_prior_rank", 16)),
        relation_entity_prior_init_scale=float(model_cfg.get("relation_entity_prior_init_scale", 0.0)),
    )
    model.struct_encoder.history_chunk_size = int(model_cfg.get("history_chunk_size", model.struct_encoder.history_chunk_size))
    model.struct_encoder.gradient_checkpoint = bool(model_cfg.get("gradient_checkpoint", False))
    return model


def ablation_grid(cfg: dict) -> list[tuple[str, dict]]:
    items = [("full", {"model": {}, "train": {}})]
    if not cfg.get("ablation", {}).get("enabled", False):
        return items
    items.extend(
        [
            ("no_gate", {"model": {"use_gate": False}, "train": {}}),
            ("no_theory_loss", {"model": {"use_variational": False}, "train": {"lambda_align": 0.0, "lambda_kl": 0.0, "lambda_meta": 0.0, "lambda_ood": 0.0}}),
            ("struct_only", {"model": {"use_sem": False, "use_cross": False, "use_gate": False}, "train": {"lambda_align": 0.0, "lambda_kl": 0.0}}),
            ("sem_only", {"model": {"use_struct": False, "use_cross": False, "use_gate": False}, "train": {"lambda_align": 0.0, "lambda_kl": 0.0}}),
            ("no_cross", {"model": {"use_cross": False}, "train": {}}),
            ("lstm_encoder", {"model": {"temporal_encoder": "lstm"}, "train": {}}),
            ("transformer_encoder", {"model": {"temporal_encoder": "transformer"}, "train": {}}),
            ("mamba_encoder", {"model": {"temporal_encoder": "mamba"}, "train": {}}),
            ("identity_time", {"model": {"time_encoder": "identity"}, "train": {}}),
            ("wavelet_time", {"model": {"time_encoder": "wavelet"}, "train": {}}),
            ("random_fourier_time", {"model": {"time_encoder": "random_fourier"}, "train": {}}),
            ("random_negative", {"model": {}, "train": {"negative_mode_train": "random", "negative_mode_eval": "random"}}),
        ]
    )
    excluded = {str(name) for name in cfg.get("ablation", {}).get("exclude", [])}
    return [(name, ablation) for name, ablation in items if name not in excluded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ssptgfm_synthetic.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override, e.g. 1 or 1,2,3.")
    parser.add_argument("--resume", action="store_true", help="Preserve and skip rows already present in partial_results.jsonl.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.seeds:
        cfg["seeds"] = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    ablation_enabled = bool(cfg.get("ablation", {}).get("enabled", False))
    validate_full_formula_config(cfg, context=f"config {args.config}", strict_default=not ablation_enabled)
    out_dir = ensure_dir(args.output or cfg.get("output_dir", "results/ssptgfm"))
    dataset = build_dataset(cfg)
    dataset.validate()
    split_cfg = cfg.get("split", {})
    scenario_cfg = cfg.get("scenario", {"name": "standard"})
    if split_cfg.get("mode", "time") == "labels":
        base_splits = split_by_labels(dataset)
        scenario_name = "label_split"
        scenario_meta = {}
    else:
        dataset, scenario = build_scenario(
            dataset,
            scenario_cfg,
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(scenario_cfg.get("seed", cfg.get("data", {}).get("seed", 1))),
        )
        base_splits = scenario.splits
        scenario_name = scenario.name
        scenario_meta = scenario.metadata
    text_cfg = cfg.get("text", {})
    requested_cuda = cfg.get("device", "cuda") == "cuda"
    if bool(cfg.get("require_cuda", requested_cuda)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")
    print(f"runtime_device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    node_text = encode_texts(
        dataset.node_texts,
        backend=str(text_cfg.get("backend", "hashing")),
        model_name=str(text_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        cache_dir=str(text_cfg.get("cache_dir", "data/processed/text_cache")),
        dim=int(text_cfg.get("dim", 384)),
        device=str(device),
    )
    rel_text = encode_texts(
        dataset.relation_texts,
        backend=str(text_cfg.get("backend", "hashing")),
        model_name=str(text_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        cache_dir=str(text_cfg.get("cache_dir", "data/processed/text_cache")),
        dim=int(text_cfg.get("dim", 384)),
        device=str(device),
    )
    node_text_t = torch.tensor(node_text, dtype=torch.float32, device=device)
    rel_text_t = torch.tensor(rel_text, dtype=torch.float32, device=device)
    seeds = [int(s) for s in cfg.get("seeds", [1, 2, 3, 4, 5])]
    ratios = [float(x) for x in cfg.get("few_shot_ratios", [1.0])]
    eval_cfg = cfg.get("eval", {})
    filtered_rank_edges = eval_cfg.get("filtered_rank_edges", 200)
    if filtered_rank_edges is not None:
        filtered_rank_edges = int(filtered_rank_edges)
    tgb_eval_edges = eval_cfg.get("tgb_eval_edges", None)
    if tgb_eval_edges is not None:
        tgb_eval_edges = int(tgb_eval_edges)
    tgb_val_rank_edges = eval_cfg.get("tgb_val_edges", cfg.get("train", {}).get("val_rank_edges", None))
    if tgb_val_rank_edges is not None:
        tgb_val_rank_edges = int(tgb_val_rank_edges)
    tgb_val_negative_sets = load_tgb_negative_sets(cfg, dataset, "val")
    tgb_test_negative_sets = load_tgb_negative_sets(cfg, dataset, "test")
    partial_path = Path(out_dir) / "partial_results.jsonl"
    all_results: list[dict] = []
    completed: dict[tuple[str, str, float, int], dict] = {}
    if args.resume and partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            all_results.append(row)
            completed[
                (
                    str(row.get("method")),
                    str(row.get("ablation")),
                    float(row.get("few_shot_ratio", 1.0)),
                    int(row.get("seed")),
                )
            ] = row
    elif partial_path.exists():
        partial_path.unlink()
    baselines = [str(x) for x in cfg.get("baselines", [])]
    for ablation_name, ablation in ablation_grid(cfg):
        if ablation_name == "full":
            full_cfg = copy.deepcopy(cfg)
            full_cfg["model"] = {**full_cfg.get("model", {}), **ablation.get("model", {})}
            full_cfg["train"] = {**full_cfg.get("train", {}), **ablation.get("train", {})}
            validate_full_formula_config(
                full_cfg,
                context=f"full variant in {args.config}",
                strict_default=not ablation_enabled,
            )
        train_cfg = make_train_config(cfg, ablation.get("train", {}))
        for ratio in ratios:
            seed_rows = []
            for seed in seeds:
                set_seed(seed)
                splits = copy.deepcopy(base_splits)
                splits.train = subsample_train_edges(splits.train, ratio)
                if "k_shot" in cfg:
                    k_cfg = cfg["k_shot"]
                    splits.train = exact_k_shot_train(splits.train, int(k_cfg.get("k", 5)), by=str(k_cfg.get("by", "relation")))
                splits.assert_no_temporal_leakage()
                row_key = ("ssptgfm", ablation_name, ratio, seed)
                row = completed.get(row_key)
                if row is None:
                    model = make_model(cfg, dataset, text_dim=node_text_t.size(1), ablation=ablation.get("model", {}))
                    print(
                        {
                            "event": "seed_start",
                            "dataset": dataset.name,
                            "ablation": ablation_name,
                            "seed": seed,
                            "device": str(device),
                            "node_text_device": str(node_text_t.device),
                        }
                    )
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    with Timer() as timer:
                        model, val_metrics = train_one_seed(
                            dataset,
                            splits,
                            model,
                            node_text_t,
                            rel_text_t,
                            train_cfg,
                            device,
                            seed,
                            tgb_val_negative_sets=tgb_val_negative_sets,
                            tgb_val_rank_edges=tgb_val_rank_edges,
                        )
                        test_metrics = evaluate_test(
                            dataset,
                            splits,
                            model,
                            node_text_t,
                            rel_text_t,
                            train_cfg,
                            device,
                            seed,
                            filtered_rank_edges=filtered_rank_edges,
                            tgb_test_negative_sets=tgb_test_negative_sets,
                            tgb_eval_edges=tgb_eval_edges,
                        )
                    row = {
                        "dataset": dataset.name,
                        "method": "ssptgfm",
                        "scenario": scenario_name,
                        "ablation": ablation_name,
                        "few_shot_ratio": ratio,
                        "seed": seed,
                        "train_edges": len(splits.train),
                        "val_edges": len(splits.val),
                        "test_edges": len(splits.test),
                        "params_trainable": count_parameters(model, trainable_only=True),
                        "params_total": count_parameters(model, trainable_only=False),
                        "rough_forward_flops": rough_forward_flops(model, len(splits.train), train_cfg.batch_size),
                        "train_eval_time_sec": timer.elapsed,
                        **cuda_memory_mb(),
                        **val_metrics,
                        **test_metrics,
                    }
                    all_results.append(row)
                    with open(partial_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    print(row)
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print({"event": "resume_skip", "method": "ssptgfm", "ablation": ablation_name, "seed": seed})
                seed_rows.append({k: v for k, v in row.items() if isinstance(v, (int, float))})
                if ablation_name != "full":
                    continue
                for baseline_name in baselines:
                    baseline_key = (baseline_name, "baseline", ratio, seed)
                    if baseline_key in completed:
                        print({"event": "resume_skip", "method": baseline_name, "seed": seed})
                        continue
                    baseline_model = build_baseline(
                        baseline_name,
                        dataset,
                        text_dim=node_text_t.size(1),
                        hidden_dim=int(cfg.get("baseline", {}).get("hidden_dim", cfg.get("model", {}).get("hidden_dim", 128))),
                    )
                    print({"event": "baseline_start", "method": baseline_name, "seed": seed}, flush=True)
                    with Timer() as b_timer:
                        baseline_model, b_val_metrics = train_baseline_one_seed(
                            dataset,
                            splits,
                            baseline_model,
                            node_text_t,
                            rel_text_t,
                            train_cfg,
                            device,
                            seed,
                        )
                        history_pool = splits.train.concat(splits.val, sort=False)
                        known_all = KnownFacts.from_edges(dataset.edges)
                        print({"event": "baseline_test_binary_start", "method": baseline_name, "seed": seed}, flush=True)
                        sampler = NegativeSampler(
                            dataset.num_nodes,
                            known_all,
                            splits.train,
                            mode=train_cfg.negative_mode_eval,
                            filter_scope=train_cfg.filter_scope,
                            seed=seed + 909,
                        )
                        b_test = evaluate_baseline_binary(
                            dataset,
                            splits.test,
                            history_pool,
                            baseline_model,
                            node_text_t,
                            rel_text_t,
                            sampler,
                            train_cfg,
                            device,
                        )
                        print({"event": "baseline_test_ranking_start", "method": baseline_name, "seed": seed}, flush=True)
                        b_rank = evaluate_baseline_ranking(
                            dataset,
                            splits.test,
                            history_pool,
                            known_all,
                            baseline_model,
                            node_text_t,
                            rel_text_t,
                            train_cfg,
                            filter_scope=train_cfg.filter_scope,
                            max_eval_edges=filtered_rank_edges,
                        )
                    b_row = {
                        "dataset": dataset.name,
                        "method": baseline_name,
                        "scenario": scenario_name,
                        "ablation": "baseline",
                        "few_shot_ratio": ratio,
                        "seed": seed,
                        "train_edges": len(splits.train),
                        "val_edges": len(splits.val),
                        "test_edges": len(splits.test),
                        "params_trainable": count_parameters(baseline_model, trainable_only=True),
                        "params_total": count_parameters(baseline_model, trainable_only=False),
                        "train_eval_time_sec": b_timer.elapsed,
                        **b_val_metrics,
                        **{f"test_{k}": v for k, v in b_test.items()},
                        **{f"test_{k}": v for k, v in b_rank.items()},
                    }
                    all_results.append(b_row)
                    with open(partial_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(b_row, ensure_ascii=False, sort_keys=True) + "\n")
                    print(b_row)
            summary = summarize_seed_metrics(seed_rows)
            save_json(
                {
                    "config": cfg,
                    "environment": env_report(),
                    "summary": summary,
                    "rows": all_results,
                    "temporal_split": {
                        "val_start_time": base_splits.val_start_time,
                        "test_start_time": base_splits.test_start_time,
                    },
                    "scenario": {"name": scenario_name, **scenario_meta},
                    "leakage_control": {
                        "history_for_train_batch": "edges with time strictly before batch min time",
                        "history_for_val": "train edges only, further restricted causally per query batch",
                        "history_for_test": "train+validation edges only, further restricted causally per query batch",
                        "text_encoder": "label-free frozen/cache-only embeddings",
                        "filtered_eval": True,
                    },
                },
                Path(out_dir) / f"{dataset.name}_{ablation_name}_ratio{ratio}.json",
            )
    save_json({"environment": env_report(), "rows": all_results}, Path(out_dir) / "all_results.json")


if __name__ == "__main__":
    main()
