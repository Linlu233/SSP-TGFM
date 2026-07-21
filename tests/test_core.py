from __future__ import annotations

import torch

from ssptgfm.data import EdgeTensor, TemporalDataset, generate_synthetic_dataset, limit_train_edges, split_by_labels, split_by_time
from ssptgfm.baseline_training import _history_index_by_time, build_baseline, evaluate_baseline_binary
from ssptgfm.features import GraphHistoryIndex
from ssptgfm.explain import explain_edge
from ssptgfm.experiment_splits import build_scenario, exact_k_shot_train
from ssptgfm.model import SSPTGFM
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler
from ssptgfm.training import (
    TrainConfig,
    _batch_rank_loss,
    _candidate_listwise_rank_loss,
    combine_scores_for_binary_eval,
    _meta_episode_loss,
    _select_eval_edges,
    _struct_features,
    _time_coherent_batches,
    _time_grouped_batches,
    causal_history_for_batch,
    train_one_seed,
)
from scripts.run_ssptgfm import validate_full_formula_config
from scripts.search_hparams import deep_update


def _strict_cfg() -> dict:
    return {
        "model": {
            "prompt_tokens": 2,
            "prompt_heads": 2,
            "relation_rank": 4,
            "adapter_rank": 4,
            "temporal_layers": 1,
            "use_struct": True,
            "use_sem": True,
            "use_cross": True,
            "use_gate": True,
            "use_variational": True,
        },
        "train": {
            "lambda_align": 0.05,
            "lambda_kl": 0.0005,
            "lambda_meta": 0.01,
            "lambda_ood": 0.001,
            "meta_lr": 0.01,
            "meta_support_size": 4,
        },
    }


def test_strict_full_formula_config_requires_all_modules_and_losses() -> None:
    cfg = _strict_cfg()
    validate_full_formula_config(cfg, context="valid")
    cfg["model"]["use_gate"] = False
    try:
        validate_full_formula_config(cfg, context="invalid")
    except ValueError as exc:
        assert "model.use_gate must be true" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict config with a disabled formula module should fail")


def test_strict_full_formula_config_requires_positive_meta_and_ood_losses() -> None:
    cfg = _strict_cfg()
    cfg["train"]["lambda_meta"] = 0.0
    cfg["train"]["lambda_ood"] = 0.0
    try:
        validate_full_formula_config(cfg, context="invalid")
    except ValueError as exc:
        message = str(exc)
        assert "train.lambda_meta must be > 0" in message
        assert "train.lambda_ood must be > 0" in message
    else:  # pragma: no cover
        raise AssertionError("strict config with zero formula losses should fail")


def test_temporal_split_is_strict() -> None:
    dataset = generate_synthetic_dataset(num_nodes=32, num_relations=3, num_edges=240, seed=4)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    splits.assert_no_temporal_leakage()
    assert float(splits.train.time.max()) < splits.val_start_time
    assert float(splits.val.time.min()) >= splits.val_start_time
    assert float(splits.val.time.max()) < splits.test_start_time
    assert float(splits.test.time.min()) >= splits.test_start_time


def test_temporal_uniform_train_limit_stays_inside_train_window() -> None:
    train = EdgeTensor.from_arrays(range(10), range(1, 11), [0] * 10, range(10))
    limited = limit_train_edges(train, 4, mode="temporal_uniform")
    assert len(limited) == 4
    assert limited.time.tolist() == [0.0, 3.0, 6.0, 9.0]
    assert float(limited.time.max()) <= float(train.time.max())


def test_causal_history_excludes_batch_time() -> None:
    edges = EdgeTensor.from_arrays([0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 0, 0], [1, 2, 3, 4])
    batch = edges.slice([2, 3])
    hist = causal_history_for_batch(edges, batch)
    assert len(hist) == 2
    assert bool(torch.all(hist.time < 3.0))


def test_time_coherent_batches_do_not_mix_query_times() -> None:
    edges = EdgeTensor.from_arrays([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [1, 1, 2, 2, 3])
    batches = _time_coherent_batches(edges, batch_size=8, shuffle=False, seed=1)
    assert len(batches) == 3
    for idx in batches:
        assert torch.unique(edges.time[idx]).numel() == 1
    grouped = _time_grouped_batches(edges, batch_size=8, shuffle_within_time=False, seed=1)
    assert [time for time, _ in grouped] == [1.0, 2.0, 3.0]
    assert [len(group_batches) for _, group_batches in grouped] == [1, 1, 1]


def test_filtered_negative_sampler_avoids_known_fact() -> None:
    edges = EdgeTensor.from_arrays([0, 0, 1], [1, 2, 2], [0, 0, 0], [1, 1, 2])
    known = KnownFacts.from_edges(edges)
    sampler = NegativeSampler(4, known, edges, mode="filtered", filter_scope="exact", seed=1)
    neg = sampler.sample(edges.slice([0]), num_neg_per_pos=20, history=edges.before(1, strict=True))
    for s, r, d, t in neg.tuples(include_time=True):
        assert not known.contains(s, r, d, t, "exact")


def test_relation_hard_negative_sampler_uses_relation_entity_pool() -> None:
    edges = EdgeTensor.from_arrays([0, 2, 3, 4], [1, 3, 4, 5], [0, 0, 1, 1], [1, 2, 3, 4])
    known = KnownFacts.from_edges(edges)
    sampler = NegativeSampler(8, known, edges, mode="relation_hard", filter_scope="exact", seed=2)
    neg = sampler.sample(edges.slice([0]), num_neg_per_pos=20, history=edges)
    rel0_heads = {0, 2}
    rel0_tails = {1, 3}
    assert len(neg) == 20
    assert any(int(s) in rel0_heads or int(d) in rel0_tails for s, d in zip(neg.src, neg.dst))
    for s, r, d, t in neg.tuples(include_time=True):
        assert not known.contains(s, r, d, t, "exact")


def test_negative_sampler_candidate_nodes_are_filtered_and_unique() -> None:
    edges = EdgeTensor.from_arrays([0, 0, 1], [1, 2, 2], [0, 0, 0], [1, 1, 2])
    known = KnownFacts.from_edges(edges)
    sampler = NegativeSampler(6, known, edges, mode="filtered", filter_scope="exact", seed=11)
    nodes = sampler.sample_candidate_nodes(0, 1, 0, 1.0, corrupt_head=False, num_negatives=3)
    assert len(nodes) == 3
    assert len(set(nodes)) == len(nodes)
    assert 1 not in nodes
    for node in nodes:
        assert node != 0
        assert not known.contains(0, 0, node, 1.0, "exact")


def test_filtered_candidate_mask_matches_contains_logic() -> None:
    edges = EdgeTensor.from_arrays([0, 0, 1], [1, 2, 2], [0, 0, 0], [1, 1, 2])
    known = KnownFacts.from_edges(edges)
    keep, true_index = known.filtered_nodes_for_query(0, 0, 1, 1.0, corrupt_head=False, num_nodes=4, scope="exact")
    candidates = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
    expected = []
    expected_true = None
    for node in range(4):
        is_true = node == 1
        if node == 0:
            continue
        if not is_true and known.contains(0, 0, node, 1.0, "exact"):
            continue
        if is_true:
            expected_true = len(expected)
        expected.append(node)
    assert candidates == expected
    assert true_index == expected_true


def test_validation_known_facts_exclude_test_edges() -> None:
    train = EdgeTensor.from_arrays([0], [1], [0], [1.0])
    val = EdgeTensor.from_arrays([1], [2], [0], [2.0])
    test = EdgeTensor.from_arrays([2], [3], [0], [3.0])
    known_val = KnownFacts.from_edges(train.concat(val))
    known_test = KnownFacts.from_edges(train.concat(val).concat(test))
    assert not known_val.contains(2, 0, 3, 3.0, "exact")
    assert known_test.contains(2, 0, 3, 3.0, "exact")


def test_candidate_features_match_edge_features() -> None:
    hist = EdgeTensor.from_arrays([0, 1, 2, 0], [1, 2, 3, 2], [0, 0, 0, 0], [1, 2, 3, 4])
    candidates = [(0, 3, 0, 5.0), (1, 3, 0, 5.0), (3, 0, 0, 5.0)]
    index = GraphHistoryIndex(4, hist)
    batch = EdgeTensor.from_arrays(
        [x[0] for x in candidates],
        [x[1] for x in candidates],
        [x[2] for x in candidates],
        [x[3] for x in candidates],
    )
    sf_edges, ood_edges = index.features_for_edges(batch)
    sf_candidates, ood_candidates = index.features_for_candidates(0, 3, candidates)
    assert torch.allclose(sf_edges, sf_candidates, atol=1e-6)
    assert torch.equal(ood_edges, ood_candidates)
    sf_tensor, ood_tensor = index.features_for_candidates_tensor(candidates, torch.device("cpu"))
    assert torch.allclose(sf_edges, sf_tensor.cpu(), atol=1e-6)
    assert torch.equal(ood_edges, ood_tensor.cpu())
    sf_sparse, ood_sparse = index.features_for_edges_sparse_ppr(batch, torch.device("cpu"))
    assert torch.allclose(sf_edges, sf_sparse.cpu(), atol=1e-6)
    assert torch.equal(ood_edges, ood_sparse.cpu())


def test_sparse_ppr_column_path_matches_edge_features() -> None:
    hist = EdgeTensor.from_arrays(
        [0, 1, 2, 3, 4, 5, 6, 0, 2],
        [1, 2, 3, 4, 5, 6, 7, 7, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    candidates = [(node, 7, 0, 10.0) for node in range(8)]
    index = GraphHistoryIndex(8, hist)
    batch = EdgeTensor.from_arrays(
        [x[0] for x in candidates],
        [x[1] for x in candidates],
        [x[2] for x in candidates],
        [x[3] for x in candidates],
    )
    sf_expected, ood_expected = index.features_for_edges(batch)
    sf_sparse, ood_sparse = index.features_for_edges_sparse_ppr(batch, torch.device("cpu"))
    assert torch.allclose(sf_expected, sf_sparse.cpu(), atol=1e-6)
    assert torch.equal(ood_expected, ood_sparse.cpu())


def test_history_prior_features_keep_legacy_dims_and_are_causal() -> None:
    hist = EdgeTensor.from_arrays([0, 0, 1], [1, 1, 2], [0, 0, 0], [1.0, 3.0, 4.0])
    index = GraphHistoryIndex(4, hist.before(3.0, strict=True))
    edge = EdgeTensor.from_arrays([0], [1], [0], [3.0])
    f12 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=12)
    f20 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=20)
    f28 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=28)
    f40 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=40)
    f42 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=42)
    f48 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=48)
    f64 = index.history_prior_features_for_edges(edge, torch.device("cpu"), feature_dim=64)
    assert f12.shape == (1, 12)
    assert f20.shape == (1, 20)
    assert f28.shape == (1, 28)
    assert f40.shape == (1, 40)
    assert f42.shape == (1, 42)
    assert f48.shape == (1, 48)
    assert f64.shape == (1, 64)
    assert torch.allclose(f12, f42[:, :12])
    assert torch.allclose(f20, f42[:, :20])
    assert torch.allclose(f28, f42[:, :28])
    assert torch.allclose(f40, f42[:, :40])
    assert torch.allclose(f42, f48[:, :42])
    assert torch.allclose(f48, f64[:, :48])
    assert torch.isclose(f42[0, 0], torch.log1p(torch.tensor(1.0)))
    assert f42[0, 20] > 0.0
    assert f42[0, 28] > 0.0
    assert f48[0, 42] > 0.0
    assert torch.isfinite(f64).all()
    assert f64[0, 48] > 0.0


def test_struct_features_preserve_edge_field_order_on_tensor_path() -> None:
    hist = EdgeTensor.from_arrays([0, 2, 3], [2, 3, 4], [0, 0, 0], [1, 2, 3])
    batch = EdgeTensor.from_arrays([0, 4], [4, 0], [1, 1], [5.0, 5.0])
    index = GraphHistoryIndex(5, hist)
    sf_expected, ood_expected = index.features_for_edges(batch)
    sf_actual, ood_actual = _struct_features(index, batch, torch.device("cpu"))
    assert torch.allclose(sf_expected, sf_actual.cpu(), atol=1e-6)
    assert torch.equal(ood_expected, ood_actual.cpu())


def test_model_forward_shapes() -> None:
    dataset = generate_synthetic_dataset(num_nodes=32, num_relations=3, num_edges=240, seed=5)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(10, 18))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    out = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood))
    assert out.final_score.shape == (len(batch),)
    assert out.gate.shape == (len(batch), 1)
    expected = out.gate.squeeze(-1) * out.struct_score + (1.0 - out.gate.squeeze(-1)) * out.sem_score
    assert torch.allclose(out.final_score, expected, atol=1e-6)
    assert out.struct_feature_score is None


def test_struct_feature_residual_adds_causal_feature_score() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=3, num_edges=160, seed=15)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(8, 14))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
        use_struct_feature_residual=True,
        struct_feature_hidden_dim=16,
        struct_feature_init_scale=0.25,
    )
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    out = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood))
    assert out.struct_feature_score is not None
    gated = out.gate.squeeze(-1) * out.struct_score + (1.0 - out.gate.squeeze(-1)) * out.sem_score
    expected = gated + model.struct_feature_scale * out.struct_feature_score
    assert torch.allclose(out.final_score, expected, atol=1e-6)


def test_history_prior_gate_scales_history_prior_contribution() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=3, num_edges=160, seed=16)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(8, 14))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    hp = index.history_prior_features_for_edges(batch, torch.device("cpu"), feature_dim=12)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
        use_history_prior=True,
        history_prior_dim=12,
        history_prior_init_scale=0.5,
        use_history_prior_gate=True,
        history_prior_gate_hidden_dim=8,
    )
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    out = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood), history_prior_features=hp)
    assert out.history_prior_score is not None
    assert out.history_prior_gate is not None
    assert torch.all(out.history_prior_gate >= 0.0)
    assert torch.all(out.history_prior_gate <= 1.0)
    gated = out.gate.squeeze(-1) * out.struct_score + (1.0 - out.gate.squeeze(-1)) * out.sem_score
    expected = gated + model.history_prior_scale * out.history_prior_gate * out.history_prior_score
    assert torch.allclose(out.final_score, expected, atol=1e-6)


def test_history_prior_gate_without_history_prior_keeps_base_score() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=3, num_edges=160, seed=17)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(8, 14))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
        use_history_prior=False,
        use_history_prior_gate=True,
    )
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    out = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood))
    assert out.history_prior_score is None
    assert out.history_prior_gate is None
    expected = out.gate.squeeze(-1) * out.struct_score + (1.0 - out.gate.squeeze(-1)) * out.sem_score
    assert torch.allclose(out.final_score, expected, atol=1e-6)


def test_relation_entity_prior_adds_relation_conditioned_entity_score() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=3, num_edges=160, seed=18)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(8, 14))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
        use_relation_entity_prior=True,
        relation_entity_prior_rank=4,
        relation_entity_prior_init_scale=0.3,
    )
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    out = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood))
    assert out.relation_entity_prior_score is not None
    gated = out.gate.squeeze(-1) * out.struct_score + (1.0 - out.gate.squeeze(-1)) * out.sem_score
    expected = gated + model.relation_entity_prior_scale * out.relation_entity_prior_score
    assert torch.allclose(out.final_score, expected, atol=1e-6)


def test_lm_baseline_does_not_compute_unused_struct_features(monkeypatch) -> None:
    dataset = generate_synthetic_dataset(num_nodes=16, num_relations=2, num_edges=80, seed=12)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    model = build_baseline("lm_mlp", dataset, text_dim=8, hidden_dim=12)
    node_text = torch.randn(dataset.num_nodes, 8)
    rel_text = torch.randn(dataset.num_relations, 8)
    sampler = NegativeSampler(
        dataset.num_nodes,
        KnownFacts.from_edges(splits.train.concat(splits.val)),
        splits.train,
        mode="filtered",
        filter_scope="exact",
        seed=3,
    )

    def fail_struct(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("lm_mlp must not compute structural features")

    monkeypatch.setattr("ssptgfm.baseline_training._struct_features", fail_struct)
    metrics = evaluate_baseline_binary(
        dataset,
        splits.val,
        splits.train,
        model,
        node_text,
        rel_text,
        sampler,
        TrainConfig(epochs=1, batch_size=16, num_neg_eval=2),
        torch.device("cpu"),
    )
    assert "auc" in metrics


def test_binary_eval_can_limit_validation_edges() -> None:
    dataset = generate_synthetic_dataset(num_nodes=16, num_relations=2, num_edges=120, seed=15)
    splits = split_by_time(dataset.edges, val_ratio=0.2, test_ratio=0.2)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=8,
        hidden_dim=16,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    sampler = NegativeSampler(
        dataset.num_nodes,
        KnownFacts.from_edges(splits.train.concat(splits.val)),
        splits.train,
        mode="filtered",
        filter_scope="exact",
        seed=5,
    )
    labels, scores = combine_scores_for_binary_eval(
        model,
        splits.val,
        splits.train,
        sampler,
        torch.randn(dataset.num_nodes, 8),
        torch.randn(dataset.num_relations, 8),
        torch.device("cpu"),
        num_neg_per_pos=2,
        batch_size=8,
        num_relations=dataset.num_relations,
        max_eval_edges=3,
    )
    assert labels.shape == scores.shape
    assert labels.shape[0] == 9


def test_batch_rank_loss_supports_bpr_and_softplus() -> None:
    scores = torch.tensor([2.0, 1.0, 0.0, -1.0])
    hinge = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=1.0, loss_type="hinge")
    bpr = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.0, loss_type="bpr")
    softplus = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.5, loss_type="softplus")
    sampled_softmax = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.0, loss_type="sampled_softmax")
    adv_bpr = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.5, loss_type="adv_bpr")
    assert float(hinge) >= 0.0
    assert float(bpr) > 0.0
    assert float(softplus) > 0.0
    assert float(sampled_softmax) > 0.0
    assert float(adv_bpr) > 0.0


def test_candidate_listwise_rank_loss_uses_model_score_path_and_backpropagates() -> None:
    dataset = generate_synthetic_dataset(num_nodes=20, num_relations=2, num_edges=120, seed=19)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    pos = splits.train.slice(slice(8, 10))
    hist = causal_history_for_batch(splits.train, pos)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sampler = NegativeSampler(dataset.num_nodes, KnownFacts.from_edges(splits.train), splits.train, mode="filtered", seed=4)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=12,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
        use_history_prior=True,
        history_prior_dim=12,
        history_prior_init_scale=0.1,
    )
    node_text = torch.randn(dataset.num_nodes, 12)
    rel_text = torch.randn(dataset.num_relations, 12)
    context = model.encode_context(hist, pos.time, node_text, rel_text)
    relation_counts = torch.bincount(hist.rel.cpu(), minlength=dataset.num_relations).float()
    loss = _candidate_listwise_rank_loss(
        model,
        pos,
        index,
        sampler,
        context,
        torch.device("cpu"),
        dataset.num_relations,
        relation_counts,
        candidate_rank_size=4,
        candidate_rank_sides="both",
        candidate_rank_queries=1,
    )
    assert loss.requires_grad
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_rank_loss_variants_backpropagate() -> None:
    scores = torch.tensor([2.0, 1.0, 0.0, -1.0], requires_grad=True)
    loss = _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.0, loss_type="sampled_softmax")
    loss = loss + _batch_rank_loss(scores, pos_count=2, neg_count=2, margin=0.5, loss_type="adv_bpr")
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_train_config_supports_temporal_uniform_validation_sampling() -> None:
    cfg = TrainConfig(val_eval_sample="temporal_uniform")
    assert cfg.val_eval_sample == "temporal_uniform"


def test_temporal_uniform_single_eval_edge_uses_middle_time() -> None:
    edges = EdgeTensor.from_arrays(range(7), range(1, 8), [0] * 7, range(7))
    selected = _select_eval_edges(edges, max_eval_edges=1, eval_sample="temporal_uniform")
    assert len(selected) == 1
    assert selected.time.tolist() == [3.0]


def test_search_deep_update_preserves_nested_config() -> None:
    cfg = {"train": {"epochs": 1, "num_neg_eval": 5}, "search": {"train_edge_limit": 10}}
    updated = deep_update(cfg, {"train": {"epochs": 2}})
    assert updated["train"]["epochs"] == 2
    assert updated["train"]["num_neg_eval"] == 5
    assert updated["search"]["train_edge_limit"] == 10


def test_baseline_history_index_cache_uses_strict_causal_history() -> None:
    history = EdgeTensor.from_arrays([0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 0, 0], [1.0, 2.0, 2.0, 4.0])
    queries = EdgeTensor.from_arrays([0, 2], [2, 4], [0, 0], [2.0, 4.0])
    cache = _history_index_by_time(5, history, queries)
    hist_2, _ = cache[2.0]
    hist_4, _ = cache[4.0]
    assert len(hist_2) == 1
    assert bool(torch.all(hist_2.time < 2.0))
    assert len(hist_4) == 3
    assert bool(torch.all(hist_4.time < 4.0))


def test_cached_context_scores_match_forward() -> None:
    dataset = generate_synthetic_dataset(num_nodes=32, num_relations=3, num_edges=240, seed=6)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    batch = splits.train.slice(slice(10, 18))
    hist = causal_history_for_batch(splits.train, batch)
    index = GraphHistoryIndex(dataset.num_nodes, hist)
    sf, ood = index.features_for_edges(batch)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=16,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    model.eval()
    node_text = torch.randn(dataset.num_nodes, 16)
    rel_text = torch.randn(dataset.num_relations, 16)
    direct = model(batch, hist, node_text, rel_text, sf, ood_s=ood, ood_e=torch.zeros_like(ood))
    ctx = model.encode_context(hist, batch.time, node_text, rel_text)
    cached = model.score_edges_from_context(batch, sf, ood, torch.zeros_like(ood), ctx)
    assert torch.allclose(direct.final_score, cached.final_score, atol=1e-6)


def test_label_split_keeps_strict_time_order() -> None:
    edges = EdgeTensor.from_arrays([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [0] * 6, [0, 1, 2, 3, 4, 5])
    dataset = TemporalDataset(
        name="labeled",
        num_nodes=6,
        num_relations=1,
        edges=edges,
        node_texts=[f"node {i}" for i in range(6)],
        relation_texts=["relation"],
        edge_split=torch.tensor([0, 0, 0, 1, 1, 2]).numpy().astype(str),
    )
    dataset.edge_split = torch.tensor([0, 0, 0, 1, 1, 2]).numpy().astype(str)
    dataset.edge_split = ["train", "train", "train", "val", "val", "test"]
    splits = split_by_labels(dataset)
    splits.assert_no_temporal_leakage()
    assert len(splits.train) == 3
    assert len(splits.val) == 2
    assert len(splits.test) == 1


def test_explanation_has_path_and_terms() -> None:
    dataset = generate_synthetic_dataset(num_nodes=16, num_relations=2, num_edges=80, seed=7)
    edge = EdgeTensor.from_arrays([0], [2], [0], [3])
    history = EdgeTensor.from_arrays([0, 1], [1, 2], [0, 0], [1, 2])
    exp = explain_edge(dataset, history, edge)
    assert exp["structural_path"] == [0, 1, 2]
    assert exp["semantic_terms"]


def test_meta_episode_loss_backpropagates() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=2, num_edges=120, seed=9)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    pos = splits.train.slice(slice(10, 22))
    hist = causal_history_for_batch(splits.train, pos)
    known = KnownFacts.from_edges(splits.train)
    neg = NegativeSampler(dataset.num_nodes, known, splits.train, mode="filtered", seed=3).sample(pos, 1, hist)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=12,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    node_text = torch.randn(dataset.num_nodes, 12)
    rel_text = torch.randn(dataset.num_relations, 12)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    cfg = TrainConfig(lambda_meta=0.01, meta_support_size=4)
    loss = _meta_episode_loss(
        model,
        pos,
        neg,
        hist,
        labels,
        node_text,
        rel_text,
        cfg,
        dataset.num_relations,
        torch.device("cpu"),
    )
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if "prompts" in n or "text_proj" in n)


def test_meta_episode_loss_backpropagates_for_single_positive() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=2, num_edges=120, seed=14)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    pos = splits.train.slice([10])
    hist = causal_history_for_batch(splits.train, pos)
    known = KnownFacts.from_edges(splits.train)
    neg = NegativeSampler(dataset.num_nodes, known, splits.train, mode="filtered", seed=3).sample(pos, 1, hist)
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=12,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    node_text = torch.randn(dataset.num_nodes, 12)
    rel_text = torch.randn(dataset.num_relations, 12)
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    cfg = TrainConfig(lambda_meta=0.01, meta_support_size=4)
    loss = _meta_episode_loss(
        model,
        pos,
        neg,
        hist,
        labels,
        node_text,
        rel_text,
        cfg,
        dataset.num_relations,
        torch.device("cpu"),
    )
    assert loss.requires_grad
    assert float(loss.detach()) > 0.0


def test_exact_k_shot_train_by_relation() -> None:
    dataset = generate_synthetic_dataset(num_nodes=32, num_relations=3, num_edges=240, seed=10)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    shot = exact_k_shot_train(splits.train, k=2, by="relation")
    counts = torch.bincount(shot.rel, minlength=dataset.num_relations)
    assert int(counts.max()) <= 2
    assert len(shot) > 0


def test_train_one_seed_updates_with_empty_initial_history() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=2, num_edges=120, seed=13)
    splits = split_by_time(dataset.edges, val_ratio=0.15, test_ratio=0.15)
    splits.train = exact_k_shot_train(splits.train, k=1, by="relation")
    model = SSPTGFM(
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        text_dim=12,
        hidden_dim=24,
        time_dim=8,
        prompt_tokens=2,
        prompt_heads=2,
        relation_rank=4,
        adapter_rank=4,
        temporal_layers=1,
    )
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    node_text = torch.randn(dataset.num_nodes, 12)
    rel_text = torch.randn(dataset.num_relations, 12)
    cfg = TrainConfig(
        epochs=1,
        batch_size=16,
        lr=0.001,
        num_neg_train=1,
        num_neg_eval=2,
        lambda_align=0.01,
        lambda_kl=0.0001,
        lambda_meta=0.01,
        lambda_ood=0.001,
        patience=1,
    )
    trained, _ = train_one_seed(dataset, splits, model, node_text, rel_text, cfg, torch.device("cpu"), seed=1)
    assert any(not torch.allclose(before[name], param.detach()) for name, param in trained.state_dict().items())


def test_new_relation_scenario_removes_holdout_from_train() -> None:
    dataset = generate_synthetic_dataset(num_nodes=40, num_relations=4, num_edges=300, seed=11)
    _, scenario = build_scenario(dataset, {"name": "new_relation", "holdout_ratio": 0.25, "seed": 1}, 0.15, 0.15, 1)
    holdout = set(scenario.metadata["holdout_relations"])
    assert holdout
    assert all(int(r) not in holdout for r in scenario.splits.train.rel.tolist())
    scenario.splits.assert_no_temporal_leakage()


def test_hallucination_stress_changes_text_not_edges() -> None:
    dataset = generate_synthetic_dataset(num_nodes=24, num_relations=2, num_edges=120, seed=12)
    stressed, scenario = build_scenario(dataset, {"name": "hallucination_stress"}, 0.15, 0.15, 1)
    assert scenario.name == "hallucination_stress"
    assert stressed.node_texts != dataset.node_texts
    assert torch.equal(stressed.edges.src, dataset.edges.src)
