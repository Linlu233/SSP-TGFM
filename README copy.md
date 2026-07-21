# SSP-TGFM

Research implementation for **Semantic-Structural Prompted Temporal Graph Foundation Model** from `SSP-TGFM.md` / `SSP-TGFM.pdf`.

The code follows the document constraints:

- strict chronological train/validation/test split;
- causal history for every train/eval batch: only events with time strictly before the query batch;
- random, historical, inductive, and filtered negative sampling modes;
- filtered ranking metrics for temporal KG-style link prediction;
- frozen/cache-only text encoder interface;
- semantic branch, structural temporal branch, relation prompts, anti-hallucination gate, alignment loss, and variational semantic prior;
- multi-seed summaries with mean/std/95% CI.

## Environment

The current workspace has CUDA PyTorch installed and verified on the V100 GPU.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy scipy pandas scikit-learn pyyaml tqdm pypdf
```

Optional frozen neural text embeddings:

```bash
python -m pip install sentence-transformers
```

Optional TGB datasets:

```bash
python -m pip install py-tgb
```

For baseline comparisons such as TGN/TGAT/TComplEx/TeRo/RE-NET, use their official implementations or add wrappers under `scripts/` while preserving the same split and evaluation protocol.

## Run

CUDA smoke run:

```bash
PYTHONPATH=. python scripts/run_ssptgfm.py --config configs/ssptgfm_smoke.yaml
```

Main synthetic experiment with 5 seeds:

```bash
PYTHONPATH=. python scripts/run_ssptgfm.py --config configs/ssptgfm_synthetic.yaml
```

Required ablation/few-shot scaffold:

```bash
PYTHONPATH=. python scripts/run_ssptgfm.py --config configs/ssptgfm_ablation.yaml
```

CSV dataset format:

```text
dataset_dir/
  edges.csv      # columns: src,dst,rel,time; optional split=train|val|test
  nodes.csv      # optional columns: id,text
  relations.csv  # optional columns: id,text
```

Set:

```yaml
data:
  name: my_dataset
  format: csv
  path: data/raw/my_dataset
```

Temporal KG conversion from common `head relation tail timestamp` files:

```bash
PYTHONPATH=. python scripts/convert_tkg_to_csv.py \
  --input-dir data/raw/ICEWS14 \
  --out data/raw/icews14_csv
```

Then copy `configs/ssptgfm_csv_template.yaml` and point `data.path` to the converted directory.
If you want to use official split labels in `edges.csv`, set `split.mode: labels`; the loader will still assert strict chronological order.

TGB/TGB 2.0 loader example:

```bash
PYTHONPATH=. python scripts/run_ssptgfm.py --config configs/ssptgfm_tgb_template.yaml
```

Result summary:

```bash
PYTHONPATH=. python scripts/summarize_results.py --results results/ssptgfm_synthetic/all_results.json
```

Evidence export for the document's explanation-quality check:

```bash
PYTHONPATH=. python scripts/explain_predictions.py --config configs/ssptgfm_smoke.yaml --out results/explanations.json
```

Protocol smoke for exact k-shot, hallucination stress, and lightweight baselines:

```bash
PYTHONPATH=. python scripts/run_ssptgfm.py --config configs/ssptgfm_protocol_smoke.yaml
```

## Leakage Controls

Validation history is restricted to training events, and test history is restricted to training plus validation events. Both paths are additionally filtered by query time, so an event at the same or future timestamp is never used as a neighbor/message for its own prediction.

Text embeddings are built from text only. Labels and split membership are not passed to the text encoder. If `sentence-transformers` is used, the encoder is frozen and embeddings are cached with backend/model metadata.

## Metrics

The runner reports binary AUC/AP/NDCG using sampled negatives and filtered MRR/Hits@1/10/50/100 by corrupting both head and tail while filtering known facts. Results are written to `results/.../*.json`.

Formula-level implementation notes are in `FORMULA_AUDIT.md`.

## Scope Notes

This repository contains the SSP-TGFM model and protocol harness. Publication-ready claims still require running the official datasets named in the document, adding official baseline wrappers, and recording exact dataset/model versions plus hyperparameter searches.
