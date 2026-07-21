from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from ssptgfm.utils import ensure_dir


def _cache_key(texts: Sequence[str], backend: str, model_name: str, dim: int) -> str:
    h = hashlib.sha256()
    h.update(backend.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(str(dim).encode("utf-8"))
    for text in texts:
        h.update(str(text).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def encode_texts(
    texts: Sequence[str],
    backend: str = "hashing",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str | Path = "data/processed/text_cache",
    dim: int = 384,
    device: str = "cpu",
    batch_size: int = 64,
    normalize_output: bool = True,
) -> np.ndarray:
    """Encode text without accessing labels or split metadata.

    The sentence-transformers backend is frozen by construction. The hashing
    backend is deterministic and intended for smoke tests or label-free
    datasets that do not ship raw text.
    """
    cache_dir = ensure_dir(cache_dir)
    key = _cache_key(texts, backend, model_name, dim)
    emb_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"
    if emb_path.exists() and meta_path.exists():
        return np.load(emb_path).astype(np.float32)

    if backend == "hashing":
        vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        mat = vectorizer.transform([str(x) for x in texts])
        emb = mat.astype(np.float32).toarray()
        if normalize_output:
            emb = normalize(emb, norm="l2", copy=False).astype(np.float32)
    elif backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers for frozen neural text encoding, "
                "or set text.backend=hashing for smoke tests."
            ) from exc
        model = SentenceTransformer(model_name, device=device)
        model.eval()
        emb = model.encode(
            list(map(str, texts)),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_output,
            show_progress_bar=True,
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown text backend: {backend}")

    np.save(emb_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "model_name": model_name,
                "dim": int(emb.shape[1]),
                "num_texts": len(texts),
                "frozen": True,
                "label_free": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return emb.astype(np.float32)
