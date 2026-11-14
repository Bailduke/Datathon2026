"""Deterministic text features: tokenisation + hashed TF-IDF embeddings.

Spec Section 2.5.5.  The hashing must be stable across processes and machines,
therefore ``zlib.crc32`` is used instead of Python's salted ``hash()``.  New
products can be embedded online with exactly the same function that produced
the offline matrix.
"""
from __future__ import annotations

import re
import zlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9àèéíòóúüçñ]{2,}", re.UNICODE)


def tokenize(text: Any) -> List[str]:
    """Lower-case unigrams and bigrams (deterministic order)."""
    if text is None:
        return []
    tokens = _TOKEN_RE.findall(str(text).lower())
    if not tokens:
        return []
    out = list(tokens)
    out.extend("%s_%s" % (a, b) for a, b in zip(tokens, tokens[1:]))
    return out


def _weighted_tokens(product: Mapping[str, Any]) -> Dict[str, float]:
    """Token -> term frequency, with structural tokens weighted by 2."""
    counts: Dict[str, float] = {}
    description = product.get("description") or ""
    for token in tokenize(description):
        counts[token] = counts.get(token, 0.0) + 1.0
    for key, prefix in (("category", "cat"), ("subcategory", "sub"),
                        ("brand", "brand")):
        value = product.get(key)
        if value is None or str(value).strip() == "":
            continue
        # Structural tokens are deliberately not split by the regex: values
        # such as "Sports & Outdoors" must hash as one stable token.
        token = "%s:%s" % (prefix, str(value).strip().lower())
        counts[token] = counts.get(token, 0.0) + 2.0
    return counts


def _hash(token: str, dim: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % dim


def _sign(token: str) -> float:
    # Bit 31 of crc32 (spec 2.5.5).  CRC32 is stable across processes.
    return 1.0 if (zlib.crc32(token.encode("utf-8")) >> 31) & 1 == 0 else -1.0


def hashed_embedding(product: Mapping[str, Any], dim: int = 512,
                     dtype: Any = np.float32) -> np.ndarray:
    """L2-normalised sparse hashed TF-IDF vector as a dense row."""
    vector = np.zeros(dim, dtype=np.float64)
    for token, tf in _weighted_tokens(product).items():
        h = _hash(token, dim)
        vector[h] += _sign(token) * (1.0 + np.log1p(tf))
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.astype(dtype)


def build_embeddings(products: Sequence[Mapping[str, Any]], dim: int = 512,
                     dtype: Any = np.float32) -> np.ndarray:
    """Embed a catalogue, returning an ``(N, dim)`` float32 matrix."""
    matrix = np.zeros((len(products), dim), dtype=dtype)
    for i, product in enumerate(products):
        matrix[i] = hashed_embedding(product, dim, dtype)
    return matrix


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_cosine(query: np.ndarray, matrix: np.ndarray, k: int = 25,
                 mask: Optional[np.ndarray] = None,
                 min_similarity: float = 0.05) -> np.ndarray:
    """Indices of the top-k cosine neighbours (assumes L2-normalised rows)."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.int64)
    scores = matrix @ query
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, scores.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < scores.size:
        idx = np.argpartition(scores, -k)[-k:]
    else:
        idx = np.arange(scores.size)
    idx = idx[scores[idx] >= min_similarity]
    return idx[np.argsort(scores[idx])[::-1]].astype(np.int64)
