"""Synthetic catalogue used by the offline unit tests (no network)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from starter.state import CSR, RuntimePolicy, Snapshot


def make_csr(n: int, neighbours: int = 3) -> CSR:
    indices = []
    data = []
    indptr = [0]
    for i in range(n):
        js = np.asarray([(i + k + 1) % n for k in range(neighbours)],
                        dtype=np.int32)
        vals = np.linspace(0.5, 0.1, neighbours).astype(np.float32)
        vals /= vals.sum()
        indices.append(js)
        data.append(vals)
        indptr.append(indptr[-1] + js.size)
    return CSR(np.asarray(indptr, dtype=np.int64),
               np.concatenate(indices).astype(np.int32),
               np.concatenate(data).astype(np.float32), n)


def make_snapshot(n: int = 5000, d: int = 512, seed: int = 7,
                  new_subcats: Optional[Tuple[int, ...]] = None) -> Snapshot:
    rng = np.random.default_rng(seed)
    ids = ["p_%06d" % i for i in range(n)]
    index = {pid: i for i, pid in enumerate(ids)}
    E = rng.normal(size=(n, d)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    category = rng.integers(0, 10, n).astype(np.int16)
    subcategory = rng.integers(0, 90, n).astype(np.int16)
    brand = rng.integers(0, 30, n).astype(np.int16)
    price = rng.uniform(2.0, 120.0, n).astype(np.float32)
    active = np.ones(n, dtype=bool)
    unavailable = np.zeros(n, dtype=bool)
    is_new = np.zeros(n, dtype=bool)
    alpha0 = rng.uniform(1.0, 5.0, n).astype(np.float32)
    beta0 = rng.uniform(500.0, 1500.0, n).astype(np.float32)
    s_slow = rng.uniform(0.0, 5.0, n).astype(np.float32)
    f_slow = rng.uniform(0.0, 100.0, n).astype(np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    # A realistic Laplace posterior is much tighter than the prior.
    Sigma = 0.05 * np.eye(24, dtype=np.float32)
    Sigma += np.full((24, 24), 0.001, dtype=np.float32)
    prices = price[price > 0]
    snap = Snapshot(
        ids=ids, index=index, n=n, category=category, subcategory=subcategory,
        brand=brand, category_names=tuple("cat%d" % i for i in range(10)),
        subcategory_names=tuple("sub%d" % i for i in range(90)),
        brand_names=tuple("brand%d" % i for i in range(30)),
        price=price, active=active, unavailable=unavailable, is_new=is_new,
        alpha0=alpha0, beta0=beta0, alpha=alpha0 + s_slow,
        beta=beta0 + f_slow, s_slow=s_slow, f_slow=f_slow, s_fast=zeros,
        f_fast=zeros, s_click=zeros, f_click=zeros, s_org=zeros, f_org=zeros,
        org_v=rng.uniform(0, 100, n).astype(np.float32),
        org_atc=rng.uniform(0, 10, n).astype(np.float32),
        org_p=rng.uniform(0, 2, n).astype(np.float32),
        feat_view=np.log1p(rng.uniform(0, 100, n)).astype(np.float32),
        feat_atc=rng.normal(size=n).astype(np.float32),
        feat_q=rng.normal(size=n).astype(np.float32),
        price_pct=rng.random(n).astype(np.float32),
        price_pct_subcat=rng.random(n).astype(np.float32), E=E,
        trans_view=make_csr(n), trans_buy=make_csr(n),
        mu_beta=np.zeros(24, dtype=np.float32), Sigma_beta=Sigma,
        feature_names=tuple("f%d" % i for i in range(24)),
        feature_mean=np.zeros(24, dtype=np.float32),
        feature_std=np.ones(24, dtype=np.float32),
        w=np.asarray([1.0, 0.72, 0.52, 0.39, 0.30, 0.23, 0.18, 0.14, 0.11,
                      0.08], dtype=np.float32),
        trend_boost=np.zeros(n, dtype=np.float32),
        new_bonus=np.zeros(n, dtype=np.float32),
        restock_bonus=np.zeros(n, dtype=np.float32),
        price_bonus=np.zeros(n, dtype=np.float32),
        best_safe=np.arange(20, dtype=np.int64), p95_price=float(
            np.percentile(prices, 95)),
        new_subcats=frozenset(new_subcats or ()),
        policy=RuntimePolicy())
    return snap
