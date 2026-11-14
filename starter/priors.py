"""Offline priors: IPS, hierarchical Beta, transitions, Laplace, persistence.

This module contains the mathematics of spec Section 2 and the two-phase
scratch protocol of Section 5.  The SQL that feeds it lives in
``offline_build.py`` so each concern can be read independently.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Tuple

import numpy as np

from .config import CFG, Config
from .rawtree import iso_utc, parse_datetime
from .state import (CSR, LRU, RuntimePolicy, Snapshot, cholesky_lower,
                    decode_array, decode_embedding_chunk,
                    encode_array, encode_embedding_chunk)
from .text import build_embeddings

LOGGER = logging.getLogger("priors")


# --------------------------------------------------------------------------- #
# 2.2 IPS + effective sample size
# --------------------------------------------------------------------------- #
def ips_effective_counts(rows: Sequence[Mapping[str, Any]],
                         w: np.ndarray, prior_mean: float,
                         cfg: Config = CFG) -> Dict[str, np.ndarray]:
    """Turn ``(product, slot, imp, pur, ips_exp, ips_suc, ips_sq)`` rows into
    effective Beta counts using position-corrected, clipped IPS weights.

    Raw IPS counts are *never* returned: only ``s_eff``/``f_eff`` (eq. 5) and
    their ESS, as required by Sections 2.2-2.3.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size < 10:
        w = np.pad(w, (0, 10 - w.size),
                   constant_values=w[-1] if w.size else 0.03)
    w = np.clip(w, cfg.position_w_min, 1.0)
    by_product: Dict[str, Dict[str, float]] = {}
    for row in rows:
        pid = str(row.get("product_id") or row.get("p") or "")
        if not pid:
            continue
        slot = int(row.get("slot") or 1)
        slot = min(max(slot, 1), 10)
        imp = float(row.get("imp") or 0.0)
        pur = float(row.get("pur") or 0.0)
        exp = float(row.get("ips_exp") or 0.0)
        suc = float(row.get("ips_suc") or 0.0)
        sq = float(row.get("ips_sq") or 0.0)
        exp_w = min(exp / w[slot - 1], imp * cfg.omega_max)
        suc_w = min(suc / w[slot - 1], imp * cfg.omega_max)
        sq_w = sq / (w[slot - 1] ** 2)
        bucket = by_product.setdefault(pid, {
            "exposure": 0.0, "success": 0.0, "sq": 0.0,
            "impressions": 0.0, "purchases": 0.0})
        bucket["exposure"] += exp_w
        bucket["success"] += suc_w
        bucket["sq"] += sq_w
        bucket["impressions"] += imp
        bucket["purchases"] += pur

    ids = list(by_product.keys())
    n = len(ids)
    s_eff = np.zeros(n, dtype=np.float32)
    f_eff = np.zeros(n, dtype=np.float32)
    ess = np.zeros(n, dtype=np.float32)
    theta = np.full(n, float(np.clip(prior_mean, 1e-4, 1 - 1e-4)),
                    dtype=np.float32)
    for i, pid in enumerate(ids):
        b = by_product[pid]
        exp, suc, sq = b["exposure"], b["success"], b["sq"]
        if exp <= 0:
            continue
        ess_i = min((exp * exp) / max(sq, 1e-9), float(cfg.ess_max))
        th = float(np.clip(suc / exp, 1e-4, 1 - 1e-4))
        ess[i] = ess_i
        theta[i] = th
        s_eff[i] = ess_i * th
        f_eff[i] = ess_i * (1.0 - th)
    return {"ids": np.asarray(ids, dtype=object), "s_eff": s_eff,
            "f_eff": f_eff, "ess": ess, "theta": theta,
            "impressions": np.asarray([by_product[p]["impressions"]
                                       for p in ids], dtype=np.float64),
            "purchases": np.asarray([by_product[p]["purchases"]
                                     for p in ids], dtype=np.float64)}


# --------------------------------------------------------------------------- #
# 2.3 Hierarchical Beta-Binomial by moments
# --------------------------------------------------------------------------- #
def _moments(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    mean = float((weights * values).sum() / total)
    var = float((weights * (values - mean) ** 2).sum() / total)
    return mean, var


def beta_from_moments(mean: float, var: float, cfg: Config = CFG
                      ) -> Tuple[float, float, float]:
    """Equations (7)-(10) with numerical clamps."""
    mean = float(np.clip(mean, cfg.mu_floor, cfg.mu_ceil))
    if var <= 1e-12:
        return mean, float(cfg.nu_min), mean * cfg.nu_min
    nu = mean * (1.0 - mean) / var - 1.0
    nu = float(np.clip(nu, cfg.nu_min, cfg.nu_max))
    alpha = max(mean * nu, cfg.alpha_floor)
    beta = max((1.0 - mean) * nu, cfg.beta_floor)
    return mean, nu, alpha


def fit_hierarchy(theta: np.ndarray, ess: np.ndarray,
                  category: np.ndarray, subcategory: np.ndarray,
                  cfg: Config = CFG) -> Dict[str, Any]:
    """Four-level empirical-Bayes hierarchy with precision shrinkage.

    Returns product priors ``alpha0``/``beta0`` and the group parameters used
    in the ``meta`` row for auditing.
    """
    theta = np.asarray(theta, dtype=np.float64)
    ess = np.asarray(ess, dtype=np.float64)
    n = theta.size
    category = np.asarray(category, dtype=np.int64)
    subcategory = np.asarray(subcategory, dtype=np.int64)

    m_g, v_g = _moments(theta, ess)
    m_g = float(np.clip(m_g, cfg.mu_floor, cfg.mu_ceil))
    _, nu_g, _ = beta_from_moments(m_g, v_g, cfg)

    cat_ids = np.unique(category)
    sub_ids = np.unique(subcategory)
    cat_mean: Dict[int, float] = {}
    cat_nu: Dict[int, float] = {}
    for c in cat_ids:
        mask = category == c
        m_c, v_c = _moments(theta[mask], ess[mask])
        if mask.sum() < 2 or v_c <= 1e-12:
            m_c, nu_c = m_g, 0.0
        else:
            m_c, nu_c, _ = beta_from_moments(m_c, v_c, cfg)
        cat_mean[int(c)] = m_c
        cat_nu[int(c)] = nu_c

    sub_parent: Dict[int, int] = {}
    for s in sub_ids:
        vals, counts = np.unique(category[subcategory == s], return_counts=True)
        sub_parent[int(s)] = int(vals[np.argmax(counts)])

    sub_mean: Dict[int, float] = {}
    sub_nu: Dict[int, float] = {}
    for s in sub_ids:
        mask = subcategory == s
        m_s, v_s = _moments(theta[mask], ess[mask])
        parent = sub_parent[int(s)]
        m_c = cat_mean.get(parent, m_g)
        nu_c = cat_nu.get(parent, 0.0)
        if mask.sum() < 2 or v_s <= 1e-12:
            m_s, nu_s = m_c, 0.0
        else:
            m_s, nu_s, _ = beta_from_moments(m_s, v_s, cfg)
        denom = nu_s + cfg.kappa_2 * nu_c
        m_star = m_c if denom <= 1e-9 else (
            nu_s * m_s + cfg.kappa_2 * nu_c * m_c) / denom
        nu_star = float(np.clip(nu_s + cfg.kappa_2 * nu_c,
                                cfg.nu_min, cfg.nu_max))
        sub_mean[int(s)] = float(np.clip(m_star, cfg.mu_floor, cfg.mu_ceil))
        sub_nu[int(s)] = nu_star

    cat_mean_star: Dict[int, float] = {}
    cat_nu_star: Dict[int, float] = {}
    for c in cat_ids:
        m_c, nu_c = cat_mean[int(c)], cat_nu[int(c)]
        denom = nu_c + cfg.kappa_1 * nu_g
        m_star = m_g if denom <= 1e-9 else (
            nu_c * m_c + cfg.kappa_1 * nu_g * m_g) / denom
        cat_mean_star[int(c)] = float(np.clip(m_star, cfg.mu_floor,
                                              cfg.mu_ceil))
        cat_nu_star[int(c)] = float(np.clip(nu_c + cfg.kappa_1 * nu_g,
                                            cfg.nu_min, cfg.nu_max))

    alpha0 = np.empty(n, dtype=np.float32)
    beta0 = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = int(subcategory[i])
        m = sub_mean.get(s, m_g)
        nu = float(np.clip(sub_nu.get(s, cfg.nu_min), 1.0,
                           cfg.n_max_product_prior))
        alpha0[i] = max(m * nu, cfg.alpha_floor)
        beta0[i] = max((1.0 - m) * nu, cfg.beta_floor)
    return {
        "alpha0": alpha0, "beta0": beta0,
        "global_mean": m_g, "global_nu": nu_g,
        "cat_mean": cat_mean_star, "cat_nu": cat_nu_star,
        "sub_mean": sub_mean, "sub_nu": sub_nu, "sub_parent": sub_parent,
    }



# --------------------------------------------------------------------------- #
# 2.4 Organic signals
# --------------------------------------------------------------------------- #
def organic_pseudo_counts(org_v: np.ndarray, org_atc: np.ndarray,
                          org_p: np.ndarray, subcategory: np.ndarray,
                          cfg: Config = CFG) -> Dict[str, np.ndarray]:
    """Equations (15)-(18): smoothed organic conversion and pseudo-counts."""
    v = np.asarray(org_v, dtype=np.float64)
    a = np.asarray(org_atc, dtype=np.float64)
    p = np.asarray(org_p, dtype=np.float64)
    sub = np.asarray(subcategory, dtype=np.int64)
    n = v.size
    a_s = 20.0
    sub_m: Dict[int, float] = {}
    for s in np.unique(sub):
        mask = sub == s
        sub_m[int(s)] = float((p[mask].sum() + 1.0) /
                              max(v[mask].sum() + 20.0, 1e-9))
    q = np.empty(n, dtype=np.float64)
    for i in range(n):
        m_s = sub_m.get(int(sub[i]), 0.0)
        q[i] = np.clip((p[i] + a_s * m_s) / (v[i] + a_s), 0.0, 1.0)
    n_org = cfg.k_org * (v + 5.0 * a)
    s_org = n_org * q
    f_org = np.maximum(n_org - s_org, 0.0)
    f_view = np.log1p(np.maximum(v, 0.0))
    # Relative ATC / purchases with respect to the subcategory mean.
    sub_v = np.zeros(n, dtype=np.float64)
    sub_a = np.zeros(n, dtype=np.float64)
    sub_p = np.zeros(n, dtype=np.float64)
    for s in np.unique(sub):
        mask = sub == s
        size = max(mask.sum(), 1)
        sub_v[mask] = v[mask].mean()
        sub_a[mask] = a[mask].mean()
        sub_p[mask] = p[mask].mean()
    f_atc = np.log1p(np.maximum(a, 0.0)) - np.log1p(np.maximum(sub_a, 0.0))
    # f_q is a logit clipped to the plausible purchase-rate range.
    f_q = np.log(np.clip(q, 0.001, 0.5)) - np.log1p(
        -np.clip(q, 0.001, 0.5))
    return {"q": q.astype(np.float32), "s_org": s_org.astype(np.float32),
            "f_org": f_org.astype(np.float32),
            "f_view": f_view.astype(np.float32),
            "f_atc": f_atc.astype(np.float32),
            "f_q": f_q.astype(np.float32), "sub_m": sub_m}


# --------------------------------------------------------------------------- #
# 2.7 Position bias: EM + isotonic PAVA
# --------------------------------------------------------------------------- #
def pava_nonincreasing(values: Sequence[float],
                       weights: Optional[Sequence[float]] = None
                       ) -> np.ndarray:
    """Pool Adjacent Violators for a non-increasing sequence."""
    values = np.asarray(values, dtype=np.float64)
    w = np.ones_like(values) if weights is None else np.asarray(
        weights, dtype=np.float64)
    if values.size == 0:
        return values
    blocks: List[List[float]] = []  # [weighted_sum, weight, count]
    for value, weight in zip(values, w):
        blocks.append([value * max(weight, 1e-9), max(weight, 1e-9), 1])
        while len(blocks) >= 2:
            prev, last = blocks[-2], blocks[-1]
            if prev[0] / prev[1] >= last[0] / last[1]:
                break
            merged = [prev[0] + last[0], prev[1] + last[1], prev[2] + last[2]]
            blocks[-2:] = [merged]
    out: List[float] = []
    for total, weight, count in blocks:
        out.extend([total / weight] * int(count))
    return np.asarray(out, dtype=np.float64)


def estimate_position_bias(rows: Sequence[Mapping[str, Any]],
                           cfg: Config = CFG) -> np.ndarray:
    """EM for ``w_k`` on aggregated product/slot impressions + PAVA.

    Falls back to the spec defaults when the log is not randomised
    (``w_2/w_1 > 0.9``), exactly as Section 2.7.4 requires.
    """
    imp = np.zeros(10, dtype=np.float64)
    clicks = np.zeros(10, dtype=np.float64)
    # product x slot matrices for the EM responsibility.
    products: Dict[str, int] = {}
    p_imp: List[List[float]] = []
    p_pur: List[List[float]] = []
    for row in rows:
        slot = min(max(int(row.get("slot") or 1), 1), 10)
        pid = str(row.get("product_id") or row.get("p") or "")
        impressions = float(row.get("imp") or 0.0)
        # The EM of Section 2.7 is a *click* model.  Use `clk` when present,
        # falling back to purchases only for callers with purchase-only rows.
        raw_clicks = row.get("clk")
        purchases = float(raw_clicks if raw_clicks is not None
                          else (row.get("pur") or 0.0))
        imp[slot - 1] += impressions
        clicks[slot - 1] += purchases
        if pid:
            if pid not in products:
                products[pid] = len(p_imp)
                p_imp.append([0.0] * 10)
                p_pur.append([0.0] * 10)
            j = products[pid]
            p_imp[j][slot - 1] += impressions
            p_pur[j][slot - 1] += purchases

    smoothed = (clicks + 0.5) / np.maximum(imp + 1.0, 1e-9)
    if smoothed[0] <= 0:
        return np.asarray(cfg.default_position_w, dtype=np.float32)
    w = smoothed / smoothed[0]
    w = np.clip(w, 1e-3, 1.0)
    imp_m = np.asarray(p_imp, dtype=np.float64)
    pur_m = np.asarray(p_pur, dtype=np.float64)
    for _ in range(50):
        denom = (imp_m * w[None, :]).sum(axis=1)
        r = np.divide(pur_m.sum(axis=1), denom,
                      out=np.zeros_like(denom), where=denom > 1e-9)
        num = (pur_m * r[:, None]).sum(axis=0)
        den = (imp_m * r[:, None]).sum(axis=0)
        new_w = np.divide(num, np.maximum(den, 1e-9))
        new_w = np.where(den > 1e-9, new_w, w)
        if new_w[0] <= 1e-9:
            break
        new_w = np.clip(new_w / new_w[0], 1e-3, 1.0)
        if np.abs(new_w - w).sum() < 1e-6:
            w = new_w
            break
        w = 0.5 * w + 0.5 * new_w         # damping for stability
    # Fallback when the log is not sufficiently randomised.
    if w.size < 2 or w[0] <= 1e-9 or (w[1] / w[0]) > 0.9:
        return np.asarray(cfg.default_position_w, dtype=np.float32)
    w = pava_nonincreasing(w, np.maximum(imp, 1.0))
    w = np.clip(w, cfg.position_w_min, 1.0)
    w = w / max(w[0], 1e-9)
    return w.astype(np.float32)



# --------------------------------------------------------------------------- #
# 2.5 Structural prior and transition matrices
# --------------------------------------------------------------------------- #
def build_structural_prior(E: np.ndarray, category: np.ndarray,
                           subcategory: np.ndarray, brand: np.ndarray,
                           cfg: Config = CFG, top_k: int = 200,
                           chunk: int = 256) -> CSR:
    """Top-``k`` structural prior ``m_ji`` for every source row.

    ``m_ji = M0 * [lambda_sub 1{sub} + lambda_brand 1{brand} +
    lambda_cat 1{cat} + lambda_sem cos(e_i,e_j)^2] / Z_j`` (eq. 22).
    The diagonal is excluded, so ``P(j|j) = 0``.
    """
    n = int(category.size)
    top_k = min(top_k, max(n - 1, 1))
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    cat = np.asarray(category, dtype=np.int64)
    sub = np.asarray(subcategory, dtype=np.int64)
    br = np.asarray(brand, dtype=np.int64)
    E = np.asarray(E, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sim = E[start:stop] @ E.T
        sim = np.clip(sim, -1.0, 1.0)
        block = (cfg.lambda_sub * (sub[start:stop, None] == sub[None, :]) +
                 cfg.lambda_brand_structural *
                 (br[start:stop, None] == br[None, :]) +
                 cfg.lambda_cat_structural *
                 (cat[start:stop, None] == cat[None, :]) +
                 cfg.lambda_sem * (sim * sim))
        for local in range(stop - start):
            block[local, start + local] = 0.0
        z = block.sum(axis=1)
        z[z <= 1e-12] = 1.0
        block *= (cfg.m0_transition / z)[:, None]
        if n > top_k:
            part = np.argpartition(-block, top_k, axis=1)[:, :top_k]
        else:
            part = np.tile(np.arange(n), (stop - start, 1))
        rows.append(np.repeat(np.arange(start, stop, dtype=np.int64),
                              part.shape[1]))
        cols.append(part.reshape(-1).astype(np.int32))
        vals.append(np.take_along_axis(block, part, axis=1).reshape(-1)
                    .astype(np.float32))
    from scipy.sparse import csr_matrix
    mat = csr_matrix((np.concatenate(vals),
                      (np.concatenate(rows), np.concatenate(cols))),
                     shape=(n, n))
    return CSR.from_scipy(mat)


def _csr_to_scipy(matrix: CSR):
    from scipy.sparse import csr_matrix
    return csr_matrix((matrix.data.copy(), matrix.indices.copy(),
                       matrix.indptr.copy()),
                      shape=(len(matrix.indptr) - 1, matrix.n_cols))


def top_k_rows(matrix: Any, k: int = 200, normalize: bool = True,
               exclude_diagonal: bool = True) -> CSR:
    """Keep the top-k destinations of every row and renormalise to sum 1."""
    from scipy.sparse import csr_matrix
    mat = matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
    n = mat.shape[0]
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for i in range(n):
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        idx = np.asarray(mat.indices[lo:hi], dtype=np.int64)
        dat = np.asarray(mat.data[lo:hi], dtype=np.float64)
        if exclude_diagonal:
            keep = idx != i
            idx, dat = idx[keep], dat[keep]
        if idx.size == 0:
            continue
        if idx.size > k:
            part = np.argpartition(dat, -k)[-k:]
            idx, dat = idx[part], dat[part]
        if normalize:
            total = dat.sum()
            if total <= 1e-12:
                continue
            dat = dat / total
        rows.append(np.full(idx.size, i, dtype=np.int64))
        cols.append(idx.astype(np.int32))
        vals.append(dat.astype(np.float32))
    if not vals:
        return CSR.empty(n)
    out = csr_matrix((np.concatenate(vals),
                      (np.concatenate(rows), np.concatenate(cols))),
                     shape=(n, n))
    return CSR.from_scipy(out)



def build_transition_from_edges(edges: Optional[Sequence[Mapping[str, Any]]],
                                structural: CSR, cfg: Config = CFG,
                                top_k: Optional[int] = None) -> CSR:
    """Combine observed co-occurrence counts with the structural prior.

    ``n_hat_ji = n_ji / max(n_i_in, 1)^alpha_pop`` (eq. 20b), then
    ``P(i|j) = (n_hat_ji + m_ji) / sum_l(...)`` (eq. 20), top-k and with the
    diagonal removed.
    """
    from scipy.sparse import csr_matrix
    n = len(structural.indptr) - 1
    top_k = top_k or cfg.trans_top_k
    struct = _csr_to_scipy(structural)
    if not edges:
        return top_k_rows(struct, top_k)
    src: List[int] = []
    dst: List[int] = []
    counts: List[float] = []
    for edge in edges:
        j = int(edge.get("src_idx", edge.get("p_idx", -1)))
        i = int(edge.get("dst_idx", edge.get("q_idx", -1)))
        if j < 0 or i < 0 or j >= n or i >= n or j == i:
            continue
        count = float(edge.get("n") or edge.get("count") or 0.0)
        src.append(j)
        dst.append(i)
        counts.append(max(count, 0.0))
    if not counts:
        return top_k_rows(struct, top_k)
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    count_arr = np.asarray(counts, dtype=np.float64)
    n_in = np.zeros(n, dtype=np.float64)
    np.add.at(n_in, dst_arr, count_arr)
    adjusted = count_arr / np.maximum(n_in[dst_arr], 1.0) ** cfg.alpha_pop
    observed = csr_matrix((adjusted, (src_arr, dst_arr)), shape=(n, n))
    combined = observed + struct
    return top_k_rows(combined, top_k, normalize=True, exclude_diagonal=True)


def blend_transitions(base: CSR, live: CSR, mu_live: float,
                      cfg: Config = CFG) -> CSR:
    """``T <- (1 - mu_live) T0 + mu_live T_live`` with row reindexing."""
    mu = float(np.clip(mu_live, 0.0, 1.0))
    b = _csr_to_scipy(base)
    l = _csr_to_scipy(live)
    out = b.multiply(1.0 - mu) + l.multiply(mu)
    return top_k_rows(out, cfg.trans_top_k)


def transition_fallback(structural: CSR, src: int, dst: int) -> float:
    """``m_{j,new} / m_j`` for products not in the top-k row (eq. 70)."""
    cols, vals = structural.row(int(src))
    total = float(structural.row_sum(int(src)))
    if total <= 1e-12:
        return 0.0
    hit = np.where(cols == int(dst))[0]
    return float(vals[hit[0]] / total) if hit.size else 0.0



# --------------------------------------------------------------------------- #
# 2.6 / 2.8 Bayesian logistic regression + global calibration
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _prior_precision(d: int, cfg: Config) -> np.ndarray:
    tau = np.full(d, cfg.prior_tau_other, dtype=np.float64)
    for idx in (10, 11):
        if idx < d:
            tau[idx] = cfg.prior_tau_price
    return np.diag(1.0 / np.maximum(tau, 1e-6) ** 2)


def _log_posterior(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
                   w: np.ndarray, mu0: np.ndarray, P0: np.ndarray) -> float:
    eta = np.clip(X @ beta, -35.0, 35.0)
    # log-likelihood without overflow: -log(1+exp(-eta)) for y=1 etc.
    ll = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
    diff = beta - mu0
    return float(ll - 0.5 * diff @ P0 @ diff)


def fit_logistic_map(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                     mu0: Optional[np.ndarray] = None,
                     Sigma0: Optional[np.ndarray] = None,
                     cfg: Config = CFG, max_iter: int = 10
                     ) -> Dict[str, Any]:
    """Laplace approximation by damped Newton / IRLS (equations 29-30)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n, d = X.shape
    if mu0 is None:
        mu0 = np.zeros(d, dtype=np.float64)
    if Sigma0 is None:
        P0 = _prior_precision(d, cfg)
    else:
        Sigma0 = np.asarray(Sigma0, dtype=np.float64)
        P0 = np.linalg.pinv(0.5 * (Sigma0 + Sigma0.T))
    beta = np.asarray(mu0, dtype=np.float64).copy()
    current = _log_posterior(beta, X, y, w, mu0, P0)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p = _sigmoid(eta)
        grad = X.T @ (w * (y - p)) - P0 @ (beta - mu0)
        s = w * p * (1.0 - p)
        H = (X.T * s) @ X + P0
        H = 0.5 * (H + H.T)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad
        step = 1.0
        improved = False
        for _ in range(10):
            candidate = beta + step * delta
            value = _log_posterior(candidate, X, y, w, mu0, P0)
            if value >= current - 1e-9:
                beta = candidate
                current = value
                improved = True
                break
            step *= 0.5
        if not improved or np.linalg.norm(step * delta) < 1e-7:
            break
    eta = np.clip(X @ beta, -35.0, 35.0)
    p = _sigmoid(eta)
    s = w * p * (1.0 - p)
    H = (X.T * s) @ X + P0
    try:
        Sigma = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(H)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return {"mu": beta.astype(np.float32),
            "Sigma": Sigma.astype(np.float32),
            "log_posterior": current, "iterations": max_iter}


def fit_calibration(base_logits: np.ndarray, y: np.ndarray, w: np.ndarray,
                    cfg: Config = CFG) -> Tuple[float, float]:
    """Grid over ``T`` and 1-D Newton for ``a`` on the temporal holdout.

    If the holdout has fewer than 5,000 purchases the spec tells us not to
    over-fit calibration: return ``(0.0, 1.0)``.
    """
    base = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if float(y.sum()) < 5000.0 or base.size == 0:
        return 0.0, 1.0
    best_t, best_a, best_loss = 1.0, 0.0, float("inf")
    total_w = max(float(w.sum()), 1e-9)
    for temperature in np.arange(0.5, 2.01, 0.1):
        a = 0.0
        for _ in range(8):
            eta = a + base / temperature
            p = _sigmoid(eta)
            grad = float(np.sum(w * (y - p)))
            hess = float(np.sum(w * p * (1.0 - p)))
            if hess <= 1e-12:
                break
            a += grad / hess
        eta = np.clip(a + base / temperature, -35.0, 35.0)
        p = _sigmoid(eta)
        loss = -float(np.sum(w * (y * np.log(np.clip(p, 1e-12, 1.0)) +
                                  (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12,
                                                             1.0))))) / total_w
        if loss < best_loss:
            best_loss, best_t, best_a = loss, float(temperature), float(a)
    return best_a, best_t


def calibration_metrics(p: np.ndarray, y: np.ndarray,
                        bins: int = 10) -> Dict[str, float]:
    """Brier score and ECE used in the offline validation report."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return {"brier": 0.0, "ece": 0.0}
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    return {"brier": brier, "ece": float(ece)}



# --------------------------------------------------------------------------- #
# Persistence: local .npz (fast start) and team_NN.scratch (truth)
# --------------------------------------------------------------------------- #
_ARRAY_FIELDS = (
    "category", "subcategory", "brand", "price", "active", "unavailable",
    "is_new", "alpha0", "beta0", "alpha", "beta", "s_slow", "f_slow",
    "s_fast", "f_fast", "s_click", "f_click", "s_org", "f_org", "org_v",
    "org_atc", "org_p", "feat_view", "feat_atc", "feat_q", "price_pct",
    "price_pct_subcat", "mu_beta", "Sigma_beta", "feature_mean",
    "feature_std", "w", "trend_boost", "new_bonus", "restock_bonus",
    "price_bonus", "best_safe")


def save_snapshot_npz(snapshot: Snapshot, path: str,
                      extra: Optional[Mapping[str, Any]] = None) -> str:
    """Save the full state locally for a <2 s restart (spec 5.2.2)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload: Dict[str, Any] = {
        "ids": np.asarray(snapshot.ids, dtype=object),
        "category_names": np.asarray(snapshot.category_names, dtype=object),
        "subcategory_names": np.asarray(snapshot.subcategory_names,
                                        dtype=object),
        "brand_names": np.asarray(snapshot.brand_names, dtype=object),
        "feature_names": np.asarray(snapshot.feature_names, dtype=object),
        "E": np.asarray(snapshot.E, dtype=np.float16),
        "trans_view_indptr": snapshot.trans_view.indptr,
        "trans_view_indices": snapshot.trans_view.indices,
        "trans_view_data": snapshot.trans_view.data,
        "trans_buy_indptr": snapshot.trans_buy.indptr,
        "trans_buy_indices": snapshot.trans_buy.indices,
        "trans_buy_data": snapshot.trans_buy.data,
        "calib_a": np.asarray(snapshot.calib_a),
        "calib_T": np.asarray(snapshot.calib_T),
        "p95_price": np.asarray(snapshot.p95_price),
        "tau_scale_view": np.asarray(snapshot.tau_scale_view),
        "tau_scale_buy": np.asarray(snapshot.tau_scale_buy),
        "ts": np.asarray(snapshot.ts),
        "new_product_set": np.asarray(sorted(snapshot.new_product_set),
                                      dtype=object),
        "new_subcats": np.asarray(sorted(snapshot.new_subcats), dtype=np.int32),
        "policy": np.asarray(json.dumps(snapshot.policy.as_dict())),
    }
    for name in _ARRAY_FIELDS:
        payload[name] = np.asarray(getattr(snapshot, name))
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def load_snapshot_npz(path: str) -> Optional[Snapshot]:
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except (OSError, ValueError):
        return None
    try:
        ids = [str(x) for x in data["ids"].tolist()]
        index = {pid: i for i, pid in enumerate(ids)}
        n = len(ids)
        kwargs: Dict[str, Any] = {"ids": ids, "index": index, "n": n}
        for name in _ARRAY_FIELDS:
            if name in data:
                value = data[name]
                if name in ("category", "subcategory", "brand"):
                    value = value.astype(np.int16)
                elif name in ("active", "unavailable", "is_new"):
                    value = value.astype(bool)
                elif name == "best_safe":
                    value = value.astype(np.int64)
                else:
                    value = value.astype(np.float32)
                kwargs[name] = value
        kwargs["E"] = data["E"].astype(np.float32)
        kwargs["trans_view"] = CSR(
            data["trans_view_indptr"].astype(np.int64),
            data["trans_view_indices"].astype(np.int32),
            data["trans_view_data"].astype(np.float32), n)
        kwargs["trans_buy"] = CSR(
            data["trans_buy_indptr"].astype(np.int64),
            data["trans_buy_indices"].astype(np.int32),
            data["trans_buy_data"].astype(np.float32), n)
        for name in ("category_names", "subcategory_names", "brand_names"):
            kwargs[name] = tuple(str(x) for x in data[name].tolist())
        kwargs["feature_names"] = tuple(str(x) for x in
                                        data["feature_names"].tolist())
        kwargs["calib_a"] = float(data["calib_a"])
        kwargs["calib_T"] = float(data["calib_T"])
        kwargs["p95_price"] = float(data["p95_price"])
        kwargs["tau_scale_view"] = float(data["tau_scale_view"])
        kwargs["tau_scale_buy"] = float(data["tau_scale_buy"])
        kwargs["ts"] = float(data["ts"])
        kwargs["new_product_set"] = frozenset(
            str(x) for x in data["new_product_set"].tolist())
        kwargs["new_subcats"] = frozenset(
            int(x) for x in data["new_subcats"].tolist())
        try:
            policy = RuntimePolicy(**json.loads(str(data["policy"])))
        except (ValueError, TypeError):
            policy = RuntimePolicy()
        kwargs["policy"] = policy
        return Snapshot(**kwargs)
    except (KeyError, ValueError, TypeError) as exc:
        LOGGER.warning("invalid local snapshot %s: %s", path, exc)
        return None


def _clean_floats(values: np.ndarray, digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in np.asarray(values).tolist()]


def _clean_ints(values: np.ndarray) -> List[int]:
    return [int(v) for v in np.asarray(values).tolist()]



def new_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _product_chunk_row(snap: Snapshot, start: int, stop: int, sid: str,
                       chunk: int) -> Dict[str, Any]:
    sl = slice(start, stop)
    return {
        "kind": "product_chunk", "snapshot_id": sid, "chunk": int(chunk),
        "ids": snap.ids[sl],
        "alpha0": _clean_floats(snap.alpha0[sl]),
        "beta0": _clean_floats(snap.beta0[sl]),
        "s_slow": _clean_floats(snap.s_slow[sl]),
        "f_slow": _clean_floats(snap.f_slow[sl]),
        "s_fast": _clean_floats(snap.s_fast[sl]),
        "f_fast": _clean_floats(snap.f_fast[sl]),
        "s_click": _clean_floats(snap.s_click[sl]),
        "f_click": _clean_floats(snap.f_click[sl]),
        "s_org": _clean_floats(snap.s_org[sl]),
        "f_org": _clean_floats(snap.f_org[sl]),
        "org_v": _clean_floats(snap.org_v[sl]),
        "org_atc": _clean_floats(snap.org_atc[sl]),
        "org_p": _clean_floats(snap.org_p[sl]),
        "feat_view": _clean_floats(snap.feat_view[sl]),
        "feat_atc": _clean_floats(snap.feat_atc[sl]),
        "feat_q": _clean_floats(snap.feat_q[sl]),
        "price": _clean_floats(snap.price[sl]),
        "price_pct": _clean_floats(snap.price_pct[sl]),
        "price_pct_subcat": _clean_floats(snap.price_pct_subcat[sl]),
        "active": _clean_ints(np.asarray(snap.active[sl], dtype=np.int8)),
        "is_new": _clean_ints(np.asarray(snap.is_new[sl], dtype=np.int8)),
        "subcategory_code": _clean_ints(snap.subcategory[sl]),
        "brand_code": _clean_ints(snap.brand[sl]),
        "category_code": _clean_ints(snap.category[sl]),
    }


def write_snapshot_to_scratch(client: Any, snap: Snapshot, cfg: Config = CFG,
                              snapshot_id: Optional[str] = None,
                              include_embeddings: bool = True,
                              include_transitions: bool = True,
                              extra_stats: Optional[Mapping[str, Any]] = None
                              ) -> str:
    """Write all chunks, then ``meta`` last (two-phase commit, spec 5.1/5.2)."""
    sid = snapshot_id or new_snapshot_id()
    table = cfg.scratch_table
    # Best-effort table creation (admin keys); restricted keys usually already
    # have team_NN.scratch provisioned by the organisers.
    try:
        client.create_table(table)
    except Exception:
        pass
    chunk = max(int(cfg.chunk_products), 16)
    product_rows: List[Dict[str, Any]] = []
    n_chunks = 0
    for start in range(0, snap.n, chunk):
        stop = min(start + chunk, snap.n)
        product_rows.append(_product_chunk_row(snap, start, stop, sid, n_chunks))
        n_chunks += 1
    client.insert_rows(table, product_rows, database=client.team_id,
                       max_rows=64)

    n_embedding_chunks = 0
    if include_embeddings:
        embedding_rows: List[Dict[str, Any]] = []
        for start in range(0, snap.n, chunk):
            stop = min(start + chunk, snap.n)
            embedding_rows.append({
                "kind": "embedding_chunk", "snapshot_id": sid,
                "chunk": n_embedding_chunks,
                "ids": snap.ids[start:stop],
                "dim": int(snap.E.shape[1]), "dtype": "float16",
                "encoding": "base64",
                "data": encode_embedding_chunk(snap.E[start:stop]),
            })
            n_embedding_chunks += 1
        client.insert_rows(table, embedding_rows, database=client.team_id,
                           max_rows=8)


    if include_transitions:
        trans_rows: List[Dict[str, Any]] = []
        for i, pid in enumerate(snap.ids):
            cols, vals = snap.trans_view.row(i)
            if cols.size:
                trans_rows.append({
                    "kind": "transition_chunk", "snapshot_id": sid,
                    "matrix": "view", "src_id": pid,
                    "dst_ids": [snap.ids[int(c)] for c in cols],
                    "p": _clean_floats(vals, 7)})
        for i, pid in enumerate(snap.ids):
            cols, vals = snap.trans_buy.row(i)
            if cols.size:
                trans_rows.append({
                    "kind": "transition_chunk", "snapshot_id": sid,
                    "matrix": "buy", "src_id": pid,
                    "dst_ids": [snap.ids[int(c)] for c in cols],
                    "p": _clean_floats(vals, 7)})
        client.insert_rows(table, trans_rows, database=client.team_id,
                           max_rows=400, max_bytes=6 * 1024 * 1024)

    client.insert_rows(table, [{
        "kind": "context", "snapshot_id": sid, "ts": iso_utc(),
        "feature_names": list(snap.feature_names),
        "mu": _clean_floats(snap.mu_beta),
        "sigma": _clean_floats(np.asarray(snap.Sigma_beta).reshape(-1)),
        "feature_mean": _clean_floats(snap.feature_mean),
        "feature_std": _clean_floats(snap.feature_std),
        "calib_a": float(snap.calib_a), "calib_T": float(snap.calib_T),
        "position_w": _clean_floats(snap.w),
        "p95_price": float(snap.p95_price),
        "tau_scale_view": float(snap.tau_scale_view),
        "tau_scale_buy": float(snap.tau_scale_buy)}],
        database=client.team_id)

    order = np.argsort(snap.trend_boost)[::-1][:500]
    trend_items = [{"id": snap.ids[int(i)],
                    "boost": round(float(snap.trend_boost[int(i)]), 6)}
                   for i in order if snap.trend_boost[int(i)] > 0]
    client.insert_rows(table, [{
        "kind": "trend", "snapshot_id": sid, "ts": iso_utc(),
        "items": trend_items}], database=client.team_id)

    stats = {
        "alpha_mean": round(float(np.mean(snap.alpha)), 6),
        "beta_mean": round(float(np.mean(snap.beta)), 6),
        "ess_mean": round(float(np.mean(snap.alpha + snap.beta)), 3),
        "n_new_products": len(snap.new_product_set),
        "w": _clean_floats(snap.w, 5),
    }
    if extra_stats:
        stats.update(dict(extra_stats))
    meta = {
        "kind": "meta", "schema_version": cfg.schema_version,
        "model_version": cfg.model_version, "snapshot_id": sid,
        "ts": iso_utc(), "n_products": int(snap.n),
        "dim_text": int(snap.E.shape[1]),
        "dim_context": int(np.asarray(snap.mu_beta).size),
        "n_product_chunks": n_chunks,
        "n_embedding_chunks": n_embedding_chunks,
        "category_names": list(snap.category_names),
        "subcategory_names": list(snap.subcategory_names),
        "brand_names": list(snap.brand_names),
        "config": {
            "rho": cfg.rho, "lambda_T_view": cfg.lambda_T_view,
            "lambda_T_buy": cfg.lambda_T_buy, "lambda_C": cfg.lambda_C,
            "temperature": float(snap.calib_T), "calib_a": float(snap.calib_a),
            "half_life_slow_days": cfg.half_life_slow_days,
            "half_life_fast_min": cfg.half_life_fast_min,
            "ess_max": int(snap.policy.ess_max),
            "regime": snap.policy.regime},
        "stats": stats,
        "config_json": json.dumps({
            "rho": cfg.rho, "lambda_T_view": cfg.lambda_T_view,
            "lambda_T_buy": cfg.lambda_T_buy, "lambda_C": cfg.lambda_C,
            "temperature": float(snap.calib_T), "calib_a": float(snap.calib_a),
            "half_life_slow_days": cfg.half_life_slow_days,
            "half_life_fast_min": cfg.half_life_fast_min,
            "ess_max": int(snap.policy.ess_max),
            "regime": snap.policy.regime}, separators=(",", ":")),
        "stats_json": json.dumps(stats, separators=(",", ":"), default=str),
    }
    client.insert_rows(table, [meta], database=client.team_id)
    LOGGER.info("wrote snapshot %s (%d products, %d chunks) to %s",
                sid, snap.n, n_chunks, table)
    return sid



def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, list) else [value]
        except ValueError:
            return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def _as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_as_float_list(item))
            elif item not in (None, ""):
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    out.append(0.0)
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _as_float_array(value: Any, default: np.ndarray) -> np.ndarray:
    values = _as_float_list(value)
    if not values:
        return default
    return np.asarray(values, dtype=np.float32)


def _apply_product_delta(arrays: Dict[str, np.ndarray],
                         index: Dict[str, int],
                         rows: Sequence[Mapping[str, Any]],
                         cfg: Config) -> None:
    for row in rows:
        ids = _as_str_list(row.get("ids"))
        if not ids:
            continue
        for pos, pid in enumerate(ids):
            idx = index.get(pid)
            if idx is None:
                continue
            for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_click",
                         "f_click", "s_org", "f_org", "price", "price_pct",
                         "price_pct_subcat", "trend_boost", "new_bonus",
                         "restock_bonus", "price_bonus"):
                values = _as_float_list(row.get(name))
                if pos < len(values):
                    arrays[name][idx] = values[pos]
            active = row.get("active")
            if isinstance(active, list) and pos < len(active):
                arrays["active"][idx] = bool(int(active[pos]))
            is_new = row.get("is_new")
            if isinstance(is_new, list) and pos < len(is_new):
                arrays["is_new"][idx] = bool(int(is_new[pos]))


def latest_meta(client: Any, cfg: Config = CFG) -> Optional[Dict[str, Any]]:
    try:
        rows = client.query(
            "SELECT * FROM %s.%s WHERE kind::String = 'meta' "
            "ORDER BY ts::DateTime64(3) DESC LIMIT 1"
            % (client.team_id, cfg.scratch_table), retries=2)
    except Exception as exc:
        LOGGER.info("no scratch meta: %s", exc)
        return None
    return rows[0] if rows else None



def load_snapshot_from_scratch(client: Any, cfg: Config = CFG
                               ) -> Optional[Snapshot]:
    """Rebuild a snapshot from scratch using the two-phase commit markers."""
    meta = latest_meta(client, cfg)
    if not meta:
        return None
    sid = str(meta.get("snapshot_id") or "")
    if not sid:
        return None
    table = "%s.%s" % (client.team_id, cfg.scratch_table)
    try:
        product_rows = client.query(
            "SELECT * FROM %s WHERE kind::String = 'product_chunk' "
            "AND snapshot_id::String = '%s' ORDER BY chunk::Int32"
            % (table, sid), retries=2)
    except Exception as exc:
        LOGGER.warning("cannot load product chunks: %s", exc)
        return None
    if not product_rows:
        return None
    ids: List[str] = []
    for row in product_rows:
        ids.extend(_as_str_list(row.get("ids")))
    if not ids:
        return None
    n = len(ids)
    index = {pid: i for i, pid in enumerate(ids)}
    defaults = {
        "category": np.zeros(n, np.int16), "subcategory": np.zeros(n, np.int16),
        "brand": np.zeros(n, np.int16), "price": np.ones(n, np.float32),
        "active": np.ones(n, bool), "unavailable": np.zeros(n, bool),
        "is_new": np.zeros(n, bool), "alpha0": np.ones(n, np.float32),
        "beta0": np.ones(n, np.float32), "alpha": np.ones(n, np.float32),
        "beta": np.ones(n, np.float32), "s_slow": np.zeros(n, np.float32),
        "f_slow": np.zeros(n, np.float32), "s_fast": np.zeros(n, np.float32),
        "f_fast": np.zeros(n, np.float32), "s_click": np.zeros(n, np.float32),
        "f_click": np.zeros(n, np.float32), "s_org": np.zeros(n, np.float32),
        "f_org": np.zeros(n, np.float32), "org_v": np.zeros(n, np.float32),
        "org_atc": np.zeros(n, np.float32), "org_p": np.zeros(n, np.float32),
        "feat_view": np.zeros(n, np.float32),
        "feat_atc": np.zeros(n, np.float32),
        "feat_q": np.zeros(n, np.float32),
        "price_pct": np.full(n, 0.5, np.float32),
        "price_pct_subcat": np.full(n, 0.5, np.float32),
        "trend_boost": np.zeros(n, np.float32),
        "new_bonus": np.zeros(n, np.float32),
        "restock_bonus": np.zeros(n, np.float32),
        "price_bonus": np.zeros(n, np.float32)}
    arrays = {k: v.copy() for k, v in defaults.items()}
    cursor = 0
    for row in product_rows:
        chunk_ids = _as_str_list(row.get("ids"))
        size = len(chunk_ids)
        sl = slice(cursor, cursor + size)
        for name in ("alpha0", "beta0", "s_slow", "f_slow", "s_fast",
                     "f_fast", "s_click", "f_click", "s_org", "f_org",
                     "org_v", "org_atc", "org_p", "feat_view", "feat_atc",
                     "feat_q", "price", "price_pct", "price_pct_subcat"):
            values = _as_float_list(row.get(name))
            if len(values) == size:
                arrays[name][sl] = np.asarray(values, dtype=np.float32)
        for name in ("active", "is_new"):
            values = row.get(name)
            if isinstance(values, (list, tuple)) and len(values) == size:
                arrays[name][sl] = np.asarray(
                    [bool(int(v)) for v in values], dtype=bool)
        for name, target in (("subcategory_code", "subcategory"),
                             ("brand_code", "brand"),
                             ("category_code", "category")):
            values = row.get(name)
            if isinstance(values, (list, tuple)) and len(values) == size:
                arrays[target][sl] = np.asarray(values, dtype=np.int16)
        cursor += size


    # Embeddings (float16 + base64 chunks).
    E = np.zeros((n, 512), dtype=np.float32)
    try:
        embed_rows = client.query(
            "SELECT * FROM %s WHERE kind::String = 'embedding_chunk' "
            "AND snapshot_id::String = '%s' ORDER BY chunk::Int32"
            % (table, sid), retries=2)
        for row in embed_rows:
            chunk_ids = _as_str_list(row.get("ids"))
            dim = int(row.get("dim") or 512)
            payload = row.get("data") or ""
            if not chunk_ids or not payload:
                continue
            block = decode_embedding_chunk(payload, len(chunk_ids), dim)
            if E.shape[1] != dim:
                E = np.zeros((n, dim), dtype=np.float32)
            start = index.get(chunk_ids[0])
            if start is not None:
                E[start:start + len(chunk_ids)] = block[:n - start]
    except Exception as exc:
        LOGGER.warning("embedding chunks unavailable: %s", exc)

    # Context model + calibration.
    mu_beta = np.zeros(cfg.context_dim, dtype=np.float32)
    Sigma = np.eye(cfg.context_dim, dtype=np.float32)
    feature_mean = np.zeros(cfg.context_dim, dtype=np.float32)
    feature_std = np.ones(cfg.context_dim, dtype=np.float32)
    feature_names: Tuple[str, ...] = ()
    calib_a, calib_T = cfg.calib_a, cfg.temperature
    w = np.asarray(cfg.default_position_w, dtype=np.float32)
    p95_price = 1.0
    tau_scale_view = 1.0
    tau_scale_buy = 1.0
    try:
        ctx_rows = client.query(
            "SELECT * FROM %s WHERE kind::String = 'context' "
            "AND snapshot_id::String = '%s' ORDER BY ts::DateTime64(3) DESC "
            "LIMIT 1" % (table, sid), retries=2)
        if ctx_rows:
            ctx = ctx_rows[0]
            mu_values = _as_float_list(ctx.get("mu"))
            if mu_values:
                mu_beta = np.asarray(mu_values, dtype=np.float32)
            sigma_values = _as_float_list(ctx.get("sigma"))
            d = mu_beta.size
            if len(sigma_values) == d * d:
                Sigma = np.asarray(sigma_values, dtype=np.float32).reshape(d, d)
            feature_mean = _as_float_array(ctx.get("feature_mean"), feature_mean)
            feature_std = _as_float_array(ctx.get("feature_std"), feature_std)
            feature_names = tuple(_as_str_list(ctx.get("feature_names")))
            calib_a = float(ctx.get("calib_a") or calib_a)
            calib_T = float(ctx.get("calib_T") or calib_T)
            w_values = _as_float_list(ctx.get("position_w"))
            if len(w_values) == 10:
                w = np.asarray(w_values, dtype=np.float32)
            p95_price = float(ctx.get("p95_price") or p95_price)
            tau_scale_view = float(ctx.get("tau_scale_view") or tau_scale_view)
            tau_scale_buy = float(ctx.get("tau_scale_buy") or tau_scale_buy)
    except Exception as exc:
        LOGGER.warning("context row unavailable: %s", exc)

    # Transition matrices.
    trans_view: Dict[int, Tuple[List[int], List[float]]] = {}
    trans_buy: Dict[int, Tuple[List[int], List[float]]] = {}
    try:
        trans_rows = client.query(
            "SELECT * FROM %s WHERE kind::String = 'transition_chunk' "
            "AND snapshot_id::String = '%s'" % (table, sid), retries=2)
        for row in trans_rows:
            src = index.get(str(row.get("src_id") or ""))
            if src is None:
                continue
            dst_ids = _as_str_list(row.get("dst_ids"))
            probs = _as_float_list(row.get("p"))
            targets = [index[pid] for pid in dst_ids if pid in index]
            probs = probs[:len(targets)]
            if not targets:
                continue
            bucket = trans_buy if str(row.get("matrix")) == "buy" else trans_view
            bucket[src] = (targets, probs)
    except Exception as exc:
        LOGGER.warning("transition chunks unavailable: %s", exc)


    def to_csr(rows: Dict[int, Tuple[List[int], List[float]]]) -> CSR:
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        for i in range(n):
            targets, probs = rows.get(i, ([], []))
            if targets:
                order = np.argsort(np.asarray(targets, dtype=np.int64))
                targets = [targets[int(o)] for o in order]
                probs = [probs[int(o)] if int(o) < len(probs) else 0.0
                         for o in order]
            indices.extend(int(t) for t in targets)
            data.extend(float(p) for p in probs)
            indptr.append(len(indices))
        return CSR(np.asarray(indptr, dtype=np.int64),
                   np.asarray(indices, dtype=np.int32),
                   np.asarray(data, dtype=np.float32), n)

    try:
        trend_rows = client.query(
            "SELECT * FROM %s WHERE kind::String = 'trend' "
            "AND snapshot_id::String = '%s' ORDER BY ts::DateTime64(3) DESC "
            "LIMIT 1" % (table, sid), retries=2)
        if trend_rows:
            items = trend_rows[0].get("items") or []
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except ValueError:
                    items = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                idx = index.get(str(item.get("id") or ""))
                if idx is not None:
                    arrays["trend_boost"][idx] = float(
                        item.get("boost") or 0.0)
    except Exception as exc:
        LOGGER.warning("trend row unavailable: %s", exc)

    meta_ts = str(meta.get("ts") or "")
    if meta_ts:
        try:
            delta_rows = client.query(
                "SELECT * FROM %s WHERE kind::String = 'product_delta' "
                "AND ts::DateTime64(3) > toDateTime64('%s', 3) "
                "ORDER BY ts::DateTime64(3)" % (table, meta_ts), retries=1)
            _apply_product_delta(arrays, index, delta_rows, cfg)
        except Exception as exc:
            LOGGER.info("no product deltas: %s", exc)

    arrays["alpha"] = arrays["alpha0"] + arrays["s_slow"] + arrays["s_fast"] + \
        arrays["s_org"]
    arrays["beta"] = arrays["beta0"] + arrays["f_slow"] + arrays["f_fast"] + \
        arrays["f_org"]
    total = arrays["alpha"] + arrays["beta"]
    cap = float(cfg.ess_max)
    over = total > cap
    if over.any():
        scale = cap / np.maximum(total[over], 1e-9)
        for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_org", "f_org"):
            arrays[name][over] *= scale
        arrays["alpha"] = arrays["alpha0"] + arrays["s_slow"] + \
            arrays["s_fast"] + arrays["s_org"]
        arrays["beta"] = arrays["beta0"] + arrays["f_slow"] + \
            arrays["f_fast"] + arrays["f_org"]


    eligible = arrays["active"] & (~arrays["unavailable"]) & (arrays["price"] > 0)
    mu = arrays["alpha"] / np.maximum(arrays["alpha"] + arrays["beta"], 1e-9)
    safe_order = np.argsort(np.where(eligible, mu, -1.0))[::-1][:10]
    best_safe = safe_order[eligible[safe_order]].astype(np.int64)
    if best_safe.size < 10:
        best_safe = np.flatnonzero(eligible)[:10].astype(np.int64)

    is_new = arrays["is_new"]
    new_product_set = frozenset(ids[i] for i in np.flatnonzero(is_new))
    new_subcats = frozenset(int(x) for x in np.unique(
        arrays["subcategory"][is_new])) if is_new.any() else frozenset()

    config = meta.get("config") or {}
    if not isinstance(config, Mapping):
        config = meta.get("config_json") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            config = {}
    regime = str(config.get("regime") or "normal")
    ess_max = int(config.get("ess_max") or cfg.ess_max)
    if regime == "final":
        policy = RuntimePolicy(mode="NORMAL", ess_max=ess_max, regime=regime,
                               temperature_mult=1.25, epsilon_explore=0.7,
                               c_ucb=0.25, c_new=0.8, eta_trend=1.0,
                               k_trend=0.01, mu_live=cfg.regime_mu_live_start)
    else:
        policy = RuntimePolicy(mode="NORMAL", ess_max=ess_max, regime=regime,
                               mu_live=(cfg.regime_mu_live_final
                                        if regime == "anneal" else 0.0))

    ts_value = parse_datetime(meta.get("ts"))
    return Snapshot(
        ids=ids, index=index, n=n,
        category=arrays["category"], subcategory=arrays["subcategory"],
        brand=arrays["brand"],
        category_names=tuple(_as_str_list(meta.get("category_names"))),
        subcategory_names=tuple(_as_str_list(meta.get("subcategory_names"))),
        brand_names=tuple(_as_str_list(meta.get("brand_names"))),
        price=arrays["price"],
        active=arrays["active"], unavailable=arrays["unavailable"],
        is_new=arrays["is_new"], alpha0=arrays["alpha0"],
        beta0=arrays["beta0"], alpha=arrays["alpha"], beta=arrays["beta"],
        s_slow=arrays["s_slow"], f_slow=arrays["f_slow"],
        s_fast=arrays["s_fast"], f_fast=arrays["f_fast"],
        s_click=arrays["s_click"], f_click=arrays["f_click"],
        s_org=arrays["s_org"], f_org=arrays["f_org"],
        org_v=arrays["org_v"], org_atc=arrays["org_atc"],
        org_p=arrays["org_p"], feat_view=arrays["feat_view"],
        feat_atc=arrays["feat_atc"], feat_q=arrays["feat_q"],
        price_pct=arrays["price_pct"],
        price_pct_subcat=arrays["price_pct_subcat"],
        E=E, trans_view=to_csr(trans_view), trans_buy=to_csr(trans_buy),
        mu_beta=mu_beta, Sigma_beta=Sigma,
        L_beta=cholesky_lower(Sigma), calib_a=calib_a, calib_T=calib_T,
        feature_names=feature_names, feature_mean=feature_mean,
        feature_std=feature_std, w=w,
        trend_boost=arrays["trend_boost"], new_bonus=arrays["new_bonus"],
        restock_bonus=arrays["restock_bonus"],
        price_bonus=arrays["price_bonus"], best_safe=best_safe,
        p95_price=p95_price, tau_scale_view=tau_scale_view,
        tau_scale_buy=tau_scale_buy, new_product_set=new_product_set,
        new_subcats=new_subcats, policy=policy,
        ts=(ts_value.timestamp() if ts_value else time.time()))



# --------------------------------------------------------------------------- #
# Incremental scratch rows and external policies
# --------------------------------------------------------------------------- #
def build_product_delta_row(snap: Snapshot, indices: Sequence[int],
                            sid: str) -> Dict[str, Any]:
    idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < snap.n)]
    row: Dict[str, Any] = {
        "kind": "product_delta", "snapshot_id": sid, "ts": iso_utc(),
        "ids": [snap.ids[int(i)] for i in idx]}
    for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_click",
                 "f_click", "s_org", "f_org", "price", "price_pct",
                 "price_pct_subcat", "trend_boost", "new_bonus",
                 "restock_bonus", "price_bonus", "alpha0", "beta0"):
        row[name] = _clean_floats(np.asarray(getattr(snap, name))[idx])
    row["active"] = _clean_ints(np.asarray(snap.active[idx], dtype=np.int8))
    row["is_new"] = _clean_ints(np.asarray(snap.is_new[idx], dtype=np.int8))
    return row


def write_product_delta(client: Any, snap: Snapshot,
                        indices: Sequence[int], cfg: Config = CFG,
                        snapshot_id: Optional[str] = None) -> int:
    if not indices:
        return 0
    return client.insert_rows(cfg.scratch_table,
                              [build_product_delta_row(
                                  snap, indices, snapshot_id or new_snapshot_id())],
                              database=client.team_id)


def write_trend_row(client: Any, snap: Snapshot, cfg: Config = CFG,
                    snapshot_id: Optional[str] = None,
                    top: int = 500) -> int:
    order = np.argsort(snap.trend_boost)[::-1][:top]
    items = [{"id": snap.ids[int(i)],
              "boost": round(float(snap.trend_boost[int(i)]), 6)}
             for i in order if snap.trend_boost[int(i)] != 0]
    return client.insert_rows(cfg.scratch_table, [{
        "kind": "trend", "snapshot_id": snapshot_id or new_snapshot_id(),
        "ts": iso_utc(), "items": items}], database=client.team_id)


def write_metrics_row(client: Any, metrics: Mapping[str, Any],
                      cfg: Config = CFG,
                      snapshot_id: Optional[str] = None) -> int:
    row = {"kind": "metrics", "snapshot_id": snapshot_id or new_snapshot_id(),
           "ts": iso_utc()}
    row.update({str(k): v for k, v in metrics.items()})
    return client.insert_rows(cfg.scratch_table, [row],
                              database=client.team_id)


def write_policy_row(client: Any, policy: Mapping[str, Any],
                     cfg: Config = CFG) -> int:
    row = {"kind": "control_tower_policy", "ts": iso_utc()}
    row.update({str(k): v for k, v in policy.items()})
    return client.insert_rows(cfg.scratch_table, [row],
                              database=client.team_id)


def read_latest_policy(client: Any, cfg: Config = CFG,
                       max_age_s: float = 900.0) -> Optional[Dict[str, Any]]:
    """Read the newest ``control_tower_policy`` row, safely and bounded."""
    try:
        rows = client.query(
            "SELECT * FROM %s.%s WHERE kind::String = "
            "'control_tower_policy' ORDER BY ts::DateTime64(3) DESC LIMIT 1"
            % (client.team_id, cfg.scratch_table), retries=1)
    except Exception:
        return None
    if not rows:
        return None
    row = {k: v for k, v in dict(rows[0]).items() if v is not None}
    ts = parse_datetime(row.get("ts"))
    if ts is not None:
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > max_age_s:
            return None
    return row


def clamp_policy(raw: Mapping[str, Any]) -> Dict[str, float]:
    """Strict clamping for anything the Control Tower publishes."""
    def _num(key: str, default: float, lo: float, hi: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError):
            value = default
        return float(np.clip(value, lo, hi))

    return {
        "restock_boost": _num("restock_boost", 1.0, 0.8, 1.25),
        "trend_scale": _num("trend_scale", 1.0, 0.5, 1.5),
        "temperature_mult": _num("temperature_mult", 1.0, 0.9, 1.15),
        "exploration_delta": _num("exploration_delta", 0.0, -0.10, 0.10),
    }

