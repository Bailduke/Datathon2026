"""Real-time inference: candidates, Thompson sampling, utility, slate order.

Spec Sections 3.3-3.8.  ``answer()`` is the only function on the critical
path.  It uses the immutable ``Snapshot`` and a per-request ``numpy`` generator,
never performs network I/O, and always returns ten unique eligible IDs when the
catalogue permits it.

Corrections applied here:
  #1 ``order_slate(..., rng)`` receives the generator explicitly.
  #2 page-dependent ``lambda_T_view`` / ``lambda_T_buy`` come from ``Config``.
  #3 ``cold_reserve`` compares ``snap.subcategory[snap.index[v]]`` codes.
  #6 unknown new products only wake the catalog worker (no HTTP in answer()).
  #7 sampling uses the pre-computed Cholesky factor ``snap.L_beta``.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import CFG, Config
from .features import (FEATURE_NAMES, ReqCtx, build_context_features,
                       build_request_context, click_penalty, price_delta,
                       price_norm, same_view_penalty)
from .state import Snapshot

EPS = 1e-5


def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (0.5 * (1.0 + np.tanh(0.5 * x))).astype(np.float32)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _mu(snap: Snapshot) -> np.ndarray:
    a = np.maximum(np.asarray(snap.alpha, dtype=np.float64), 1e-9)
    b = np.maximum(np.asarray(snap.beta, dtype=np.float64), 1e-9)
    return (a / (a + b)).astype(np.float32)


def eligible_mask(snap: Snapshot, bought: Sequence[int] = ()) -> np.ndarray:
    ok = snap.eligible_mask()
    if bought:
        try:
            ok[np.fromiter((int(i) for i in bought), dtype=np.int64)] = False
        except (ValueError, IndexError):
            pass
    return ok


def best_safe_candidates(snap: Snapshot, bought: Sequence[int] = (),
                         limit: int = 10) -> List[str]:
    """Eligible IDs, starting from the pre-computed best sellers."""
    ok = eligible_mask(snap, bought)
    out: List[str] = []
    seen = set()
    for idx in np.asarray(snap.best_safe, dtype=np.int64):
        if 0 <= int(idx) < snap.n and ok[int(idx)]:
            pid = snap.ids[int(idx)]
            if pid not in seen:
                out.append(pid)
                seen.add(pid)
                if len(out) >= limit:
                    return out
    remaining = np.flatnonzero(ok)
    if remaining.size:
        mu = _mu(snap)
        top = remaining[np.argsort(mu[remaining])[::-1]]
        for idx in top:
            pid = snap.ids[int(idx)]
            if pid not in seen:
                out.append(pid)
                seen.add(pid)
                if len(out) >= limit:
                    break
    return out


def fill_safe(snap: Snapshot, bought: Sequence[int],
              existing: Sequence[str], need: int = 10) -> List[str]:
    """Repair a slate to ``need`` unique, in-stock, non-bought IDs."""
    out: List[str] = []
    seen = set()
    for pid in existing:
        idx = snap.index.get(str(pid))
        if idx is None or idx in seen:
            continue
        if not snap.active[idx] or snap.unavailable[idx] or snap.price[idx] <= 0:
            continue
        if bought and idx in bought:
            continue
        out.append(str(pid))
        seen.add(idx)
        if len(out) >= need:
            return out
    for pid in best_safe_candidates(snap, bought, need):
        idx = snap.index.get(pid)
        if idx is None or idx in seen:
            continue
        out.append(pid)
        seen.add(idx)
        if len(out) >= need:
            break
    return out


def validate_items(items: Sequence[str], snap: Snapshot,
                   bought: Sequence[int] = ()) -> bool:
    if len(items) != 10 or len(set(items)) != 10:
        return False
    bought_set = set(int(i) for i in bought)
    for pid in items:
        idx = snap.index.get(str(pid))
        if idx is None or idx in bought_set:
            return False
        if not snap.active[idx] or snap.unavailable[idx] or snap.price[idx] <= 0:
            return False
    return True


def repair_items(items: Sequence[str], snap: Snapshot,
                 bought: Sequence[int] = ()) -> List[str]:
    return fill_safe(snap, bought, items, 10)


def _top_indices(scores: np.ndarray, k: int, valid: np.ndarray,
                 selected: np.ndarray, positive: bool = False) -> List[int]:
    if k <= 0:
        return []
    mask = valid & (~selected)
    if positive:
        mask &= scores > 0
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    if idx.size <= k:
        order = idx[np.argsort(scores[idx])[::-1]]
    else:
        part = np.argpartition(scores[idx], -k)[-k:]
        order = idx[part[np.argsort(scores[idx[part]])[::-1]]]
    return [int(i) for i in order]


def build_candidates(snap: Snapshot, valid: np.ndarray, rc: ReqCtx,
                     rng: np.random.Generator, cfg: Config,
                     policy: Any) -> np.ndarray:
    """Generate up to 100 candidates with per-source quotas (Section 3.3)."""
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.size != snap.n:
        mask = np.zeros(snap.n, dtype=bool)
        mask[valid.astype(np.int64)] = True
        valid = mask
    total = max(int(policy.n_candidates), 10)
    scale = total / 100.0
    quota_trans = max(5, int(35 * scale))
    quota_content = max(4, int(20 * scale))
    quota_trend = max(2, int(10 * scale))
    quota_prior = max(4, int(15 * scale))
    quota_explore = max(2, int(10 * scale))
    quota_safe = max(2, 10)

    chosen: List[int] = []
    selected = np.zeros(snap.n, dtype=bool)

    def add(indices: Sequence[int], cap: int) -> None:
        for idx in indices:
            idx = int(idx)
            if len(chosen) >= total:
                return
            if selected[idx] or not valid[idx]:
                continue
            chosen.append(idx)
            selected[idx] = True

    # 1) Session transitions (the strongest immediate-intent signal).
    tau_comb = np.asarray(rc.tau_view + rc.tau_buy, dtype=np.float32)
    add(_top_indices(tau_comb, quota_trans, valid, selected, True),
        quota_trans)
    # 2) Content similarity to the session centroid.
    if rc.c_t.size:
        content = (snap.E @ rc.c_t).astype(np.float32)
        add(_top_indices(content, quota_content, valid, selected), quota_content)
    # 3) Live trends (disabled by DEGRADED / SAFE policies).
    if policy.enable_trends:
        trend = np.asarray(snap.trend_boost, dtype=np.float32)
        add(_top_indices(trend, quota_trend, valid, selected, True), quota_trend)
    # 4) Hierarchical priors within the session categories (global if cold).
    mu = _mu(snap)
    if rc.idx_views:
        cats = np.unique(np.asarray(snap.category[
            np.asarray(rc.idx_views, dtype=np.int64)], dtype=np.int64))
        cat_mask = np.isin(np.asarray(snap.category, dtype=np.int64), cats)
        cat_valid = valid & cat_mask
    else:
        cat_valid = valid
    add(_top_indices(mu, quota_prior, cat_valid, selected), quota_prior)
    # 5) Exploration: Beta samples where ESS is low or the product is new.
    if policy.enable_exploration:
        ess = np.asarray(snap.alpha + snap.beta, dtype=np.float32)
        explore_mask = valid & ((ess < 50) | np.asarray(snap.is_new, dtype=bool))
        explore_idx = np.flatnonzero(explore_mask)
        if explore_idx.size:
            a = np.asarray(snap.alpha[explore_idx], dtype=np.float64)
            b = np.asarray(snap.beta[explore_idx], dtype=np.float64)
            samples = rng.beta(np.maximum(a, 1e-6), np.maximum(b, 1e-6))
            order = np.argsort(samples)[::-1][:quota_explore]
            add(explore_idx[order].tolist(), quota_explore)
    # 6) Safety: best sellers in stock.
    safe = [int(i) for i in np.asarray(snap.best_safe, dtype=np.int64)
            if 0 <= int(i) < snap.n]
    add(safe, quota_safe)

    # 7) Cold-start pool: new products touched by the session, their content
    #    neighbours, and their subcategory / brand peers (Section 3.7.2).
    if rc.cold_reserve or rc.cold:
        new_indices = [int(i) for i in snap.index.values()
                       if snap.is_new[int(i)] and valid[int(i)]]
        touched = [int(i) for i in rc.idx_views if snap.is_new[int(i)]]
        direct = [int(i) for pid, _ in rc.views
                  for i in [snap.index.get(pid)]
                  if i is not None and snap.is_new[int(i)] and valid[int(i)]]
        pool = list(dict.fromkeys(touched + direct))[:4]
        if not pool:
            pool = new_indices[:2]
        for idx in pool:
            add([idx], 1)
            if not rc.c_t.size:
                continue
            sim = (snap.E @ snap.E[idx]).astype(np.float32)
            add(_top_indices(sim, 25, valid, selected), 25)
            sub = int(snap.subcategory[idx])
            brand = int(snap.brand[idx])
            peers = valid & ((np.asarray(snap.subcategory) == sub) |
                             (np.asarray(snap.brand) == brand))
            add(_top_indices(mu, 10, peers, selected), 10)
            if len(chosen) >= total:
                break

    # Ensure a minimal pool even if every source was empty.
    if len(chosen) < 10:
        add([int(i) for i in valid[:max(100, total)]], max(100, total))

    if not chosen:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(chosen[:max(total, 10)], dtype=np.int64)
    # If truncation dropped all cold products, append a couple of them so the
    # cold reservation in order_slate() can still fire.
    if (rc.cold_reserve or rc.cold) and arr.size:
        new_local = [int(i) for i in np.asarray(snap.is_new, dtype=bool)
                     .nonzero()[0] if valid[int(i)]]
        if new_local and not any(bool(snap.is_new[int(i)]) for i in arr):
            arr = np.concatenate([arr[:-2] if arr.size >= 2 else arr[:0],
                                  np.asarray(new_local[:2], dtype=np.int64)])
    return arr.astype(np.int64)



def deterministic_scores(cand: np.ndarray, rc: ReqCtx, snap: Snapshot,
                         cfg: Config = CFG, policy: Any = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
    """Deterministic part of ``eta`` (Section 3.4.2, equation 38)."""
    policy = policy or snap.policy
    Z = build_context_features(cand, rc, snap)
    mu = _mu(snap)[cand]
    eta = logit(mu).astype(np.float32) + (Z @ np.asarray(
        snap.mu_beta, dtype=np.float32))
    eta = eta + rc.lambda_T_view * np.log1p(
        np.asarray(rc.tau_view[cand], dtype=np.float32))
    eta = eta + rc.lambda_T_buy * np.log1p(
        np.asarray(rc.tau_buy[cand], dtype=np.float32))
    content = (snap.E[cand] @ rc.c_t).astype(np.float32) if rc.c_t.size else \
        np.zeros(len(cand), dtype=np.float32)
    eta = eta + cfg.lambda_C * content
    eta = eta + cfg.lambda_price * price_delta(cand, rc, snap)
    if policy.enable_trends:
        eta = eta + float(policy.eta_trend) * np.asarray(
            snap.trend_boost[cand], dtype=np.float32) * float(
                policy.external_trend_scale)
    eta = eta + np.asarray(snap.new_bonus[cand], dtype=np.float32)
    eta = eta + np.asarray(snap.restock_bonus[cand], dtype=np.float32) * \
        float(policy.external_restock)
    eta = eta + np.asarray(snap.price_bonus[cand], dtype=np.float32)
    eta = eta - same_view_penalty(cand, rc, snap, cfg.same_view_delta)
    eta = eta - click_penalty(cand, rc, snap, cfg.click_penalty)

    t_eff = max(float(snap.calib_T) * float(policy.temperature_mult) *
                float(policy.external_temperature_mult), 1e-3)
    eta = float(snap.calib_a) + eta / t_eff
    p = expit(eta)
    U = p * (1.0 + cfg.rho * price_norm(cand, snap))
    return Z, eta.astype(np.float32), p, U.astype(np.float32)


def sample_scores(cand: np.ndarray, Z: np.ndarray, eta_det: np.ndarray,
                  rc: ReqCtx, snap: Snapshot, rng: np.random.Generator,
                  cfg: Config = CFG, policy: Any = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """One Thompson sample per product and one contextual draw per request.

    Correction #7: ``snap.L_beta`` is the Cholesky factor computed by the
    worker whenever ``Sigma_beta`` changes, so this is just ``mu + L @ z``.
    """
    policy = policy or snap.policy
    cand = np.asarray(cand, dtype=np.int64)
    alpha = np.maximum(np.asarray(snap.alpha[cand], dtype=np.float64), 1e-6)
    beta = np.maximum(np.asarray(snap.beta[cand], dtype=np.float64), 1e-6)
    theta = rng.beta(alpha, beta).astype(np.float32)
    mu = (alpha / (alpha + beta)).astype(np.float32)

    d = int(np.asarray(snap.mu_beta).size)
    z = rng.standard_normal(d).astype(np.float32)
    L = np.asarray(snap.L_beta, dtype=np.float32)
    beta_sample = np.asarray(snap.mu_beta, dtype=np.float32) + L @ z

    t_eff = max(float(snap.calib_T) * float(policy.temperature_mult) *
                float(policy.external_temperature_mult), 1e-3)
    delta = (logit(theta) - logit(mu) +
             Z @ (beta_sample - np.asarray(snap.mu_beta, dtype=np.float32))
             ) / t_eff
    eta = eta_det.astype(np.float32) + delta.astype(np.float32)
    p = expit(eta)
    U = p * (1.0 + cfg.rho * price_norm(cand, snap))
    return p, U.astype(np.float32)



def order_slate(cand: np.ndarray, p: np.ndarray, U: np.ndarray,
                U_mean: np.ndarray, snap: Snapshot, cfg: Config,
                cold_reserve: bool, rng: np.random.Generator,
                policy: Any = None) -> List[int]:
    """Greedy slate: exploit slot 1, MMR slots 2-8, explore 9-10.

    ``rng`` is an explicit argument (correction #1); the function never reads a
    global generator, which keeps synthetic latency tests reproducible.
    """
    policy = policy or snap.policy
    n = len(cand)
    if n == 0:
        return []
    if n <= 10:
        return list(range(n))

    sim = (snap.E[cand] @ snap.E[cand].T).astype(np.float32)
    np.clip(sim, 0.0, 1.0, out=sim)
    ess = np.asarray(snap.alpha[cand] + snap.beta[cand], dtype=np.float32)
    sub = np.asarray(snap.subcategory[cand], dtype=np.int64)
    cat = np.asarray(snap.category[cand], dtype=np.int64)
    brand = np.asarray(snap.brand[cand], dtype=np.int64)

    chosen: List[int] = []
    first = int(np.argmax(U_mean if cfg.exploit_slot1 else U))
    chosen.append(first)
    avail = np.ones(n, dtype=bool)
    avail[first] = False
    max_sim = sim[first].copy()

    sub_counts: dict = {int(sub[first]): 1}
    cat_counts: dict = {int(cat[first]): 1}
    brand_counts: dict = {int(brand[first]): 1}
    lambda_div = float(policy.lambda_div)

    def quota_ok(j: int, k: int) -> bool:
        s, c, b = int(sub[j]), int(cat[j]), int(brand[j])
        if sub_counts.get(s, 0) + 1 > 3:
            return False
        if cat_counts.get(c, 0) + 1 > 5:
            return False
        if k <= 5 and brand_counts.get(b, 0) + 1 > 2:
            return False
        return True

    def accept(j: int, k: int) -> None:
        chosen.append(j)
        avail[j] = False
        s, c, b = int(sub[j]), int(cat[j]), int(brand[j])
        sub_counts[s] = sub_counts.get(s, 0) + 1
        cat_counts[c] = cat_counts.get(c, 0) + 1
        brand_counts[b] = brand_counts.get(b, 0) + 1

    for k in range(2, 11):
        if len(chosen) >= 10:
            break
        if (k >= 9 and policy.enable_exploration and
                rng.random() < float(policy.epsilon_explore)):
            pool = [j for j in range(n) if avail[j] and
                    (ess[j] < 50 or bool(snap.is_new[cand[j]]))]
            if pool:
                pool_arr = np.asarray(pool, dtype=np.int64)
                score = U[pool_arr] * (1.0 + float(policy.c_ucb) /
                                       np.sqrt(ess[pool_arr] + 1.0))
                top_n = min(20, pool_arr.size)
                top = pool_arr[np.argsort(score)[::-1][:top_n]]
                order = top[rng.permutation(top.size)]
                picked = None
                for j in order:
                    if quota_ok(int(j), k):
                        picked = int(j)
                        break
                if picked is None:
                    picked = int(order[0])
                accept(picked, k)
                max_sim = np.maximum(max_sim, sim[picked])
                continue

        score = U * (1.0 - lambda_div * max_sim) if k <= 8 else U.copy()
        candidates = np.flatnonzero(avail)
        if candidates.size == 0:
            break
        candidates = candidates[np.argsort(score[candidates])[::-1]]
        picked = None
        for j in candidates:
            if quota_ok(int(j), k):
                picked = int(j)
                break
        if picked is None:                    # relax soft quotas, never empty
            picked = int(candidates[0])
        accept(picked, k)
        max_sim = np.maximum(max_sim, sim[picked])

    return chosen[:10]


def apply_cold_reservation(chosen: Sequence[int], cand: np.ndarray,
                           p: np.ndarray, snap: Snapshot,
                           cfg: Config = CFG) -> List[int]:
    """Reserve slots 3-6 for a new product / new-subcategory product."""
    out = list(int(x) for x in chosen)
    if len(out) < 4 or len(cand) == 0:
        return out

    def is_cold_pos(j: int) -> bool:
        idx = int(cand[j])
        return bool(snap.is_new[idx]) or (
            int(snap.subcategory[idx]) in snap.new_subcats)

    if is_cold_pos(out[0]):
        for slot in (2, 3, 4, 5):
            if slot < len(out) and not is_cold_pos(out[slot]):
                out[0], out[slot] = out[slot], out[0]
                break

    threshold = 0.7 * float(p[out[-1]]) if out else 0.0
    cold_available = [j for j in range(len(cand))
                      if j not in out and is_cold_pos(j)
                      and float(p[j]) >= threshold]
    if not cold_available:
        return out
    best_new = max(cold_available, key=lambda j: float(p[j]))
    targets = [slot for slot in (2, 3, 4, 5) if slot < len(out)]
    if not targets:
        return out
    weak = min(targets, key=lambda slot: float(p[out[slot]]))
    out[weak] = best_new
    deduped: List[int] = []
    seen = set()
    for j in out:
        if j not in seen:
            deduped.append(j)
            seen.add(j)
    if len(deduped) < 10:
        for j in range(len(cand)):
            if j not in seen:
                deduped.append(j)
                seen.add(j)
                if len(deduped) >= 10:
                    break
    return deduped[:10]



def score_candidates(cand: np.ndarray, rc: ReqCtx, snap: Snapshot,
                     rng: np.random.Generator, cfg: Config = CFG,
                     sampled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning ``(p, U)`` for the notebook."""
    Z, eta, p, U = deterministic_scores(cand, rc, snap, cfg, snap.policy)
    if sampled:
        return sample_scores(cand, Z, eta, rc, snap, rng, cfg, snap.policy)
    return p, U


def answer(req: Mapping[str, Any], snap: Snapshot,
           rng: np.random.Generator, cfg: Config = CFG) -> List[str]:
    """Return exactly ten unique, eligible product IDs for one request.

    This is the critical path: no network, no locks, no exceptions.  If
    anything goes wrong it degrades to the in-stock best-seller list.
    """
    bought_seq: Sequence[int] = ()
    try:
        now = req.get("_now")
        if not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
        rc = build_request_context(req, snap, cfg, now)
        bought = snap.user_bought.get(rc.user, frozenset()) or frozenset()
        bought_seq = tuple(int(i) for i in bought)
        policy = snap.policy
        if policy.mode == "SAFE":
            return fill_safe(snap, bought_seq, [], 10)

        ok = eligible_mask(snap, bought_seq)
        valid = np.flatnonzero(ok)
        if valid.size < 10:
            return fill_safe(snap, bought_seq, [], 10)

        cand = build_candidates(snap, valid, rc, rng, cfg, policy)
        if cand.size < 10:
            extra = [int(i) for i in valid if int(i) not in set(cand.tolist())]
            cand = np.asarray(list(cand) + extra[:max(10, 100 - cand.size)],
                              dtype=np.int64)
        if cand.size == 0:
            return fill_safe(snap, bought_seq, [], 10)

        Z, eta_det, p_det, U_det = deterministic_scores(
            cand, rc, snap, cfg, policy)
        p, U = sample_scores(cand, Z, eta_det, rc, snap, rng, cfg, policy)
        chosen = order_slate(cand, p, U, U_det, snap, cfg,
                             rc.cold_reserve, rng, policy)
        if rc.cold_reserve:
            chosen = apply_cold_reservation(chosen, cand, p, snap, cfg)

        out: List[str] = []
        seen = set()
        for local in chosen:
            idx = int(cand[local])
            if idx in seen or not ok[idx]:
                continue
            seen.add(idx)
            out.append(snap.ids[idx])
        if len(out) < 10:
            out = fill_safe(snap, bought_seq, out, 10)
        return out[:10]
    except Exception:  # reliability: never break the request loop
        try:
            return fill_safe(snap, bought_seq, [], 10)
        except Exception:
            return []


def answer_with_debug(req: Mapping[str, Any], snap: Snapshot,
                      rng: np.random.Generator, cfg: Config = CFG
                      ) -> Tuple[List[str], Dict[str, Any]]:
    """Same as ``answer`` but returns timing/decomposition for the 1% log."""
    started = time.perf_counter()
    items = answer(req, snap, rng, cfg)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return items, {"latency_ms": elapsed_ms, "n_items": len(items),
                   "model_version": cfg.model_version}

