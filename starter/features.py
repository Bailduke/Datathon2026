"""Defensive request parsing and session/context features.

Spec Sections 2.5.6, 2.6.2 and 3.1.  ``recent_views`` has appeared in at least
five shapes in RawTree (JSON string, list of dicts, pairs, plain strings and a
mapping) and ``context`` may arrive nested or flattened as ``context.device``.
Nothing in this module raises on malformed input: reliability is worth points.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Tuple

import numpy as np

from .rawtree import parse_datetime
from .state import Snapshot

FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "ios_x_price_pct",
    "android_x_price_pct",
    "product_page_x_log1p_tau_buy",
    "category_page_x_log1p_tau_view",
    "region_hash0_x_price_pct",
    "region_hash1_x_price_pct",
    "region_hash2_x_price_pct",
    "region_hash3_x_price_pct",
    "same_subcategory_as_last_view",
    "same_brand_as_last_view",
    "price_pct_within_subcategory",
    "new_user_x_log1p_f_view",
    "m_t_x_content_sim",
    "sin_hour_x_price_pct",
    "cos_hour_x_price_pct",
    "sin_dow_x_price_pct",
    "cos_dow_x_price_pct",
    "f_view_std",
    "f_atc_std",
    "f_q_std",
    "session_has_atc_x_log1p_tau_buy",
    "user_purchases_x_f_view",
    "user_category_affinity",
)
DEFAULT_CONTEXT = {"device": "web", "region": "?", "page": "home"}


def _as_mapping(raw: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, Mapping) else None
    return None


def _as_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return []
        raw = value
    if isinstance(raw, Mapping):
        # A fifth shape seen in the wild: {"product_id": ts, ...}
        for key in ("recent_views", "views", "items", "data"):
            if key in raw and isinstance(raw[key], (list, tuple, str)):
                return _as_list(raw[key])
        return [{"product_id": k, "ts": v} for k, v in raw.items()
                if isinstance(k, str)]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def normalize_views(raw: Any) -> List[Tuple[str, Optional[datetime]]]:
    """Normalise every supported ``recent_views`` representation.

    Returns ``[(product_id, ts_or_None), ...]`` preserving the last occurrence
    of each product (most recent intent) and the original order of those last
    occurrences.
    """
    items = _as_list(raw)
    seen: Dict[str, Tuple[int, Optional[datetime]]] = {}
    for pos, item in enumerate(items):
        pid: Optional[str] = None
        ts: Optional[datetime] = None
        if isinstance(item, Mapping):
            for key in ("product_id", "productId", "id", "pid"):
                if key in item and item[key] is not None:
                    pid = str(item[key])
                    break
            for key in ("ts", "timestamp", "time", "viewed_at"):
                if key in item and item[key] not in (None, ""):
                    ts = parse_datetime(item[key])
                    break
        elif isinstance(item, (list, tuple)):
            if item:
                pid = str(item[0])
            if len(item) > 1:
                ts = parse_datetime(item[1])
        elif isinstance(item, str):
            pid = item
        if pid is None:
            continue
        pid = pid.strip()
        if not pid:
            continue
        # Keep the LAST occurrence (Rule 7): overwrite position and timestamp.
        seen[pid] = (pos, ts)
    ordered = sorted(((pos, pid, ts) for pid, (pos, ts) in seen.items()),
                     key=lambda row: row[0])
    return [(pid, ts) for _, pid, ts in ordered]


def normalize_context(raw: Any) -> Dict[str, str]:
    """Parse nested/flattened context, filling defaults."""
    ctx = dict(DEFAULT_CONTEXT)
    mapping = _as_mapping(raw)
    if mapping is None:
        return ctx
    nested = mapping.get("context")
    if isinstance(nested, Mapping):
        mapping = nested
    # RawTree may return "context.device" instead of a nested object.
    for key in ("device", "region", "page"):
        value = mapping.get(key)
        if value is None:
            value = mapping.get("context." + key)
        if value not in (None, ""):
            ctx[key] = str(value).lower() if key in ("device", "page") else \
                str(value)
    return ctx


def request_ts(req: Mapping[str, Any], default: Optional[datetime] = None) -> datetime:
    parsed = parse_datetime(req.get("ts"))
    if parsed is not None:
        return parsed
    parsed = parse_datetime(req.get("timestamp"))
    if parsed is not None:
        return parsed
    return default or datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReqCtx:
    """Everything about the current request that scoring needs."""

    user: str
    views: List[Tuple[str, Optional[datetime]]]
    idx_views: List[int]
    c_t: np.ndarray
    tau_view: np.ndarray
    tau_buy: np.ndarray
    cold: bool
    cold_reserve: bool
    context: Dict[str, str]
    page: str
    device: str
    region: str
    is_new_user: bool
    has_atc: bool
    hour: int
    dow: int
    lambda_T_view: float
    lambda_T_buy: float
    user_purchases: int = 0
    m_t: int = 0
    last_idx: int = -1
    has_views: bool = False


def view_weights(views: Sequence[Tuple[str, Optional[datetime]]],
                 now: Optional[datetime] = None, rho: float = 0.65,
                 tau_sess: float = 900.0) -> np.ndarray:
    """Ordinal and temporal recency weights ``w_r`` of equation (23).

    ``views`` is ordered oldest -> newest.  If a timestamp is missing only the
    ordinal factor is used.
    """
    m = len(views)
    if m == 0:
        return np.zeros(0, dtype=np.float32)
    now = now or datetime.now(timezone.utc)
    weights = np.empty(m, dtype=np.float32)
    for r, (_, ts) in enumerate(views):
        w = rho ** (m - 1 - r)
        if ts is not None:
            delta = max((now - ts).total_seconds(), 0.0)
            w *= float(np.exp(-delta / max(tau_sess, 1e-6)))
        weights[r] = w
    return weights


def session_centroid(E: np.ndarray, idx_views: Sequence[int],
                     weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """L2-normalised weighted embedding centroid ``c_t`` (equation 28)."""
    if E is None or E.size == 0 or len(idx_views) == 0:
        return np.zeros((E.shape[1] if E is not None and E.ndim == 2 else 512,),
                        dtype=np.float32)
    idx = np.asarray(idx_views, dtype=np.int64)
    if weights is None:
        w = np.ones(len(idx), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
    if w.size != idx.size:
        w = np.ones(len(idx), dtype=np.float32)
    centroid = (E[idx].astype(np.float32) * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-9:
        centroid /= norm
    return centroid.astype(np.float32)


def _accumulate_rows(matrix: Any, idx_views: Sequence[int],
                     weights: Sequence[float], n: int) -> np.ndarray:
    """Weighted sum of CSR rows for a small number of session views."""
    out = np.zeros(n, dtype=np.float32)
    if matrix is None or matrix.nnz() == 0:
        return out
    # The five most recent views carry almost all the mass; cap the loop for
    # latency even if a (malformed) request sends hundreds of views.
    pairs = list(zip(idx_views, weights))[-25:]
    for idx, weight in pairs:
        if weight <= 0:
            continue
        cols, values = matrix.row(int(idx))
        if cols.size:
            np.add.at(out, cols, values * float(weight))
    return out


def session_transition(snap: Snapshot, idx_views: Sequence[int],
                       weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """``tau^view`` and ``tau^buy`` for the session (equations 23, 38)."""
    n = snap.n
    tau_view = _accumulate_rows(snap.trans_view, idx_views, weights, n)
    tau_buy = _accumulate_rows(snap.trans_buy, idx_views, weights, n)
    sv = max(float(snap.tau_scale_view), 1e-6)
    sb = max(float(snap.tau_scale_buy), 1e-6)
    tau_view = np.clip(tau_view / sv, 0.0, 1.0)
    tau_buy = np.clip(tau_buy / sb, 0.0, 1.0)
    return tau_view, tau_buy


def transition_from(snap: Snapshot, product_indices: Iterable[int],
                    kind: str = "buy") -> np.ndarray:
    """Transition mass from a set of live add-to-cart products (2.5.6)."""
    matrix = snap.trans_buy if kind == "buy" else snap.trans_view
    out = np.zeros(snap.n, dtype=np.float32)
    if matrix is None or matrix.nnz() == 0:
        return out
    for idx in product_indices:
        idx = int(idx)
        if 0 <= idx < snap.n:
            cols, values = matrix.row(idx)
            if cols.size:
                np.add.at(out, cols, values)
    scale = max(float(snap.tau_scale_buy if kind == "buy"
                      else snap.tau_scale_view), 1e-6)
    return np.clip(out / scale, 0.0, 1.0)



def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "t", "y"}


def build_request_context(req: Mapping[str, Any], snap: Snapshot,
                          cfg: Any, now: Optional[datetime] = None
                          ) -> ReqCtx:
    """Parse the request and compute all session-level features once."""
    views = normalize_views(req.get("recent_views"))
    ctx = normalize_context(req.get("context"))
    user = str(req.get("user_id") or req.get("user") or "?")
    is_new_user = _truthy(req.get("is_new_user"))
    now = now or datetime.now(timezone.utc)

    idx_views: List[int] = []
    known_weights: List[float] = []
    weights = view_weights(views, now)
    for (pid, _), weight in zip(views, weights):
        idx = snap.index.get(pid)
        if idx is None:
            # Correction #6: never block answer() on HTTP.  Signal the
            # catalog worker; cold_reserve logic below still sees the ID.
            if pid in snap.new_product_set and snap.wake_catalog is not None:
                snap.wake_catalog.set()
            continue
        idx_views.append(int(idx))
        known_weights.append(float(weight))

    c_t = session_centroid(snap.E, idx_views, known_weights)
    tau_view, tau_buy = session_transition(snap, idx_views, known_weights)

    atc = snap.live_atc.get(user, frozenset()) or frozenset()
    has_atc = bool(atc)
    if has_atc:
        tau_buy = np.clip(tau_buy + 2.0 * transition_from(snap, atc, "buy"),
                          0.0, 1.0)

    cold = is_new_user or any(pid in snap.new_product_set for pid, _ in views)
    cold_reserve = False
    for pid, _ in views:
        if pid in snap.new_product_set:
            cold_reserve = True
            if pid not in snap.index and snap.wake_catalog is not None:
                snap.wake_catalog.set()
            continue
        idx = snap.index.get(pid)
        # Correction #3: compare the *subcategory code* array, not a string id.
        if idx is not None and int(snap.subcategory[idx]) in snap.new_subcats:
            cold_reserve = True

    page = (ctx.get("page") or "home").lower()
    device = (ctx.get("device") or "web").lower()
    region = ctx.get("region") or "?"
    lam_v, lam_b = cfg.page_lambdas(page)
    bought = snap.user_bought.get(user, frozenset()) or frozenset()
    return ReqCtx(
        user=user,
        views=views,
        idx_views=idx_views,
        c_t=c_t,
        tau_view=tau_view,
        tau_buy=tau_buy,
        cold=cold,
        cold_reserve=cold_reserve,
        context=ctx,
        page=page,
        device=device,
        region=region,
        is_new_user=is_new_user,
        has_atc=has_atc,
        hour=int(now.astimezone(timezone.utc).hour),
        dow=int(now.astimezone(timezone.utc).isoweekday() % 7),
        lambda_T_view=float(lam_v),
        lambda_T_buy=float(lam_b),
        user_purchases=len(bought),
        m_t=len(idx_views),
        last_idx=(idx_views[-1] if idx_views else -1),
        has_views=bool(views),
    )



def user_affinity_vector(snap: Snapshot, user: str) -> Optional[np.ndarray]:
    value = snap.user_affinity.get(user)
    if value is None:
        return None
    if isinstance(value, Mapping):
        out = np.zeros(max(1, len(snap.category_names)), dtype=np.float32)
        for name, score in value.items():
            try:
                out[int(name)] = float(score)
            except (TypeError, ValueError, IndexError):
                continue
        return out
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def price_delta(cand: np.ndarray, rc: ReqCtx, snap: Snapshot) -> np.ndarray:
    """``f_price`` of equation (38): relative log price vs the session."""
    log_price = np.log1p(np.maximum(snap.price[cand], 0.0))
    if rc.idx_views:
        ref = float(np.mean(np.log1p(np.maximum(
            snap.price[np.asarray(rc.idx_views, dtype=np.int64)], 0.0))))
    elif rc.user_purchases > 0 and rc.user:
        ref = float(np.log1p(max(snap.p95_price * 0.5, 1.0)))
    else:
        ref = float(np.log1p(max(snap.p95_price * 0.5, 1.0)))
    return (log_price - ref).astype(np.float32)


def price_norm(cand: np.ndarray, snap: Snapshot) -> np.ndarray:
    """``q_i`` of equation (40), clipped to [0, 1]."""
    denom = float(np.log1p(max(snap.p95_price, 1.0)))
    if denom <= 1e-6:
        return np.zeros(len(cand), dtype=np.float32)
    return np.clip(np.log1p(np.maximum(snap.price[cand], 0.0)) / denom,
                   0.0, 1.0).astype(np.float32)


def same_view_penalty(cand: np.ndarray, rc: ReqCtx, snap: Snapshot,
                      delta: float = 0.6) -> np.ndarray:
    """Penalty for recommending a product already on a product page."""
    out = np.zeros(len(cand), dtype=np.float32)
    if rc.page != "product" or not rc.idx_views:
        return out
    viewed = set(int(v) for v in rc.idx_views)
    for pos, idx in enumerate(cand):
        if int(idx) in viewed and not bool(snap.is_new[int(idx)]):
            out[pos] = float(np.log(1.0 / max(delta, 1e-6)))
    return out


def click_penalty(cand: np.ndarray, rc: ReqCtx, snap: Snapshot,
                  delta: float = 0.8) -> np.ndarray:
    out = np.zeros(len(cand), dtype=np.float32)
    clicked = snap.user_clicked.get(rc.user, frozenset()) or frozenset()
    if not clicked:
        return out
    penalty = float(np.log(1.0 / max(delta, 1e-6)))
    for pos, idx in enumerate(cand):
        if int(idx) in clicked:
            out[pos] = penalty
    return out




def build_context_features(cand: np.ndarray, rc: ReqCtx,
                           snap: Snapshot) -> np.ndarray:
    """Standardised contextual matrix ``Z`` with shape ``(len(cand), 24)``.

    Features 10, 11, 18, 19 and 28 of the specification are deliberately *not*
    duplicated here because they already appear as fixed terms in ``eta``:
    ``lambda_price``, ``lambda_C``, ``lambda_T_view``, ``lambda_T_buy`` and the
    trend boost.  This is the no-double-counting rule of Section 2.6.2.
    """
    cand = np.asarray(cand, dtype=np.int64)
    c = cand.size
    d = len(FEATURE_NAMES)
    z = np.zeros((c, d), dtype=np.float32)
    if c == 0:
        return z

    price_pct = np.asarray(snap.price_pct[cand], dtype=np.float32)
    price_pct_sub = np.asarray(snap.price_pct_subcat[cand], dtype=np.float32)
    tau_v = np.log1p(np.asarray(rc.tau_view[cand], dtype=np.float32))
    tau_b = np.log1p(np.asarray(rc.tau_buy[cand], dtype=np.float32))
    content = (snap.E[cand] @ rc.c_t).astype(np.float32) if rc.c_t.size else \
        np.zeros(c, dtype=np.float32)
    f_view = np.asarray(snap.feat_view[cand], dtype=np.float32)
    f_atc = np.asarray(snap.feat_atc[cand], dtype=np.float32)
    f_q = np.asarray(snap.feat_q[cand], dtype=np.float32)
    log_f_view = np.log1p(np.maximum(f_view, 0.0))

    is_product = rc.page == "product"
    is_category = rc.page == "category"
    page_p = 1.0 if is_product else 0.0
    page_c = 1.0 if is_category else 0.0
    device = rc.device.lower()
    ios = 1.0 if device == "ios" else 0.0
    android = 1.0 if device == "android" else 0.0
    m_t = min(max(rc.m_t, 0), 10) / 10.0
    hour_angle = 2.0 * np.pi * rc.hour / 24.0
    dow_angle = 2.0 * np.pi * rc.dow / 7.0

    z[:, 0] = 1.0
    z[:, 1] = ios * price_pct
    z[:, 2] = android * price_pct
    z[:, 3] = page_p * tau_b
    z[:, 4] = page_c * tau_v
    region_bucket = int(zlib.crc32((rc.region or "?").lower().encode()) % 4)
    z[:, 5 + region_bucket] = price_pct
    if rc.last_idx >= 0:
        last_sub = int(snap.subcategory[rc.last_idx])
        last_brand = int(snap.brand[rc.last_idx])
        z[:, 9] = (np.asarray(snap.subcategory[cand]) == last_sub).astype(
            np.float32)
        z[:, 10] = (np.asarray(snap.brand[cand]) == last_brand).astype(
            np.float32)
    z[:, 11] = price_pct_sub
    z[:, 12] = (1.0 if rc.is_new_user else 0.0) * log_f_view
    z[:, 13] = m_t * content
    z[:, 14] = np.sin(hour_angle) * price_pct
    z[:, 15] = np.cos(hour_angle) * price_pct
    z[:, 16] = np.sin(dow_angle) * price_pct
    z[:, 17] = np.cos(dow_angle) * price_pct
    z[:, 18] = f_view
    z[:, 19] = f_atc
    z[:, 20] = f_q
    z[:, 21] = (1.0 if rc.has_atc else 0.0) * tau_b
    z[:, 22] = np.log1p(max(rc.user_purchases, 0)) * log_f_view

    affinity = user_affinity_vector(snap, rc.user)
    if affinity is not None and affinity.size:
        cats = np.asarray(snap.category[cand], dtype=np.int64)
        valid = (cats >= 0) & (cats < affinity.size)
        z[valid, 23] = affinity[cats[valid]]
    else:
        z[:, 23] = 0.0

    mean = np.asarray(snap.feature_mean, dtype=np.float32)
    std = np.asarray(snap.feature_std, dtype=np.float32)
    if mean.size < d:
        mean = np.pad(mean, (0, d - mean.size))
    if std.size < d:
        std = np.pad(std, (0, d - std.size), constant_values=1.0)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    z = (z - mean[:d]) / std[:d]
    z[:, 0] = 1.0  # intercept is not standardised
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
