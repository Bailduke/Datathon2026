"""Asynchronous workers: request/push, feedback, events, catalog, persist,
health and the regime shift.

All worker mutations happen on ``Engine.mutable``; the inference path only
sees the ``Snapshot`` published in one atomic reference assignment.  Sets
stored in caches are always frozensets updated copy-on-write (correction #4),
so ``answer()`` can iterate them while a worker is running.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

import numpy as np

from .config import CFG, Config
from .features import (FEATURE_NAMES, build_context_features,
                       build_request_context, normalize_views,
                       session_centroid, session_transition, view_weights)
from .priors import (blend_transitions, build_product_delta_row,
                     build_structural_prior, build_transition_from_edges,
                     clamp_policy, fit_logistic_map, fit_calibration,
                     organic_pseudo_counts, read_latest_policy,
                     save_snapshot_npz, top_k_rows, write_metrics_row,
                     write_product_delta, write_snapshot_to_scratch,
                     write_trend_row)
from .rawtree import (EmptyTableError, QueryTimeoutError, iso_utc,
                      parse_datetime, to_jsonable)
from .scoring import (answer, best_safe_candidates, eligible_mask,
                      fill_safe, repair_items, validate_items)
from .state import (CSR, LRU, RuntimePolicy, Snapshot, StateHolder,
                    cholesky_lower)
from .text import hashed_embedding, build_embeddings

LOGGER = logging.getLogger("workers")
EPS = 1e-9


# --------------------------------------------------------------------------- #
# SQL used by the live loops (Sections 5.7 and 5.8)
# --------------------------------------------------------------------------- #
SQL_REQUESTS = """
SELECT request_id::String AS request_id,
       ts::DateTime64(3) AS ts,
       user_id::String AS user_id,
       toString(is_new_user) IN ('true', '1') AS is_new_user,
       context,
       recent_views
FROM shop.requests
WHERE ts > now() - INTERVAL {window} SECOND
ORDER BY ts::DateTime64(3) ASC
LIMIT {limit}"""

SQL_FEEDBACK = """
SELECT request_id::String AS request_id,
       ts::DateTime64(3) AS ts,
       shown::Array(String) AS shown,
       served_by::String AS served_by,
       fallback_reason::String AS fallback_reason,
       toFloat64OrZero(toString(latency_ms)) AS latency_ms,
       ifNull(toString(clicked), '') AS clicked,
       ifNull(toString(purchased), '') AS purchased,
       toFloat64OrZero(toString(revenue_eur)) AS revenue_eur
FROM {db}.feedback
WHERE request_id != '__init__'
  AND ts > now() - INTERVAL {window} SECOND
ORDER BY ts::DateTime64(3) DESC
LIMIT {limit}"""

SQL_EVENTS = """
SELECT ts::DateTime64(3) AS ts,
       event_type::String AS event_type,
       user_id::String AS user_id,
       session_id::String AS session_id,
       product_id::String AS product_id,
       toFloat64OrZero(toString(price_eur)) AS price_eur,
       device::String AS device,
       region::String AS region
FROM shop.events
WHERE ts > toDateTime64('{watermark}', 3) - INTERVAL 2 SECOND
ORDER BY ts::DateTime64(3)
LIMIT 5000"""

SQL_CATALOG_UPDATES = """
SELECT product_id::String AS product_id,
       ts::DateTime64(3) AS ts,
       change::String AS change,
       toString(category) AS category,
       toString(subcategory) AS subcategory,
       toString(brand) AS brand,
       toFloat64OrZero(toString(price_eur)) AS price_eur,
       toString(description) AS description,
       toString(is_active) AS active_s
FROM shop.catalog_updates
WHERE ts > toDateTime64('{watermark}', 3) - INTERVAL 5 SECOND
ORDER BY ts::DateTime64(3) ASC
LIMIT 2000"""

SQL_CATALOG = """
WITH src AS (
    SELECT product_id::String AS product_id,
           toString(category) AS category,
           toString(subcategory) AS subcategory,
           toString(brand) AS brand,
           toFloat64OrZero(toString(price_eur)) AS price_eur,
           toString(description) AS description,
           toDateTime64('1970-01-01 00:00:00', 3) AS updated_at,
           toUInt8(0) AS prio,
           toString(is_active) AS active_s
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String,
           toString(category),
           toString(subcategory),
           toString(brand),
           toFloat64OrZero(toString(price_eur)),
           toString(description),
           ts::DateTime64(3),
           toUInt8(1),
           toString(is_active)
    FROM shop.catalog_updates
)
SELECT product_id, category, subcategory, brand, price_eur, description,
       (active_s = 'true' OR active_s = '1') AS is_active, updated_at
FROM src
ORDER BY product_id, prio DESC, updated_at DESC
LIMIT 1 BY product_id"""

SQL_OUT_OF_STOCK = """
SELECT product_id::String AS product_id,
       argMax(change::String, ts::DateTime64(3)) AS last_change
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING last_change = 'out_of_stock'"""


# --------------------------------------------------------------------------- #
# Mutable worker state (never published directly)
# --------------------------------------------------------------------------- #
@dataclass
class MutableState:
    cfg: Config
    ids: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)
    n: int = 0

    category: np.ndarray = None
    subcategory: np.ndarray = None
    brand: np.ndarray = None
    price: np.ndarray = None
    active: np.ndarray = None
    unavailable: np.ndarray = None
    is_new: np.ndarray = None
    alpha0: np.ndarray = None
    beta0: np.ndarray = None
    s_slow: np.ndarray = None
    f_slow: np.ndarray = None
    s_fast: np.ndarray = None
    f_fast: np.ndarray = None
    s_click: np.ndarray = None
    f_click: np.ndarray = None
    s_org: np.ndarray = None
    f_org: np.ndarray = None
    org_v: np.ndarray = None
    org_atc: np.ndarray = None
    org_p: np.ndarray = None
    feat_view: np.ndarray = None
    feat_atc: np.ndarray = None
    feat_q: np.ndarray = None
    price_pct: np.ndarray = None
    price_pct_subcat: np.ndarray = None
    E: np.ndarray = None
    trans_view: CSR = None
    trans_buy: CSR = None
    mu_beta: np.ndarray = None
    Sigma_beta: np.ndarray = None
    feature_mean: np.ndarray = None
    feature_std: np.ndarray = None
    w: np.ndarray = None
    trend_boost: np.ndarray = None
    new_bonus: np.ndarray = None
    restock_bonus: np.ndarray = None
    price_bonus: np.ndarray = None
    best_safe: np.ndarray = None
    last_change_ts: np.ndarray = None

    category_names: List[str] = field(default_factory=list)
    subcategory_names: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)

    new_product_set: Set[str] = field(default_factory=set)
    new_subcats: Set[int] = field(default_factory=set)

    live_atc: LRU = None
    user_bought: LRU = None
    user_clicked: LRU = None
    user_affinity: LRU = None
    req_meta: LRU = None
    seen_feedback: LRU = None
    seen_events: LRU = None
    submitted: LRU = None
    seen_requests: LRU = None

    ring_view: np.ndarray = None
    ring_atc: np.ndarray = None
    ring_purchase: np.ndarray = None
    ring_mark: np.ndarray = None
    last_event_ts: float = 0.0
    last_catalog_ts: float = 0.0
    live_edges: Dict[Tuple[int, int], float] = field(default_factory=dict)
    session_views: LRU = None

    dirty: Set[int] = field(default_factory=set)
    feedback_obs: List[Tuple[np.ndarray, float, float]] = field(
        default_factory=list)
    recent_feedback: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    circuit: str = "NORMAL"
    circuit_errors: int = 0
    regime: str = "normal"
    regime_started: float = 0.0
    external_policy: Dict[str, float] = field(default_factory=dict)
    last_decay: float = 0.0
    last_variance: float = 0.0
    last_refit: float = 0.0
    last_persist: float = 0.0
    last_full_persist: float = 0.0
    last_reconcile: float = 0.0
    last_trend: float = 0.0
    last_metrics_persist: float = 0.0
    cusum: float = 0.0
    cusum_ref: float = 0.04
    feedback_first: bool = True
    lock: Any = field(default_factory=threading.RLock)


def _zeros(n: int, dtype=np.float32) -> np.ndarray:
    return np.zeros(n, dtype=dtype)


def _code_for(names: List[str], value: Any) -> int:
    text = str(value or "").strip() or "?"
    try:
        return names.index(text)
    except ValueError:
        names.append(text)
        return len(names) - 1



def make_mutable(snapshot: Snapshot, cfg: Config = CFG) -> MutableState:
    """Allocate pre-sized buffers (N_MAX) and copy a published snapshot."""
    n_max = int(cfg.n_max)
    n = int(snapshot.n)
    d = int(snapshot.E.shape[1]) if snapshot.E is not None else cfg.text_dim
    m = MutableState(cfg=cfg)
    m.ids = list(snapshot.ids) + [""] * max(0, n_max - n)
    m.index = dict(snapshot.index)
    m.n = n

    def copy_array(name: str, dtype=np.float32, fill: float = 0.0):
        arr = np.full(n_max, fill, dtype=dtype)
        source = getattr(snapshot, name)
        if source is not None:
            arr[:n] = np.asarray(source, dtype=dtype)
        return arr

    m.category = copy_array("category", np.int16)
    m.subcategory = copy_array("subcategory", np.int16)
    m.brand = copy_array("brand", np.int16)
    m.price = copy_array("price", np.float32, 1.0)
    m.active = copy_array("active", bool, True)
    m.unavailable = copy_array("unavailable", bool, False)
    m.is_new = copy_array("is_new", bool, False)
    m.alpha0 = copy_array("alpha0", np.float32, 1.0)
    m.beta0 = copy_array("beta0", np.float32, 1.0)
    for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_click",
                 "f_click", "s_org", "f_org", "org_v", "org_atc", "org_p",
                 "feat_view", "feat_atc", "feat_q", "trend_boost",
                 "new_bonus", "restock_bonus", "price_bonus"):
        setattr(m, name, copy_array(name, np.float32))
    m.price_pct = copy_array("price_pct", np.float32, 0.5)
    m.price_pct_subcat = copy_array("price_pct_subcat", np.float32, 0.5)
    m.E = np.zeros((n_max, d), dtype=np.float32)
    if snapshot.E is not None:
        m.E[:n] = snapshot.E
    m.trans_view = snapshot.trans_view
    m.trans_buy = snapshot.trans_buy
    m.mu_beta = np.asarray(snapshot.mu_beta, dtype=np.float32).copy()
    m.Sigma_beta = np.asarray(snapshot.Sigma_beta, dtype=np.float32).copy()
    m.feature_mean = np.asarray(snapshot.feature_mean, dtype=np.float32).copy()
    m.feature_std = np.asarray(snapshot.feature_std, dtype=np.float32).copy()
    m.w = np.asarray(snapshot.w, dtype=np.float32).copy()
    m.best_safe = np.asarray(snapshot.best_safe, dtype=np.int64).copy()
    m.last_change_ts = np.zeros(n_max, dtype=np.float64)

    m.category_names = list(snapshot.category_names)
    m.subcategory_names = list(snapshot.subcategory_names)
    m.brand_names = list(snapshot.brand_names)
    m.new_product_set = set(snapshot.new_product_set)
    m.new_subcats = set(int(x) for x in snapshot.new_subcats)

    m.live_atc = LRU(cfg.n_max * 4, cfg.live_atc_ttl_s)
    m.user_bought = LRU(200000, cfg.bought_ttl_s)
    m.user_clicked = LRU(200000, cfg.clicked_ttl_s)
    m.user_affinity = LRU(200000, 7 * 86400.0)
    m.req_meta = LRU(200000, cfg.req_meta_ttl_s)
    m.seen_feedback = LRU(200000, 3600.0)
    m.seen_events = LRU(500000, 600.0)
    m.submitted = LRU(200000, 900.0)
    m.seen_requests = LRU(200000, 900.0)
    m.session_views = LRU(200000, cfg.live_atc_ttl_s)

    ring = int(cfg.trend_ring_minutes)
    m.ring_view = np.zeros((ring, n_max), dtype=np.int32)
    m.ring_atc = np.zeros((ring, n_max), dtype=np.int32)
    m.ring_purchase = np.zeros((ring, n_max), dtype=np.int32)
    m.ring_mark = np.zeros(ring, dtype=np.int64)
    m.metrics = {
        "requests": 0, "answered": 0, "pushed": 0, "failed": 0,
        "fallbacks": 0, "purchases": 0, "impressions": 0,
        "latencies": [], "new_slate": 0, "slates": 0,
        "errors": 0, "queries": 0,
    }
    return m



# --------------------------------------------------------------------------- #
# Engine: shared state + publication + regime policy
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, client: Any, cfg: Config, snapshot: Snapshot) -> None:
        self.client = client
        self.cfg = cfg
        self.mutable = make_mutable(snapshot, cfg)
        self.mutable.last_event_ts = time.time() - 300.0
        self.mutable.last_catalog_ts = time.time() - 5.0
        self.mutable.last_persist = time.time()
        self.mutable.last_full_persist = time.time()
        self.stop = threading.Event()
        self.wake_catalog = threading.Event()
        self.outbox: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=max(cfg.outbox_max, 100))
        self.rng = np.random.default_rng(cfg.seed)
        self.state = StateHolder(self.build_snapshot())
        self.threads: List[threading.Thread] = []
        self.started_at = time.time()

    # ------------------------------------------------------------ policy
    def _policy(self) -> RuntimePolicy:
        m = self.mutable
        cfg = self.cfg
        mode = m.circuit if m.circuit in ("NORMAL", "DEGRADED", "SAFE",
                                          "RECOVER") else "NORMAL"
        n_candidates = cfg.n_candidates
        enable_trends = True
        enable_exploration = True
        temperature_mult = 1.0
        epsilon = cfg.epsilon_explore
        c_ucb = cfg.c_ucb
        c_new = cfg.c_new
        ess_max = cfg.ess_max
        eta_trend = cfg.eta_trend
        k_trend = cfg.k_trend
        lambda_div = cfg.lambda_div
        mu_live = 0.0
        regime = m.regime

        if regime == "final":
            temperature_mult, epsilon, c_ucb, c_new = 1.25, 0.7, 0.25, 0.8
            ess_max, eta_trend, k_trend = cfg.regime_ess_max, 1.0, 0.01
            elapsed = max(0.0, time.time() - m.regime_started)
            mu_live = cfg.regime_mu_live_start + (
                cfg.regime_mu_live_final - cfg.regime_mu_live_start) * min(
                    elapsed / max(cfg.regime_anneal_s, 1.0), 1.0)
        elif regime == "anneal":
            elapsed = max(0.0, time.time() - m.regime_started)
            progress = float(np.clip(
                (elapsed - cfg.regime_anneal_s) / max(cfg.regime_anneal_s,
                                                      1.0), 0.0, 1.0))
            temperature_mult = 1.25 + (1.0 - 1.25) * progress
            epsilon = 0.7 + (cfg.epsilon_explore - 0.7) * progress
            c_ucb = 0.25 + (cfg.c_ucb - 0.25) * progress
            c_new = 0.8 + (cfg.c_new - 0.8) * progress
            ess_max = int(cfg.regime_ess_max + (
                cfg.regime_anneal_ess_max - cfg.regime_ess_max) * progress)
            eta_trend = 1.0 + (cfg.eta_trend - 1.0) * progress
            k_trend = 0.01 + (cfg.k_trend - 0.01) * progress
            mu_live = cfg.regime_mu_live_final

        if mode == "DEGRADED":
            n_candidates = min(n_candidates, 50)
            enable_trends = False
            enable_exploration = False
        external = m.external_policy or {}
        restock = float(external.get("restock_boost", 1.0))
        trend_scale = float(external.get("trend_scale", 1.0))
        ext_temp = float(external.get("temperature_mult", 1.0))
        epsilon = float(np.clip(epsilon + float(
            external.get("exploration_delta", 0.0)), 0.0, 0.9))
        return RuntimePolicy(
            mode=mode, n_candidates=int(n_candidates),
            enable_trends=enable_trends,
            enable_exploration=enable_exploration,
            temperature_mult=float(temperature_mult),
            epsilon_explore=float(epsilon), c_ucb=float(c_ucb),
            c_new=float(c_new), ess_max=int(ess_max),
            eta_trend=float(eta_trend), k_trend=float(k_trend),
            lambda_div=float(lambda_div), mu_live=float(mu_live),
            regime=regime, regime_started=float(m.regime_started),
            external_restock=restock, external_trend_scale=trend_scale,
            external_temperature_mult=ext_temp, updated_at=time.time())


    def posterior(self, policy: Optional[RuntimePolicy] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        m = self.mutable
        policy = policy or self._policy()
        n = m.n
        s = m.s_slow[:n] + m.s_fast[:n] + m.s_org[:n]
        f = m.f_slow[:n] + m.f_fast[:n] + m.f_org[:n]
        alpha = m.alpha0[:n] + s
        beta = m.beta0[:n] + f
        total = alpha + beta
        over = total > float(policy.ess_max)
        if over.any():
            scale = float(policy.ess_max) / np.maximum(total[over], EPS)
            s = s.copy()
            f = f.copy()
            s[over] *= scale
            f[over] *= scale
            alpha = m.alpha0[:n] + s
            beta = m.beta0[:n] + f
        return alpha.astype(np.float32), beta.astype(np.float32)

    def _tau_scales(self) -> Tuple[float, float]:
        m = self.mutable
        n = m.n
        view_sums = np.diff(m.trans_view.indptr[:n + 1])
        buy_sums = np.diff(m.trans_buy.indptr[:n + 1])
        sv = float(np.percentile(view_sums, 95)) if view_sums.size else 1.0
        sb = float(np.percentile(buy_sums, 95)) if buy_sums.size else 1.0
        return max(sv, 1e-6), max(sb, 1e-6)

    def build_snapshot(self) -> Snapshot:
        m = self.mutable
        cfg = self.cfg
        with m.lock:
            policy = self._policy()
            n = m.n
            alpha, beta = self.posterior(policy)
            mu = alpha / np.maximum(alpha + beta, EPS)
            new_bonus = np.zeros(n, dtype=np.float32)
            ess = alpha + beta
            if n:
                mask = np.asarray(m.is_new[:n], dtype=bool)
                new_bonus[mask] = (policy.c_new /
                                   np.sqrt(ess[mask] + 1.0)).astype(np.float32)
            eligible = (m.active[:n] & (~m.unavailable[:n]) &
                        (m.price[:n] > 0))
            if eligible.any():
                order = np.argsort(np.where(eligible, mu, -1.0))[::-1][:10]
                best_safe = order[eligible[order]].astype(np.int64)
            else:
                best_safe = np.zeros(0, dtype=np.int64)
            if best_safe.size < 10:
                best_safe = np.flatnonzero(eligible)[:10].astype(np.int64)
            sv, sb = self._tau_scales()
            prices = m.price[:n][eligible] if eligible.any() else m.price[:n]
            p95 = float(np.percentile(prices, 95)) if prices.size else 1.0
            return Snapshot(
                ids=list(m.ids[:n]), index=dict(m.index), n=n,
                category=m.category[:n].copy(),
                subcategory=m.subcategory[:n].copy(),
                brand=m.brand[:n].copy(),
                category_names=tuple(m.category_names),
                subcategory_names=tuple(m.subcategory_names),
                brand_names=tuple(m.brand_names),
                price=m.price[:n].copy(), active=m.active[:n].copy(),
                unavailable=m.unavailable[:n].copy(),
                is_new=m.is_new[:n].copy(),
                alpha0=m.alpha0[:n].copy(), beta0=m.beta0[:n].copy(),
                alpha=alpha, beta=beta,
                s_slow=m.s_slow[:n].copy(), f_slow=m.f_slow[:n].copy(),
                s_fast=m.s_fast[:n].copy(), f_fast=m.f_fast[:n].copy(),
                s_click=m.s_click[:n].copy(), f_click=m.f_click[:n].copy(),
                s_org=m.s_org[:n].copy(), f_org=m.f_org[:n].copy(),
                org_v=m.org_v[:n].copy(), org_atc=m.org_atc[:n].copy(),
                org_p=m.org_p[:n].copy(), feat_view=m.feat_view[:n].copy(),
                feat_atc=m.feat_atc[:n].copy(), feat_q=m.feat_q[:n].copy(),
                price_pct=m.price_pct[:n].copy(),
                price_pct_subcat=m.price_pct_subcat[:n].copy(),
                E=m.E[:n], trans_view=m.trans_view, trans_buy=m.trans_buy,
                mu_beta=np.asarray(m.mu_beta, dtype=np.float32).copy(),
                Sigma_beta=np.asarray(m.Sigma_beta, dtype=np.float32).copy(),
                calib_a=cfg.calib_a, calib_T=cfg.temperature,
                feature_names=tuple(FEATURE_NAMES),
                feature_mean=np.asarray(m.feature_mean,
                                        dtype=np.float32).copy(),
                feature_std=np.asarray(m.feature_std,
                                       dtype=np.float32).copy(),
                w=np.asarray(m.w, dtype=np.float32).copy(),
                trend_boost=m.trend_boost[:n].copy(),
                new_bonus=new_bonus, restock_bonus=m.restock_bonus[:n].copy(),
                price_bonus=m.price_bonus[:n].copy(), best_safe=best_safe,
                p95_price=p95, tau_scale_view=sv, tau_scale_buy=sb,
                live_atc=m.live_atc, user_bought=m.user_bought,
                user_clicked=m.user_clicked, user_affinity=m.user_affinity,
                new_product_set=frozenset(m.new_product_set),
                new_subcats=frozenset(int(x) for x in m.new_subcats),
                submitted=m.submitted, policy=policy, ts=time.time(),
                wake_catalog=self.wake_catalog)

    def publish(self) -> None:
        self.state.publish(self.build_snapshot())


    # ---------------------------------------------------------- feedback
    def decay(self) -> None:
        m = self.mutable
        cfg = self.cfg
        with m.lock:
            n = m.n
            gs, gf = cfg.gamma_slow_min, cfg.gamma_fast_min
            m.s_slow[:n] *= gs
            m.f_slow[:n] *= gs
            m.s_fast[:n] *= gf
            m.f_fast[:n] *= gf
            m.s_click[:n] *= gs
            m.f_click[:n] *= gs
            m.price_bonus[:n] *= 0.5 ** (1.0 / 120.0)
            m.restock_bonus[:n] *= 0.5 ** (1.0 / 60.0)
        m.last_decay = time.time()

    def maybe_variance_reinject(self) -> None:
        m = self.mutable
        cfg = self.cfg
        if time.time() - m.last_variance < cfg.variance_reinject_s:
            return
        lam_s = 0.90 if m.regime == "final" else cfg.lambda_sigma
        lam_m = 0.99 if m.regime == "final" else cfg.lambda_mu
        with m.lock:
            d = m.Sigma_beta.shape[0]
            Sigma0 = np.eye(d, dtype=np.float32)
            m.Sigma_beta = (lam_s * m.Sigma_beta +
                            (1.0 - lam_s) * Sigma0).astype(np.float32)
            m.mu_beta = (lam_m * m.mu_beta).astype(np.float32)
        m.last_variance = time.time()

    def maybe_refit(self) -> None:
        m = self.mutable
        cfg = self.cfg
        if (len(m.feedback_obs) < cfg.refit_feedback_min or
                time.time() - m.last_refit < cfg.refit_interval_s):
            return
        obs = m.feedback_obs[-5000:]
        X = np.asarray([o[0] for o in obs], dtype=np.float64)
        y = np.asarray([o[1] for o in obs], dtype=np.float64)
        w = np.asarray([o[2] for o in obs], dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 50:
            return
        try:
            fit = fit_logistic_map(
                X, y, w, mu0=np.asarray(m.mu_beta, dtype=np.float64),
                Sigma0=np.asarray(m.Sigma_beta, dtype=np.float64),
                cfg=cfg, max_iter=6)
        except Exception as exc:
            LOGGER.warning("refit failed: %s", exc)
            return
        if np.all(np.isfinite(fit["mu"])) and np.all(np.isfinite(fit["Sigma"])):
            with m.lock:
                m.mu_beta = fit["mu"]
                m.Sigma_beta = fit["Sigma"]
            m.last_refit = time.time()
            LOGGER.info("context refit on %d observations", X.shape[0])

    def _adf_update(self, idx: int, y: float, omega: float,
                    req: Optional[Mapping[str, Any]]) -> None:
        """Sequential ADF / Sherman-Morrison contextual update."""
        if req is None:
            return
        m = self.mutable
        try:
            rc = build_request_context(req, self.state.snapshot, self.cfg)
            z = build_context_features(np.asarray([idx], dtype=np.int64), rc,
                                       self.state.snapshot)[0]
        except Exception:
            return
        snap = self.state.snapshot
        mu_all = snap.alpha / np.maximum(snap.alpha + snap.beta, EPS)
        p_i = float(np.clip(mu_all[idx], 1e-5, 1 - 1e-5))
        offset = float(np.log(p_i) - np.log1p(-p_i))
        z64 = z.astype(np.float64)
        with m.lock:
            Sigma = np.asarray(m.Sigma_beta, dtype=np.float64)
            mu = np.asarray(m.mu_beta, dtype=np.float64)
            eta = offset + float(z64 @ mu)
            p = float(0.5 * (1.0 + np.tanh(0.5 * float(np.clip(eta, -35, 35)))))
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            s = omega * p * (1.0 - p)
            Sz = Sigma @ z64
            denom = 1.0 + s * float(z64 @ Sz)
            if denom <= 1e-12:
                return
            Sigma_new = Sigma - (s * np.outer(Sz, Sz)) / denom
            mu_new = mu + Sigma_new @ z64 * omega * (y - p)
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
            if np.any(np.diag(Sigma_new) < 1e-10):
                Sigma_new += 1e-8 * np.eye(Sigma_new.shape[0])
            try:
                np.linalg.cholesky(Sigma_new)
            except np.linalg.LinAlgError:
                d = Sigma_new.shape[0]
                Sigma_new = np.eye(d, dtype=np.float64)
                mu_new = np.zeros(d, dtype=np.float64)
            m.Sigma_beta = Sigma_new.astype(np.float32)
            m.mu_beta = np.clip(mu_new, -8.0, 8.0).astype(np.float32)
        m.feedback_obs.append((z.astype(np.float32), float(y), float(omega)))
        if len(m.feedback_obs) > 20000:
            del m.feedback_obs[:5000]


    def on_feedback_one(self, row: Mapping[str, Any]) -> None:
        m = self.mutable
        cfg = self.cfg
        rid = str(row.get("request_id") or "")
        if not rid or rid == "__init__":
            return
        shown = [str(x) for x in (row.get("shown") or [])]
        clicked = str(row.get("clicked") or "")
        purchased = str(row.get("purchased") or "")
        served_by = str(row.get("served_by") or "")
        meta = m.req_meta.get(rid) or {}
        user = str(meta.get("user_id") or "?")
        w = np.asarray(m.w, dtype=np.float64)

        if served_by == "fallback":
            m.metrics["fallbacks"] = m.metrics.get("fallbacks", 0) + 1
        if purchased:
            idx = m.index.get(purchased)
            if idx is not None and user != "?":
                m.user_bought.add_to_set(user, int(idx), max_items=5)
                m.dirty.add(int(idx))
            m.metrics["purchases"] = m.metrics.get("purchases", 0) + 1
        if clicked:
            idx = m.index.get(clicked)
            if idx is not None:
                m.dirty.add(int(idx))
                if user != "?":
                    m.user_clicked.add_to_set(user, int(idx), max_items=20)
        m.metrics["impressions"] = m.metrics.get("impressions", 0) + len(shown)
        lat = float(row.get("latency_ms") or 0.0)
        if lat > 0:
            latencies = m.metrics.setdefault("latencies", [])
            latencies.append(lat)
            if len(latencies) > 2000:
                del latencies[:-1000]

        if served_by != "team" or not shown:
            return
        rho_fast = (cfg.rho_fast_final if m.regime == "final" else
                    cfg.rho_fast_anneal if m.regime == "anneal" else
                    cfg.rho_fast_normal)
        ds, df = 1.0 - rho_fast, rho_fast
        with m.lock:
            indices = [m.index.get(pid) for pid in shown]
            for slot, idx in enumerate(indices):
                if idx is None:
                    continue
                wk = float(w[min(slot, 9)])
                if clicked == shown[slot]:
                    m.s_click[idx] += 1.0
                else:
                    m.f_click[idx] += cfg.c_neg_click * wk
                m.dirty.add(int(idx))
            pos_purchase = shown.index(purchased) if purchased in shown else -1
            pos_click = shown.index(clicked) if clicked in shown else -1
            for slot, idx in enumerate(indices):
                if idx is None:
                    continue
                wk = float(w[min(slot, 9)])
                if pos_purchase == slot:
                    m.s_slow[idx] += ds
                    m.s_fast[idx] += df
                elif pos_purchase < 0 and pos_click == slot:
                    m.f_slow[idx] += ds
                    m.f_fast[idx] += df
                else:
                    m.f_slow[idx] += ds * cfg.c_neg_purchase * wk
                    m.f_fast[idx] += df * cfg.c_neg_purchase * wk
            if user != "?" and (purchased or clicked):
                pid = purchased or clicked
                idx = m.index.get(pid)
                if idx is not None:
                    aff = m.user_affinity.get(user)
                    cats = max(len(m.category_names), 10)
                    if not isinstance(aff, np.ndarray) or aff.size < cats:
                        aff = np.zeros(cats, dtype=np.float32)
                    else:
                        aff = aff.copy()
                    aff[int(m.category[idx])] += 1.0
                    m.user_affinity.set(user, aff)
        obs_idx = m.index.get(purchased) if purchased else m.index.get(clicked)
        if obs_idx is not None:
            self._adf_update(int(obs_idx), 1.0 if purchased else 0.0,
                             1.0 if purchased else 0.5, meta.get("req"))

    def feedback_loop(self) -> None:
        while not self.stop.is_set():
            try:
                sql = SQL_FEEDBACK.format(
                    window=int(self.cfg.feedback_window_s),
                    limit=int(self.cfg.feedback_limit),
                    db=self.client.team_id)
                rows = self.client.query(sql)
                if self.mutable.feedback_first:
                    # On a restart the persisted snapshot already contains
                    # the previous 30 min; only replay a short overlap to
                    # avoid double-counting feedback (spec 4.2.6).
                    cutoff = time.time() - 180.0
                    rows = [r for r in rows
                            if (parse_datetime(r.get("ts")) or
                                datetime.now(timezone.utc)).timestamp() >= cutoff]
                    self.mutable.feedback_first = False
                fresh = [r for r in rows
                         if str(r.get("request_id")) not in
                         self.mutable.seen_feedback]
                for row in fresh:
                    rid = str(row.get("request_id"))
                    self.mutable.seen_feedback.add(rid)
                    try:
                        self.on_feedback_one(row)
                    except Exception as exc:
                        LOGGER.exception("feedback row %s: %s", rid, exc)
                self.mutable.recent_feedback = (
                    self.mutable.recent_feedback + fresh)[-2000:]
                if time.time() - self.mutable.last_decay >= 60.0:
                    self.decay()
                self.maybe_variance_reinject()
                self.maybe_refit()
                if fresh:
                    self.publish()
            except Exception as exc:
                LOGGER.warning("feedback_loop: %s", exc)
                self.mutable.circuit_errors += 1
            self.stop.wait(self.cfg.feedback_poll_s)


    # ------------------------------------------------------------ events
    def _ring_sum(self, minutes_back_start: int, minutes_back_end: int,
                  ring: np.ndarray) -> np.ndarray:
        m = self.mutable
        n = m.n
        size = m.ring_mark.size
        current = int(time.time() // 60)
        out = np.zeros(m.cfg.n_max, dtype=np.float64)
        for k in range(minutes_back_start, minutes_back_end):
            slot = (current - k) % size
            if int(m.ring_mark[slot]) == current - k:
                out += ring[slot]
        return out[:n]

    def on_events(self, rows: Sequence[Mapping[str, Any]]) -> int:
        m = self.mutable
        size = m.ring_mark.size
        processed = 0
        for row in rows:
            ts = parse_datetime(row.get("ts"))
            if ts is None:
                continue
            key = (str(row.get("session_id") or ""),
                   str(row.get("product_id") or ""),
                   str(row.get("event_type") or ""), ts.isoformat())
            if key in m.seen_events:
                continue
            m.seen_events.add(key)
            pid = str(row.get("product_id") or "")
            idx = m.index.get(pid)
            if idx is None:
                continue
            minute = int(ts.timestamp() // 60)
            slot = minute % size
            if int(m.ring_mark[slot]) != minute:
                m.ring_mark[slot] = minute
                m.ring_view[slot, :m.n] = 0
                m.ring_atc[slot, :m.n] = 0
                m.ring_purchase[slot, :m.n] = 0
            etype = str(row.get("event_type") or "")
            if etype == "view":
                m.ring_view[slot, idx] += 1
                sid = str(row.get("session_id") or "")
                if sid:
                    prev = m.session_views.get(sid) or frozenset()
                    for other in list(prev)[-10:]:
                        if other == idx:
                            continue
                        m.live_edges[(int(other), int(idx))] = (
                            m.live_edges.get((int(other), int(idx)), 0.0) + 1.0)
                        m.live_edges[(int(idx), int(other))] = (
                            m.live_edges.get((int(idx), int(other)), 0.0) + 1.0)
                    m.session_views.set(sid, frozenset(prev | {int(idx)}))
            elif etype == "add_to_cart":
                m.ring_atc[slot, idx] += 1
                user = str(row.get("user_id") or "?")
                if user != "?":
                    m.live_atc.add_to_set(user, int(idx), max_items=30)
            elif etype == "purchase":
                m.ring_purchase[slot, idx] += 1
                user = str(row.get("user_id") or "?")
                if user != "?":
                    m.user_bought.add_to_set(user, int(idx), max_items=5)
                m.dirty.add(int(idx))
            value = ts.timestamp()
            if value > m.last_event_ts:
                m.last_event_ts = value
            processed += 1
        if processed and time.time() - getattr(m, "last_edge_decay", 0.0) > 600:
            factor = 0.5 ** (600.0 / 1800.0)
            m.live_edges = {k: v * factor for k, v in m.live_edges.items()
                            if v * factor >= 0.5}
            m.last_edge_decay = time.time()
        return processed


    def recompute_trends(self) -> None:
        m = self.mutable
        cfg = self.cfg
        n = m.n
        if n == 0:
            return
        w15, w180 = cfg.trend_window_min, cfg.trend_baseline_min
        V15 = self._ring_sum(0, w15, m.ring_view)
        A15 = self._ring_sum(0, w15, m.ring_atc)
        P15 = self._ring_sum(0, w15, m.ring_purchase)
        V180 = self._ring_sum(w15, w15 + w180, m.ring_view)
        A180 = self._ring_sum(w15, w15 + w180, m.ring_atc)
        P180 = self._ring_sum(w15, w15 + w180, m.ring_purchase)
        sub = m.subcategory[:n].astype(np.int64)
        n_sub = int(sub.max()) + 1 if n else 1
        Vsub = np.bincount(sub, weights=V180, minlength=n_sub)
        Psub = np.bincount(sub, weights=P180, minlength=n_sub)
        m_s = (Psub + 1.0) / (Vsub + 50.0)
        lamb_base = (P180 + 50.0 * m_s[sub]) / (V180 + 50.0)
        lamb_now = (P15 + 50.0 * lamb_base) / (V15 + 50.0)
        s_i = P15 + 0.25 * A15
        e_i = (P180 + 0.25 * A180) / 12.0
        s_sub = np.bincount(sub, weights=s_i, minlength=n_sub)
        e_sub = np.bincount(sub, weights=e_i, minlength=n_sub)
        L_sub = (s_sub + 1.0) / (e_sub + 1.0)
        L_comb = (s_i + 1.0) / (e_i + 1.0)
        n_i = s_i + e_i
        L_shrunk = (n_i * L_comb + cfg.trend_shrink_kappa * L_sub[sub]) / (
            n_i + cfg.trend_shrink_kappa)
        gate = np.clip(s_i / 3.0, 0.0, 1.0)
        boost = (cfg.eta_trend * np.tanh(np.log(np.clip(
            L_shrunk, 0.25, 8.0))) * gate)
        zscore = (P15 - lamb_base * V15) / np.sqrt(lamb_base * V15 + 1.0)
        significant = (zscore > 2.0) & (P15 >= 3)
        boost = np.where(significant, boost * 1.3, boost)
        new_hot = m.is_new[:n] & ((A15 >= 3) | (P15 >= 1))
        boost = np.where(new_hot, boost * 1.5, boost)
        m.trend_boost[:n] = np.clip(boost, -cfg.trend_boost_clip,
                                    cfg.trend_boost_clip).astype(np.float32)
        total15 = float(V15.sum())
        total180 = float(V180.sum())
        # Only test a stable mix: an empty/short ring is not a regime change.
        if (m.regime == "normal" and total15 >= 30.0 and total180 >= 100.0
                and time.time() - self.started_at > 600.0):
            p15 = V15 / max(total15, 1.0)
            p180 = V180 / max(total180, 1.0)
            sub15 = np.bincount(sub, weights=p15, minlength=n_sub)
            sub180 = np.bincount(sub, weights=p180, minlength=n_sub)
            tv = 0.5 * float(np.abs(sub15 - sub180).sum())
            if tv > cfg.tv_threshold:
                LOGGER.warning("TV detector fired: %.3f", tv)
                self.regime_shift(reason="tv")

    def events_loop(self) -> None:
        while not self.stop.is_set():
            try:
                watermark = iso_utc(datetime.fromtimestamp(
                    self.mutable.last_event_ts, tz=timezone.utc))
                watermark = watermark.replace("T", " ").replace("Z", "")
                rows = self.client.query(SQL_EVENTS.format(
                    watermark=watermark))
                processed = self.on_events(rows)
                if processed or time.time() - self.mutable.last_trend >= 10:
                    self.recompute_trends()
                    self.mutable.last_trend = time.time()
                    self.publish()
            except EmptyTableError:
                pass
            except Exception as exc:
                LOGGER.warning("events_loop: %s", exc)
                self.mutable.circuit_errors += 1
            self.stop.wait(self.cfg.events_poll_s)


    # ------------------------------------------------------------ catalog
    def _structural_row(self, idx: int, emb: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
        m = self.mutable
        cfg = self.cfg
        n = m.n
        if n == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        sim = np.clip(m.E[:n] @ emb, -1.0, 1.0)
        cat = int(m.category[idx])
        sub = int(m.subcategory[idx])
        brand = int(m.brand[idx])
        values = (cfg.lambda_sub * (m.subcategory[:n] == sub) +
                  cfg.lambda_brand_structural * (m.brand[:n] == brand) +
                  cfg.lambda_cat_structural * (m.category[:n] == cat) +
                  cfg.lambda_sem * sim * sim).astype(np.float64)
        values[idx] = 0.0
        k = min(cfg.trans_top_k, max(n - 1, 1))
        part = (np.argpartition(-values, k)[:k] if n > k
                else np.arange(n))
        weights = values[part]
        total = float(weights.sum())
        if total <= 1e-12:
            weights = np.ones_like(weights) / max(weights.size, 1)
        else:
            weights = weights / total
        return part.astype(np.int32), weights.astype(np.float32)

    def _rebuild_csr(self, rows: Dict[int, Tuple[np.ndarray, np.ndarray]],
                     n: int) -> CSR:
        indptr = [0]
        indices: List[int] = []
        data: List[float] = []
        for i in range(n):
            cols, vals = rows.get(i, (np.zeros(0, np.int32),
                                      np.zeros(0, np.float32)))
            order = np.argsort(cols)
            indices.extend(int(cols[int(o)]) for o in order)
            data.extend(float(vals[int(o)]) for o in order)
            indptr.append(len(indices))
        return CSR(np.asarray(indptr, dtype=np.int64),
                   np.asarray(indices, dtype=np.int32),
                   np.asarray(data, dtype=np.float32), n)


    def add_new_product(self, row: Mapping[str, Any]) -> Optional[int]:
        m = self.mutable
        cfg = self.cfg
        pid = str(row.get("product_id") or "")
        if not pid or pid in m.index:
            return m.index.get(pid)
        if m.n >= cfg.n_max:
            LOGGER.error("catalog capacity reached (%d)", m.n)
            return None
        cfg_cat = _code_for(m.category_names, row.get("category"))
        cfg_sub = _code_for(m.subcategory_names, row.get("subcategory"))
        cfg_brand = _code_for(m.brand_names, row.get("brand"))
        price = float(row.get("price_eur") or 0.0)
        active_text = str(row.get("active_s", row.get("is_active", "true"))
                          if row.get("active_s") is not None
                          else row.get("is_active", "true")).lower()
        active = active_text in ("true", "1", "t", "yes")
        idx = m.n
        emb = hashed_embedding(row, m.E.shape[1], np.float32)
        m.ids[idx] = pid
        m.index[pid] = idx
        m.category[idx] = cfg_cat
        m.subcategory[idx] = cfg_sub
        m.brand[idx] = cfg_brand
        m.price[idx] = max(price, 0.0)
        m.active[idx] = active
        m.unavailable[idx] = (not active) or price <= 0
        m.is_new[idx] = True
        m.E[idx] = emb
        m.last_change_ts[idx] = time.time()
        m.new_product_set.add(pid)
        m.new_subcats.add(cfg_sub)
        m.dirty.add(idx)

        n_old = m.n
        alpha_all, beta_all = self.posterior()
        mu_all = alpha_all / np.maximum(alpha_all + beta_all, EPS)
        if n_old:
            sim = np.clip(m.E[:n_old] @ emb, -1.0, 1.0)
            k = min(25, n_old)
            part = (np.argpartition(-sim, k - 1)[:k] if n_old > k
                    else np.arange(n_old))
            part = part[sim[part] > 0.05]
            if part.size:
                weights = np.maximum(sim[part], 0.0) ** 2
                mu_knn = float((weights * mu_all[part]).sum() /
                               max(weights.sum(), 1e-9))
            else:
                mu_knn = float(mu_all.mean())
            same_sub = m.subcategory[:n_old] == cfg_sub
            same_brand = m.brand[:n_old] == cfg_brand
            mu_sub = (float(mu_all[same_sub].mean()) if same_sub.any()
                      else float(mu_all.mean()))
            mu_brand = (float(mu_all[same_brand].mean())
                        if same_brand.sum() >= 30 else mu_sub)
            mu_glob = float(mu_all.mean())
        else:
            mu_knn = mu_sub = mu_brand = mu_glob = 0.02
        mu_prior = float(np.clip(
            0.45 * mu_knn + 0.30 * mu_sub + 0.15 * mu_brand + 0.10 * mu_glob,
            cfg.mu_floor, cfg.mu_ceil))
        ess_new = 6.0 if m.regime == "final" else float(cfg.ess_new)
        m.alpha0[idx] = mu_prior * ess_new
        m.beta0[idx] = (1.0 - mu_prior) * ess_new
        for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_click",
                     "f_click", "s_org", "f_org", "org_v", "org_atc",
                     "org_p", "trend_boost", "price_bonus"):
            getattr(m, name)[idx] = 0.0
        m.restock_bonus[idx] = (0.15 if row.get("change") == "back_in_stock"
                                else 0.0)
        m.feat_view[idx] = 0.0
        m.feat_atc[idx] = 0.0
        m.feat_q[idx] = 0.0
        prices = m.price[:n_old][m.price[:n_old] > 0]
        m.price_pct[idx] = (float(np.searchsorted(np.sort(prices), price) /
                                  max(prices.size, 1)) if prices.size else 0.5)
        sub_mask = m.subcategory[:n_old] == cfg_sub
        sub_prices = m.price[:n_old][sub_mask & (m.price[:n_old] > 0)]
        m.price_pct_subcat[idx] = (
            float(np.searchsorted(np.sort(sub_prices), price) /
                  max(sub_prices.size, 1)) if sub_prices.size else 0.5)
        m.n = n_old + 1


        cols, vals = self._structural_row(idx, emb)
        view_rows: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        buy_rows: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for j in range(n_old):
            cj, vj = m.trans_view.row(j)
            if cj.size:
                view_rows[j] = (cj.copy(), vj.copy())
            cj, vj = m.trans_buy.row(j)
            if cj.size:
                buy_rows[j] = (cj.copy(), vj.copy())
        view_rows[idx] = (cols.copy(), vals.copy())
        buy_rows[idx] = (cols.copy(), vals.copy())
        if n_old:
            nearest = np.argsort(-(m.E[:n_old] @ emb))[:25]
            for j in nearest:
                j = int(j)
                for bucket in (view_rows, buy_rows):
                    cj, vj = bucket.get(j, (np.zeros(0, np.int32),
                                            np.zeros(0, np.float32)))
                    if np.any(cj == idx):
                        continue
                    extra = 0.5 * float(vj.mean()) if vj.size else 0.01
                    cj = np.append(cj, idx).astype(np.int32)
                    vj = np.append(vj, extra).astype(np.float32)
                    bucket[j] = (cj, (vj / max(float(vj.sum()), 1e-9))
                                 .astype(np.float32))
        m.trans_view = self._rebuild_csr(view_rows, m.n)
        m.trans_buy = self._rebuild_csr(buy_rows, m.n)
        LOGGER.info("new product %s -> idx %d (mu_prior=%.5f, ESS=%.0f)",
                    pid, idx, mu_prior, ess_new)
        return idx

    def apply_catalog_update(self, row: Mapping[str, Any]) -> bool:
        m = self.mutable
        cfg = self.cfg
        pid = str(row.get("product_id") or "")
        if not pid:
            return False
        change = str(row.get("change") or "")
        ts = parse_datetime(row.get("ts"))
        ts_value = ts.timestamp() if ts else time.time()
        idx = m.index.get(pid)
        if idx is None:
            # Idempotent unknown product: only a complete new_product row can
            # create it; otherwise wait for the new_product / reconciliation.
            if change == "new_product" or (row.get("category") and
                                           row.get("subcategory")):
                return self.add_new_product(row) is not None
            return False
        if ts_value + 1e-6 < float(m.last_change_ts[idx]):
            return False
        m.last_change_ts[idx] = ts_value
        changed = False
        active_raw = row.get("active_s")
        if active_raw is None:
            active_raw = row.get("is_active")
        active_text = str(active_raw).lower() if active_raw is not None else ""
        new_active = True if active_text in ("true", "1") else (
            False if active_text in ("false", "0") else bool(m.active[idx]))
        if new_active != bool(m.active[idx]):
            m.active[idx] = new_active
            changed = True
        price = float(row.get("price_eur") or 0.0)
        if change == "price_change":
            if price > 0:
                old = float(m.price[idx])
                ratio = price / max(old, 1e-6)
                if abs(ratio - 1.0) > 1e-5:
                    m.price[idx] = price
                    m.price_bonus[idx] = float(np.clip(-0.5 * np.log(ratio),
                                                       -0.4, 0.4))
                    changed = True
            elif not m.unavailable[idx]:
                m.unavailable[idx] = True
                changed = True
                LOGGER.warning("corrupt price for %s -> unavailable", pid)
        elif change == "out_of_stock":
            if not m.unavailable[idx]:
                m.unavailable[idx] = True
                changed = True
        elif change == "back_in_stock":
            if m.unavailable[idx]:
                m.unavailable[idx] = False
                m.restock_bonus[idx] = 0.15
                changed = True
        elif change == "new_product":
            pass
        elif price > 0 and abs(price - float(m.price[idx])) > 1e-4:
            m.price[idx] = price
            changed = True
        if changed:
            m.dirty.add(int(idx))
        return changed
        return False


    def reconcile_catalog(self) -> int:
        """Full catalogue + stock truth (Section 4.6.6), every 60 s."""
        m = self.mutable
        changed = 0
        try:
            rows = self.client.query(SQL_CATALOG, heavy=True)
        except Exception as exc:
            LOGGER.warning("catalog reconcile failed: %s", exc)
            m.circuit_errors += 1
            return 0
        with m.lock:
            for row in rows:
                row = dict(row)
                row.setdefault("change", "reconcile")
                row.setdefault("ts", iso_utc())
                if self.apply_catalog_update(row):
                    changed += 1
        try:
            oos = self.client.query(SQL_OUT_OF_STOCK, heavy=True)
            with m.lock:
                for row in oos:
                    idx = m.index.get(str(row.get("product_id") or ""))
                    if idx is not None and not m.unavailable[idx]:
                        m.unavailable[idx] = True
                        m.dirty.add(int(idx))
                        changed += 1
        except Exception as exc:
            LOGGER.info("stock truth unavailable: %s", exc)
        m.last_reconcile = time.time()
        if changed:
            LOGGER.info("catalog reconciled: %d changes", changed)
        return changed

    def catalog_loop(self) -> None:
        self.reconcile_catalog()
        self.publish()
        while not self.stop.is_set():
            try:
                watermark = iso_utc(datetime.fromtimestamp(
                    self.mutable.last_catalog_ts - 5.0, tz=timezone.utc))
                watermark = watermark.replace("T", " ").replace("Z", "")
                rows = self.client.query(SQL_CATALOG_UPDATES.format(
                    watermark=watermark))
                rows = sorted(rows, key=lambda r: parse_datetime(
                    r.get("ts")) or datetime.now(timezone.utc))
                changed = 0
                with self.mutable.lock:
                    for row in rows:
                        try:
                            changed += int(self.apply_catalog_update(row))
                        except Exception as exc:
                            LOGGER.warning("catalog row: %s", exc)
                        ts = parse_datetime(row.get("ts"))
                        if ts is not None:
                            self.mutable.last_catalog_ts = max(
                                self.mutable.last_catalog_ts, ts.timestamp())
                    if rows:
                        self.mutable.last_catalog_ts = max(
                            self.mutable.last_catalog_ts, time.time() - 5.0)
                if changed:
                    self.publish()
                if time.time() - self.mutable.last_reconcile >= 60.0:
                    self.reconcile_catalog()
                    self.publish()
                if self.wake_catalog.is_set():
                    self.wake_catalog.clear()
                    # The next iteration starts 5 minutes back to pick up the
                    # product that answer() could not resolve.
                    self.mutable.last_catalog_ts = time.time() - 300.0
                    continue
            except EmptyTableError:
                pass
            except Exception as exc:
                LOGGER.warning("catalog_loop: %s", exc)
                self.mutable.circuit_errors += 1
            self.stop.wait(self.cfg.catalog_poll_s)


    # ---------------------------------------------------------- requests
    def _now(self) -> datetime:
        try:
            return self.client.now()
        except Exception:
            return datetime.now(timezone.utc)

    def request_loop(self) -> None:
        m = self.mutable
        cfg = self.cfg
        while not self.stop.is_set():
            try:
                rows = self.client.query(SQL_REQUESTS.format(
                    window=int(cfg.request_window_s),
                    limit=int(cfg.request_limit)))
                fresh = [r for r in rows
                         if str(r.get("request_id")) not in m.seen_requests
                         and str(r.get("request_id")) != "__init__"]
                if len(fresh) > cfg.backlog_threshold:
                    fresh.sort(key=lambda r: parse_datetime(r.get("ts")) or
                               datetime.now(timezone.utc), reverse=True)
                    fresh = fresh[:cfg.max_answers_per_cycle]
                now = self._now()
                for req in fresh:
                    rid = str(req.get("request_id"))
                    ts = parse_datetime(req.get("ts"))
                    if ts is not None:
                        age = (now - ts).total_seconds()
                        if age > cfg.deadline_ms / 1000.0:
                            m.seen_requests.add(rid)
                            continue
                    req = dict(req)
                    req["_now"] = now
                    started = time.perf_counter()
                    try:
                        snap = self.state.snapshot
                        items = answer(req, snap, self.rng, cfg)
                    except Exception as exc:
                        LOGGER.exception("answer %s: %s", rid, exc)
                        m.metrics["errors"] = m.metrics.get("errors", 0) + 1
                        continue
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    m.metrics["requests"] = m.metrics.get("requests", 0) + 1
                    m.metrics["answered"] = m.metrics.get("answered", 0) + 1
                    latencies = m.metrics.setdefault("latencies", [])
                    latencies.append(latency_ms)
                    if len(latencies) > 2000:
                        del latencies[:-1000]
                    m.seen_requests.add(rid)
                    user = str(req.get("user_id") or "?")
                    bought = tuple(int(i) for i in (
                        self.state.snapshot.user_bought.get(user) or ()))
                    m.req_meta.set(rid, {
                        "user_id": user,
                        "is_new_user": bool(req.get("is_new_user")),
                        "context": req.get("context"),
                        "ts": req.get("ts"), "bought": bought,
                        "req": {"request_id": rid, "user_id": user,
                                "context": req.get("context"),
                                "recent_views": req.get("recent_views"),
                                "is_new_user": req.get("is_new_user"),
                                "ts": req.get("ts")}})
                    ans = {"request_id": rid, "items": items,
                           "model_version": cfg.model_version,
                           "ts": req.get("ts"),
                           "enqueued": time.monotonic(),
                           "user_id": user, "bought": bought}
                    try:
                        self.outbox.put_nowait(ans)
                    except queue.Full:
                        try:
                            self.outbox.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.outbox.put_nowait(ans)
                        except queue.Full:
                            m.metrics["failed"] = m.metrics.get("failed",
                                                               0) + 1
            except EmptyTableError:
                pass
            except Exception as exc:
                LOGGER.warning("request_loop: %s", exc)
                m.circuit_errors += 1
                self.stop.wait(0.5)
            self.stop.wait(cfg.request_poll_s)


    def push_batch(self, batch: Sequence[Mapping[str, Any]]) -> int:
        m = self.mutable
        cfg = self.cfg
        if not batch:
            return 0
        now = self._now()
        snap = self.state.snapshot
        valid: List[Dict[str, Any]] = []
        for ans in batch:
            ts = parse_datetime(ans.get("ts"))
            if ts is not None and (now - ts).total_seconds() * 1000.0 > \
                    cfg.deadline_ms:
                continue
            bought = tuple(int(i) for i in (ans.get("bought") or ()))
            items = repair_items(list(ans.get("items") or []), snap, bought)
            if not validate_items(items, snap, bought):
                continue
            rid = str(ans.get("request_id"))
            if rid in m.submitted:
                continue
            valid.append({"request_id": rid, "items": items,
                          "model_version": cfg.model_version})
        if not valid:
            return 0
        try:
            written = self.client.insert_rows(cfg.submissions_table, valid,
                                              timeout=cfg.insert_timeout_s)
        except Exception:
            for ans in reversed(batch):
                try:
                    self.outbox.put_nowait(ans)
                except queue.Full:
                    break
            raise
        for ans in valid:
            m.submitted.add(ans["request_id"])
        m.metrics["pushed"] = m.metrics.get("pushed", 0) + written
        if written != len(valid):
            LOGGER.warning("push mismatch: wrote %d of %d", written, len(valid))
        return written

    def push_loop(self) -> None:
        cfg = self.cfg
        while not self.stop.is_set():
            batch: List[Dict[str, Any]] = []
            deadline = time.monotonic() + cfg.push_batch_s
            while len(batch) < 50 and time.monotonic() < deadline:
                try:
                    timeout = max(0.01, deadline - time.monotonic())
                    batch.append(self.outbox.get(timeout=timeout))
                except queue.Empty:
                    break
            if batch:
                try:
                    self.push_batch(batch)
                except Exception as exc:
                    LOGGER.warning("push_loop: %s", exc)
                    self.mutable.circuit_errors += 1
                    self.stop.wait(0.5)
            self.stop.wait(0.05)


    # ---------------------------------------------------------- persistence
    def persist(self, full: bool = False) -> bool:
        m = self.mutable
        cfg = self.cfg
        snap = self.state.snapshot
        sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ok = True
        try:
            if m.dirty:
                write_product_delta(self.client, snap, sorted(m.dirty), cfg,
                                    snapshot_id=sid)
            write_trend_row(self.client, snap, cfg, snapshot_id=sid)
            metrics = self.compute_metrics()
            write_metrics_row(self.client, metrics, cfg, snapshot_id=sid)
        except Exception as exc:
            LOGGER.warning("delta persist failed: %s", exc)
            ok = False
        if full or time.time() - m.last_full_persist >= cfg.full_snapshot_s:
            try:
                directory = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), cfg.local_state_dir)
                path = os.path.join(directory, "latest.npz")
                save_snapshot_npz(snap, path)
                archive = os.path.join(directory, "snapshot_%s.npz" % sid)
                try:
                    if os.path.exists(archive):
                        os.remove(archive)
                except OSError:
                    pass
                # A cheap atomic copy of the latest file is enough for restart.
                write_snapshot_to_scratch(self.client, snap, cfg,
                                          snapshot_id=sid)
                m.last_full_persist = time.time()
            except Exception as exc:
                LOGGER.warning("full persist failed: %s", exc)
                ok = False
        m.dirty.clear()
        m.last_persist = time.time()
        return ok

    def persist_loop(self) -> None:
        while not self.stop.is_set():
            time.sleep(min(self.cfg.persist_s, 5.0))
            if self.stop.is_set():
                break
            if time.time() - self.mutable.last_persist >= self.cfg.persist_s:
                self.persist(full=False)

    # --------------------------------------------------------------- health
    def compute_metrics(self) -> Dict[str, Any]:
        m = self.mutable
        recent = [r for r in m.recent_feedback
                  if (parse_datetime(r.get("ts")) is not None and
                      (datetime.now(timezone.utc) -
                       parse_datetime(r.get("ts"))).total_seconds() < 300)]
        requests = len(recent)
        fallbacks = sum(1 for r in recent
                        if str(r.get("served_by")) == "fallback")
        purchases = sum(1 for r in recent if str(r.get("purchased") or ""))
        feedback_latencies = [float(r.get("latency_ms") or 0.0)
                              for r in recent]
        feedback_latencies = [x for x in feedback_latencies if x > 0]
        latencies = feedback_latencies or m.metrics.get("latencies", [])
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        snap = self.state.snapshot
        return {
            "requests": int(m.metrics.get("requests", 0)),
            "answered": int(m.metrics.get("answered", 0)),
            "pushed": int(m.metrics.get("pushed", 0)),
            "fallback_rate": round(fallbacks / max(requests, 1), 5),
            "latency_p50": round(p50, 2), "latency_p95": round(p95, 2),
            "latency_p99": round(p99, 2),
            "conversion_5m": round(purchases / max(requests, 1), 5),
            "new_products": len(snap.new_product_set),
            "n_products": int(snap.n),
            "circuit": m.circuit, "regime": m.regime,
            "live_edges": len(m.live_edges),
            "clock_offset_s": round(float(getattr(self.client,
                                                  "clock_offset", 0.0)), 3),
            "model_version": self.cfg.model_version,
        }

    def update_circuit(self, metrics: Mapping[str, Any]) -> None:
        m = self.mutable
        cfg = self.cfg
        fr = float(metrics.get("fallback_rate", 0.0))
        own = m.metrics.get("latencies", [])
        p99 = float(np.percentile(own, 99)) if own else 0.0
        if m.circuit == "NORMAL":
            # ``p99`` here is *our* answer()/pipeline latency.  The feedback
            # latency also includes the organiser's simulator poll and is only
            # used for monitoring (metrics["latency_p99"]).
            if fr > cfg.safe_fallback_rate or \
                    m.circuit_errors >= cfg.circuit_errors or \
                    p99 > cfg.safe_p99_ms:
                m.circuit = "SAFE"
            elif fr > cfg.degraded_fallback_rate or p99 > cfg.degraded_p99_ms:
                m.circuit = "DEGRADED"
        elif m.circuit in ("DEGRADED", "SAFE"):
            if fr < cfg.recover_fallback_rate and p99 < cfg.recover_p99_ms:
                m.circuit = "RECOVER"
        elif m.circuit == "RECOVER":
            m.circuit = "NORMAL"
            m.circuit_errors = 0


    def regime_shift(self, reason: str = "clock") -> bool:
        """Accelerate adaptation at the domain change (Section 4.7)."""
        m = self.mutable
        if m.regime in ("final", "anneal"):
            return False
        with m.lock:
            m.s_slow[:m.n] *= 0.25
            m.f_slow[:m.n] *= 0.25
            d = m.Sigma_beta.shape[0]
            m.Sigma_beta = (0.5 * m.Sigma_beta +
                            0.5 * np.eye(d, dtype=np.float32)).astype(
                                np.float32)
            m.mu_beta = (0.8 * m.mu_beta).astype(np.float32)
            m.regime = "final"
            m.regime_started = time.time()
            m.last_variance = time.time()
        if m.live_edges:
            edges = [{"src_idx": a, "dst_idx": b, "n": n}
                     for (a, b), n in m.live_edges.items()]
            try:
                live_view = build_transition_from_edges(
                    edges, m.trans_view, self.cfg)
                m.trans_view = blend_transitions(
                    m.trans_view, live_view, self.cfg.regime_mu_live_start)
            except Exception as exc:
                LOGGER.warning("live transition blend failed: %s", exc)
        LOGGER.warning("REGIME SHIFT (%s): fast buffer on, Sigma inflated, "
                       "exploration up", reason)
        self.publish()
        try:
            write_metrics_row(self.client,
                              {"event": "regime_shift", "reason": reason},
                              self.cfg)
        except Exception:
            pass
        return True

    def should_shift_by_clock(self) -> bool:
        year, month, day, hour, minute = self.cfg.regime_utc
        trigger = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= trigger

    def health_loop(self) -> None:
        m = self.mutable
        cfg = self.cfg
        while not self.stop.is_set():
            try:
                metrics = self.compute_metrics()
                self.update_circuit(metrics)
                if self.should_shift_by_clock() and m.regime == "normal":
                    self.regime_shift(reason="clock")
                if (m.regime == "final" and
                        time.time() - m.regime_started >= cfg.regime_anneal_s):
                    m.regime = "anneal"
                    LOGGER.info("regime annealing started")
                # CUSUM on 5-minute conversion.
                recent = [r for r in m.recent_feedback
                          if (parse_datetime(r.get("ts")) is not None and
                              (datetime.now(timezone.utc) - parse_datetime(
                                  r.get("ts"))).total_seconds() < 300)]
                if len(recent) >= 20:
                    req = len(recent)
                    pur = sum(1 for r in recent if str(r.get("purchased") or ""))
                    rate = pur / max(req, 1)
                    m.cusum_ref = 0.995 * m.cusum_ref + 0.005 * rate
                    m.cusum = max(0.0, m.cusum + (m.cusum_ref - rate) -
                                  cfg.cusum_delta)
                    if m.cusum > max(5.0 * m.cusum_ref, 0.05):
                        LOGGER.warning("CUSUM fired (%.3f)", m.cusum)
                        self.regime_shift(reason="cusum")
                        m.cusum = 0.0
                if time.time() - getattr(m, "last_policy_read", 0.0) >= 30.0:
                    policy_row = read_latest_policy(self.client, cfg)
                    m.external_policy = clamp_policy(policy_row or {})
                    m.last_policy_read = time.time()
                self.publish()
                if time.time() - m.last_metrics_persist >= 60.0:
                    try:
                        write_metrics_row(self.client, metrics, cfg)
                    except Exception:
                        pass
                    m.last_metrics_persist = time.time()
            except Exception as exc:
                LOGGER.warning("health_loop: %s", exc)
            self.stop.wait(cfg.health_poll_s)

    # -------------------------------------------------------------- threads
    def start(self) -> None:
        targets = [
            ("request", self.request_loop), ("push", self.push_loop),
            ("feedback", self.feedback_loop), ("events", self.events_loop),
            ("catalog", self.catalog_loop), ("persist", self.persist_loop),
            ("health", self.health_loop)]
        for name, target in targets:
            thread = threading.Thread(target=target, name="bayes-%s" % name,
                                      daemon=True)
            thread.start()
            self.threads.append(thread)
        LOGGER.info("engine started with %d workers", len(self.threads))

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop.set()
        self.wake_catalog.set()
        for thread in self.threads:
            thread.join(timeout=timeout)
        self.threads = []
        try:
            self.persist(full=True)
        except Exception as exc:
            LOGGER.warning("final persist failed: %s", exc)

