"""Immutable snapshot, atomic holder, TTL caches and embedding codecs.

The inference path reads a single reference to a ``Snapshot``.  Workers build a
new snapshot and publish it with one assignment (atomic under the CPython GIL),
so ``answer()`` never needs a long lock and never observes a half-written
state (spec Sections 1.4, 4.6.8).
"""
from __future__ import annotations

import base64
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Generic, Iterable, List, Mapping
from typing import Optional, Sequence, Tuple, TypeVar

import numpy as np

K = TypeVar("K")
V = TypeVar("V")


# --------------------------------------------------------------------------- #
# TTL caches (used for dedupe, user_bought, user_clicked, ...)
# --------------------------------------------------------------------------- #
class LRU(Generic[K, V]):
    """Thread-safe LRU map with optional TTL.

    ``add_to_set`` is copy-on-write: values stored by workers are frozensets,
    and the new value is published as ``old | {item}``.  ``answer()`` may hold
    the old frozenset while a worker updates the map; iterating it can never
    raise ``RuntimeError: Set changed size during iteration`` (correction #4).
    """

    def __init__(self, maxsize: int = 200000, ttl: Optional[float] = None):
        self.maxsize = int(maxsize)
        self.ttl = float(ttl) if ttl is not None else None
        self._data: "OrderedDict[K, Tuple[float, V]]" = OrderedDict()
        self._lock = threading.RLock()

    def _expired(self, touched: float, now: float) -> bool:
        return self.ttl is not None and (now - touched) > self.ttl

    def _prune_locked(self, now: float) -> None:
        if self.ttl is not None:
            stale = [k for k, (t, _) in self._data.items()
                     if self._expired(t, now)]
            for key in stale:
                self._data.pop(key, None)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return default
            touched, value = item
            if self._expired(touched, now):
                self._data.pop(key, None)
                return default
            self._data.move_to_end(key)
            return value

    def set(self, key: K, value: V) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            self._prune_locked(now)

    def add(self, key: K, value: Any = True) -> None:
        self.set(key, value)

    def add_to_set(self, key: K, item: Any,
                   max_items: Optional[int] = None) -> FrozenSet[Any]:
        """Copy-on-write insertion into a frozenset value."""
        now = time.time()
        with self._lock:
            current = self._data.get(key)
            old: FrozenSet[Any] = frozenset()
            if current is not None and not self._expired(current[0], now):
                old = current[1] if isinstance(current[1], frozenset) else \
                    frozenset(current[1] or ())
            new = old | {item}
            if max_items is not None and len(new) > max_items:
                new = frozenset(list(new)[-max_items:])
            self._data[key] = (now, new)
            self._data.move_to_end(key)
            self._prune_locked(now)
            return new

    def __contains__(self, key: K) -> bool:
        return self.get(key, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._prune_locked(time.time())
            return len(self._data)

    def pop(self, key: K, default: Any = None) -> Any:
        with self._lock:
            item = self._data.pop(key, None)
            return default if item is None else item[1]

    def keys(self) -> List[K]:
        with self._lock:
            self._prune_locked(time.time())
            return list(self._data.keys())

    def items(self) -> List[Tuple[K, V]]:
        with self._lock:
            self._prune_locked(time.time())
            return [(k, v) for k, (_, v) in self._data.items()]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# --------------------------------------------------------------------------- #
# CSR transition matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CSR:
    """Small immutable CSR matrix with the two operations scoring needs."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_cols: int

    @classmethod
    def empty(cls, n_rows: int, n_cols: Optional[int] = None) -> "CSR":
        n_cols = n_rows if n_cols is None else n_cols
        return cls(indptr=np.zeros(n_rows + 1, dtype=np.int64),
                   indices=np.zeros(0, dtype=np.int32),
                   data=np.zeros(0, dtype=np.float32),
                   n_cols=int(n_cols))

    @classmethod
    def from_scipy(cls, matrix: Any) -> "CSR":
        matrix = matrix.tocsr()
        return cls(indptr=np.asarray(matrix.indptr, dtype=np.int64),
                   indices=np.asarray(matrix.indices, dtype=np.int32),
                   data=np.asarray(matrix.data, dtype=np.float32),
                   n_cols=int(matrix.shape[1]))

    def row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        return self.indices[lo:hi], self.data[lo:hi]

    def row_sum(self, i: int) -> float:
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        return float(self.data[lo:hi].sum())

    def matvec(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_cols, dtype=np.float32)
        data, indices, indptr = self.data, self.indices, self.indptr
        for i in range(len(indptr) - 1):
            lo, hi = int(indptr[i]), int(indptr[i + 1])
            if hi > lo:
                out[indices[lo:hi]] += data[lo:hi] * x[i]
        return out

    def nnz(self) -> int:
        return int(self.data.size)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.indptr, self.indices, self.data


# --------------------------------------------------------------------------- #
# Runtime policy (mutable knobs published by health_loop)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimePolicy:
    mode: str = "NORMAL"                 # NORMAL / DEGRADED / SAFE / RECOVER
    n_candidates: int = 100
    enable_trends: bool = True
    enable_exploration: bool = True
    temperature_mult: float = 1.0
    epsilon_explore: float = 0.5
    c_ucb: float = 0.15
    c_new: float = 0.6
    ess_max: int = 5000
    eta_trend: float = 0.7
    k_trend: float = 0.005
    lambda_div: float = 0.20
    mu_live: float = 0.0
    regime: str = "normal"               # normal / final / anneal
    regime_started: float = 0.0
    external_restock: float = 1.0
    external_trend_scale: float = 1.0
    external_temperature_mult: float = 1.0
    updated_at: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_POLICY = RuntimePolicy()



@dataclass(frozen=True)
class Snapshot:
    """Immutable, self-contained state read by ``answer()``."""

    ids: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)
    n: int = 0

    category: np.ndarray = None          # type: ignore[assignment]
    subcategory: np.ndarray = None       # type: ignore[assignment]
    brand: np.ndarray = None             # type: ignore[assignment]
    category_names: Tuple[str, ...] = ()
    subcategory_names: Tuple[str, ...] = ()
    brand_names: Tuple[str, ...] = ()

    price: np.ndarray = None             # type: ignore[assignment]
    active: np.ndarray = None            # type: ignore[assignment]
    unavailable: np.ndarray = None       # type: ignore[assignment]
    is_new: np.ndarray = None            # type: ignore[assignment]

    alpha0: np.ndarray = None            # type: ignore[assignment]
    beta0: np.ndarray = None             # type: ignore[assignment]
    alpha: np.ndarray = None             # type: ignore[assignment]
    beta: np.ndarray = None              # type: ignore[assignment]

    s_slow: np.ndarray = None            # type: ignore[assignment]
    f_slow: np.ndarray = None            # type: ignore[assignment]
    s_fast: np.ndarray = None            # type: ignore[assignment]
    f_fast: np.ndarray = None            # type: ignore[assignment]
    s_click: np.ndarray = None           # type: ignore[assignment]
    f_click: np.ndarray = None           # type: ignore[assignment]
    s_org: np.ndarray = None             # type: ignore[assignment]
    f_org: np.ndarray = None             # type: ignore[assignment]

    E: np.ndarray = None                 # type: ignore[assignment]
    trans_view: CSR = None               # type: ignore[assignment]
    trans_buy: CSR = None                # type: ignore[assignment]

    # Organic aggregates / standardised ranking features (Section 2.4.4)
    org_v: np.ndarray = None             # type: ignore[assignment]
    org_atc: np.ndarray = None           # type: ignore[assignment]
    org_p: np.ndarray = None             # type: ignore[assignment]
    feat_view: np.ndarray = None         # type: ignore[assignment]
    feat_atc: np.ndarray = None          # type: ignore[assignment]
    feat_q: np.ndarray = None            # type: ignore[assignment]
    price_pct: np.ndarray = None         # type: ignore[assignment]
    price_pct_subcat: np.ndarray = None  # type: ignore[assignment]

    mu_beta: np.ndarray = None           # type: ignore[assignment]
    Sigma_beta: np.ndarray = None        # type: ignore[assignment]
    L_beta: np.ndarray = None            # type: ignore[assignment]
    calib_a: float = 0.0
    calib_T: float = 1.0
    feature_names: Tuple[str, ...] = ()
    feature_mean: np.ndarray = None      # type: ignore[assignment]
    feature_std: np.ndarray = None       # type: ignore[assignment]

    w: np.ndarray = None                 # type: ignore[assignment]
    trend_boost: np.ndarray = None       # type: ignore[assignment]
    new_bonus: np.ndarray = None         # type: ignore[assignment]
    restock_bonus: np.ndarray = None     # type: ignore[assignment]
    price_bonus: np.ndarray = None       # type: ignore[assignment]
    best_safe: np.ndarray = None         # type: ignore[assignment]
    p95_price: float = 1.0
    tau_scale_view: float = 1.0
    tau_scale_buy: float = 1.0

    live_atc: LRU = field(default_factory=lambda: LRU(200000, 1800.0))
    user_bought: LRU = field(default_factory=lambda: LRU(200000, 30 * 86400))
    user_clicked: LRU = field(default_factory=lambda: LRU(200000, 3600.0))
    user_affinity: LRU = field(default_factory=lambda: LRU(200000, 7 * 86400))

    new_product_set: FrozenSet[str] = frozenset()
    new_subcats: FrozenSet[int] = frozenset()
    submitted: LRU = field(default_factory=lambda: LRU(100000, 900.0))
    policy: RuntimePolicy = DEFAULT_POLICY
    ts: float = 0.0
    # Set by ``answer()`` when a recent view references a product that is in
    # new_product_set but not yet in the index (correction #6).  It is an
    # event, never a synchronous HTTP request inside the request path.
    wake_catalog: Optional[threading.Event] = None

    # ------------------------------------------------------------------ API
    def __post_init__(self) -> None:
        n = int(self.n)
        if self.category is None:
            object.__setattr__(self, "category", np.zeros(n, dtype=np.int16))
        if self.subcategory is None:
            object.__setattr__(self, "subcategory",
                               np.zeros(n, dtype=np.int16))
        if self.brand is None:
            object.__setattr__(self, "brand", np.zeros(n, dtype=np.int16))
        if self.price is None:
            object.__setattr__(self, "price", np.ones(n, dtype=np.float32))
        if self.active is None:
            object.__setattr__(self, "active", np.ones(n, dtype=bool))
        if self.unavailable is None:
            object.__setattr__(self, "unavailable", np.zeros(n, dtype=bool))
        if self.is_new is None:
            object.__setattr__(self, "is_new", np.zeros(n, dtype=bool))
        for name in ("alpha0", "beta0", "alpha", "beta"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.ones(n, dtype=np.float32))
        for name in ("s_slow", "f_slow", "s_fast", "f_fast", "s_click",
                     "f_click", "s_org", "f_org", "trend_boost", "new_bonus",
                     "restock_bonus", "price_bonus"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(n, dtype=np.float32))
        if self.E is None:
            object.__setattr__(self, "E", np.zeros((n, 512), dtype=np.float32))
        for name in ("org_v", "org_atc", "org_p", "feat_view", "feat_atc",
                     "feat_q"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(n, dtype=np.float32))
        if self.price_pct is None:
            object.__setattr__(self, "price_pct",
                               np.full(n, 0.5, dtype=np.float32))
        if self.price_pct_subcat is None:
            object.__setattr__(self, "price_pct_subcat",
                               np.full(n, 0.5, dtype=np.float32))
        if self.trans_view is None:
            object.__setattr__(self, "trans_view", CSR.empty(n))
        if self.trans_buy is None:
            object.__setattr__(self, "trans_buy", CSR.empty(n))
        if self.mu_beta is None:
            object.__setattr__(self, "mu_beta", np.zeros(24, dtype=np.float32))
        if self.Sigma_beta is None:
            d = int(self.mu_beta.size)
            object.__setattr__(self, "Sigma_beta",
                               np.eye(d, dtype=np.float32))
        if self.L_beta is None:
            object.__setattr__(self, "L_beta",
                               cholesky_lower(self.Sigma_beta))
        if self.feature_mean is None:
            object.__setattr__(self, "feature_mean",
                               np.zeros(int(self.mu_beta.size),
                                        dtype=np.float32))
        if self.feature_std is None:
            object.__setattr__(self, "feature_std",
                               np.ones(int(self.mu_beta.size),
                                       dtype=np.float32))
        if self.w is None:
            object.__setattr__(self, "w", np.asarray(
                [1.0, 0.72, 0.52, 0.39, 0.30, 0.23, 0.18, 0.14, 0.11, 0.08],
                dtype=np.float32))
        if self.best_safe is None:
            object.__setattr__(self, "best_safe", np.arange(n, dtype=np.int64))

    def eligible_mask(self) -> np.ndarray:
        return self.active & (~self.unavailable) & (self.price > 0)

    def is_cold_product(self, pid: str) -> bool:
        return pid in self.new_product_set

    def subcat_is_new(self, idx: int) -> bool:
        return int(self.subcategory[idx]) in self.new_subcats


def cholesky_lower(matrix: np.ndarray) -> np.ndarray:
    """Cholesky with a safe jitter/identity fallback (correction #7)."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return np.eye(int(a.size) if a.ndim else 1, dtype=np.float32)
    d = a.shape[0]
    a = 0.5 * (a + a.T)
    for jitter in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            return np.linalg.cholesky(a + jitter * np.eye(d)).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
    return np.eye(d, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Atomic holder + embedding codecs
# --------------------------------------------------------------------------- #
class StateHolder:
    """Single-reference publication; assignment is atomic under the GIL."""

    def __init__(self, snapshot: Snapshot):
        self._snap = snapshot
        self._lock = threading.Lock()

    @property
    def snapshot(self) -> Snapshot:
        return self._snap

    @snapshot.setter
    def snapshot(self, value: Snapshot) -> None:
        self._snap = value

    def publish(self, snapshot: Snapshot) -> None:
        self._snap = snapshot


def encode_embedding_chunk(matrix: np.ndarray) -> str:
    """float16 + base64 (spec 5.1.3)."""
    return base64.b64encode(
        np.asarray(matrix, dtype=np.float16).tobytes()).decode("ascii")


def decode_embedding_chunk(payload: str, n_rows: int, dim: int) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float16).reshape(
        n_rows, dim).astype(np.float32)


def encode_array(values: np.ndarray, dtype: Any = np.float16) -> str:
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode(
        "ascii")


def decode_array(payload: str, dtype: Any = np.float16) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload.encode("ascii")),
                         dtype=dtype)


def decode_float_array(values: Sequence[Any], dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=dtype)

