"""Configuration for the Bayes-TS real-time recommender (spec v3, Section 6.4).

Every hyper-parameter used by the engine lives here.  Values can be overridden
from the environment with the ``BAYES_<FIELD>`` convention (e.g.
``BAYES_RHO=0.04``), which keeps the operational deployment configurable
without touching code.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple


TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _coerce(value: str, default: Any) -> Any:
    """Convert an environment string to the type of ``default``."""
    if isinstance(default, bool):
        return value.strip().lower() in TRUTHY
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, (tuple, list)):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(float(p) if isinstance(default[0], float) else
                     int(p) if isinstance(default[0], int) else p
                     for p in parts)
    return value


def load_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """Load a ``.env`` file without overwriting real environment variables."""
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), ".env"))
    loaded: Dict[str, str] = {}
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    loaded[key] = value
                    os.environ.setdefault(key, value)
        except OSError:
            continue
        break
    return loaded


@dataclass
class Config:
    """All tunable constants of the v3 "Bayes-TS" specification."""

    # --- identity ---------------------------------------------------------
    model_version: str = "v3-bayes-ts"
    schema_version: int = 3

    # --- candidates and ordering -----------------------------------------
    n_candidates: int = 100
    lambda_T: float = 0.8          # legacy aggregate (kept for compatibility)
    lambda_T_view: float = 0.8     # home / category default
    lambda_T_buy: float = 0.3      # home / category default
    lambda_C: float = 0.5
    lambda_price: float = -0.25
    lambda_div: float = 0.20
    rho: float = 0.06
    # Section 3.4.2: page-specific transition weights (correction #2)
    page_weights_home: Tuple[float, float] = (0.8, 0.3)
    page_weights_category: Tuple[float, float] = (0.8, 0.3)
    page_weights_product: Tuple[float, float] = (0.4, 0.9)

    # --- exploration ------------------------------------------------------
    explore_slots: int = 2
    epsilon_explore: float = 0.5
    c_ucb: float = 0.15
    c_new: float = 0.6
    exploit_slot1: bool = True

    # --- priors -----------------------------------------------------------
    ess_max: int = 5000
    ess_new: int = 8
    kappa_1: float = 0.5
    kappa_2: float = 0.5
    n_max_product_prior: int = 300
    k_org: float = 0.01
    alpha_floor: float = 0.05
    beta_floor: float = 0.05
    mu_floor: float = 0.001
    mu_ceil: float = 0.25
    nu_min: float = 0.5
    nu_max: float = 500.0
    m0_transition: float = 50.0
    lambda_sub: float = 1.0
    lambda_brand_structural: float = 0.6
    lambda_cat_structural: float = 0.25
    lambda_sem: float = 0.8
    alpha_pop: float = 0.5
    lambda_cart: float = 0.6
    lambda_purchase: float = 1.0
    trans_top_k: int = 200
    text_dim: int = 512

    # --- offline IPS ------------------------------------------------------
    hist_days_ips: int = 60
    hist_days_transitions: int = 30
    omega_max: float = 50.0
    propensity_floor: float = 0.01
    holdout_days: int = 7
    default_position_w: Tuple[float, ...] = (
        1.00, 0.72, 0.52, 0.39, 0.30, 0.23, 0.18, 0.14, 0.11, 0.08)
    position_w_min: float = 0.03

    # --- decay ------------------------------------------------------------
    half_life_slow_days: float = 7.0
    half_life_fast_min: float = 20.0
    half_life_hist_days: float = 21.0
    restock_half_life_s: float = 3600.0
    price_bonus_half_life_s: float = 7200.0

    # --- contextual model -------------------------------------------------
    temperature: float = 1.0
    calib_a: float = 0.0
    lambda_sigma: float = 0.97
    lambda_mu: float = 0.995
    context_dim: int = 24
    prior_tau_price: float = 0.5
    prior_tau_other: float = 1.0
    refit_feedback_min: int = 500
    refit_interval_s: float = 1800.0
    variance_reinject_s: float = 300.0

    # --- trends -----------------------------------------------------------
    eta_trend: float = 0.7
    k_trend: float = 0.005
    trend_window_min: int = 15
    trend_baseline_min: int = 180
    trend_ring_minutes: int = 240
    trend_shrink_kappa: float = 10.0
    trend_boost_clip: float = 0.8
    cusum_delta: float = 0.005
    tv_threshold: float = 0.25

    # --- feedback ---------------------------------------------------------
    c_neg_click: float = 0.05
    c_neg_purchase: float = 0.02
    rho_fast_normal: float = 0.0
    rho_fast_final: float = 0.6
    rho_fast_anneal: float = 0.3
    click_penalty: float = 0.8
    same_view_delta: float = 0.6
    bought_ttl_s: float = 30 * 24 * 3600.0
    clicked_ttl_s: float = 3600.0
    live_atc_ttl_s: float = 1800.0
    req_meta_ttl_s: float = 2700.0
    feedback_window_s: float = 1800.0
    feedback_limit: int = 10000

    # --- operation --------------------------------------------------------
    request_poll_s: float = 0.25
    push_batch_s: float = 0.35
    feedback_poll_s: float = 1.5
    events_poll_s: float = 10.0
    catalog_poll_s: float = 0.75
    persist_s: float = 60.0
    full_snapshot_s: float = 3600.0
    health_poll_s: float = 10.0
    deadline_ms: float = 2900.0
    query_timeout_s: float = 10.0
    heavy_query_timeout_s: float = 35.0
    insert_timeout_s: float = 5.0
    query_max_execution_s: int = 25
    http_retries: int = 4
    insert_batch_rows: int = 500
    insert_max_bytes: int = 7 * 1024 * 1024
    outbox_max: int = 2000
    request_window_s: float = 60.0
    request_limit: int = 300
    max_answers_per_cycle: int = 50
    backlog_threshold: int = 30

    # --- reliability ------------------------------------------------------
    degraded_fallback_rate: float = 0.01
    degraded_p99_ms: float = 1200.0
    safe_fallback_rate: float = 0.03
    safe_p99_ms: float = 2000.0
    recover_fallback_rate: float = 0.005
    recover_p99_ms: float = 800.0
    circuit_errors: int = 3

    # --- regime shift -----------------------------------------------------
    regime_utc: Tuple[int, int, int, int, int] = (2026, 11, 15, 9, 0)
    regime_ess_max: int = 500
    regime_anneal_ess_max: int = 2000
    regime_mu_live_start: float = 0.3
    regime_mu_live_final: float = 0.7
    regime_anneal_s: float = 1800.0

    # --- storage ----------------------------------------------------------
    n_max: int = 8000
    chunk_products: int = 128
    local_state_dir: str = "state"
    scratch_table: str = "scratch"
    feedback_table: str = "feedback"
    submissions_table: str = "submissions"
    seed: int = 20261115

    # ------------------------------------------------------------------ API
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None,
                 prefix: str = "BAYES_") -> "Config":
        """Build a config, applying ``BAYES_<FIELD>`` environment overrides."""
        source = os.environ if env is None else env
        cfg = cls()
        for f in fields(cls):
            key = prefix + f.name.upper()
            if key not in source:
                continue
            default = getattr(cfg, f.name)
            try:
                setattr(cfg, f.name, _coerce(str(source[key]), default))
            except (TypeError, ValueError):
                continue
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view used in the ``meta`` scratch row."""
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        return out

    def page_lambdas(self, page: str) -> Tuple[float, float]:
        """Transition weights ``(lambda_T_view, lambda_T_buy)`` by page."""
        page = (page or "home").lower()
        if page == "product":
            return self.page_weights_product
        if page == "category":
            return self.page_weights_category
        return self.page_weights_home

    @property
    def gamma_slow_min(self) -> float:
        return 0.5 ** (1.0 / (self.half_life_slow_days * 24.0 * 60.0))

    @property
    def gamma_fast_min(self) -> float:
        return 0.5 ** (1.0 / self.half_life_fast_min)

    @property
    def gamma_day(self) -> float:
        return 0.5 ** (1.0 / self.half_life_hist_days)


CFG = Config()
