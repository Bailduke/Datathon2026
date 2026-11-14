"""Offline builder for the Bayes-TS priors.

Usage::

    python offline_build.py                 # full build + scratch + local .npz
    python offline_build.py --quick         # lighter sampling / 8 partitions
    python offline_build.py --no-scratch    # only the local snapshot

The heavy queries append ``SETTINGS max_execution_time = 25`` and are
partitioned by ``cityHash64(...) % N`` when they approach the server limit.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Tuple

import numpy as np

from starter.config import CFG, Config, load_dotenv
from starter.features import (FEATURE_NAMES, build_context_features,
                              build_request_context)
from starter.priors import (beta_from_moments, build_structural_prior,
                            build_transition_from_edges, estimate_position_bias,
                            fit_calibration, fit_hierarchy, fit_logistic_map,
                            ips_effective_counts, latest_meta,
                            organic_pseudo_counts, save_snapshot_npz,
                            write_snapshot_to_scratch)
from starter.rawtree import (EmptyTableError, QueryTimeoutError, RawTreeClient,
                             iso_utc, parse_datetime)
from starter.scoring import best_safe_candidates
from starter.state import CSR, RuntimePolicy, Snapshot, cholesky_lower
from starter.text import build_embeddings
from starter import workers

LOGGER = logging.getLogger("offline_build")

SQL_SLOT = """
WITH s AS (
    SELECT shown::Array(String) AS ids,
           propensity::Array(Float64) AS props,
           ifNull(toString(purchased), '') AS purchased,
           ifNull(toString(clicked), '') AS clicked
    FROM shop.history_sessions
    WHERE ts >= now() - INTERVAL {days} DAY
)
SELECT product_id::String AS product_id, slot,
       count() AS imp, countIf(purchased = product_id) AS pur,
       countIf(clicked = product_id) AS clk,
       sum(1.0 / greatest(e, 0.01)) AS ips_exp,
       sum(toUInt8(purchased = product_id) / greatest(e, 0.01)) AS ips_suc,
       sum(1.0 / pow(greatest(e, 0.01), 2)) AS ips_sq
FROM (
    SELECT ids, props, purchased, clicked, slot,
           arrayElement(ids, slot) AS product_id,
           arrayElement(props, slot) AS e
    FROM s ARRAY JOIN arrayEnumerate(ids) AS slot
)
WHERE {partition}
GROUP BY product_id, slot"""

SQL_ORGANIC = """
SELECT product_id::String AS product_id,
       sumIf(exp(-dateDiff('day', ts::DateTime64(3), now()) / {hl}),
             event_type::String = 'view') AS v,
       sumIf(exp(-dateDiff('day', ts::DateTime64(3), now()) / {hl}),
             event_type::String = 'add_to_cart') AS a,
       sumIf(exp(-dateDiff('day', ts::DateTime64(3), now()) / {hl}),
             event_type::String = 'purchase') AS p,
       countIf(event_type::String = 'view') AS v_raw,
       countIf(event_type::String = 'add_to_cart') AS a_raw,
       countIf(event_type::String = 'purchase') AS p_raw
FROM shop.history_events
WHERE ts >= now() - INTERVAL {days} DAY
  AND {partition}
GROUP BY product_id"""

SQL_COVIEW = """
WITH v AS (
    SELECT session_id::String AS sid, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type::String = 'view'
      AND ts >= now() - INTERVAL {days} DAY
      AND cityHash64(session_id::String) % {parts} = {part}
)
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM v AS a
INNER JOIN v AS b ON a.sid = b.sid
WHERE a.product_id != b.product_id
GROUP BY p, q
ORDER BY n DESC
LIMIT 200 BY p"""

SQL_COBUY = """
WITH v AS (
    SELECT session_id::String AS sid, product_id::String AS product_id,
           ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE event_type::String = 'view'
      AND ts >= now() - INTERVAL {days} DAY
      AND cityHash64(session_id::String) % {parts} = {part}
),
pur AS (
    SELECT session_id::String AS sid, product_id::String AS product_id,
           ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE event_type::String = 'purchase'
      AND ts >= now() - INTERVAL {days} DAY
      AND cityHash64(session_id::String) % {parts} = {part}
)
SELECT v.product_id AS src_id, pur.product_id AS dst_id, count() AS n
FROM v
INNER JOIN pur ON pur.sid = v.sid
WHERE v.product_id != pur.product_id
  AND v.ts <= pur.ts
GROUP BY src_id, dst_id
ORDER BY n DESC
LIMIT 200 BY src_id"""

SQL_COCART = """
WITH v AS (
    SELECT session_id::String AS sid, product_id::String AS product_id,
           ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE event_type::String = 'view'
      AND ts >= now() - INTERVAL {days} DAY
      AND cityHash64(session_id::String) % {parts} = {part}
),
cart AS (
    SELECT session_id::String AS sid, product_id::String AS product_id,
           ts::DateTime64(3) AS ts
    FROM shop.history_events
    WHERE event_type::String = 'add_to_cart'
      AND ts >= now() - INTERVAL {days} DAY
      AND cityHash64(session_id::String) % {parts} = {part}
)
SELECT v.product_id AS src_id, cart.product_id AS dst_id, count() AS n
FROM v
INNER JOIN cart ON cart.sid = v.sid
WHERE v.product_id != cart.product_id
  AND v.ts <= cart.ts
GROUP BY src_id, dst_id
ORDER BY n DESC
LIMIT 200 BY src_id"""


SQL_SESSIONS = """
SELECT request_id::String AS request_id,
       ts::DateTime64(3) AS ts,
       user_id::String AS user_id,
       context,
       recent_views,
       shown::Array(String) AS shown,
       ifNull(toString(clicked), '') AS clicked,
       ifNull(toString(purchased), '') AS purchased
FROM shop.history_sessions
WHERE {window}
  AND cityHash64(request_id::String) % {parts} = 0
LIMIT {limit}"""

SQL_NEW_PRODUCTS = """
SELECT product_id::String AS product_id, toString(category) AS category,
       toString(subcategory) AS subcategory, toString(brand) AS brand,
       toFloat64OrZero(toString(price_eur)) AS price_eur,
       toString(description) AS description,
       ts::DateTime64(3) AS ts
FROM shop.catalog_updates
WHERE change::String = 'new_product'
ORDER BY ts::DateTime64(3)"""


def _partition_sql(template: str, parts: int, part: int,
                   product_expr: str = "product_id") -> str:
    if parts <= 1:
        return template.format(partition="1 = 1")
    return template.format(partition="cityHash64(%s::String) %% %d = %d" % (
        product_expr, parts, part))


def fetch_catalog(client: RawTreeClient, cfg: Config) -> List[Dict[str, Any]]:
    rows = client.query(workers.SQL_CATALOG, heavy=True, timeout=40.0)
    LOGGER.info("catalog: %d products", len(rows))
    return rows


def fetch_slot_rows(client: RawTreeClient, cfg: Config,
                    parts: int = 1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part in range(max(parts, 1)):
        partition = ("cityHash64(product_id::String) %% %d = %d" %
                     (parts, part)) if parts > 1 else "1 = 1"
        sql = SQL_SLOT.replace("{days}", str(cfg.hist_days_ips)).replace(
            "{partition}", partition)
        try:
            got = client.query(sql, heavy=True)
            rows.extend(got)
            LOGGER.info("slot rows part %d/%d: %d", part + 1, parts, len(got))
        except Exception as exc:
            LOGGER.warning("slot partition %d failed: %s", part, exc)
    return rows


def fetch_organic(client: RawTreeClient, cfg: Config,
                  parts: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part in range(max(parts, 1)):
        partition = ("cityHash64(product_id::String) %% %d = %d" %
                     (parts, part)) if parts > 1 else "1 = 1"
        sql = SQL_ORGANIC.format(days=cfg.hist_days_ips,
                                 hl=cfg.half_life_hist_days,
                                 partition=partition)
        try:
            got = client.query(sql, heavy=True)
            rows.extend(got)
            LOGGER.info("organic part %d/%d: %d", part + 1, parts, len(got))
        except Exception as exc:
            LOGGER.warning("organic partition %d failed: %s", part, exc)
    return rows


def fetch_edges(client: RawTreeClient, cfg: Config, template: str,
                parts: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part in range(max(parts, 1)):
        sql = template.format(days=cfg.hist_days_transitions, parts=parts,
                              part=part)
        try:
            got = client.query(sql, heavy=True, timeout=40.0)
            rows.extend(got)
            LOGGER.info("edges %s part %d/%d: %d", template.split()[1],
                        part + 1, parts, len(got))
        except Exception as exc:
            LOGGER.warning("edge partition %s/%d failed: %s", template[:30],
                           part, exc)
    return rows


def fetch_sessions(client: RawTreeClient, cfg: Config, window: str,
                   parts: int = 20, limit: int = 120000
                   ) -> List[Dict[str, Any]]:
    try:
        rows = client.query(SQL_SESSIONS.format(
            window=window, parts=max(parts, 1), limit=limit), heavy=True,
            timeout=40.0)
        LOGGER.info("sessions (%s): %d rows", window, len(rows))
        return rows
    except Exception as exc:
        LOGGER.warning("session query failed: %s", exc)
        return []


def fetch_new_products(client: RawTreeClient) -> List[Dict[str, Any]]:
    try:
        return client.query(SQL_NEW_PRODUCTS, heavy=True)
    except Exception:
        return []



def catalog_arrays(products: Sequence[Mapping[str, Any]], cfg: Config
                   ) -> Dict[str, Any]:
    ids: List[str] = []
    rows: List[Mapping[str, Any]] = []
    for row in products:
        pid = str(row.get("product_id") or "")
        if pid and pid not in ids:
            ids.append(pid)
            rows.append(row)
    n = len(ids)
    index = {pid: i for i, pid in enumerate(ids)}
    cat_names: List[str] = []
    sub_names: List[str] = []
    brand_names: List[str] = []
    cat = np.zeros(n, np.int16)
    sub = np.zeros(n, np.int16)
    brand = np.zeros(n, np.int16)
    price = np.ones(n, np.float32)
    active = np.ones(n, bool)
    for i, row in enumerate(rows):
        for value, names, target in (
                (row.get("category"), cat_names, cat),
                (row.get("subcategory"), sub_names, sub),
                (row.get("brand"), brand_names, brand)):
            text = str(value or "?")
            if text not in names:
                names.append(text)
            target[i] = names.index(text)
        try:
            price[i] = max(float(row.get("price_eur") or 0.0), 0.0)
        except (TypeError, ValueError):
            price[i] = 0.0
        active[i] = str(row.get("is_active", True)).lower() not in (
            "false", "0")
    E = build_embeddings(rows, cfg.text_dim)
    return {"ids": ids, "rows": rows, "index": index, "n": n,
            "category": cat, "subcategory": sub, "brand": brand,
            "category_names": cat_names, "subcategory_names": sub_names,
            "brand_names": brand_names, "price": price, "active": active,
            "E": E}


def compute_price_percentiles(price: np.ndarray, subcategory: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    n = price.size
    pct = np.full(n, 0.5, dtype=np.float32)
    pct_sub = np.full(n, 0.5, dtype=np.float32)
    if n:
        order = np.argsort(price)
        pct[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for s in np.unique(subcategory):
            mask = subcategory == s
            idx = np.flatnonzero(mask)
            if idx.size:
                idx = idx[np.argsort(price[idx])]
                pct_sub[idx] = np.linspace(0.0, 1.0, idx.size,
                                           dtype=np.float32)
    return pct, pct_sub


def make_snapshot(arrays: Dict[str, Any], alpha0: np.ndarray,
                  beta0: np.ndarray, s_slow: np.ndarray, f_slow: np.ndarray,
                  s_org: np.ndarray, f_org: np.ndarray, org_v: np.ndarray,
                  org_atc: np.ndarray, org_p: np.ndarray, feat_view: np.ndarray,
                  feat_atc: np.ndarray, feat_q: np.ndarray,
                  price_pct: np.ndarray, price_pct_sub: np.ndarray,
                  E: np.ndarray, trans_view: CSR, trans_buy: CSR,
                  mu_beta: np.ndarray, Sigma_beta: np.ndarray,
                  feature_mean: np.ndarray, feature_std: np.ndarray,
                  calib_a: float, calib_T: float, w: np.ndarray,
                  trend_boost: np.ndarray, price_bonus: np.ndarray,
                  restock_bonus: np.ndarray, new_bonus: np.ndarray,
                  best_safe: np.ndarray, p95_price: float,
                  new_product_set: Iterable[str],
                  new_subcats: Iterable[int],
                  unavailable: np.ndarray, ts: Optional[float] = None
                  ) -> Snapshot:
    alpha = alpha0 + s_slow + s_org
    beta = beta0 + f_slow + f_org
    prices = arrays["price"][arrays["price"] > 0]
    if not p95_price:
        p95_price = float(np.percentile(prices, 95)) if prices.size else 1.0
    return Snapshot(
        ids=arrays["ids"], index=arrays["index"], n=arrays["n"],
        category=arrays["category"], subcategory=arrays["subcategory"],
        brand=arrays["brand"], category_names=tuple(arrays["category_names"]),
        subcategory_names=tuple(arrays["subcategory_names"]),
        brand_names=tuple(arrays["brand_names"]), price=arrays["price"],
        active=arrays["active"], unavailable=unavailable,
        is_new=np.asarray([pid in set(new_product_set)
                           for pid in arrays["ids"]], dtype=bool),
        alpha0=alpha0, beta0=beta0, alpha=alpha.astype(np.float32),
        beta=beta.astype(np.float32), s_slow=s_slow, f_slow=f_slow,
        s_fast=np.zeros(arrays["n"], np.float32),
        f_fast=np.zeros(arrays["n"], np.float32),
        s_click=np.zeros(arrays["n"], np.float32),
        f_click=np.zeros(arrays["n"], np.float32), s_org=s_org, f_org=f_org,
        org_v=org_v, org_atc=org_atc, org_p=org_p, feat_view=feat_view,
        feat_atc=feat_atc, feat_q=feat_q, price_pct=price_pct,
        price_pct_subcat=price_pct_sub, E=E, trans_view=trans_view,
        trans_buy=trans_buy, mu_beta=mu_beta, Sigma_beta=Sigma_beta,
        L_beta=cholesky_lower(Sigma_beta), calib_a=calib_a, calib_T=calib_T,
        feature_names=FEATURE_NAMES, feature_mean=feature_mean,
        feature_std=feature_std, w=w, trend_boost=trend_boost,
        new_bonus=new_bonus, restock_bonus=restock_bonus,
        price_bonus=price_bonus, best_safe=best_safe, p95_price=p95_price,
        new_product_set=frozenset(new_product_set),
        new_subcats=frozenset(int(x) for x in new_subcats),
        policy=RuntimePolicy(), ts=(ts if ts is not None else time.time()))



def build_training_data(sessions: Sequence[Mapping[str, Any]], snap: Snapshot,
                        cfg: Config, max_rows: int = 800000,
                        return_offset: bool = False):
    """Turn historical slates into weighted Bernoulli rows (eq. 27)."""
    X: List[np.ndarray] = []
    y: List[float] = []
    weights: List[float] = []
    offsets: List[float] = []
    w = np.asarray(snap.w, dtype=np.float64)
    mu = snap.alpha0 / np.maximum(snap.alpha0 + snap.beta0, 1e-9)
    for row in sessions:
        shown = [str(x) for x in (row.get("shown") or [])]
        if not shown:
            continue
        ts = parse_datetime(row.get("ts")) or datetime.now(timezone.utc)
        req = {"request_id": row.get("request_id"),
               "user_id": row.get("user_id"), "context": row.get("context"),
               "recent_views": row.get("recent_views"),
               "is_new_user": False, "ts": row.get("ts")}
        try:
            rc = build_request_context(req, snap, cfg, ts)
        except Exception as exc:
            import os as _os
            if _os.environ.get("BAYES_DEBUG_TRAIN"):
                print("RC FAIL", repr(exc))
            continue
        idx = [snap.index.get(pid) for pid in shown]
        valid_positions = [(pos, i) for pos, i in enumerate(idx)
                           if i is not None]
        if not valid_positions:
            continue
        positions = np.asarray([p for p, _ in valid_positions], dtype=np.int64)
        indices = np.asarray([i for _, i in valid_positions], dtype=np.int64)
        Z = build_context_features(indices, rc, snap)
        purchased = str(row.get("purchased") or "")
        clicked = str(row.get("clicked") or "")
        for local, (pos, i) in enumerate(valid_positions):
            yv = 1.0 if purchased == shown[pos] else 0.0
            wt = 1.0 if (yv == 1.0 or clicked == shown[pos]) else \
                float(w[min(pos, 9)])
            X.append(Z[local])
            y.append(yv)
            weights.append(wt)
            p_i = float(np.clip(mu[i], 1e-5, 1 - 1e-5))
            offsets.append(float(np.log(p_i) - np.log1p(-p_i)))
            if len(X) >= max_rows:
                break
        if len(X) >= max_rows:
            break
    if not X:
        empty = (np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
                 np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
        return empty + ((np.zeros(0, dtype=np.float32),) if return_offset
                        else ())
    result = (np.asarray(X, dtype=np.float32),
              np.asarray(y, dtype=np.float32),
              np.asarray(weights, dtype=np.float32))
    if return_offset:
        result = result + (np.asarray(offsets, dtype=np.float32),)
    return result


def train_context_model(sessions: Sequence[Mapping[str, Any]], snap: Snapshot,
                        cfg: Config) -> Dict[str, Any]:
    X, y, wts = build_training_data(sessions, snap, cfg)
    if X.shape[0] < 1000:
        LOGGER.warning("not enough sessions for logistic model (%d rows)",
                       X.shape[0])
        return {"mu": np.zeros(len(FEATURE_NAMES), np.float32),
                "Sigma": np.eye(len(FEATURE_NAMES), dtype=np.float32),
                "feature_mean": np.zeros(len(FEATURE_NAMES), np.float32),
                "feature_std": np.ones(len(FEATURE_NAMES), np.float32),
                "rows": X.shape[0]}
    feature_mean = X.mean(axis=0).astype(np.float32)
    feature_std = X.std(axis=0).astype(np.float32)
    feature_mean[0] = 0.0
    feature_std[0] = 1.0
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(
        np.float32)
    Xs = (X - feature_mean) / feature_std
    Xs[:, 0] = 1.0
    LOGGER.info("fitting contextual logistic on %d rows (conversion %.4f)",
                X.shape[0], float(y.mean()) if y.size else 0.0)
    fit = fit_logistic_map(Xs, y, wts, cfg=cfg, max_iter=10)
    fit.update({"feature_mean": feature_mean, "feature_std": feature_std,
                "rows": int(X.shape[0])})
    return fit


def calibrate_model(sessions: Sequence[Mapping[str, Any]], snap: Snapshot,
                    fit: Mapping[str, Any], cfg: Config) -> Dict[str, float]:
    from dataclasses import replace as dataclass_replace
    from starter.priors import calibration_metrics
    snap2 = dataclass_replace(
        snap, mu_beta=np.asarray(fit["mu"], dtype=np.float32),
        feature_mean=np.asarray(fit["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(fit["feature_std"], dtype=np.float32))
    X, y, wts, offsets = build_training_data(
        sessions, snap2, cfg, max_rows=400000, return_offset=True)
    if X.shape[0] == 0:
        return {"a": 0.0, "T": 1.0, "brier": 0.0, "ece": 0.0, "rows": 0}
    base = np.asarray(offsets, dtype=np.float32) + X @ np.asarray(
        fit["mu"], dtype=np.float32)
    a, temperature = fit_calibration(base, y, wts, cfg)
    p = 0.5 * (1.0 + np.tanh(0.5 * (a + base / max(temperature, 1e-3))))
    metrics = calibration_metrics(p, y)
    LOGGER.info("calibration: a=%.4f T=%.3f brier=%.5f ece=%.4f (%d rows)",
                a, temperature, metrics["brier"], metrics["ece"], X.shape[0])
    return {"a": float(a), "T": float(temperature),
            "brier": metrics["brier"], "ece": metrics["ece"],
            "rows": int(X.shape[0])}



SQL_LAST_DAY = "SELECT toDate(max(ts::DateTime64(3))) AS last_day FROM shop.history_sessions"


def _best_safe_indices(snap: Snapshot) -> np.ndarray:
    ids = best_safe_candidates(snap, (), 10)
    return np.asarray([snap.index[p] for p in ids if p in snap.index],
                      dtype=np.int64)


def build(client: RawTreeClient, cfg: Config, args: argparse.Namespace
          ) -> Tuple[Snapshot, Dict[str, Any]]:
    t0 = time.time()
    products = fetch_catalog(client, cfg)
    arrays = catalog_arrays(products, cfg)
    n = arrays["n"]
    LOGGER.info("catalog arrays: %d products, %d categories, %d subcategories",
                n, len(set(arrays["category"])),
                len(set(arrays["subcategory"])))
    parts = int(args.partitions)
    slot_rows = fetch_slot_rows(client, cfg, parts=1)
    if not slot_rows:
        slot_rows = []
    w = estimate_position_bias(slot_rows, cfg)
    LOGGER.info("position weights: %s", np.round(w, 3).tolist())
    ips = ips_effective_counts(slot_rows, w, 0.02, cfg)
    ips_map = {str(pid): i for i, pid in enumerate(ips["ids"])}

    organic_rows = fetch_organic(client, cfg, parts=min(parts, 4))
    org_map = {str(r.get("product_id")): r for r in organic_rows}
    org_v = np.zeros(n, dtype=np.float32)
    org_atc = np.zeros(n, dtype=np.float32)
    org_p = np.zeros(n, dtype=np.float32)
    for pid, row in org_map.items():
        idx = arrays["index"].get(pid)
        if idx is None:
            continue
        org_v[idx] = float(row.get("v") or 0.0)
        org_atc[idx] = float(row.get("a") or 0.0)
        org_p[idx] = float(row.get("p") or 0.0)

    theta = np.full(n, 0.02, dtype=np.float64)
    ess = np.zeros(n, dtype=np.float64)
    s_slow = np.zeros(n, dtype=np.float32)
    f_slow = np.zeros(n, dtype=np.float32)
    for pid, j in ips_map.items():
        idx = arrays["index"].get(pid)
        if idx is None:
            continue
        theta[idx] = float(ips["theta"][j])
        ess[idx] = float(ips["ess"][j])
        s_slow[idx] = float(ips["s_eff"][j])
        f_slow[idx] = float(ips["f_eff"][j])
    hierarchy = fit_hierarchy(theta.astype(np.float32),
                              ess.astype(np.float32),
                              arrays["category"], arrays["subcategory"], cfg)
    alpha0 = hierarchy["alpha0"].copy()
    beta0 = hierarchy["beta0"].copy()
    org = organic_pseudo_counts(org_v, org_atc, org_p,
                                arrays["subcategory"], cfg)
    s_org, f_org = org["s_org"], org["f_org"]

    new_products = fetch_new_products(client)
    new_ids = set()
    new_subs = set()
    for row in new_products:
        pid = str(row.get("product_id") or "")
        idx = arrays["index"].get(pid)
        if idx is None:
            continue
        new_ids.add(pid)
        sub_code = int(arrays["subcategory"][idx])
        new_subs.add(sub_code)
        mu_sub = float(hierarchy["sub_mean"].get(
            sub_code, hierarchy["global_mean"]))
        alpha0[idx] = max(mu_sub * cfg.ess_new, cfg.alpha_floor)
        beta0[idx] = max((1.0 - mu_sub) * cfg.ess_new, cfg.beta_floor)
    LOGGER.info("new products: %d", len(new_ids))

    price_pct, price_pct_sub = compute_price_percentiles(
        arrays["price"], arrays["subcategory"])
    try:
        unavailable_ids = {str(r.get("product_id"))
                           for r in client.query(
                               "SELECT product_id::String AS product_id FROM "
                               "shop.catalog_updates WHERE change::String IN "
                               "('out_of_stock','back_in_stock') GROUP BY "
                               "product_id HAVING argMax(change::String, "
                               "ts::DateTime64(3)) = 'out_of_stock'",
                               heavy=True)}
    except Exception:
        unavailable_ids = set()
    unavailable = np.asarray([pid in unavailable_ids
                              for pid in arrays["ids"]], dtype=bool)

    if args.skip_transitions:
        trans_view = CSR.empty(n)
        trans_buy = CSR.empty(n)
    else:
        LOGGER.info("building structural prior (top-%d)", cfg.trans_top_k)
        structural = build_structural_prior(
            arrays["E"], arrays["category"], arrays["subcategory"],
            arrays["brand"], cfg, top_k=cfg.trans_top_k)
        if args.skip_cooc:
            trans_view = build_transition_from_edges(None, structural, cfg)
            trans_buy = trans_view
        else:
            view_rows = fetch_edges(client, cfg, SQL_COVIEW, parts=parts)
            buy_rows = fetch_edges(client, cfg, SQL_COBUY, parts=parts)
            cart_rows = fetch_edges(client, cfg, SQL_COCART, parts=parts)
            view_edges = [{"src_idx": arrays["index"].get(str(r.get("p"))),
                           "dst_idx": arrays["index"].get(str(r.get("q"))),
                           "n": float(r.get("n") or 0.0)}
                          for r in view_rows]
            combined: Dict[Tuple[int, int], float] = {}
            for rows, weight in ((buy_rows, cfg.lambda_purchase),
                                 (cart_rows, cfg.lambda_cart)):
                for row in rows:
                    a = arrays["index"].get(str(row.get("src_id")))
                    b = arrays["index"].get(str(row.get("dst_id")))
                    if a is None or b is None or a == b:
                        continue
                    combined[(a, b)] = combined.get((a, b), 0.0) + \
                        weight * float(row.get("n") or 0.0)
            buy_edges = [{"src_idx": a, "dst_idx": b, "n": v}
                         for (a, b), v in combined.items()]
            trans_view = build_transition_from_edges(
                view_edges, structural, cfg)
            trans_buy = build_transition_from_edges(
                buy_edges, structural, cfg)
            LOGGER.info("transitions: view nnz=%d buy nnz=%d",
                        trans_view.nnz(), trans_buy.nnz())


    zero = np.zeros(n, dtype=np.float32)
    tmp = make_snapshot(
        arrays, alpha0, beta0, s_slow, f_slow, s_org, f_org, org_v,
        org_atc, org_p, org["f_view"], org["f_atc"], org["f_q"],
        price_pct, price_pct_sub, arrays["E"], trans_view, trans_buy,
        np.zeros(len(FEATURE_NAMES), np.float32),
        np.eye(len(FEATURE_NAMES), dtype=np.float32),
        np.zeros(len(FEATURE_NAMES), np.float32),
        np.ones(len(FEATURE_NAMES), np.float32), 0.0, 1.0, w,
        zero, zero, zero, zero, zero,
        0.0, new_ids, new_subs, unavailable)
    tmp = __import__("dataclasses").replace(
        tmp, best_safe=_best_safe_indices(tmp))

    import dataclasses
    try:
        last_row = client.query(SQL_LAST_DAY, heavy=True)
        last_day = parse_datetime(last_row[0].get("last_day")) if last_row \
            else datetime.now(timezone.utc)
    except Exception as exc:
        LOGGER.warning("cannot read last history day: %s", exc)
        last_day = datetime.now(timezone.utc)
    cut = last_day - __import__("datetime").timedelta(days=cfg.holdout_days)
    cut_text = cut.strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info("temporal cut: %s", cut_text)
    train_sessions = fetch_sessions(
        client, cfg, "ts < toDateTime64('%s', 3)" % cut_text,
        parts=int(args.session_parts), limit=int(args.sessions))
    holdout_sessions = fetch_sessions(
        client, cfg, "ts >= toDateTime64('%s', 3)" % cut_text,
        parts=int(args.session_parts), limit=max(int(args.sessions) // 3,
                                                 10000))
    fit = train_context_model(train_sessions, tmp, cfg)
    calib = calibrate_model(holdout_sessions, tmp, fit, cfg)
    LOGGER.info("context fit rows=%d, calibration=%s",
                int(fit.get("rows", 0)), calib)

    mu_beta = np.asarray(fit["mu"], dtype=np.float32)
    Sigma_beta = np.asarray(fit["Sigma"], dtype=np.float32)
    feature_mean = np.asarray(fit["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(fit["feature_std"], dtype=np.float32)
    p95 = float(np.percentile(arrays["price"][arrays["price"] > 0], 95)) \
        if (arrays["price"] > 0).any() else 1.0
    final = make_snapshot(
        arrays, alpha0, beta0, s_slow, f_slow, s_org, f_org, org_v,
        org_atc, org_p, org["f_view"], org["f_atc"], org["f_q"],
        price_pct, price_pct_sub, arrays["E"], trans_view, trans_buy,
        mu_beta, Sigma_beta, feature_mean, feature_std,
        float(calib["a"]), float(calib["T"]), w, zero, zero, zero, zero,
        zero, p95, new_ids, new_subs, unavailable)
    new_bonus = np.zeros(n, dtype=np.float32)
    if new_ids:
        mask = np.asarray([pid in new_ids for pid in arrays["ids"]], dtype=bool)
        new_bonus[mask] = (cfg.c_new / np.sqrt(
            final.alpha[mask] + final.beta[mask] + 1.0)).astype(np.float32)
    final = dataclasses.replace(final, best_safe=_best_safe_indices(final),
                                new_bonus=new_bonus)

    stats = {
        "n_products": int(n), "n_sessions_train": len(train_sessions),
        "n_sessions_holdout": len(holdout_sessions),
        "context_rows": int(fit.get("rows", 0)),
        "calib_a": float(calib["a"]), "calib_T": float(calib["T"]),
        "ece": float(calib["ece"]), "w": np.round(w, 4).tolist(),
        "trans_view_nnz": int(trans_view.nnz()),
        "trans_buy_nnz": int(trans_buy.nnz()),
        "new_products": len(new_ids), "unavailable": int(unavailable.sum()),
    }
    LOGGER.info("build finished in %.1f s: %s", time.time() - t0, stats)
    return final, stats



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Bayes-TS offline priors")
    parser.add_argument("--partitions", type=int, default=4,
                        help="hash partitions for heavy edge queries")
    parser.add_argument("--session-parts", type=int, default=20,
                        help="hash modulus for session sampling")
    parser.add_argument("--sessions", type=int, default=120000,
                        help="maximum training sessions")
    parser.add_argument("--quick", action="store_true",
                        help="lighter sampling for a fast smoke build")
    parser.add_argument("--skip-transitions", action="store_true")
    parser.add_argument("--skip-cooc", action="store_true")
    parser.add_argument("--no-scratch", action="store_true")
    parser.add_argument("--output", default="",
                        help="local .npz path (default state/latest.npz)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.quick:
        args.partitions = min(args.partitions, 4)
        args.session_parts = max(args.session_parts, 40)
        args.sessions = min(args.sessions, 30000)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    cfg = Config.from_env()
    try:
        rt = RawTreeClient(
            api_url=os.environ.get("RAWTREE_API_URL",
                                   "https://api.rawtree.com"),
            api_key=os.environ.get("RAWTREE_API_KEY", ""),
            team_id=os.environ.get("TEAM_ID", "team_03"),
            heavy_timeout=cfg.heavy_query_timeout_s,
            max_execution_s=cfg.query_max_execution_s)
    except ValueError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    try:
        snapshot, stats = build(rt, cfg, args)
    except Exception:
        LOGGER.exception("offline build failed")
        return 1
    base = Path(__file__).resolve().parent
    out = Path(args.output) if args.output else (
        base / cfg.local_state_dir / "latest.npz")
    save_snapshot_npz(snapshot, str(out))
    stats_path = out.parent / "latest_stats.json"
    try:
        import json
        stats_path.write_text(json.dumps(stats, indent=2, default=str))
    except OSError:
        pass
    LOGGER.info("local snapshot: %s", out)
    if not args.no_scratch:
        try:
            sid = write_snapshot_to_scratch(rt, snapshot, cfg,
                                            extra_stats=stats)
            meta = None
            for _ in range(10):
                meta = latest_meta(rt, cfg)
                if meta is not None and str(meta.get("snapshot_id")) == sid:
                    break
                time.sleep(1.0)
            if meta is None or str(meta.get("snapshot_id")) != sid:
                raise RuntimeError("meta commit marker missing")
            LOGGER.info("scratch snapshot committed: %s (%s products)",
                        sid, meta.get("n_products"))
        except Exception:
            LOGGER.exception("scratch write failed")
            return 1
    print("offline build OK: %d products, %d view-nnz, %d buy-nnz, "
          "T=%.2f a=%.3f" % (
              stats["n_products"], stats["trans_view_nnz"],
              stats["trans_buy_nnz"], stats["calib_T"], stats["calib_a"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

