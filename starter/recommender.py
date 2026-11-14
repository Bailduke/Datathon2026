"""Bayes-TS entry point, compatible with the original starter API.

Run::

    python starter/recommender.py            # unattended service
    python starter/recommender.py --once     # one pull/predict/push cycle

The notebook imports ``best_sellers`` and ``recommend`` exactly as before; both
are thin wrappers around the same ``answer()`` code that runs live.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

if __package__ in (None, ""):
    # Allow ``python starter/recommender.py`` without PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import workers
from starter.config import CFG, Config, load_dotenv
from starter.features import FEATURE_NAMES
from starter.priors import (build_structural_prior, fit_hierarchy,
                            load_snapshot_from_scratch, load_snapshot_npz)
from starter.rawtree import RawTreeClient, iso_utc, parse_datetime, to_jsonable
from starter.scoring import answer, fill_safe
from starter.state import CSR, Snapshot
from starter.text import build_embeddings

LOGGER = logging.getLogger("recommender")

load_dotenv()
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_03")
_CLIENT: Optional[RawTreeClient] = None
_ENGINE: Optional[workers.Engine] = None
_COMPAT_SNAPSHOT: Optional[Snapshot] = None
_COMPAT_LOCK = threading.Lock()

SALES = """
SELECT product_id::String AS product_id, count() AS n
FROM shop.history_events
WHERE event_type::String = 'purchase'{sales}
GROUP BY product_id"""

NEW_REQUESTS = """
SELECT request_id::String AS request_id, recent_views
FROM shop.requests
ORDER BY ts::DateTime64(3) DESC
LIMIT 50"""

OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

CATALOG = workers.SQL_CATALOG


def client() -> RawTreeClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = RawTreeClient(api_url=API,
                                api_key=os.environ.get("RAWTREE_API_KEY", ""),
                                team_id=TEAM)
    return _CLIENT


def query(sql: str, heavy: bool = False) -> List[Dict[str, Any]]:
    return client().query(sql, heavy=heavy)


def best_sellers(until: Optional[str] = None) -> List[str]:
    """Best-selling products (optionally before ``until``), as in the starter."""
    cut = " AND ts < toDateTime64('%s', 3)" % until if until else ""
    rows = query(SALES.format(sales=cut), heavy=True)
    sales = sorted(((int(r.get("n") or 0), str(r.get("product_id") or ""))
                    for r in rows), reverse=True)
    return [pid for _, pid in sales if pid]


def out_of_stock() -> Set[str]:
    return {str(r.get("product_id")) for r in query(OUT_OF_STOCK)}


def pull(seen: Set[str]) -> List[Dict[str, Any]]:
    fresh = []
    for req in query(NEW_REQUESTS):
        rid = str(req.get("request_id"))
        if rid not in seen:
            seen.add(rid)
            fresh.append(req)
    return fresh


def _snapshot_for_compat(best: Sequence[str],
                         unavailable: Optional[Set[str]] = None) -> Snapshot:
    """Neutral snapshot built from the catalogue, for the notebook wrapper."""
    global _COMPAT_SNAPSHOT
    with _COMPAT_LOCK:
        if _COMPAT_SNAPSHOT is not None:
            return _COMPAT_SNAPSHOT
        unavailable = set(unavailable or ())
        try:
            rows = query(CATALOG, heavy=True)
        except Exception as exc:
            LOGGER.warning("catalog unavailable for notebook snapshot: %s", exc)
            rows = [{"product_id": pid, "price_eur": 1.0} for pid in best]
        by_id = {str(r.get("product_id")): r for r in rows
                 if r.get("product_id")}
        ids: List[str] = list(by_id.keys())
        products: List[Dict[str, Any]] = [by_id[pid] for pid in ids]
        for pid in best:
            if pid and pid not in by_id:
                ids.append(pid)
                products.append({"product_id": pid, "price_eur": 1.0})
        if not ids:
            ids = list(best)
            products = [{"product_id": pid, "price_eur": 1.0} for pid in best]
        n = len(ids)
        index = {pid: i for i, pid in enumerate(ids)}
        category_names: List[str] = []
        sub_names: List[str] = []
        brand_names: List[str] = []
        cat = np.zeros(n, np.int16)
        sub = np.zeros(n, np.int16)
        brand = np.zeros(n, np.int16)
        price = np.ones(n, np.float32)
        active = np.ones(n, bool)
        for i, product in enumerate(products):
            for value, names, target in (
                    (product.get("category"), category_names, cat),
                    (product.get("subcategory"), sub_names, sub),
                    (product.get("brand"), brand_names, brand)):
                text = str(value or "?")
                if text not in names:
                    names.append(text)
                target[i] = names.index(text)
            try:
                price[i] = max(float(product.get("price_eur") or 1.0), 0.01)
            except (TypeError, ValueError):
                price[i] = 1.0
            active[i] = str(product.get("is_active", True)).lower() not in (
                "false", "0")
        E = build_embeddings([{"description": p.get("description"),
                               "category": p.get("category"),
                               "subcategory": p.get("subcategory"),
                               "brand": p.get("brand")} for p in products],
                             CFG.text_dim)
        hierarchy = fit_hierarchy(np.full(n, 0.02, dtype=np.float32),
                                  np.full(n, 1.0, dtype=np.float32),
                                  cat, sub, CFG)
        unavail = np.zeros(n, bool)
        for pid in unavailable:
            if pid in index:
                unavail[index[pid]] = True
        s_slow = np.zeros(n, np.float32)
        for rank, pid in enumerate(best[:500]):
            if pid in index:
                s_slow[index[pid]] += 20.0 / (1.0 + rank / 50.0)
        prices = price[price > 0]
        snap = Snapshot(
            ids=ids, index=index, n=n, category=cat, subcategory=sub,
            brand=brand, category_names=tuple(category_names),
            subcategory_names=tuple(sub_names), brand_names=tuple(brand_names),
            price=price, active=active, unavailable=unavail,
            alpha0=hierarchy["alpha0"], beta0=hierarchy["beta0"],
            alpha=(hierarchy["alpha0"] + s_slow).astype(np.float32),
            beta=hierarchy["beta0"],
            s_slow=s_slow, f_slow=np.zeros(n, np.float32),
            s_fast=np.zeros(n, np.float32), f_fast=np.zeros(n, np.float32),
            s_click=np.zeros(n, np.float32), f_click=np.zeros(n, np.float32),
            s_org=np.zeros(n, np.float32), f_org=np.zeros(n, np.float32),
            org_v=np.zeros(n, np.float32), org_atc=np.zeros(n, np.float32),
            org_p=np.zeros(n, np.float32),
            feat_view=np.zeros(n, np.float32),
            feat_atc=np.zeros(n, np.float32),
            feat_q=np.zeros(n, np.float32),
            price_pct=np.full(n, 0.5, np.float32),
            price_pct_subcat=np.full(n, 0.5, np.float32), E=E,
            trans_view=CSR.empty(n), trans_buy=CSR.empty(n),
            mu_beta=np.zeros(len(FEATURE_NAMES), np.float32),
            Sigma_beta=np.eye(len(FEATURE_NAMES), dtype=np.float32),
            feature_names=FEATURE_NAMES,
            feature_mean=np.zeros(len(FEATURE_NAMES), np.float32),
            feature_std=np.ones(len(FEATURE_NAMES), np.float32),
            w=np.asarray(CFG.default_position_w, np.float32),
            trend_boost=np.zeros(n, np.float32),
            new_bonus=np.zeros(n, np.float32),
            restock_bonus=np.zeros(n, np.float32),
            price_bonus=np.zeros(n, np.float32),
            best_safe=np.asarray([index[p] for p in best if p in index][:10],
                                 dtype=np.int64),
            p95_price=float(np.percentile(prices, 95)) if prices.size else 1.0)
        _COMPAT_SNAPSHOT = snap
        return snap


def recommend(recent_views: Any, best: Sequence[str],
              unavailable: Optional[Set[str]] = None) -> List[str]:
    """The part the notebook replaces: 10 IDs through the live ``answer()``."""
    snap = _snapshot_for_compat(best, unavailable)
    req = {"request_id": "notebook", "user_id": "notebook_user",
           "context": {"device": "web", "region": "?", "page": "home"},
           "recent_views": recent_views, "is_new_user": False}
    return answer(req, snap, np.random.default_rng(), CFG)



def predict(requests: Sequence[Mapping[str, Any]], best: Sequence[str],
            unavailable: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """One submission per request (starter-compatible)."""
    global _ENGINE
    if _ENGINE is not None:
        snap = _ENGINE.state.snapshot
        rng = _ENGINE.rng
    else:
        snap = _snapshot_for_compat(best, unavailable)
        rng = np.random.default_rng(CFG.seed)
    out = []
    for req in requests:
        rid = str(req.get("request_id"))
        out.append({"request_id": rid,
                    "items": answer(req, snap, rng, CFG),
                    "model_version": CFG.model_version})
    return out


def push(answers: Sequence[Mapping[str, Any]]) -> None:
    """Batch insert into team_NN.submissions (starter-compatible)."""
    client().insert_rows(CFG.submissions_table, list(answers),
                         timeout=CFG.insert_timeout_s)


def build_live_bootstrap(cfg: Config = CFG) -> Snapshot:
    """Catalogue + neutral priors when neither .npz nor scratch exist."""
    rows = query(CATALOG, heavy=True)
    products: List[Dict[str, Any]] = []
    ids: List[str] = []
    for row in rows:
        pid = str(row.get("product_id") or "")
        if not pid:
            continue
        ids.append(pid)
        products.append(row)
    if not products:
        return Snapshot(ids=[], index={}, n=0, feature_names=FEATURE_NAMES)
    n = len(products)
    index = {pid: i for i, pid in enumerate(ids)}
    category_names: List[str] = []
    sub_names: List[str] = []
    brand_names: List[str] = []
    cat = np.zeros(n, np.int16)
    sub = np.zeros(n, np.int16)
    brand = np.zeros(n, np.int16)
    price = np.ones(n, np.float32)
    active = np.ones(n, bool)
    for i, product in enumerate(products):
        for value, names, target in (
                (product.get("category"), category_names, cat),
                (product.get("subcategory"), sub_names, sub),
                (product.get("brand"), brand_names, brand)):
            text = str(value or "?")
            if text not in names:
                names.append(text)
            target[i] = names.index(text)
        try:
            price[i] = max(float(product.get("price_eur") or 0.0), 0.0)
        except (TypeError, ValueError):
            price[i] = 0.0
        active[i] = str(product.get("is_active", True)).lower() not in (
            "false", "0")
    try:
        unavailable_ids = out_of_stock()
    except Exception:
        unavailable_ids = set()
    unavailable = np.zeros(n, bool)
    for pid in unavailable_ids:
        if pid in index:
            unavailable[index[pid]] = True
    E = build_embeddings(products, cfg.text_dim)
    hierarchy = fit_hierarchy(np.full(n, 0.02, dtype=np.float32),
                              np.full(n, 1.0, dtype=np.float32), cat, sub,
                              cfg)
    prices = price[price > 0]
    order = np.argsort(price)
    price_pct = np.empty(n, np.float32)
    if n:
        price_pct[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    else:
        price_pct = np.zeros(0, np.float32)
    price_pct_sub = price_pct.copy()
    try:
        trans = build_structural_prior(E, cat, sub, brand, cfg,
                                       top_k=cfg.trans_top_k)
    except Exception as exc:
        LOGGER.warning("structural prior failed: %s", exc)
        trans = CSR.empty(n)
    alpha0 = hierarchy["alpha0"]
    beta0 = hierarchy["beta0"]
    mu = alpha0 / np.maximum(alpha0 + beta0, 1e-9)
    eligible = active & (~unavailable) & (price > 0)
    safe = np.argsort(np.where(eligible, mu, -1.0))[::-1][:10]
    best_safe = safe[eligible[safe]].astype(np.int64)
    if best_safe.size < 10:
        best_safe = np.flatnonzero(eligible)[:10].astype(np.int64)
    return Snapshot(
        ids=ids, index=index, n=n, category=cat, subcategory=sub, brand=brand,
        category_names=tuple(category_names),
        subcategory_names=tuple(sub_names), brand_names=tuple(brand_names),
        price=price, active=active, unavailable=unavailable,
        alpha0=alpha0, beta0=beta0, alpha=alpha0.copy(), beta=beta0.copy(),
        s_slow=np.zeros(n, np.float32), f_slow=np.zeros(n, np.float32),
        s_fast=np.zeros(n, np.float32), f_fast=np.zeros(n, np.float32),
        s_click=np.zeros(n, np.float32), f_click=np.zeros(n, np.float32),
        s_org=np.zeros(n, np.float32), f_org=np.zeros(n, np.float32),
        org_v=np.zeros(n, np.float32), org_atc=np.zeros(n, np.float32),
        org_p=np.zeros(n, np.float32), feat_view=np.zeros(n, np.float32),
        feat_atc=np.zeros(n, np.float32), feat_q=np.zeros(n, np.float32),
        price_pct=price_pct, price_pct_subcat=price_pct_sub, E=E,
        trans_view=trans, trans_buy=trans,
        mu_beta=np.zeros(len(FEATURE_NAMES), np.float32),
        Sigma_beta=np.eye(len(FEATURE_NAMES), dtype=np.float32),
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(len(FEATURE_NAMES), np.float32),
        feature_std=np.ones(len(FEATURE_NAMES), np.float32),
        w=np.asarray(cfg.default_position_w, np.float32),
        trend_boost=np.zeros(n, np.float32),
        new_bonus=np.zeros(n, np.float32),
        restock_bonus=np.zeros(n, np.float32),
        price_bonus=np.zeros(n, np.float32), best_safe=best_safe,
        p95_price=float(np.percentile(prices, 95)) if prices.size else 1.0)


def bootstrap_snapshot(cfg: Config = CFG) -> Snapshot:
    """Load .npz -> scratch -> live catalogue, in that order."""
    state_dir = Path(__file__).resolve().parent.parent / cfg.local_state_dir
    local = state_dir / "latest.npz"
    snap = load_snapshot_npz(str(local))
    if snap is not None and snap.n > 0:
        age = time.time() - float(snap.ts or 0.0)
        if age < 6 * 3600:
            LOGGER.info("loaded local snapshot (%d products, age %.1f min)",
                        snap.n, age / 60.0)
            return snap
        LOGGER.warning("local snapshot older than 6 h; ignoring")
    try:
        snap = load_snapshot_from_scratch(client(), cfg)
    except Exception as exc:
        LOGGER.warning("scratch load failed: %s", exc)
        snap = None
    if snap is not None and snap.n > 0:
        LOGGER.info("loaded scratch snapshot (%d products)", snap.n)
        return snap
    LOGGER.info("building live bootstrap snapshot")
    snap = build_live_bootstrap(cfg)
    LOGGER.info("bootstrap snapshot ready (%d products)", snap.n)
    return snap


def _process_once(engine: workers.Engine, push_answers: bool = True) -> int:
    """One pull -> answer -> push cycle for integration checks."""
    cfg = engine.cfg
    try:
        engine.reconcile_catalog()
        engine.publish()
    except Exception as exc:
        LOGGER.warning("reconcile in --once: %s", exc)
    try:
        rows = client().query(workers.SQL_REQUESTS.format(
            window=int(cfg.request_window_s), limit=int(cfg.request_limit)))
    except Exception:
        rows = []
    if not rows:
        try:
            rows = query(NEW_REQUESTS)
        except Exception:
            rows = []
    now = client().now()
    batch: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for req in rows:
        rid = str(req.get("request_id") or "")
        if not rid or rid == "__init__" or rid in seen:
            continue
        seen.add(rid)
        ts = parse_datetime(req.get("ts"))
        age = (now - ts).total_seconds() if ts else 0.0
        if age > cfg.deadline_ms / 1000.0:
            continue
        req = dict(req)
        req["_now"] = now
        started = time.perf_counter()
        items = answer(req, engine.state.snapshot, engine.rng, cfg)
        latency_ms = (time.perf_counter() - started) * 1000.0
        user = str(req.get("user_id") or "?")
        bought = tuple(int(i) for i in (
            engine.state.snapshot.user_bought.get(user) or ()))
        LOGGER.info("one-shot %s: %d items in %.2f ms", rid, len(items),
                    latency_ms)
        batch.append({"request_id": rid, "items": items,
                      "model_version": cfg.model_version,
                      "ts": req.get("ts"),
                      "enqueued": time.monotonic(), "user_id": user,
                      "bought": bought})
        if len(batch) >= cfg.max_answers_per_cycle:
            break
    if push_answers and batch:
        written = engine.push_batch(batch)
        LOGGER.info("one-shot pushed %d answers", written)
    return len(batch)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bayes-TS recommender")
    parser.add_argument("--once", action="store_true",
                        help="one pull/predict/push cycle and exit")
    parser.add_argument("--no-push", action="store_true",
                        help="answer but do not write submissions")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="run for N seconds then shut down cleanly")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    global _ENGINE
    cfg = Config.from_env()
    snap = bootstrap_snapshot(cfg)
    if snap.n == 0:
        LOGGER.error("empty catalogue: cannot start")
        return 2
    _ENGINE = workers.Engine(client(), cfg, snap)
    if args.once:
        n = _process_once(_ENGINE, push_answers=not args.no_push)
        return 0 if n >= 0 else 1

    def _stop(signum: int, frame: Any) -> None:
        LOGGER.info("signal %s received, shutting down", signum)
        if _ENGINE is not None:
            _ENGINE.stop.set()
            _ENGINE.wake_catalog.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    _ENGINE.start()
    started = time.time()
    try:
        while not _ENGINE.stop.is_set():
            time.sleep(1.0)
            if args.duration and time.time() - started >= args.duration:
                break
    finally:
        _ENGINE.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

