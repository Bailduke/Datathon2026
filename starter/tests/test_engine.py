"""Validity, latency, online update and cold-start tests (spec 6.5.1)."""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import offline_build
from starter import workers
from starter.config import CFG
from starter.features import (build_request_context, normalize_context,
                              normalize_views)
from starter.scoring import (answer, build_candidates, eligible_mask,
                            order_slate, validate_items)
from starter.state import LRU
from starter.tests.factory import make_snapshot


@pytest.fixture(scope="module")
def snapshot():
    return make_snapshot()


@pytest.fixture()
def engine(snapshot):
    return workers.Engine(None, CFG, snapshot)


def test_parse_recent_views_five_formats():
    ts = "2026-11-14T13:14:18.336000000"
    formats = [
        json.dumps([{"product_id": "a", "ts": ts}, {"product_id": "b"}]),
        [{"product_id": "a", "ts": ts}, {"product_id": "b"}],
        [["a", ts], ["b", None]],
        ["a", "b"],
        {"a": ts, "b": None},
    ]
    for raw in formats:
        views = normalize_views(raw)
        assert [pid for pid, _ in views] == ["a", "b"]
    assert [p for p, _ in normalize_views(["a", "b", "a"])] == ["b", "a"]
    assert normalize_views(None) == []
    assert normalize_views("not json") == []


def test_context_parsing_nested_and_flattened():
    assert normalize_context({"device": "ios"})["device"] == "ios"
    assert normalize_context({"context.device": "android",
                              "context.page": "product"})["page"] == "product"
    assert normalize_context(None)["device"] == "web"
    assert normalize_context("{}")["region"] == "?"


def test_page_transition_weights():
    assert CFG.page_lambdas("home") == (0.8, 0.3)
    assert CFG.page_lambdas("category") == (0.8, 0.3)
    assert CFG.page_lambdas("product") == (0.4, 0.9)


def test_filter_excludes_unavailable_inactive_bought(snapshot, engine):
    snap = snapshot
    i0, i1, i2, i3 = 0, 1, 2, 3
    engine.apply_catalog_update({
        "product_id": snap.ids[i0], "change": "out_of_stock",
        "ts": "2026-11-14T13:00:00Z"})
    engine.mutable.active[i1] = False
    engine.mutable.price[i2] = 0.0
    engine.publish()
    engine.mutable.user_bought.add_to_set("u_filter", int(i3), max_items=5)
    req = {"request_id": "r_filter", "user_id": "u_filter",
           "context": {"device": "web", "page": "home"},
           "recent_views": [{"product_id": snap.ids[10]}]}
    items = answer(req, engine.state.snapshot, np.random.default_rng(1))
    assert len(items) == 10 and len(set(items)) == 10
    assert snap.ids[i0] not in items
    assert snap.ids[i1] not in items
    assert snap.ids[i2] not in items
    assert snap.ids[i3] not in items
    assert validate_items(items, engine.state.snapshot, (i3,))


def test_always_ten_and_latency_p99(snapshot):
    rng = np.random.default_rng(2026)
    latencies = []
    views_pool = [snapshot.ids[i] for i in range(0, 200)]
    start = time.perf_counter()
    for k in range(10000):
        m = int(rng.integers(0, 4))
        views = [{"product_id": views_pool[int(rng.integers(0, 200))],
                  "ts": "2026-11-14T13:00:00Z"} for _ in range(m)]
        req = {"request_id": "r%d" % k, "user_id": "u%d" % (k % 500),
               "is_new_user": bool(k % 7 == 0),
               "context": {"device": ["web", "ios", "android"][k % 3],
                           "page": ["home", "category", "product"][k % 3],
                           "region": ["Catalunya", "Madrid", "?"][k % 3]},
               "recent_views": views}
        t0 = time.perf_counter()
        items = answer(req, snapshot, rng)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert len(items) == 10, (k, items)
        assert len(set(items)) == 10
        for pid in items:
            idx = snapshot.index[pid]
            assert snapshot.active[idx]
            assert not snapshot.unavailable[idx]
            assert snapshot.price[idx] > 0
    elapsed = time.perf_counter() - start
    p99 = float(np.percentile(latencies, 99))
    assert p99 < 50.0, "p99=%.2f ms" % p99
    assert 10000 / elapsed > 30.0


def _feedback(engine, rid, shown, purchased="", clicked="", user="u1"):
    engine.mutable.req_meta.set(rid, {
        "user_id": user,
        "req": {"request_id": rid, "user_id": user,
                "context": {"device": "web", "page": "home"},
                "recent_views": [], "is_new_user": False,
                "ts": "2026-11-14T13:00:00Z"}})
    engine.on_feedback_one({
        "request_id": rid, "shown": shown, "served_by": "team",
        "fallback_reason": "", "latency_ms": 20.0,
        "clicked": clicked, "purchased": purchased, "revenue_eur": 0.0,
        "ts": "2026-11-14T13:00:01Z"})


def test_beta_update_position_bias(engine, snapshot):
    m = engine.mutable
    shown = [snapshot.ids[i] for i in range(10)]
    m.s_slow[:] = 0.0
    m.f_slow[:] = 0.0
    _feedback(engine, "r_purchase", shown, purchased=shown[0],
              clicked=shown[0])
    assert m.s_slow[0] == pytest.approx(1.0, abs=1e-6)
    assert m.f_slow[9] == pytest.approx(0.02 * 0.08, abs=1e-6)
    m.s_slow[:] = 0.0
    m.f_slow[:] = 0.0
    _feedback(engine, "r_none10", shown)
    f_slot10 = float(m.f_slow[9])
    m.f_slow[:] = 0.0
    _feedback(engine, "r_none1", shown)
    f_slot1 = float(m.f_slow[0])
    assert f_slot10 < f_slot1
    assert f_slot10 == pytest.approx(0.02 * 0.08, abs=1e-6)
    assert f_slot1 == pytest.approx(0.02, abs=1e-6)


def test_new_product_initialisation(engine):
    row = {"product_id": "p_brand_new", "category": "cat1",
           "subcategory": "sub1", "brand": "brand1",
           "price_eur": 19.99, "description": "A brand new yoga mat",
           "is_active": "true", "change": "new_product",
           "ts": "2026-11-14T13:20:00Z", "active_s": "true"}
    idx = engine.add_new_product(row)
    assert idx is not None
    assert "p_brand_new" in engine.mutable.index
    assert engine.mutable.is_new[idx]
    ess = float(engine.mutable.alpha0[idx] + engine.mutable.beta0[idx])
    assert 6.0 <= ess <= 8.0
    engine.publish()
    snap = engine.state.snapshot
    req = {"request_id": "r_cold", "user_id": "u_new",
           "context": {"device": "ios", "page": "product"},
           "recent_views": [{"product_id": "p_brand_new"}],
           "is_new_user": True}
    rc = build_request_context(req, snap, CFG)
    assert rc.cold and rc.cold_reserve
    valid = np.flatnonzero(eligible_mask(snap))
    cand = build_candidates(snap, valid, rc, np.random.default_rng(3), CFG,
                            snap.policy)
    assert idx in cand
    items = answer(req, snap, np.random.default_rng(4))
    assert len(items) == 10 and len(set(items)) == 10


def test_out_of_stock_and_back_in_stock(engine, snapshot):
    pid = snapshot.ids[0]
    engine.apply_catalog_update({"product_id": pid,
                                 "change": "out_of_stock",
                                 "ts": "2026-11-14T13:30:00Z"})
    engine.publish()
    assert engine.state.snapshot.unavailable[0]
    items = answer({"request_id": "r_stock", "user_id": "u",
                    "context": {"page": "home"}, "recent_views": []},
                   engine.state.snapshot, np.random.default_rng(5))
    assert pid not in items and len(items) == 10
    engine.apply_catalog_update({"product_id": pid,
                                 "change": "back_in_stock",
                                 "ts": "2026-11-14T13:31:00Z"})
    engine.publish()
    assert not engine.state.snapshot.unavailable[0]
    assert engine.state.snapshot.active[0]
    assert engine.state.snapshot.price[0] > 0


def test_cold_reserve_uses_subcategory_code(snapshot):
    code = int(snapshot.subcategory[5])
    snap = make_snapshot(new_subcats=(code,))
    req = {"request_id": "r_sub", "user_id": "u",
           "context": {"page": "category"},
           "recent_views": [{"product_id": snap.ids[5]}]}
    rc = build_request_context(req, snap, CFG)
    assert rc.cold_reserve
    assert int(snap.subcategory[snap.index[snap.ids[5]]]) in snap.new_subcats



def test_regime_shift(engine):
    before_slow = float(engine.mutable.s_slow[:engine.mutable.n].sum())
    before_trace = float(np.trace(engine.mutable.Sigma_beta))
    assert engine.regime_shift("test")
    assert not engine.regime_shift("test")
    assert float(engine.mutable.s_slow[:engine.mutable.n].sum()) == \
        pytest.approx(0.25 * before_slow, rel=1e-5)
    assert float(np.trace(engine.mutable.Sigma_beta)) > before_trace
    snap = engine.state.snapshot
    assert snap.policy.regime == "final"
    assert snap.policy.temperature_mult >= 1.25
    assert snap.policy.epsilon_explore >= 0.7


def test_order_slate_rng_is_explicit(snapshot):
    params = inspect.signature(order_slate).parameters
    assert "rng" in params
    cand = np.arange(100, dtype=np.int64)
    p = np.linspace(0.2, 0.01, 100).astype(np.float32)
    U = p.copy()
    rng = np.random.default_rng(9)
    chosen = order_slate(cand, p, U, U, snapshot, CFG, False, rng)
    assert len(chosen) == 10
    assert len(set(chosen)) == 10


def test_cholesky_factor_cached(snapshot):
    assert np.allclose(snapshot.L_beta @ snapshot.L_beta.T,
                       snapshot.Sigma_beta, atol=1e-5)
    from starter.features import build_request_context
    from starter.scoring import deterministic_scores, sample_scores
    req = {"request_id": "r_l", "user_id": "u",
           "context": {"page": "home"},
           "recent_views": [{"product_id": snapshot.ids[1]}]}
    rc = build_request_context(req, snapshot, CFG)
    cand = np.arange(20, dtype=np.int64)
    Z, eta, p, U = deterministic_scores(cand, rc, snapshot, CFG,
                                        snapshot.policy)
    p2, U2 = sample_scores(cand, Z, eta, rc, snapshot,
                           np.random.default_rng(2), CFG, snapshot.policy)
    assert p2.shape == (20,) and np.all(p2 > 0) and np.all(p2 < 1)


def test_frozenset_copy_on_write():
    lru = LRU(10, 60)
    old = lru.add_to_set("u", 1)
    new = lru.add_to_set("u", 2)
    assert old == frozenset({1})
    assert new == frozenset({1, 2})
    assert lru.get("u") == frozenset({1, 2})
    for _ in old:
        pass


def test_snapshot_is_frozen(snapshot):
    with pytest.raises(FrozenInstanceError):
        snapshot.n = 1  # type: ignore[misc]


def test_cobuy_sql_is_causal():
    sql = offline_build.SQL_COBUY
    assert "ts::DateTime64(3) AS ts" in sql
    assert "AND v.ts <= pur.ts" in sql
    assert "LIMIT 200 BY src_id" in sql

