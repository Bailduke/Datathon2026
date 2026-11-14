"""RawTree Control Tower agent (designed as an independent process).

The agent never sits in ``answer()``'s critical path.  It watches the organic
stream (``shop.events``), catalogue changes (``shop.catalog_updates``) and our
private feedback (``team_NN.feedback``), maintains a digital twin of the plant
in two schema-free tables (``team_NN.factory_inventory`` and
``team_NN.production_orders``), records its SQL investigations and decisions in
``team_NN.agent_traces``, and publishes a *strictly clamped* policy row
(``kind = 'control_tower_policy'``) that ``health_loop`` reads.

Run::

    python control_tower_agent.py --once
    python control_tower_agent.py --interval 15
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from starter.config import CFG, Config, load_dotenv
from starter.priors import clamp_policy, write_policy_row
from starter.rawtree import RawTreeClient, iso_utc, parse_datetime, to_jsonable

LOGGER = logging.getLogger("control_tower")

SQL_EVENTS_15M = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views_15m,
       countIf(event_type::String = 'add_to_cart') AS atc_15m,
       countIf(event_type::String = 'purchase') AS purchases_15m
FROM shop.events
WHERE ts > now() - INTERVAL 15 MINUTE
GROUP BY product_id"""

SQL_EVENTS_180M = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views_180m,
       countIf(event_type::String = 'purchase') AS purchases_180m
FROM shop.events
WHERE ts > now() - INTERVAL 180 MINUTE
  AND ts <= now() - INTERVAL 15 MINUTE
GROUP BY product_id"""

SQL_CATALOG_UPDATES = """
SELECT product_id::String AS product_id,
       ts::DateTime64(3) AS ts,
       change::String AS change,
       toString(category) AS category,
       toString(subcategory) AS subcategory,
       toFloat64OrZero(toString(price_eur)) AS price_eur,
       toString(is_active) AS active_s
FROM shop.catalog_updates
WHERE ts > now() - INTERVAL 6 HOUR
ORDER BY ts::DateTime64(3) DESC
LIMIT 2000"""

SQL_FEEDBACK = """
SELECT count() AS requests,
       countIf(purchased != '') AS purchases,
       countIf(served_by::String = 'fallback') AS fallbacks,
       quantile(0.99)(toFloat64OrZero(toString(latency_ms))) AS p99_ms,
       sum(toFloat64OrZero(toString(revenue_eur))) AS revenue
FROM {db}.feedback
WHERE request_id != '__init__'
  AND ts > now() - INTERVAL 30 MINUTE"""

SQL_CATALOG_SUMMARY = """
SELECT toString(category) AS category,
       toString(subcategory) AS subcategory,
       count() AS products,
       avg(toFloat64OrZero(toString(price_eur))) AS avg_price
FROM shop.catalog
GROUP BY category, subcategory"""

SQL_EXISTING_ORDERS = """
SELECT product_id::String AS product_id, qty::Int32 AS qty,
       status::String AS status
FROM {db}.production_orders
WHERE status::String = 'open'
ORDER BY ts::DateTime64(3) DESC
LIMIT 500"""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ControlTowerAgent:
    def __init__(self, client: RawTreeClient, cfg: Config = CFG) -> None:
        self.client = client
        self.cfg = cfg
        self.inventory: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, float] = {}
        self.last_policy: Dict[str, float] = {}
        self.started = time.time()
        self.table_ok: Dict[str, bool] = {
            "factory_inventory": True, "production_orders": True,
            "agent_traces": True}
        self.ensure_tables()

    def ensure_tables(self) -> None:
        """Try to create the side-challenge tables; fall back to scratch."""
        for table in ("factory_inventory", "production_orders",
                      "agent_traces"):
            try:
                self.client.create_table(table)
                LOGGER.info("table ready: %s", table)
            except Exception as exc:
                LOGGER.info("table %s not creatable (%s); using scratch "
                            "fallback rows", table, exc)

    def _insert(self, table: str, rows: Sequence[Mapping[str, Any]],
                kind: str) -> int:
        if not rows:
            return 0
        if self.table_ok.get(table, True):
            try:
                return self.client.insert_rows(
                    table, list(rows), database=self.client.team_id)
            except Exception as exc:
                self.table_ok[table] = False
                LOGGER.warning("insert into %s failed (%s); falling back to "
                               "scratch kind=%s", table, exc, kind)
        fallback = []
        for row in rows:
            item = dict(row)
            item["kind"] = kind
            item.setdefault("ts", iso_utc())
            fallback.append(item)
        return self.client.insert_rows(self.cfg.scratch_table, fallback,
                                        database=self.client.team_id)

    # ---------------------------------------------------------- ingestion
    def refresh_catalog(self) -> Tuple[int, str]:
        rows = self.client.query(SQL_CATALOG_SUMMARY, heavy=True)
        for row in rows:
            sub = str(row.get("subcategory") or "?")
            self.inventory.setdefault("__sub__" + sub, {
                "product_id": "__sub__" + sub, "subcategory": sub,
                "category": row.get("category"),
                "products": int(_num(row.get("products"))),
                "avg_price": _num(row.get("avg_price")), "stock": 0.0})
        return len(rows), SQL_CATALOG_SUMMARY.strip()

    def refresh_updates(self) -> Tuple[int, str]:
        try:
            rows = self.client.query(SQL_CATALOG_UPDATES, heavy=True)
        except Exception as exc:
            LOGGER.info("catalog updates unavailable: %s", exc)
            return 0, SQL_CATALOG_UPDATES.strip()
        for row in rows:
            pid = str(row.get("product_id") or "")
            if not pid:
                continue
            item = self.inventory.setdefault(pid, {
                "product_id": pid, "stock": 120.0, "safety_stock": 25.0,
                "demand_15m": 0.0, "demand_180m": 0.0,
                "stockout_risk": 0.0})
            item["category"] = row.get("category")
            item["subcategory"] = row.get("subcategory")
            item["price_eur"] = _num(row.get("price_eur"),
                                     item.get("price_eur", 0.0))
            item["last_change"] = row.get("change")
            item["last_change_ts"] = row.get("ts")
            if str(row.get("change")) == "out_of_stock":
                item["stock"] = 0.0
            elif str(row.get("change")) == "back_in_stock":
                item["stock"] = max(_num(item.get("stock")), 60.0)
        return len(rows), SQL_CATALOG_UPDATES.strip()

    def refresh_events(self) -> Tuple[int, str]:
        try:
            now_rows = self.client.query(SQL_EVENTS_15M, heavy=True)
            base_rows = self.client.query(SQL_EVENTS_180M, heavy=True)
        except Exception as exc:
            LOGGER.info("events unavailable: %s", exc)
            return 0, SQL_EVENTS_15M.strip()
        base = {str(r.get("product_id")): r for r in base_rows}
        total_purchases = sum(_num(r.get("purchases_15m"))
                              for r in now_rows)
        for row in now_rows:
            pid = str(row.get("product_id") or "")
            if not pid:
                continue
            item = self.inventory.setdefault(pid, {
                "product_id": pid, "stock": 120.0, "safety_stock": 25.0})
            views = _num(row.get("views_15m"))
            atc = _num(row.get("atc_15m"))
            purchases = _num(row.get("purchases_15m"))
            b = base.get(pid, {})
            item["views_15m"] = views
            item["atc_15m"] = atc
            item["purchases_15m"] = purchases
            item["views_180m"] = _num(b.get("views_180m"))
            item["purchases_180m"] = _num(b.get("purchases_180m"))
            demand = purchases + 0.2 * atc
            item["demand_15m"] = demand
            item["demand_180m"] = _num(b.get("purchases_180m"))
            item["demand_share"] = purchases / max(total_purchases, 1.0)
            item["stock"] = max(_num(item.get("stock"), 120.0) - demand, 0.0)
            expected = max(demand * 4.0, 1.0)
            item["stockout_risk"] = float(np.clip(
                1.0 - _num(item.get("stock"), 0.0) /
                (expected + _num(item.get("safety_stock"), 25.0)), 0.0, 1.0))
        return len(now_rows), (SQL_EVENTS_15M.strip() + "\n" +
                               SQL_EVENTS_180M.strip())

    def refresh_feedback(self) -> Tuple[Dict[str, float], str]:
        try:
            rows = self.client.query(
                SQL_FEEDBACK.replace("{db}", self.client.team_id), heavy=True)
        except Exception as exc:
            LOGGER.info("feedback unavailable: %s", exc)
            rows = []
        row = rows[0] if rows else {}
        requests = max(_num(row.get("requests")), 1.0)
        metrics = {
            "requests": _num(row.get("requests")),
            "purchases": _num(row.get("purchases")),
            "fallbacks": _num(row.get("fallbacks")),
            "fallback_rate": _num(row.get("fallbacks")) / requests,
            "conversion": _num(row.get("purchases")) / requests,
            "p99_ms": _num(row.get("p99_ms")),
            "revenue": _num(row.get("revenue")),
        }
        return metrics, SQL_FEEDBACK.strip()


    # ------------------------------------------------------------ planning
    def plan_orders(self) -> List[Dict[str, Any]]:
        if not self.table_ok.get("production_orders", True):
            self.open_orders = {}
        else:
            try:
                rows = self.client.query(
                    SQL_EXISTING_ORDERS.replace("{db}", self.client.team_id),
                    heavy=True)
                self.open_orders = {str(r.get("product_id")):
                                    _num(r.get("qty")) for r in rows}
            except Exception:
                self.open_orders = {}
        scored = []
        for pid, item in self.inventory.items():
            if pid.startswith("__sub__"):
                continue
            risk = _num(item.get("stockout_risk"))
            demand = _num(item.get("demand_15m"))
            stock = _num(item.get("stock"))
            if risk <= 0.35 and demand <= 0:
                continue
            target = max(4.0 * demand + _num(item.get("safety_stock"), 25.0),
                         30.0)
            qty = max(int(round(target - stock)), 0)
            if qty <= 0 or pid in self.open_orders:
                continue
            scored.append({
                "product_id": pid,
                "subcategory": item.get("subcategory"),
                "qty": qty,
                "priority": round(float(risk * (1.0 + demand)), 4),
                "stockout_risk": round(risk, 4),
                "lead_time_min": 45 if risk > 0.7 else 90,
                "reason": "stockout_risk" if risk > 0.7 else "demand_lift",
                "status": "open", "ts": iso_utc(),
                "origin": "control_tower_agent"})
        scored.sort(key=lambda r: r["priority"], reverse=True)
        return scored[:60]

    def decide_policy(self, feedback: Mapping[str, float]) -> Dict[str, Any]:
        risks = np.asarray([_num(i.get("stockout_risk"))
                            for k, i in self.inventory.items()
                            if not k.startswith("__sub__")])
        high_risk = float((risks > 0.7).mean()) if risks.size else 0.0
        demands = np.asarray([_num(i.get("demand_share"))
                              for k, i in self.inventory.items()
                              if not k.startswith("__sub__")])
        concentration = float(np.max(demands)) if demands.size else 0.0
        fallback = _num(feedback.get("fallback_rate"))
        conversion = _num(feedback.get("conversion"))
        raw = {
            "restock_boost": 1.0 + 0.25 * high_risk,
            "trend_scale": 1.0 + 0.30 * concentration,
            "temperature_mult": 1.0 + (0.10 if fallback > 0.02 else
                                       -0.03 if conversion > 0.05 else 0.0),
            "exploration_delta": float(np.clip(
                0.08 * high_risk - 0.05 * conversion, -0.08, 0.08)),
        }
        policy = clamp_policy(raw)
        policy.update({
            "ts": iso_utc(),
            "high_stockout_share": round(high_risk, 4),
            "demand_concentration": round(concentration, 4),
            "fallback_rate": round(fallback, 5),
            "conversion_30m": round(conversion, 5),
            "reasoning": ("stockout pressure raises restock boost; demand "
                          "concentration raises trend scale; poor reliability "
                          "raises temperature; conversion dampens exploration"),
            "agent": "control_tower"})
        self.last_policy = policy
        return policy


    # -------------------------------------------------------------- output
    def write_state(self, orders: Sequence[Mapping[str, Any]],
                    policy: Mapping[str, Any]) -> None:
        inventory_rows = []
        for pid, item in list(self.inventory.items()):
            if pid.startswith("__sub__"):
                continue
            row = {"kind": "factory_inventory", "ts": iso_utc(),
                   "product_id": pid}
            row.update({k: to_jsonable(v) for k, v in item.items()
                        if k != "product_id"})
            inventory_rows.append(row)
        inventory_rows = sorted(
            inventory_rows,
            key=lambda r: -_num(r.get("stockout_risk")))[:250]
        if inventory_rows:
            self._insert("factory_inventory", inventory_rows,
                         "factory_inventory")
        if orders:
            self._insert("production_orders", [dict(o) for o in orders],
                         "production_orders")
        write_policy_row(self.client, policy, self.cfg)

    def write_trace(self, sql_used: Sequence[str],
                    observations: Mapping[str, Any],
                    decisions: Mapping[str, Any]) -> None:
        row = {
            "kind": "agent_trace", "ts": iso_utc(),
            "agent": "control_tower",
            "sql_investigations": list(sql_used),
            "observations": to_jsonable(observations),
            "decisions": to_jsonable(decisions),
            "policy": self.last_policy}
        try:
            self._insert("agent_traces", [row], "agent_trace")
        except Exception as exc:
            LOGGER.warning("cannot write trace: %s", exc)

    # ---------------------------------------------------------------- cycle
    def run_cycle(self) -> Dict[str, Any]:
        sql_used: List[str] = []
        n_sub, sql = self.refresh_catalog()
        sql_used.append(sql)
        n_upd, sql = self.refresh_updates()
        sql_used.append(sql)
        n_ev, sql = self.refresh_events()
        sql_used.append(sql)
        feedback, sql = self.refresh_feedback()
        sql_used.append(sql)
        orders = self.plan_orders()
        policy = self.decide_policy(feedback)
        self.write_state(orders, policy)
        observations = {
            "subcategories": n_sub, "catalog_updates": n_upd,
            "products_with_events": n_ev, "feedback_30m": feedback,
            "inventory_tracked": len([k for k in self.inventory
                                      if not k.startswith("__sub__")]),
            "open_orders_before": len(self.open_orders)}
        decisions = {"production_orders": len(orders),
                     "top_orders": orders[:10], "policy": policy}
        self.write_trace(sql_used, observations, decisions)
        LOGGER.info("control tower cycle: %d orders, policy=%s",
                    len(orders), policy)
        return {"observations": observations, "decisions": decisions}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RawTree Control Tower agent")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    cfg = Config.from_env()
    client = RawTreeClient(
        api_url=os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
        api_key=os.environ.get("RAWTREE_API_KEY", ""),
        team_id=os.environ.get("TEAM_ID", "team_03"))
    agent = ControlTowerAgent(client, cfg)
    stop = {"flag": False}

    def _stop(signum: int, frame: Any) -> None:
        LOGGER.info("signal %s received", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop["flag"]:
        try:
            agent.run_cycle()
        except Exception as exc:
            LOGGER.exception("control tower cycle failed: %s", exc)
        if args.once:
            break
        time.sleep(max(args.interval, 5.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

