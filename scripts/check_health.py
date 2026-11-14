"""Operational health checks for the Bayes-TS deployment (read-only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter.config import CFG, load_dotenv  # noqa: E402
from starter.priors import latest_meta, read_latest_policy  # noqa: E402
from starter.rawtree import RawTreeClient  # noqa: E402


def main() -> int:
    load_dotenv()
    client = RawTreeClient(
        api_url=os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
        api_key=os.environ.get("RAWTREE_API_KEY", ""),
        team_id=os.environ.get("TEAM_ID", "team_03"))
    team = client.team_id
    print("== snapshot ==")
    meta = latest_meta(client, CFG)
    if meta:
        stats = meta.get("stats") or meta.get("stats_json") or {}
        if isinstance(stats, str):
            try:
                import json
                stats = json.loads(stats)
            except ValueError:
                stats = {}
        print("snapshot:", meta.get("snapshot_id"), "products:",
              meta.get("n_products"), "chunks:", meta.get("n_product_chunks"),
              "new:", stats.get("n_new_products", stats.get("new_products"))
              if isinstance(stats, dict) else "?")
        print("position_w:", stats.get("w") if isinstance(stats, dict) else "?")
    else:
        print("no snapshot yet")
    print("== feedback (30 min) ==")
    try:
        rows = client.query("""
            SELECT count() AS requests,
                   countIf(purchased != '') AS purchases,
                   countIf(served_by::String = 'fallback') AS fallbacks,
                   quantile(0.5)(toFloat64OrZero(toString(latency_ms))) AS p50,
                   quantile(0.99)(toFloat64OrZero(toString(latency_ms))) AS p99
            FROM %s.feedback
            WHERE request_id != '__init__'
              AND ts > now() - INTERVAL 30 MINUTE""" % team, heavy=True)
        print(rows[0] if rows else {})
    except Exception as exc:
        print("feedback unavailable:", exc)
    print("== submissions ==")
    try:
        rows = client.query("SELECT count() AS n FROM %s.submissions" % team,
                            heavy=True)
        print(rows[0] if rows else {})
    except Exception as exc:
        print("submissions unavailable:", exc)
    print("== control tower policy ==")
    print(read_latest_policy(client, CFG) or "none/unexpired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
