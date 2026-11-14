"""One-shot integration smoke test against the RawTree practice stream.

    python scripts/integration_smoke.py            # answer 5 fresh requests, no push
    python scripts/integration_smoke.py --push     # also write submissions

It verifies timing and 10 unique in-stock IDs; it never blocks the service.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from starter import workers  # noqa: E402
from starter.config import Config, load_dotenv  # noqa: E402
from starter.recommender import bootstrap_snapshot, client  # noqa: E402
from starter.rawtree import parse_datetime  # noqa: E402
from starter.scoring import answer, validate_items  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    cfg = Config.from_env()
    rt = client()
    snap = bootstrap_snapshot(cfg)
    engine = workers.Engine(rt, cfg, snap)
    rows = rt.query(workers.SQL_REQUESTS.format(window=60, limit=300))
    now = rt.now()
    ok = 0
    batch = []
    for row in rows:
        if ok >= args.limit:
            break
        rid = str(row.get("request_id"))
        ts = parse_datetime(row.get("ts"))
        if not rid or rid == "__init__" or ts is None:
            continue
        age_ms = (now - ts).total_seconds() * 1000.0
        if age_ms > cfg.deadline_ms:
            continue
        req = dict(row)
        req["_now"] = now
        t0 = time.perf_counter()
        items = answer(req, engine.state.snapshot, engine.rng, cfg)
        latency = (time.perf_counter() - t0) * 1000.0
        valid = validate_items(items, engine.state.snapshot)
        print("%s age=%.0fms answer=%.2fms items=%d valid=%s"
              % (rid, age_ms, latency, len(items), valid))
        ok += int(valid)
        batch.append({"request_id": rid, "items": items,
                      "model_version": cfg.model_version,
                      "ts": row.get("ts"), "enqueued": time.monotonic(),
                      "user_id": str(row.get("user_id") or "?"),
                      "bought": ()})
    if args.push and batch:
        written = engine.push_batch(batch)
        print("pushed", written)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
