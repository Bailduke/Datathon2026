"""Thread-safe RawTree HTTP client.

Design notes (spec Sections 4.8, 5.9):

* ``requests.Session`` is *not* thread-safe, so every thread gets its own
  session through a ``threading.local`` (keep-alive is what keeps latency low).
* Heavy offline queries append ``SETTINGS max_execution_time = 25`` so the
  server cancels them before the 30 s gateway timeout.  Callers can then
  partition by hash/day and retry (see ``offline_build.py``).
* Errors are classified: 429 (respect ``Retry-After``), 413 (split payload),
  408/5xx (exponential backoff with jitter) and empty-table errors
  (``Unknown expression identifier``) which are *not* retried forever.
* The client maintains a server-clock offset (``server_now - local_now_utc``)
  so request ages are measured with the organiser clock, as required.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import requests

LOGGER = logging.getLogger("rawtree")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class RawTreeError(RuntimeError):
    """Base class for RawTree transport errors."""


class EmptyTableError(RawTreeError):
    """The table does not exist yet (pre-stream) or a column is unknown."""


class PayloadTooLargeError(RawTreeError):
    """413: the caller must split the batch."""


class QueryTimeoutError(RawTreeError):
    """408 / max_execution_time exceeded: partition the query."""


# --------------------------------------------------------------------------- #
# Datetime helpers (UTC everywhere; data timestamps are naive UTC strings)
# --------------------------------------------------------------------------- #
_DT_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T]"
    r"(?P<time>\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d{1,9}))?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse RawTree/ISO timestamps defensively into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    match = _DT_RE.match(text)
    if not match:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else \
            parsed.replace(tzinfo=timezone.utc)
    frac = (match.group("frac") or "0")[:6].ljust(6, "0")
    base = "%sT%s.%s" % (match.group("date"), match.group("time"), frac)
    tz = match.group("tz") or "+00:00"
    if tz == "Z":
        tz = "+00:00"
    elif ":" not in tz:
        tz = tz[:3] + ":" + tz[3:]
    try:
        parsed = datetime.fromisoformat(base + tz)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def iso_utc(value: Any = None, milliseconds: bool = True) -> str:
    """Format an aware/naive datetime as an ISO-8601 UTC string."""
    dt = parse_datetime(value) if value is not None else \
        datetime.now(timezone.utc)
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if milliseconds:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + \
            "%03dZ" % (dt.microsecond // 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and datetimes to JSON-safe Python."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (ValueError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except (ValueError, TypeError):
            pass
    return str(value)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class RawTreeClient:
    """Minimal, robust client for the two RawTree endpoints we need."""

    def __init__(self, api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 team_id: Optional[str] = None,
                 query_timeout: float = 10.0,
                 heavy_timeout: float = 35.0,
                 insert_timeout: float = 5.0,
                 max_retries: int = 4,
                 max_execution_s: int = 25) -> None:
        import os
        self.api_url = (api_url or os.environ.get(
            "RAWTREE_API_URL", "https://api.rawtree.com")).rstrip("/")
        self.api_key = api_key or os.environ.get("RAWTREE_API_KEY", "")
        self.team_id = team_id or os.environ.get("TEAM_ID", "")
        if not self.api_key:
            raise ValueError("RAWTREE_API_KEY is not set")
        if not self.team_id:
            raise ValueError("TEAM_ID is not set")
        self.query_timeout = query_timeout
        self.heavy_timeout = heavy_timeout
        self.insert_timeout = insert_timeout
        self.max_retries = max_retries
        self.max_execution_s = max_execution_s
        self._local = threading.local()
        self._clock_lock = threading.Lock()
        self._clock_offset = 0.0
        self._clock_checked = 0.0
        self.stats: Dict[str, int] = {
            "queries": 0, "query_errors": 0, "inserts": 0,
            "insert_errors": 0, "retries": 0,
        }
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------ HTTP
    def session(self) -> requests.Session:
        """One ``requests.Session`` per thread (keep-alive, thread-safe)."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "bayes-ts/3.0",
            })
            self._local.session = session
        return session

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.05), 10.0)
        base = min(0.5 * (2 ** attempt), 5.0)
        return base * (0.75 + 0.5 * random.random())

    @staticmethod
    def _is_empty_table(response: requests.Response) -> bool:
        if response.status_code not in (400, 404, 422):
            return False
        text = response.text.lower()
        return ("unknown expression identifier" in text or
                "unknown table" in text or "doesn't exist" in text or
                "does not exist" in text or "no such table" in text)

    def _post(self, url: str, payload: Any, timeout: float,
              allow_empty_table: bool,
              max_retries: Optional[int] = None) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        attempts = self.max_retries if max_retries is None else max_retries
        for attempt in range(attempts + 1):
            self._bump("queries")
            try:
                response = self.session().post(url, json=payload,
                                               timeout=timeout)
            except requests.RequestException as exc:
                last_error = RawTreeError(str(exc))
                if attempt >= attempts:
                    break
                self._bump("retries")
                time.sleep(self._backoff(attempt))
                continue

            status = response.status_code
            if status < 400:
                try:
                    body = response.json()
                except ValueError as exc:
                    last_error = RawTreeError("invalid JSON: %s" % exc)
                    break
                if isinstance(body, dict):
                    return body
                return {"data": body}

            if status == 429:
                retry_after = response.headers.get("retry-after")
                try:
                    retry_after_f = float(retry_after) if retry_after else None
                except ValueError:
                    retry_after_f = None
                last_error = RawTreeError("429 rate limit")
                if attempt < attempts:
                    self._bump("retries")
                    time.sleep(self._backoff(attempt, retry_after_f))
                    continue
            elif status == 413:
                raise PayloadTooLargeError(response.text[:500])
            elif status in (408, 504):
                last_error = QueryTimeoutError(response.text[:500])
                if attempt < attempts:
                    self._bump("retries")
                    time.sleep(self._backoff(attempt))
                    continue
            elif status >= 500:
                last_error = RawTreeError("HTTP %d: %s" %
                                          (status, response.text[:300]))
                if attempt < attempts:
                    self._bump("retries")
                    time.sleep(self._backoff(attempt))
                    continue
            else:
                if allow_empty_table and self._is_empty_table(response):
                    raise EmptyTableError(response.text[:500])
                last_error = RawTreeError("HTTP %d: %s" %
                                          (status, response.text[:500]))
                break
        self._bump("query_errors")
        raise last_error or RawTreeError("RawTree request failed")

    # ----------------------------------------------------------------- query
    def query(self, sql: str, heavy: bool = False,
              timeout: Optional[float] = None,
              retries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run a SELECT and return the ``data`` list of row objects."""
        text = sql.strip().rstrip(";")
        if heavy and "settings" not in text.lower():
            text += "\nSETTINGS max_execution_time = %d" % self.max_execution_s
        body = self._post("%s/v1/query" % self.api_url, {"sql": text},
                          timeout or (self.heavy_timeout if heavy
                                      else self.query_timeout),
                          allow_empty_table=True, max_retries=retries)
        data = body.get("data", [])
        return data if isinstance(data, list) else []

    # --------------------------------------------------------------- clock
    def server_time(self, force: bool = False) -> datetime:
        """Server clock (UTC), cached for 60 s, with graceful fallback."""
        now_mono = time.monotonic()
        with self._clock_lock:
            if (not force and self._clock_checked and
                    now_mono - self._clock_checked < 60.0):
                return datetime.now(timezone.utc) + timedelta(
                    seconds=self._clock_offset)
        local = datetime.now(timezone.utc)
        try:
            rows = self.query("SELECT now() AS server_now", retries=1)
            server = parse_datetime(rows[0].get("server_now")) if rows else None
        except RawTreeError:
            server = None
        if server is not None:
            offset = (server - local).total_seconds()
            with self._clock_lock:
                self._clock_offset = offset
                self._clock_checked = now_mono
        return datetime.now(timezone.utc) + timedelta(
            seconds=self._clock_offset)

    @property
    def clock_offset(self) -> float:
        with self._clock_lock:
            return self._clock_offset

    def now(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(
            seconds=self.clock_offset)

    # ---------------------------------------------------------------- tables
    def create_table(self, name: str, database: Optional[str] = None,
                     sorting_key: Optional[str] = None) -> Dict[str, Any]:
        """POST /v1/tables (requires admin).  Raises on 403/400."""
        payload: Dict[str, Any] = {"name": name}
        if sorting_key:
            payload["sorting_key"] = sorting_key
        response = self.session().post(
            "%s/v1/tables" % self.api_url,
            params={"database": database or self.team_id},
            json=payload, timeout=self.insert_timeout)
        if response.status_code >= 400:
            raise RawTreeError("HTTP %d: %s" % (
                response.status_code, response.text[:300]))
        try:
            return response.json()
        except ValueError:
            return {}

    # ---------------------------------------------------------------- insert
    def insert_rows(self, table: str, rows: Sequence[Mapping[str, Any]],
                    timeout: Optional[float] = None,
                    database: Optional[str] = None,
                    max_rows: Optional[int] = None,
                    max_bytes: Optional[int] = None) -> int:
        """Insert JSON rows in size-bounded batches; returns written rows.

        Handles 413 by recursively halving the batch and 429/5xx with backoff.
        """
        if not rows:
            return 0
        database = database or self.team_id
        max_rows = max_rows or 500
        max_bytes = max_bytes or (7 * 1024 * 1024)
        url = "%s/v1/tables/%s" % (self.api_url, table)
        params = {"database": database}
        written = 0

        def send(batch: List[Dict[str, Any]]) -> int:
            if not batch:
                return 0
            body = json.dumps([to_jsonable(row) for row in batch],
                              separators=(",", ":"), allow_nan=False)
            if len(body.encode("utf-8")) > max_bytes and len(batch) > 1:
                mid = len(batch) // 2
                return send(batch[:mid]) + send(batch[mid:])
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                self._bump("inserts")
                try:
                    response = self.session().post(
                        url, params=params, data=body,
                        timeout=timeout or self.insert_timeout)
                except requests.RequestException as exc:
                    last_error = RawTreeError(str(exc))
                    if attempt < self.max_retries:
                        self._bump("retries")
                        time.sleep(self._backoff(attempt))
                        continue
                    break
                if response.status_code < 400:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    raw = payload.get("written_rows", payload.get("rows", 0))
                    try:
                        return int(raw)
                    except (TypeError, ValueError):
                        return len(batch)
                if response.status_code == 413:
                    if len(batch) == 1:
                        raise PayloadTooLargeError(response.text[:300])
                    mid = len(batch) // 2
                    return send(batch[:mid]) + send(batch[mid:])
                if response.status_code == 429:
                    ra = response.headers.get("retry-after")
                    try:
                        raf = float(ra) if ra else None
                    except ValueError:
                        raf = None
                    last_error = RawTreeError("429 rate limit")
                    if attempt < self.max_retries:
                        self._bump("retries")
                        time.sleep(self._backoff(attempt, raf))
                        continue
                    break
                last_error = RawTreeError("HTTP %d: %s" %
                                          (response.status_code,
                                           response.text[:300]))
                if response.status_code >= 500 and attempt < self.max_retries:
                    self._bump("retries")
                    time.sleep(self._backoff(attempt))
                    continue
                break
            self._bump("insert_errors")
            raise last_error or RawTreeError("insert failed")

        batch: List[Dict[str, Any]] = []
        size = 0
        for row in rows:
            clean = to_jsonable(row)
            encoded = len(json.dumps(clean, separators=(",", ":")).encode())
            if batch and (len(batch) >= max_rows or size + encoded > max_bytes):
                written += send(batch)
                batch, size = [], 0
            batch.append(clean)
            size += encoded
        written += send(batch)
        return written

