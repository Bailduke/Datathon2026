"""
Quadre de Comandament Econòmic i de Gestió de Magatzem · Datathon FME 2026 (RawTree)
====================================================================================

Executa:  streamlit run dashboard.py

Totes les xifres surten de consultes SQL en viu sobre RawTree (base `shop` i la base
de l'equip). No hi ha cap dada inventada ni cap marge suposat: la rendibilitat de les rebaixes
s'expressa com el cost màxim per unitat que les faria rendibles, calculat només amb preus i vendes reals.

Regles RawTree aplicades a totes les consultes:
  * totes les columnes són Dynamic → càsting explícit (::String, ::Bool, ::Float64, ::DateTime64(3))
  * mai SHOW TABLES
  * {TEAM}.feedback sempre amb WHERE request_id::String != '__init__'
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# 0. ENTORN I CONNEXIÓ
# ──────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_env(path: str) -> None:
    """Carrega el .env sense dependre de python-dotenv (no sobreescriu variables ja definides)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_env(os.path.join(ROOT, ".env"))
_load_env(os.path.join(os.path.dirname(ROOT), ".env"))
_load_env(os.path.join(os.path.dirname(ROOT), "datathon-fme-2026", ".env"))
# El .env del kit fa servir TEAM_ID; l'starter llegeix RAWTREE_TEAM / TEAM.
if os.getenv("TEAM_ID"):
    os.environ.setdefault("RAWTREE_TEAM", os.environ["TEAM_ID"])
TEAM = os.getenv("TEAM_ID") or os.getenv("RAWTREE_TEAM") or "team_03"

# ── Recommenders de l'equip (carpeta tengo-aurilla-v0, al costat d'aquest projecte) ──
AURILLA_DIR = os.path.join(os.path.dirname(ROOT), "tengo-aurilla-v0")
_VERSION_FILE = re.compile(r"recommender-v(\d+)\.py$")


def recommender_files() -> List[Tuple[int, str]]:
    """[(número de versió, ruta)] dels recommenders de tengo-aurilla-v0, del més antic al més nou."""
    out = []
    for path in glob.glob(os.path.join(AURILLA_DIR, "recommender-v*.py")):
        m = _VERSION_FILE.search(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def _load_latest_recommender():
    """Importa el recommender més nou (només defineix funcions i configuració: no arrenca cap bucle)."""
    files = recommender_files()
    if not files:
        return None, ""
    version, path = files[-1]
    name = f"aurilla_recommender_v{version}"
    if name in sys.modules:
        return sys.modules[name], os.path.basename(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module, os.path.basename(path)


API_URL = os.getenv("RAWTREE_API_URL", "https://api.rawtree.com/v1/query")
_api_http = requests.Session()


def _api_query(sql: str, timeout: float = 90.0) -> List[Dict[str, Any]]:
    """Adaptador a l'API oficial (POST /v1/query, JSON {"sql"}, Bearer). Mateix SQL, mateixes dades."""
    key = os.getenv("RAWTREE_API_KEY") or os.getenv("RAWTREE_TOKEN") or ""
    resp = _api_http.post(
        API_URL,
        json={"sql": sql.strip().rstrip(";")},
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"RawTree {resp.status_code}: {resp.text[:400]}")
    return resp.json().get("data", [])


@st.cache_resource(show_spinner=False)
def _backend() -> Tuple[str, Callable[..., List[Dict[str, Any]]]]:
    """
    Tria com es connecta a RawTree, per ordre:
      1. el client RawTree del recommender més nou de tengo-aurilla-v0 (el que fa servir l'equip en viu);
      2. `starter.recommender.query`;
      3. l'adaptador propi de l'API oficial.
    """
    try:
        rec, fname = _load_latest_recommender()
        if rec is not None:
            rec.load_env()
            client = rec.RawTree(os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
                                 os.environ["RAWTREE_API_KEY"], os.environ.get("TEAM_ID", TEAM))
            client.query("SELECT 1 AS ok", timeout=15.0)
            return f"tengo-aurilla-v0/{fname}", client.query
    except Exception:
        pass
    try:
        from starter.recommender import query as starter_query  # noqa: WPS433
        starter_query("SELECT 1 AS ok", timeout=5.0)
        return "starter.recommender.query", starter_query
    except Exception:
        pass
    _api_query("SELECT 1 AS ok", timeout=15.0)  # si també falla, l'excepció arriba a la UI
    return "api.rawtree.com/v1/query", _api_query


def run_sql(sql: str) -> pd.DataFrame:
    sql = sql.replace("{TEAM}", TEAM).strip().rstrip(";")
    _, fn = _backend()
    return pd.DataFrame(fn(sql, timeout=90.0))


# ──────────────────────────────────────────────────────────────────────────────
# 1. UTILITATS
# ──────────────────────────────────────────────────────────────────────────────
def _lit(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _in(expr: str, values: Sequence[str]) -> str:
    """Filtre IN; llista buida = sense filtre."""
    if not values:
        return "1"
    return f"{expr} IN ({', '.join(_lit(v) for v in values)})"


def _num(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _ensure(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Garanteix que existeixin les columnes encara que la consulta torni buida."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(cols))
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return float(a) / float(b) if b and np.isfinite(b) and float(b) != 0.0 else default
    except Exception:
        return default


def eur(x: float, decimals: int = 0) -> str:
    s = f"{x:,.{decimals}f}".replace(",", "·").replace(".", ",").replace("·", ".")
    return f"{s} €"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f} %".replace(".", ",")


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONSULTES (totes en caché 30 s)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def q_dimensions() -> Dict[str, List[str]]:
    cats = run_sql("SELECT DISTINCT category::String AS v FROM shop.catalog ORDER BY v")
    regs = run_sql("SELECT DISTINCT region::String AS v FROM shop.history_events WHERE region::String != '' ORDER BY v")
    devs = run_sql("SELECT DISTINCT device::String AS v FROM shop.history_events WHERE device::String != '' ORDER BY v")
    return {
        "category": cats["v"].tolist() if not cats.empty else [],
        "region": regs["v"].tolist() if not regs.empty else [],
        "device": devs["v"].tolist() if not devs.empty else [],
    }


@st.cache_data(ttl=30, show_spinner=False)
def q_windows() -> pd.DataFrame:
    """Finestres temporals: dia en viu (shop.events) i històric (shop.history_events)."""
    df = run_sql(
        """
        SELECT
          (SELECT min(ts::DateTime64(3)) FROM shop.events)         AS live_start,
          now64(3)                                                  AS now,
          (SELECT min(ts::DateTime64(3)) FROM shop.history_events) AS hist_start,
          (SELECT max(ts::DateTime64(3)) FROM shop.history_events) AS hist_end
        """
    )
    for c in ("live_start", "now", "hist_start", "hist_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def q_catalog() -> pd.DataFrame:
    """Catàleg vigent = shop.catalog + últim canvi de shop.catalog_updates per producte."""
    df = run_sql(
        """
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(brand, v)       AS brand,
               argMax(price_eur, v)   AS price_eur,
               argMax(is_active, v)   AS is_active,
               min(created_at)        AS created_at
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, brand::String AS brand,
                   price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
                   created_at::DateTime64(3) AS created_at, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String,
                   price_eur::Float64, is_active::Bool, created_at::DateTime64(3),
                   toString(ts::DateTime64(3))
            FROM shop.catalog_updates
        )
        GROUP BY product_id
        """
    )
    df = _ensure(df, ["product_id", "category", "subcategory", "brand", "price_eur", "is_active", "created_at"])
    df = _num(df, ["price_eur"])
    df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1"])
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def q_stock_changes() -> pd.DataFrame:
    df = run_sql(
        """
        SELECT product_id::String AS product_id, change::String AS change, ts::DateTime64(3) AS ts
        FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock')
        ORDER BY product_id, ts
        """
    )
    df = _ensure(df, ["product_id", "change", "ts"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def _event_filters(alias: str, regions: Tuple[str, ...], devices: Tuple[str, ...]) -> str:
    p = f"{alias}." if alias else ""
    return f"{_in(p + 'region::String', regions)} AND {_in(p + 'device::String', devices)}"


def _ts_filter(alias: str, start: str, end: str) -> str:
    """Interval [start, end) sobre ts (UTC). Cadenes buides = sense límit."""
    p = f"{alias}." if alias else ""
    conds = []
    if start:
        conds.append(f"{p}ts::DateTime64(3) >= toDateTime64({_lit(start)}, 3)")
    if end:
        conds.append(f"{p}ts::DateTime64(3) < toDateTime64({_lit(end)}, 3)")
    return " AND ".join(conds) or "1"


@st.cache_data(ttl=30, show_spinner=False)
def q_live_products(regions: Tuple[str, ...], devices: Tuple[str, ...], start: str = "", end: str = "") -> pd.DataFrame:
    df = run_sql(
        f"""
        SELECT product_id::String AS product_id,
               countIf(event_type::String = 'view')        AS views,
               countIf(event_type::String = 'add_to_cart') AS carts,
               countIf(event_type::String = 'purchase')    AS purchases,
               sumIf(price_eur::Float64, event_type::String = 'purchase') AS revenue
        FROM shop.events
        WHERE {_event_filters('', regions, devices)} AND {_ts_filter('', start, end)}
        GROUP BY product_id
        """
    )
    df = _ensure(df, ["product_id", "views", "carts", "purchases", "revenue"])
    return _num(df, ["views", "carts", "purchases", "revenue"])


@st.cache_data(ttl=30, show_spinner=False)
def q_hist_products(regions: Tuple[str, ...], devices: Tuple[str, ...], days: int) -> pd.DataFrame:
    df = run_sql(
        f"""
        SELECT product_id::String AS product_id,
               countIf(event_type::String = 'view')        AS h_views,
               countIf(event_type::String = 'add_to_cart') AS h_carts,
               countIf(event_type::String = 'purchase')    AS h_purchases,
               sumIf(price_eur::Float64, event_type::String = 'purchase') AS h_revenue,
               countIf(event_type::String = 'view'        AND ts::DateTime64(3) > now64(3) - INTERVAL {int(days)} DAY) AS r_views,
               countIf(event_type::String = 'add_to_cart' AND ts::DateTime64(3) > now64(3) - INTERVAL {int(days)} DAY) AS r_carts,
               countIf(event_type::String = 'purchase'    AND ts::DateTime64(3) > now64(3) - INTERVAL {int(days)} DAY) AS r_purchases
        FROM shop.history_events
        WHERE {_event_filters('', regions, devices)}
        GROUP BY product_id
        """
    )
    cols = ["h_views", "h_carts", "h_purchases", "h_revenue", "r_views", "r_carts", "r_purchases"]
    df = _ensure(df, ["product_id", *cols])
    return _num(df, cols)


def _feedback_from(regions: Tuple[str, ...], devices: Tuple[str, ...]) -> Tuple[str, str]:
    """FROM + WHERE de feedback; regió i dispositiu venen del context de shop.requests."""
    base_where = "f.request_id::String != '__init__'"
    if not regions and not devices:
        return f"{{TEAM}}.feedback AS f", base_where
    frm = (
        "{TEAM}.feedback AS f "
        "INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String"
    )
    where = (
        f"{base_where} AND {_in('r.context.region::String', regions)} "
        f"AND {_in('r.context.device::String', devices)}"
    )
    return frm, where


@st.cache_data(ttl=30, show_spinner=False)
def q_feedback(regions: Tuple[str, ...], devices: Tuple[str, ...], start: str = "", end: str = "") -> pd.DataFrame:
    frm, where = _feedback_from(regions, devices)
    where += f" AND {_ts_filter('f', start, end)}"
    df = run_sql(
        f"""
        SELECT f.served_by::String       AS served_by,
               f.fallback_reason::String AS fallback_reason,
               f.purchased::String       AS purchased,
               count()                                 AS impressions,
               countIf(f.clicked::String != '')        AS clicks,
               countIf(f.purchased::String != '')      AS purchases,
               sum(f.revenue_eur::Float64)             AS revenue
        FROM {frm}
        WHERE {where}
        GROUP BY served_by, fallback_reason, purchased
        """
    )
    df = _ensure(df, ["served_by", "fallback_reason", "purchased", "impressions", "clicks", "purchases", "revenue"])
    return _num(df, ["impressions", "clicks", "purchases", "revenue"])


@st.cache_data(ttl=30, show_spinner=False)
def q_carousel_clicks(regions: Tuple[str, ...], devices: Tuple[str, ...]) -> pd.DataFrame:
    frm, where = _feedback_from(regions, devices)
    df = run_sql(
        f"""
        SELECT f.clicked::String AS product_id, count() AS carousel_clicks
        FROM {frm}
        WHERE {where} AND f.clicked::String != ''
        GROUP BY product_id
        """
    )
    df = _ensure(df, ["product_id", "carousel_clicks"])
    return _num(df, ["carousel_clicks"])


_CATEGORY_MAP = """
    SELECT product_id AS pid, any(category) AS category
    FROM (
        SELECT product_id::String AS product_id, category::String AS category FROM shop.catalog
        UNION ALL
        SELECT product_id::String, category::String FROM shop.catalog_updates
        WHERE change::String = 'new_product'
    )
    GROUP BY pid
"""


@st.cache_data(ttl=30, show_spinner=False)
def q_category_region(source: str, regions: Tuple[str, ...], devices: Tuple[str, ...], start: str = "",
                      end: str = "") -> pd.DataFrame:
    table = "shop.events" if source == "live" else "shop.history_events"
    tsf = _ts_filter("e", start, end) if source == "live" else "1"
    df = run_sql(
        f"""
        SELECT c.category AS category, e.region::String AS region,
               countIf(e.event_type::String = 'view')     AS views,
               countIf(e.event_type::String = 'purchase') AS purchases,
               sumIf(e.price_eur::Float64, e.event_type::String = 'purchase') AS revenue
        FROM {table} AS e
        INNER JOIN ({_CATEGORY_MAP}) AS c ON c.pid = e.product_id::String
        WHERE {_event_filters('e', regions, devices)} AND {tsf}
        GROUP BY category, region
        """
    )
    df = _ensure(df, ["category", "region", "views", "purchases", "revenue"])
    return _num(df, ["views", "purchases", "revenue"])


_FIRST_PRICE_CHANGE = """
    SELECT product_id::String AS pid,
           toString(min(ts::DateTime64(3))) AS cohort,
           argMin(price_eur::Float64, ts::DateTime64(3)) AS new_price
    FROM shop.catalog_updates
    WHERE change::String = 'price_change'
    GROUP BY pid
"""


@st.cache_data(ttl=30, show_spinner=False)
def q_price_events() -> pd.DataFrame:
    """
    Campanyes de preu per categoria: cada tanda de price_change amb el preu nou respecte a l'original.
    ratio < 1 → comença una rebaixa; ratio ≈ 1 → el preu torna a l'original (fi de la rebaixa).
    """
    df = run_sql(
        """
        SELECT toString(u.ts::DateTime64(3)) AS ts, u.category::String AS category, count() AS n,
               avg(u.price_eur::Float64 / c0.price_eur::Float64) AS ratio,
               avg(c0.price_eur::Float64) AS old_price
        FROM shop.catalog_updates AS u
        INNER JOIN shop.catalog AS c0 ON c0.product_id::String = u.product_id::String
        WHERE u.change::String = 'price_change' AND c0.price_eur::Float64 > 0
        GROUP BY ts, category
        HAVING n >= 10
        ORDER BY ts
        """
    )
    df = _ensure(df, ["ts", "category", "n", "ratio", "old_price"])
    df = _num(df, ["n", "ratio", "old_price"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def q_price_change_sales(regions: Tuple[str, ...], devices: Tuple[str, ...]) -> pd.DataFrame:
    """Vendes per franja de 30 min: cohorts rebaixats (per hora del canvi) i grup de control (mai rebaixat)."""
    df = run_sql(
        f"""
        SELECT cat.category AS category,
               if(pc.pid = '', 'control', pc.cohort) AS cohort,
               toString(toStartOfInterval(e.ts::DateTime64(3), INTERVAL 30 MINUTE)) AS slot,
               countIf(e.event_type::String = 'purchase') AS units,
               sumIf(e.price_eur::Float64, e.event_type::String = 'purchase') AS revenue
        FROM shop.events AS e
        INNER JOIN (SELECT product_id::String AS pid, category::String AS category FROM shop.catalog) AS cat
            ON cat.pid = e.product_id::String
        LEFT JOIN ({_FIRST_PRICE_CHANGE}) AS pc ON pc.pid = e.product_id::String
        WHERE {_event_filters('e', regions, devices)}
        GROUP BY category, cohort, slot
        """
    )
    df = _ensure(df, ["category", "cohort", "slot", "units", "revenue"])
    df = _num(df, ["units", "revenue"])
    df["slot"] = pd.to_datetime(df["slot"], errors="coerce")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def q_store_slots(regions: Tuple[str, ...], devices: Tuple[str, ...]) -> pd.DataFrame:
    """Compres i facturació orgàniques per franja de 30 min i categoria, separant els productes nous d'avui."""
    df = run_sql(
        f"""
        SELECT toString(toStartOfInterval(e.ts::DateTime64(3), INTERVAL 30 MINUTE)) AS slot,
               c.category AS category,
               countIf(e.event_type::String = 'purchase')                   AS purchases,
               countIf(e.event_type::String = 'purchase' AND np.pid != '')  AS new_purchases,
               sumIf(e.price_eur::Float64, e.event_type::String = 'purchase') AS revenue
        FROM shop.events AS e
        LEFT JOIN ({_CATEGORY_MAP}) AS c ON c.pid = e.product_id::String
        LEFT JOIN (SELECT DISTINCT product_id::String AS pid FROM shop.catalog_updates
                   WHERE change::String = 'new_product') AS np ON np.pid = e.product_id::String
        WHERE {_event_filters('e', regions, devices)}
        GROUP BY slot, category
        """
    )
    df = _ensure(df, ["slot", "category", "purchases", "new_purchases", "revenue"])
    df = _num(df, ["purchases", "new_purchases", "revenue"])
    df["slot"] = pd.to_datetime(df["slot"], errors="coerce")
    return df.sort_values("slot")


@st.cache_data(ttl=30, show_spinner=False)
def q_feedback_hourly(regions: Tuple[str, ...], devices: Tuple[str, ...], start: str = "", end: str = "") -> pd.DataFrame:
    """Peticions del carrusel per hora i per qui les ha servit (per veure les aturades del recommender)."""
    frm, where = _feedback_from(regions, devices)
    where += f" AND {_ts_filter('f', start, end)}"
    df = run_sql(
        f"""
        SELECT toString(toStartOfHour(f.ts::DateTime64(3))) AS hour,
               f.served_by::String AS served_by, f.fallback_reason::String AS fallback_reason,
               count() AS impressions, sum(f.revenue_eur::Float64) AS revenue
        FROM {frm}
        WHERE {where}
        GROUP BY hour, served_by, fallback_reason
        """
    )
    df = _ensure(df, ["hour", "served_by", "fallback_reason", "impressions", "revenue"])
    df = _num(df, ["impressions", "revenue"])
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    return df.sort_values("hour")


def prefetch(jobs: Dict[str, Tuple[Callable[..., Any], tuple]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Executa les consultes en paral·lel (cadascuna continua passant per la seva caché)."""
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

        ctx = get_script_run_ctx()
    except Exception:  # versions antigues de Streamlit
        add_script_run_ctx, ctx = None, None

    def call(fn: Callable[..., Any], args: tuple) -> Any:
        if add_script_run_ctx and ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        return fn(*args)

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {k: pool.submit(call, fn, args) for k, (fn, args) in jobs.items()}
        for key, fut in futures.items():
            try:
                results[key] = fut.result()
            except Exception as exc:  # una taula fallida no tomba la resta del quadre
                errors[key] = str(exc)[:300]
                results[key] = pd.DataFrame()
    return results, errors


# ──────────────────────────────────────────────────────────────────────────────
# 3. LÒGICA DE NEGOCI
# ──────────────────────────────────────────────────────────────────────────────
def stockout_metrics(changes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Per producte i dins del període [start, end): nº de ruptures, hores sense estoc i estat al final del període.
    Un producte que ja estava sense estoc en començar el període compta com una ruptura i només suma les hores
    que cauen dins del període.
    """
    cols = ["product_id", "oos_count", "oos_hours", "currently_oos", "last_oos"]
    if changes.empty or end <= start:
        return pd.DataFrame(columns=cols)

    def overlap(a: pd.Timestamp, b: pd.Timestamp) -> float:
        return max((min(b, end) - max(a, start)).total_seconds(), 0) / 3600.0

    rows = []
    for pid, grp in changes.sort_values(["product_id", "ts"]).groupby("product_id", sort=False):
        oos_since, count, hours, last_oos, carried, checked = None, 0, 0.0, pd.NaT, False, False
        for change, ts in zip(grp["change"], grp["ts"]):
            if pd.isna(ts):
                continue
            if ts >= end:
                break
            if not checked and ts >= start:
                checked, carried = True, oos_since is not None
            if change == "out_of_stock":
                if ts >= start:
                    count += 1
                    last_oos = ts
                if oos_since is None:
                    oos_since = ts
            elif change == "back_in_stock" and oos_since is not None:
                hours += overlap(oos_since, ts)
                oos_since = None
        if not checked:
            carried = oos_since is not None
        if oos_since is not None:
            hours += overlap(oos_since, end)
        count += int(carried)
        if count or hours:
            rows.append((pid, count, hours, oos_since is not None, last_oos))
    return pd.DataFrame(rows, columns=cols)


ACTION_UP, ACTION_KEEP, ACTION_DOWN = "Augmentar lot de compra", "Mantenir", "Reduir compra"


def build_product_table(
    cat: pd.DataFrame,
    live: pd.DataFrame,
    hist: pd.DataFrame,
    clicks: pd.DataFrame,
    stock: pd.DataFrame,
    live_hours: float,
    hist_hours: float,
) -> pd.DataFrame:
    prod = cat.copy()
    for other in (live, hist, clicks, stock):
        if not other.empty:
            prod = prod.merge(other, on="product_id", how="left")
    for c in ["views", "carts", "purchases", "revenue", "h_views", "h_carts", "h_purchases", "h_revenue",
              "r_views", "r_carts", "r_purchases", "carousel_clicks", "oos_count", "oos_hours"]:
        prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(0.0) if c in prod.columns else 0.0
    prod["currently_oos"] = prod["currently_oos"].fillna(False).astype(bool) if "currently_oos" in prod else False
    if "last_oos" not in prod:
        prod["last_oos"] = pd.NaT

    # Intensitat del dia en viu respecte a l'històric (el simulador d'avui és molt més dens).
    live_rate_total = safe_div(prod["purchases"].sum(), live_hours)
    hist_rate_total = safe_div(prod["h_purchases"].sum(), hist_hours)
    intensity = safe_div(live_rate_total, hist_rate_total, default=1.0)

    in_stock_h = (live_hours - prod["oos_hours"]).clip(lower=0.0)
    live_rate = prod["purchases"] / in_stock_h.where(in_stock_h > 0, np.nan)
    live_cart_rate = prod["carts"] / in_stock_h.where(in_stock_h > 0, np.nan)
    hist_rate = (prod["h_purchases"] / hist_hours * intensity) if hist_hours > 0 else 0.0
    hist_cart_rate = (prod["h_carts"] / hist_hours * intensity) if hist_hours > 0 else 0.0
    use_live = in_stock_h >= 1.0  # amb menys d'1 h amb estoc avui, el ritme en viu no és fiable

    prod["sales_rate_h"] = np.where(use_live, live_rate, hist_rate)
    prod["cart_rate_h"] = np.where(use_live, live_cart_rate, hist_cart_rate)
    prod["sales_rate_h"] = pd.to_numeric(prod["sales_rate_h"], errors="coerce").fillna(0.0)
    prod["cart_rate_h"] = pd.to_numeric(prod["cart_rate_h"], errors="coerce").fillna(0.0)
    prod["rate_source"] = np.where(use_live, "període", "històric")
    prod["potential_rev_h"] = prod["sales_rate_h"] * prod["price_eur"]
    prod["lost_units"] = prod["sales_rate_h"] * prod["oos_hours"]
    prod["lost_revenue"] = prod["lost_units"] * prod["price_eur"]
    prod["demand_pct"] = prod["sales_rate_h"].rank(pct=True, method="average").fillna(0.0)

    up = (prod["oos_count"] >= 1) & ((prod["demand_pct"] >= 0.5) | (prod["oos_count"] >= 2))
    down = (prod["oos_count"] == 0) & (prod["demand_pct"] <= 0.2)
    prod["action"] = np.select([up, down], [ACTION_UP, ACTION_DOWN], default=ACTION_KEEP)

    lost_q75 = prod.loc[prod["lost_revenue"] > 0, "lost_revenue"].quantile(0.75) if (prod["lost_revenue"] > 0).any() else np.inf
    high = up & (prod["currently_oos"] | (prod["lost_revenue"] >= lost_q75))
    mid = (up & ~high) | ((prod["oos_count"] >= 1) & ~up)
    prod["urgency"] = np.select([high, mid], ["🔴 Alta", "🟡 Mitjana"], default="🟢 Baixa")
    prod["urgency_rank"] = np.select([high, mid], [0, 1], default=2)
    return prod


HANGOVER_H = 4.0  # hores després del retorn al preu original que es vigilen (dissolució de l'efecte)


def price_windows(events: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Finestres reals de rebaixa per categoria: des de la baixada fins que el preu torna a l'original."""
    cols = ["category", "start", "end", "open", "discount", "products", "old_price"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cat, g in events.sort_values("ts").groupby("category"):
        current = None
        for e in g.itertuples(index=False):
            if e.ratio < 0.99 and current is None:  # comença una rebaixa
                current = dict(category=cat, start=e.ts, discount=1 - e.ratio, products=e.n, old_price=e.old_price)
            elif e.ratio >= 0.99 and current is not None:  # torna al preu original
                rows.append({**current, "end": e.ts, "open": False})
                current = None
        if current is not None:
            rows.append({**current, "end": now, "open": True})
    return pd.DataFrame(rows, columns=cols)


def price_change_analysis(
    events: pd.DataFrame,
    sales: pd.DataFrame,
    categories: Tuple[str, ...],
    live_start: pd.Timestamp,
    now: pd.Timestamp,
    period: Tuple[pd.Timestamp, pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Per a cada categoria rebaixada:
      * cada franja es classifica com a «abans», «durant» (finestra real de rebaixa), «després» (les
        HANGOVER_H hores següents al retorn al preu original, només com a comprovació) o «normal»;
      * pes = vendes dels productes rebaixats ÷ vendes del control a les franges a preu complet (abans + normal);
      * esperat(franja) = pes × vendes del control a la franja (unitats i facturació a preu original);
      * la rendibilitat es mesura NOMÉS a la finestra de rebaixa;
      * amb `period`, es mesuren les rebaixes que COMENCEN dins del període; la referència es calcula amb tot el
        dia i n'exclou totes les rebaixes (també les de fora del període, marcades com a «fora»).
    El control són els productes que no s'han rebaixat mai: el flux de compradors és constant, però el
    catàleg inicial perd quota durant el dia per l'entrada de productes nous, i el control ho corregeix.
    """
    empty = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    win = price_windows(events, now)
    if win.empty or sales.empty:
        return empty
    if categories:
        win = win[win["category"].isin(categories)]
    win_all = win
    if period is not None:
        win = win[(win["start"] >= period[0]) & (win["start"] < period[1])]
    if win.empty:
        return empty

    slot = timedelta(minutes=30)
    sales = sales[(sales["slot"] >= live_start.floor("30min")) & (sales["slot"] + slot <= now)]  # franges completes
    ctrl = sales[sales["cohort"] == "control"].groupby("slot")["units"].sum().sort_index()
    disc = sales[sales["cohort"] != "control"].groupby(["category", "slot"])[["units", "revenue"]].sum()
    if ctrl.empty:
        return empty

    rows, skipped = [], []
    hang = timedelta(hours=HANGOVER_H)
    in_period = set(zip(win["category"], win["start"]))
    for cat in win["category"].unique():
        wg = win_all[win_all["category"] == cat]
        first = wg["start"].min()
        try:
            obs = disc.loc[cat].reindex(ctrl.index).fillna(0.0)
        except KeyError:
            skipped.append(cat)
            continue
        pre = ctrl.index < first
        phase = np.where(pre, "abans", "normal").astype(object)
        for w in wg.itertuples(index=False):
            inside = (cat, w.start) in in_period
            after = (ctrl.index >= w.end) & (ctrl.index < w.end + hang) & (phase == "normal") & (not w.open)
            phase[after] = "després" if inside else "fora"
        for w in wg.itertuples(index=False):
            inside = (cat, w.start) in in_period
            phase[(ctrl.index >= w.start) & (ctrl.index < w.end)] = "durant" if inside else "fora"
        # Pes de partida i preu mitjà a preu complet amb TOTES les franges sense rebaixa ni ressaca (abans i
        # normal). Amb només les franges d'abans (1,5 h a Pets) el pes surt un 10 % massa alt i tot el dia
        # posterior sembla «per sota», cosa que es confondria amb una ressaca inexistent.
        full = np.isin(phase, ["abans", "normal"])
        u_full, c_full = obs["units"][full].sum(), ctrl[full].sum()
        if u_full <= 0 or c_full <= 0:
            skipped.append(cat)
            continue
        weight = u_full / c_full
        p_full = safe_div(obs["revenue"][full].sum(), u_full)
        exp_u = weight * ctrl
        rows.append(pd.DataFrame({
            "category": cat, "slot": ctrl.index, "phase": phase,
            "obs_units": obs["units"].values, "exp_units": exp_u.values,
            "obs_rev": obs["revenue"].values, "exp_rev": (exp_u * p_full).values, "sale_price": p_full,
        }))
    if not rows:
        return empty
    long = pd.concat(rows, ignore_index=True)

    def summarize(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        out = df.groupby(keys)[["obs_units", "exp_units", "obs_rev", "exp_rev"]].sum()
        return out.unstack("phase") if "phase" in keys else out

    agg = long[long["phase"].isin(["durant", "després", "normal"])]
    by_cat = summarize(agg, ["category", "phase"])
    by_cat.columns = [f"{m}_{p}" for m, p in by_cat.columns]
    by_cat = by_cat.reset_index()
    for ph in ("durant", "després", "normal"):
        for m in ("obs_units", "exp_units", "obs_rev", "exp_rev"):
            if f"{m}_{ph}" not in by_cat:
                by_cat[f"{m}_{ph}"] = 0.0
    agg = long[long["phase"] == "durant"]
    meta = win.groupby("category").agg(
        windows=("start", "size"), discount=("discount", "mean"),
        products=("products", "max"), old_price=("old_price", "mean"), open=("open", "any"),
    ).reset_index()
    # Preu mitjà de venda a preu complet (el que paga de veritat el client per unitat, no la mitjana del catàleg).
    sale = long.groupby("category")["sale_price"].first()
    by_cat = meta.merge(by_cat, on="category", how="left").fillna(0.0)
    by_cat["sale_price"] = by_cat["category"].map(sale)
    weights = agg.groupby("category")["exp_units"].sum().reindex(sale.index).fillna(0.0)
    totals = {
        "windows": int(len(win)), "products": float(meta["products"].sum()),
        "discount": float(win["discount"].mean()),
        "sale_price": float(np.average(sale, weights=weights)) if weights.sum() > 0 else float(sale.mean()),
        "first_start": win["start"].min(), "skipped": skipped,
    }
    for ph in ("durant", "després"):
        sub = long[long["phase"] == ph]
        for m in ("obs_units", "exp_units", "obs_rev", "exp_rev"):
            totals[f"{m}_{ph}"] = float(sub[m].sum())
    return long, by_cat, win, totals


def cost_threshold(d_rev: float, d_units: float, price: float) -> Tuple[str, float]:
    """
    Cost unitari que fa que la rebaixa no perdi diners, només amb dades reals.
    Benefici extra = ΔFacturació − cost × ΔUnitats  (ΔFacturació ja inclou el preu rebaixat).
      ΔU > 0: rendible si cost < ΔR/ΔU   (si ΔR ≤ 0, cap cost la fa rendible)
      ΔU < 0: rendible si cost > ΔR/ΔU   (estalvi de les unitats no venudes; si el llindar ≥ preu, impossible)
    """
    if d_units > 0:
        return ("never", np.nan) if d_rev <= 0 else ("below", d_rev / d_units)
    if d_units < 0:
        if d_rev >= 0:
            return "always", np.nan
        thr = d_rev / d_units
        return ("never", np.nan) if thr >= price else ("above", thr)
    return ("always", np.nan) if d_rev >= 0 else ("never", np.nan)


def cost_verdict(kind: str, value: float, price: float) -> str:
    if kind == "never":
        return "❌ No rendible amb cap cost"
    if kind == "always":
        return "✅ Rendible amb qualsevol cost"
    share = safe_div(value, price)
    if kind == "below":
        if value >= price:
            return "✅ Rendible amb qualsevol cost inferior al preu"
        return f"✅ Rendible si el cost unitari < {eur(value, 2)} ({pct(share, 0)} del preu de venda)"
    return f"⚠️ Només rendible si el cost unitari > {eur(value, 2)} ({pct(share, 0)} del preu de venda)"


def discount_index(long: pd.DataFrame, win: pd.DataFrame, before_h: float = 2.0, after_h: float = 6.0) -> pd.DataFrame:
    """
    Corba mitjana de totes les rebaixes: índex real ÷ esperat × 100 (100 = sense efecte), alineada a l'hora
    d'inici de cada rebaixa. Un índex (i no una suma) evita els salts quan les rebaixes cobreixen hores diferents.
    Les franges que són dins d'una altra rebaixa de la mateixa categoria s'exclouen.
    """
    cols = ["rel_h", "obs", "exp", "index", "n"]
    if long.empty or win.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    for w in win.itertuples(index=False):
        d = long[long["category"] == w.category]
        if d.empty:
            continue
        rel = (d["slot"] - w.start).dt.total_seconds() / 3600.0
        inside = (d["slot"] >= w.start) & (d["slot"] < w.end)
        keep = (rel >= -before_h) & (rel < after_h) & (d["phase"] != "fora") & ((d["phase"] != "durant") | inside)
        parts.append(pd.DataFrame({"rel_h": rel[keep], "obs": d["obs_units"][keep], "exp": d["exp_units"][keep]}))
    if not parts:
        return pd.DataFrame(columns=cols)
    g = pd.concat(parts).groupby("rel_h").agg(obs=("obs", "sum"), exp=("exp", "sum"), n=("obs", "size")).reset_index()
    g["index"] = 100 * g["obs"] / g["exp"].replace(0, np.nan)
    return g


# ──────────────────────────────────────────────────────────────────────────────
# 4. ESTIL (tema fosc, pantalla única sense scroll pensada per a 1920×1080)
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "bg": "#0d1117", "card": "#151a22", "border": "#262d38", "grid": "#212834",
    "text": "#e8eaed", "text2": "#a9b1bd", "muted": "#6e7781",
    "team": "#3987e5", "fallback": "#d95926", "neutral": "#8b949e",
    "good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b", "serious": "#ec835a",
}
SEQ_BLUE = [[0.0, "#132033"], [0.2, "#104281"], [0.45, "#1c5cab"], [0.7, "#3987e5"], [0.88, "#86b6ef"], [1.0, "#cde2fb"]]
ACTION_COLOR = {ACTION_UP: C["critical"], ACTION_KEEP: C["neutral"], ACTION_DOWN: C["team"]}
TYPE_LOOKED, TYPE_UNSEEN = "Es mira però no es compra", "Gairebé ningú el mira"
TYPE_COLOR = {TYPE_LOOKED: C["fallback"], TYPE_UNSEEN: C["neutral"]}
DEAD_EXPECTED_MIN = 3.0   # vendes esperades a partir de les quals vendre'n zero ja no és atzar (Poisson: P(0) < 5 %)
TILE_H = 408       # alçada de cada bloc de la portada (2 files caben en 1080 px sense scroll)
PLOT_CFG = {"displayModeBar": False}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {C['bg']}; }}
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], #MainMenu, footer {{ display: none !important; }}
.block-container {{ padding: .8rem 1.5rem .3rem 1.5rem !important; max-width: 100% !important; }}
div[data-testid="stVerticalBlock"] {{ gap: .55rem; }}
.app-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .12em;
               text-transform: uppercase; color: {C['muted']}; }}
.app-title {{ font-size: 1.35rem; font-weight: 700; color: {C['text']}; line-height: 1.2; }}
.tile-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .1em;
                text-transform: uppercase; color: {C['text2']}; }}
.big-label {{ font-size: .78rem; color: {C['text2']}; }}
.big {{ font-size: 2.25rem; font-weight: 700; color: {C['text']}; line-height: 1.15; font-variant-numeric: tabular-nums; }}
.big.down, .kpi-value.down {{ color: {C['critical']}; }}
.big.up, .kpi-value.up {{ color: {C['good']}; }}
.sub {{ font-size: .8rem; color: {C['muted']}; }}
.sub b {{ color: {C['text2']}; font-weight: 600; }}
.kpi {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px;
        padding: .75rem 1rem .7rem 1rem; height: 100%; }}
.kpi-label {{ font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .08em;
              text-transform: uppercase; color: {C['text2']}; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {C['text']}; margin: .2rem 0 .1rem 0;
              font-variant-numeric: tabular-nums; }}
.kpi-foot {{ font-size: .76rem; color: {C['muted']}; line-height: 1.35; }}
.kpi-foot b {{ color: {C['text2']}; font-weight: 600; }}
.up {{ color: {C['good']} !important; }} .down {{ color: {C['critical']} !important; }}
.note {{ background: {C['card']}; border: 1px solid {C['border']}; border-left: 3px solid {C['team']};
         border-radius: 8px; padding: .5rem .85rem; color: {C['text2']}; font-size: .8rem; line-height: 1.4; }}
.note b {{ color: {C['text']}; }}
.split {{ display: flex; height: 10px; border-radius: 6px; overflow: hidden; margin: .45rem 0 .3rem 0; }}
.legend-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 .3rem 0 .1rem; }}
div[class*="st-key-tile_"] {{ background: {C['card']}; border-radius: 14px; }}
div[class*="st-key-tile_"] .stButton button {{ background: transparent; border: 1px solid {C['border']};
    color: {C['team']}; font-weight: 600; padding: .15rem .6rem; min-height: 0; }}
div[class*="st-key-tile_"] .stButton button:hover {{ border-color: {C['team']}; background: rgba(57,135,229,.08); }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ padding: .35rem .9rem; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {C['border']}; border-radius: 10px; }}
div.st-key-period_box button {{ justify-content: flex-start; background: {C['card']}; border: 1px solid {C['border']};
    min-height: 2.5rem; }}
div.st-key-period_box button:hover {{ border-color: {C['team']}; }}
div.st-key-advice_box {{ position: fixed; right: 1.6rem; bottom: 1.4rem; z-index: 1000; width: auto !important; }}
div.st-key-advice_box button {{ width: 60px; height: 60px; border-radius: 50%; font-size: 1.7rem; padding: 0;
    background: #2a2410; border: 1px solid {C['warning']}; box-shadow: 0 0 18px rgba(250,178,25,.35); }}
div.st-key-advice_box button:hover {{ box-shadow: 0 0 26px rgba(250,178,25,.6); border-color: {C['warning']}; }}
div[data-testid="stPopoverBody"] {{ min-width: 440px; max-width: 540px; }}
.adv-title {{ font-weight: 700; font-size: 1.02rem; color: {C['warning']}; }}
.adv-scope {{ font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: {C['muted']}; margin: .1rem 0 .5rem 0; }}
.adv-item {{ border-top: 1px solid {C['border']}; padding: .55rem 0 .45rem 0; }}
.adv-act {{ font-weight: 600; color: {C['text']}; font-size: .9rem; }}
.adv-why {{ color: {C['text2']}; font-size: .8rem; line-height: 1.45; margin-top: .2rem; }}
.adv-why b {{ color: {C['text']}; }}
</style>
"""


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True, margin: Dict[str, int] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=C["text2"], size=12),
        margin=margin or dict(l=8, r=8, t=30, b=8),
        showlegend=legend,
        separators=",.",  # decimal "," i milers "." (format català)
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["card"], bordercolor=C["border"], font=dict(color=C["text"], family="Inter")),
    )
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["border"], tickfont=dict(color=C["text2"]))
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["border"], tickfont=dict(color=C["text2"]))
    fig.update_annotations(font=dict(color=C["text"], size=12))
    return fig


def kpi_card(label: str, value: str, foot: str = "", tone: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value {tone}">{value}</div>'
            f'<div class="kpi-foot">{foot}</div></div>')


def kpi_row(cards: Sequence[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def warn_errors(errors: Dict[str, str], keys: Sequence[str]) -> None:
    for k in keys:
        if k in errors:
            st.warning(f"No s'ha pogut carregar `{k}`: {errors[k]}")


def delta_html(cur: float, ref: float, fmt: Callable[[float], str], suffix: str) -> str:
    if not ref:
        return ""
    d = cur - ref
    return f'<span class="{"up" if d >= 0 else "down"}">{"▲" if d >= 0 else "▼"} {fmt(abs(d))}</span> {suffix}'


def signed_eur(x: float) -> str:
    return ("+" if x >= 0 else "−") + eur(abs(x))


def fmt_lift(x) -> str:
    return "—" if x is None or not np.isfinite(x) else ("+" if x >= 0 else "") + pct(x)


def go_to(view: str) -> None:
    st.query_params["v"] = view


def remember(widget_key: str, store_key: str) -> None:
    """Guarda el valor d'un control que només existeix en una vista (Streamlit l'esborra en canviar de vista)."""
    st.session_state[store_key] = st.session_state[widget_key]


# ──────────────────────────────────────────────────────────────────────────────
# 5. GRÀFICS REUTILITZABLES
# ──────────────────────────────────────────────────────────────────────────────
def fig_priority_donut(prod: pd.DataFrame, height: int) -> go.Figure:
    oos = prod[prod["oos_count"] > 0]
    counts = oos["urgency"].value_counts()
    labels = ["🔴 Alta", "🟡 Mitjana"]
    vals = [int(counts.get(l, 0)) for l in labels]
    fig = go.Figure(go.Pie(
        labels=["Alta: reposar ja", "Mitjana: vigilar"], values=vals, hole=0.66, sort=False,
        marker=dict(colors=[C["critical"], C["warning"]], line=dict(color=C["card"], width=2)),
        textinfo="value", textfont=dict(color=C["bg"], size=13),
        hovertemplate="%{label}: %{value} productes (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{num(sum(vals))}</b><br><span style='font-size:11px'>amb ruptures</span>",
                       showarrow=False, font=dict(size=20, color=C["text"]))
    style_fig(fig, height=height, margin=dict(l=4, r=4, t=4, b=4))
    fig.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", yanchor="top"))
    return fig


def fig_lost_bars(df: pd.DataFrame, label_col: str, height: int, top: int = 8) -> go.Figure:
    d = df[df["lost_revenue"] > 0].sort_values("lost_revenue", ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["lost_revenue"], y=d[label_col], orientation="h", marker_color=C["critical"], marker_line_width=0,
        text=[eur(v) for v in d["lost_revenue"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=C["text"]), hovertemplate="%{y}: %{x:,.0f} € perduts<extra></extra>",
    ))
    style_fig(fig, height=height, legend=False, margin=dict(l=4, r=40, t=6, b=4))
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(d["lost_revenue"].max() if not d.empty else 1, 1) * 1.3])
    fig.update_yaxes(showgrid=False)
    return fig


def fig_discount_index(idx: pd.DataFrame, height: int, compact: bool = False, dur: float = 2.0,
                       post: float = HANGOVER_H) -> go.Figure:
    """Corba mitjana: 2 h abans, la rebaixa (durada real) i les `post` hores després de tornar al preu original."""
    fig = go.Figure()
    top = max(160.0, float(idx["index"].max()) + 10) if not idx.empty else 160.0
    low = min(70.0, float(idx["index"].min()) - 10) if not idx.empty else 70.0
    fig.add_vrect(x0=0, x1=dur, fillcolor=C["warning"], opacity=0.14, line_width=0)
    fig.add_vrect(x0=dur, x1=dur + post, fillcolor=C["neutral"], opacity=0.08, line_width=0)
    fig.add_hline(y=100, line=dict(color=C["muted"], dash="dot", width=1))
    if not idx.empty:
        fig.add_trace(go.Scatter(
            x=idx["rel_h"] + 0.25, y=idx["index"], mode="lines+markers", name="Real ÷ esperat",
            line=dict(color=C["team"], width=2.5), marker=dict(size=6),
            hovertemplate="%{x:+.2f} h · índex %{y:.0f}<extra></extra>"))
    fig.add_annotation(x=dur / 2, y=top, text="rebaixa activa", showarrow=False, yanchor="top",
                       font=dict(color=C["warning"], size=11))
    fig.add_annotation(x=dur + post / 2, y=top, text=f"{post:.0f} h després (preu original)", showarrow=False,
                       yanchor="top", font=dict(color=C["text2"], size=11))
    fig.add_annotation(x=-2, y=100, text="100 = sense efecte", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(color=C["muted"], size=10))
    style_fig(fig, height=height, legend=False, margin=dict(l=4, r=8, t=8, b=4))
    fig.update_xaxes(range=[-2, dur + post], dtick=1, title=None if compact else "Hores des de l'inici de la rebaixa",
                     ticksuffix=" h")
    fig.update_yaxes(range=[low, top], title=None if compact else "Vendes (100 = esperat sense rebaixa)")
    return fig


def fig_dead_by_category(dead: pd.DataFrame, height: int) -> go.Figure:
    fig = go.Figure()
    if dead.empty:
        return style_fig(fig, height=height, legend=False)
    order = dead["category"].value_counts().sort_values().index.tolist()
    for t in (TYPE_UNSEEN, TYPE_LOOKED):
        cnt = dead[dead["dead_type"] == t]["category"].value_counts().reindex(order).fillna(0)
        fig.add_trace(go.Bar(x=cnt.values, y=order, orientation="h", name=t, marker_color=TYPE_COLOR[t],
                             marker_line=dict(color=C["card"], width=1),
                             hovertemplate="%{y}: %{x:.0f} productes<extra>" + t + "</extra>"))
    style_fig(fig, height=height, margin=dict(l=4, r=8, t=28, b=4))
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. APP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quadre econòmic i de magatzem", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

# Les dades es consulten en obrir (o recarregar) la pàgina i queden congelades per a tota la sessió: navegar
# entre pantalles no torna a consultar RawTree. Només s'actualitzen amb F5 o amb el botó ⟳.
if "_session_started" not in st.session_state:
    st.session_state["_session_started"] = True
    st.session_state["_snap"] = {}    # (regions, dispositius, dies d'estoc mort) → dades consultades
    st.session_state["_memo"] = {}    # càlculs derivats de cada instantània
    st.cache_data.clear()             # una sessió nova no reaprofita dades d'una sessió anterior

try:
    backend, _ = _backend()
    if "_dims" not in st.session_state:          # opcions dels filtres, també congelades per sessió
        st.session_state["_dims"] = q_dimensions()
    dims = st.session_state["_dims"]
except Exception as exc:
    st.error(f"No s'ha pogut connectar a RawTree. Revisa el fitxer .env (TEAM_ID, RAWTREE_API_KEY).\n\n{exc}")
    st.stop()

VIEWS = {"home": "Quadre de comandament", "retorn": "Retorn econòmic", "estoc": "Gestió d'estoc",
         "rebaixes": "Rebaixes", "mort": "Estoc mort"}
view = st.query_params.get("v", "home")
if view not in VIEWS:
    view = "home"

# ── Instant de referència de la sessió: fixa «ara» per a tots els períodes ───
if "_win" not in st.session_state:
    st.session_state["_win"] = q_windows()
win = st.session_state["_win"]
if win.empty or pd.isna(win.iloc[0].get("now")):
    st.session_state.pop("_win", None)
    st.error("No s'ha pogut llegir la finestra temporal de RawTree. Torna-ho a provar amb F5.")
    st.stop()
w0 = win.iloc[0]
now = w0["now"]
live_start = w0["live_start"] if pd.notna(w0["live_start"]) else now.normalize()
live_hours = max((now - live_start).total_seconds() / 3600.0, 1e-6)
hist_hours = max((w0["hist_end"] - w0["hist_start"]).total_seconds() / 3600.0, 0.0) if pd.notna(w0["hist_start"]) else 0.0

# ── Període d'anàlisi: les últimes N hores (de 1 a 24), triades amb una barra lliscant ──
MAX_HOURS = 24


def period_bounds(hours: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return max(now - pd.Timedelta(hours=hours), live_start), now


def _sql_ts(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3]


# ── Barra superior: títol, període, filtres globals i refresc ────────────────
bar = st.columns([2.9, 1.45, 1.75, 1.75, 1.25, 0.45], vertical_alignment="center", gap="small")
with bar[0]:
    if view == "home":
        head_ph = st.empty()
    else:
        b1, b2 = st.columns([1, 3.0], vertical_alignment="center")
        b1.button("← Tornar", key="back", on_click=go_to, args=("home",), width="stretch")
        head_ph = b2.empty()
hours = int(st.session_state.get("f_hours", MAX_HOURS))
with bar[1].container(key="period_box"):
    with st.popover(f"🕒 {'Última hora' if hours == 1 else f'Últimes {hours} h'}", width="stretch",
                    help="Interval de temps de totes les xifres. L'històric de 60 dies es manté com a referència."):
        st.markdown("<div class='adv-title' style='color:inherit'>Període d'anàlisi</div>", unsafe_allow_html=True)
        hours = st.slider("Hores enrere", 1, MAX_HOURS, MAX_HOURS, step=1, key="f_hours", format="%d h",
                          label_visibility="collapsed")
        st.caption("De l'última hora a les últimes 24 hores. Cada període nou es consulta un cop (2–3 s) "
                   "i després queda congelat.")
period = "Última hora" if hours == 1 else f"Últimes {hours} h"
sel_cat = tuple(bar[2].multiselect("Categoria", dims["category"], key="f_cat", placeholder="Totes les categories",
                                   label_visibility="collapsed"))
sel_reg = tuple(bar[3].multiselect("Regió", dims["region"], key="f_reg", placeholder="Totes les regions",
                                   label_visibility="collapsed"))
sel_dev = tuple(bar[4].multiselect("Dispositiu", dims["device"], key="f_dev", placeholder="Tots els dispositius",
                                   label_visibility="collapsed"))
if bar[5].button("⟳", key="refresh", help="Tornar a consultar les dades a RawTree (també ho fa recarregar la pàgina)",
                 width="stretch"):
    for k in ("_snap", "_memo"):
        st.session_state[k] = {}
    for k in ("_dims", "_win"):
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.rerun()

p_start, p_end = period_bounds(hours)
p_empty = p_end <= p_start
p_hours = max((p_end - p_start).total_seconds() / 3600.0, 1e-6)
P_IN = "en l'última hora" if hours == 1 else f"en les últimes {hours} hores"
P_OF = "de l'última hora" if hours == 1 else f"de les últimes {hours} hores"
ps, pe = _sql_ts(p_start), _sql_ts(max(p_end, p_start))

# ── Càrrega en paral·lel: dades fixes de la sessió + dades del període ───────
dead_days = int(st.session_state.get("dead_days_v", 7))
dead_max_sales = int(st.session_state.get("dead_max_v", 0))
snap_key = (sel_reg, sel_dev, dead_days, ps, pe)
_snaps = st.session_state.setdefault("_snap", {})


def load(key: tuple, jobs: Dict[str, Tuple[Callable[..., Any], tuple]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    if key in _snaps:
        return _snaps[key]
    with st.spinner("Consultant RawTree…"):
        res = prefetch(jobs)
    if not res[1]:   # una càrrega amb errors no es congela: es torna a provar en el següent clic
        _snaps[key] = res
    return res


base, err_base = load(("base", sel_reg, sel_dev, dead_days), {
    "catalog": (q_catalog, ()),
    "stock": (q_stock_changes, ()),
    "hist": (q_hist_products, (sel_reg, sel_dev, dead_days)),
    "clicks": (q_carousel_clicks, (sel_reg, sel_dev)),
    # L'estoc mort és global de magatzem i mira els últims N dies: no depèn del període, la regió ni el dispositiu.
    "live_all": (q_live_products, ((), ())),
    "hist_all": (q_hist_products, ((), (), dead_days)),
    "clicks_all": (q_carousel_clicks, ((), ())),
    "heat_hist": (q_category_region, ("hist", sel_reg, sel_dev)),
    "pc_events": (q_price_events, ()),
    "pc_sales": (q_price_change_sales, (sel_reg, sel_dev)),
    "store": (q_store_slots, (sel_reg, sel_dev)),
})
per, err_per = load(("period", sel_reg, sel_dev, ps, pe), {
    "live": (q_live_products, (sel_reg, sel_dev, ps, pe)),
    "feedback": (q_feedback, (sel_reg, sel_dev, ps, pe)),
    "fb_hourly": (q_feedback_hourly, (sel_reg, sel_dev, ps, pe)),
    "heat_live": (q_category_region, ("live", sel_reg, sel_dev, ps, pe)),
})
data, errors = {**base, **per}, {**err_base, **err_per}

cat = data["catalog"]
if cat.empty:
    st.error("El catàleg és buit o no s'ha pogut carregar.")
    warn_errors(errors, ["catalog"])
    st.stop()
cat_of = cat.set_index("product_id")["category"]

_p_end = max(p_end, p_start)
_p_txt = (f"{p_start:%d/%m %H:%M}–{_p_end:%H:%M}" if p_start.date() == _p_end.date()
          else f"{p_start:%d/%m %H:%M} – {_p_end:%d/%m %H:%M}")
head_html = (f"<div class='app-eyebrow'>Datathon FME 2026 · RawTree · {TEAM} · {period}: {_p_txt} UTC · "
             f"dades de les {now:%H:%M} UTC "
             f"(F5 o ⟳ per actualitzar)</div>")
head_ph.markdown(head_html + f"<div class='app-title'>{VIEWS[view] if view != 'home' else 'Quadre econòmic i de magatzem'}</div>",
                 unsafe_allow_html=True)
if p_empty:
    st.info(f"El període «{period}» no té dades. Tria un període més llarg.")


# ── Càlculs compartits (es fan un sol cop per instantània i filtre de categoria) ─
def compute_shared() -> Dict[str, Any]:
  stock = stockout_metrics(data["stock"], p_start, p_end)
  products = build_product_table(cat, data["live"], data["hist"], data["clicks"], stock, p_hours, hist_hours)
  if sel_cat:
      products = products[products["category"].isin(sel_cat)]

  fb = data["feedback"].copy()
  if not fb.empty:
      fb["purchased_cat"] = fb["purchased"].map(cat_of)
      fb_sales = fb[fb["purchased"] != ""]
      if sel_cat:
          fb_sales = fb_sales[fb_sales["purchased_cat"].isin(sel_cat)]
  else:
      fb_sales = fb
  org_rev, org_pur, org_views = products["revenue"].sum(), products["purchases"].sum(), products["views"].sum()
  car_rev = fb_sales["revenue"].sum() if not fb_sales.empty else 0.0
  car_pur = fb_sales["purchases"].sum() if not fb_sales.empty else 0.0
  total_rev, total_pur = org_rev + car_rev, org_pur + car_pur
  ticket, conv = safe_div(total_rev, total_pur), safe_div(org_pur, org_views)
  h_ticket = safe_div(products["h_revenue"].sum(), products["h_purchases"].sum())
  h_conv = safe_div(products["h_purchases"].sum(), products["h_views"].sum())
  lost_total = products["lost_revenue"].sum()
  n_oos_products = int((products["oos_count"] > 0).sum())
  n_now_oos = int(products["currently_oos"].sum())

  store = data["store"]
  if not store.empty:
      store = store[(store["slot"] >= live_start.floor("30min")) & (store["slot"] + timedelta(minutes=30) <= now)]
  # Gràfics de facturació per franja: només les franges completes dins del període.
  store_p = (store[(store["slot"] >= p_start.floor("30min")) & (store["slot"] + timedelta(minutes=30) <= p_end)]
             if not store.empty else store)
  store_cat = store_p[store_p["category"].isin(sel_cat)] if (sel_cat and not store_p.empty) else store_p
  org_slots = (store_cat.groupby("slot")[["revenue", "purchases"]].sum().reset_index()
               if not store_cat.empty else pd.DataFrame(columns=["slot", "revenue", "purchases"]))

  dead = build_product_table(cat, data["live_all"], data["hist_all"], data["clicks_all"], stock, live_hours, hist_hours)
  if sel_cat:
      dead = dead[dead["category"].isin(sel_cat)]
  n_active = int(dead["is_active"].sum())
  dead["sales_win"] = dead["r_purchases"] + dead["purchases"]
  dead["carts_win"] = dead["r_carts"] + dead["carts"]
  dead["views_win"] = dead["r_views"] + dead["views"]
  # Conversió típica de cada categoria a la finestra (vendes ÷ vistes de tots els productes actius).
  _act = dead[dead["is_active"]]
  cat_conv = (_act.groupby("category")["sales_win"].sum() / _act.groupby("category")["views_win"].sum().replace(0, np.nan)).fillna(0.0)
  overall_conv = safe_div(_act["sales_win"].sum(), _act["views_win"].sum())
  too_new = dead["created_at"].notna() & (dead["created_at"] > now - timedelta(days=dead_days))
  dead = dead[dead["is_active"] & ~too_new & (dead["sales_win"] <= dead_max_sales)
              & (dead["carts_win"] <= max(dead_max_sales, 1)) & (dead["carousel_clicks"] == 0)].copy()
  # Vendes que s'esperarien amb les seves vistes i la conversió de la categoria. Si n'esperàvem ≥ 3 i no se n'ha venut
  # cap, no és mala sort (P < 5 %): alguna cosa frena la compra. Si n'esperàvem menys, té massa poques vistes per jutjar.
  dead["exp_sales"] = dead["views_win"] * dead["category"].map(cat_conv).fillna(overall_conv)
  dead["dead_type"] = np.where(dead["exp_sales"] >= DEAD_EXPECTED_MIN, TYPE_LOOKED, TYPE_UNSEEN)

  pc_long, pc_by, pc_win, pc_tot = price_change_analysis(data["pc_events"], data["pc_sales"], sel_cat, live_start, now,
                                                         period=(p_start, p_end))
  _closed = pc_win[~pc_win["open"]] if not pc_win.empty else pc_win
  pc_dur = (float(((_closed["end"] - _closed["start"]).dt.total_seconds() / 3600).median())
            if not _closed.empty else 2.0)
  pc_idx = discount_index(pc_long, pc_win, 2.0, pc_dur + HANGOVER_H)
  if pc_tot:
      pc_dU = pc_tot["obs_units_durant"] - pc_tot["exp_units_durant"]
      pc_dR = pc_tot["obs_rev_durant"] - pc_tot["exp_rev_durant"]
      pc_lift = safe_div(pc_tot["obs_units_durant"], pc_tot["exp_units_durant"]) - 1 if pc_tot["exp_units_durant"] else None
      pc_kind, pc_val = cost_threshold(pc_dR, pc_dU, pc_tot["sale_price"])
  else:
      pc_dU = pc_dR = 0.0
      pc_lift, pc_kind, pc_val = None, "never", np.nan
  return {k: v for k, v in locals().items() if not k.startswith("_")}


_memo = st.session_state.setdefault("_memo", {})
_memo_key = (snap_key, sel_cat, dead_max_sales)   # snap_key ja inclou el període
if _memo_key not in _memo:
    _memo[_memo_key] = compute_shared()
globals().update(_memo[_memo_key])   # products, dead, pc_* ... per a les pantalles


def cost_short(kind: str, value: float) -> str:
    return {"never": "Cap cost", "always": "Qualsevol"}.get(kind, ("&lt; " if kind == "below" else "&gt; ") + eur(value, 2))


# ══════════════════════════════════════════════════════════════════════════════
# BOMBETA DE CONSELL: accions recomanades calculades amb les dades i els filtres actuals
# ══════════════════════════════════════════════════════════════════════════════
Advice = List[Tuple[str, str]]   # [(acció, per què amb xifres)]


def scope_label(category: str = None, extra: str = "", with_period: bool = True) -> str:
    """Text de l'àmbit al qual s'apliquen els consells (filtres globals + filtre propi de la pantalla)."""
    cats = [category] if category else list(sel_cat)
    parts = ([period] if with_period else []) + [", ".join(cats) if cats else "Totes les categories"]
    if sel_reg:
        parts.append(", ".join(sel_reg))
    if sel_dev:
        parts.append(", ".join(sel_dev))
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def advice_retorn() -> Advice:
    items: List[Tuple[float, str, str]] = []   # (impacte per ordenar, acció, per què)
    # 1. Disponibilitat del recommender
    if not fb.empty:
        total_imp = fb["impressions"].sum()
        fb_imp = fb.loc[fb["served_by"] == "fallback", "impressions"].sum()
        share = safe_div(fb_imp, total_imp)
        # El carrusel no és d'una categoria: els euros per carrusel es calculen amb totes les compres del carrusel.
        agg = fb.groupby("served_by")[["impressions", "revenue"]].sum()
        rpm = {s: safe_div(agg.loc[s, "revenue"], agg.loc[s, "impressions"]) * 1000 for s in agg.index}
        upside = max((rpm.get("team", 0) - rpm.get("fallback", 0)) / 1000 * fb_imp, 0)
        down_h = 0
        fh = data["fb_hourly"]
        if not fh.empty:
            tot_h = fh.groupby("hour")["impressions"].sum()
            miss_h = fh[fh["fallback_reason"] == "missing"].groupby("hour")["impressions"].sum()
            down_h = int(((miss_h.reindex(tot_h.index).fillna(0) / tot_h) > 0.5).sum())
        if share > 0.03:
            items.append((upside, "Evitar que el recommender s'aturi",
                          f"La reserva ha atès el {pct(share)} dels carrusels"
                          + (f", concentrats en {down_h} hores en què el recommender estava aturat" if down_h else "")
                          + f". El nostre motor factura {eur(rpm.get('team', 0))} per 1.000 carrusels i la reserva "
                            f"{eur(rpm.get('fallback', 0))}: atendre'ls nosaltres hauria aportat uns {eur(upside)}. "
                            "Un procés que el reiniciï automàticament si cau ho evita."))
    # 2. Tiquet mitjà respecte a l'històric
    if h_ticket and ticket < h_ticket * 0.98:
        loss = (h_ticket - ticket) * total_pur
        why = (f"{P_IN[0].upper() + P_IN[1:]}, {eur(ticket, 2)} per compra davant de {eur(h_ticket, 2)} de l'històric "
               f"({pct(safe_div(ticket, h_ticket) - 1)}); amb {num(total_pur)} compres, són uns {eur(loss)} menys.")
        if pc_tot:
            n_w = pc_tot["windows"]
            why += (f" Coincideix amb {n_w} {'rebaixa' if n_w == 1 else 'rebaixes'} d'un {pct(pc_tot['discount'], 0)}: "
                    f"{'revisa-la' if n_w == 1 else 'revisa-les'} a Rebaixes.")
        items.append((loss, "Revisar què fa baixar el tiquet mitjà", why))
    # 3. Conversió respecte a l'històric
    if h_conv and conv < h_conv * 0.97:
        lost_buys = (h_conv - conv) * org_views
        items.append((lost_buys * ticket, "Investigar la caiguda de conversió",
                      f"{pct(conv, 2)} de les vistes acaben en compra, davant del {pct(h_conv, 2)} habitual: "
                      f"són unes {num(lost_buys)} compres menys. Mira si coincideix amb ruptures d'estoc a Gestió d'estoc."))
    # 4. Geografia: on pesa més la categoria (o la combinació categoria × regió més destacada)
    heat = data["heat_live"]
    if not heat.empty and heat["purchases"].sum() > 0:
        reg_tot = heat.groupby("region")["purchases"].sum()
        cats = list(sel_cat) or heat.groupby("category")["purchases"].sum().sort_values(ascending=False).index.tolist()
        best = None
        for c in cats:
            ch = heat[heat["category"] == c].groupby("region")["purchases"].sum().reindex(reg_tot.index).fillna(0)
            g_share = safe_div(ch.sum(), reg_tot.sum())
            if g_share <= 0:
                continue
            idx = (ch / reg_tot.replace(0, np.nan)) / g_share
            idx = idx[reg_tot >= reg_tot.median() * 0.3].dropna()
            if idx.empty:
                continue
            r = idx.idxmax()
            if best is None or idx[r] > best[2]:
                best = (c, r, idx[r], safe_div(ch[r], reg_tot[r]), g_share)
        if best and best[2] > 1.1:
            c, r, ix, s_r, s_g = best
            items.append((ix * 1000, f"Reforçar l'estoc de {c} al magatzem de {r}",
                          f"A {r}, {c} és el {pct(s_r)} de les compres, davant del {pct(s_g)} a la resta de la botiga "
                          "(" + f"{ix:.1f}".replace(".", ",") + " vegades més). Distribuir l'estoc segons aquesta barreja redueix "
                          "ruptures locals."))
    # 5. Ruptures d'estoc
    if lost_total > 0:
        items.append((lost_total, "Prioritzar la reposició dels productes sense estoc",
                      f"Les ruptures {P_OF} han costat {eur(lost_total)} en {num(n_oos_products)} productes "
                      f"({num(n_now_oos)} encara sense estoc). La bombeta de Gestió d'estoc diu quins reposar primer."))
    items.sort(key=lambda x: -x[0])
    return [(a, w) for _, a, w in items[:4]]


def advice_estoc(pv_prod: pd.DataFrame, oos: pd.DataFrame, page_cat: str) -> Advice:
    if oos.empty:
        return [("Mantenir les compres com fins ara",
                 f"Cap producte d'aquest àmbit s'ha quedat sense estoc {P_IN}: el nivell de compra actual és suficient.")]
    out: Advice = []
    now_oos = oos[oos["currently_oos"]].sort_values("potential_rev_h", ascending=False)
    if not now_oos.empty:
        top = now_oos.head(3)
        out.append((f"Reposar ja {'aquests productes' if len(now_oos) <= 3 else f'els {num(len(now_oos))} productes'} sense estoc",
                    "Cada hora que estan sense estoc es deixen de facturar uns "
                    f"{eur(now_oos['potential_rev_h'].sum())}. Comença per "
                    + ", ".join(f"<b>{r.product_id}</b> ({r.category}, {eur(r.potential_rev_h)}/h)" for r in top.itertuples())
                    + "."))
    repeat = oos[oos["oos_count"] >= 2].sort_values("lost_revenue", ascending=False)
    if not repeat.empty:
        out.append((f"Augmentar el lot de compra de {num(len(repeat))} productes que es queden sense estoc sovint",
                    f"Han trencat estoc 2 o més vegades {P_IN} i han perdut {eur(repeat['lost_revenue'].sum())} "
                    f"({pct(safe_div(repeat['lost_revenue'].sum(), oos['lost_revenue'].sum()), 0)} de les pèrdues). "
                    f"El lot actual no cobreix la demanda: el que més perd és <b>{repeat.iloc[0]['product_id']}</b> "
                    f"({eur(repeat.iloc[0]['lost_revenue'])})."))
    if page_cat == "Totes":
        by_c = oos.groupby("category")["lost_revenue"].sum().sort_values(ascending=False)
        if len(by_c) > 1 and by_c.iloc[0] > 0:
            out.append((f"Començar la reposició per {by_c.index[0]}",
                        f"Concentra el {pct(safe_div(by_c.iloc[0], by_c.sum()), 0)} dels ingressos perduts "
                        f"({eur(by_c.iloc[0])}), per davant de {by_c.index[1]} ({eur(by_c.iloc[1])})."))
    else:
        cat_h = safe_div(oos["lost_revenue"].sum(), oos["oos_hours"].sum())
        all_oos = products[products["oos_count"] > 0]
        glob_h = safe_div(all_oos["lost_revenue"].sum(), all_oos["oos_hours"].sum())
        if glob_h > 0:
            rel = "més" if cat_h > glob_h else "menys"
            out.append((f"{'Prioritzar' if cat_h > glob_h else 'No prioritzar'} aquesta categoria en la reposició",
                        f"Una hora sense estoc aquí costa {eur(cat_h)}, {rel} que la mitjana de la botiga ({eur(glob_h)})."))
    down = pv_prod[pv_prod["action"] == ACTION_DOWN]
    if not down.empty:
        out.append((f"Reduir la compra de {num(len(down))} productes",
                    "No s'han quedat mai sense estoc i són al 20 % amb menys demanda: es pot comprar menys quantitat "
                    "sense risc de perdre vendes. Són a la Llista de reposició, acció «Reduir compra»."))
    return out[:4]


def advice_rebaixes() -> Advice:
    if pc_long.empty or pc_by.empty:
        return [("No treure conclusions sobre rebaixes en aquest àmbit",
                 f"Cap rebaixa amb prou dades ha començat {P_IN} amb aquests filtres: no se'n pot mesurar l'efecte.")]
    tb = pc_by.copy()
    tb["dU"] = tb["obs_units_durant"] - tb["exp_units_durant"]
    tb["dR"] = tb["obs_rev_durant"] - tb["exp_rev_durant"]
    tb["u_dur"] = tb["obs_units_durant"] / tb["exp_units_durant"].replace(0, np.nan) - 1
    tb["u_aft"] = tb["obs_units_després"] / tb["exp_units_després"].replace(0, np.nan) - 1
    tb = tb[tb["exp_units_durant"] > 0]
    items: List[Tuple[float, str, str]] = []
    for r in tb.itertuples():
        kind, val = cost_threshold(r.dR, r.dU, r.sale_price)
        if kind == "never":
            items.append((abs(r.dR) + 1e6, f"No repetir la rebaixa a {r.category}",
                          f"Ven un {pct(r.u_dur, 0)} més d'unitats, però factura {eur(abs(r.dR))} menys del que hauria "
                          "facturat sense rebaixa: cap cost de producte la fa rendible."))
        elif kind == "below":
            items.append((r.dR, f"Repetir la rebaixa a {r.category} només si cada producte costa menys de {eur(val, 2)}",
                          f"Durant la rebaixa ven un {pct(r.u_dur, 0)} més i factura {eur(r.dR)} més. Per sobre de "
                          f"{eur(val, 2)} de cost per unitat ({pct(safe_div(val, r.sale_price), 0)} del preu de venda), "
                          "perd diners."))
        if np.isfinite(r.u_aft) and r.u_aft < -0.07:
            items.append((abs(r.u_aft) * 1e5, f"Escurçar o espaiar les rebaixes a {r.category}",
                          f"En tornar al preu original ven un {pct(abs(r.u_aft), 0)} menys del normal durant "
                          f"{HANGOVER_H:.0f} h: la rebaixa avança compres que s'haurien fet igualment, en lloc de crear-ne de noves."))
    items.sort(key=lambda x: -x[0])
    out = [(a, w) for _, a, w in items[:3]]
    if not store.empty and pd.notna(pc_tot.get("first_start")):
        sl = store.groupby("slot")["purchases"].sum()
        pre_s, post_s = sl[sl.index < pc_tot["first_start"]], sl[sl.index >= pc_tot["first_start"]]
        if not pre_s.empty and not post_s.empty and abs(safe_div(post_s.mean(), pre_s.mean()) - 1) < 0.03:
            out.append(("Fer servir les rebaixes per moure estoc concret, no per fer créixer les vendes",
                        f"Les compres totals de la botiga no canvien amb les rebaixes ({num(pre_s.mean())} per franja "
                        f"abans i {num(post_s.mean())} després): les rebaixes només desplacen compres d'uns productes a uns altres."))
    return out[:4]


def advice_mort() -> Advice:
    if dead.empty:
        return [("Mantenir l'assortiment",
                 f"Cap producte actiu ha passat {dead_days} dies sense vendes ni intenció de compra.")]
    out: Advice = []
    looked = dead[dead["dead_type"] == TYPE_LOOKED].sort_values("exp_sales", ascending=False)
    unseen = dead[dead["dead_type"] == TYPE_UNSEEN]
    if not looked.empty:
        out.append((f"Provar una rebaixa en {num(len(looked))} productes que es miren però no es compren",
                    f"Amb les seves vistes n'haurien d'haver venut unes {num(looked['exp_sales'].sum())} unitats i no "
                    "n'han venut cap: alguna cosa (sovint el preu) frena la compra. Comença per "
                    + ", ".join(f"<b>{r.product_id}</b> ({num(r.views_win)} vistes)" for r in looked.head(3).itertuples())
                    + ". Si rebaixats tampoc es venen, retira'ls."))
    if not unseen.empty:
        out.append((f"Deixar de comprar {num(len(unseen))} productes que gairebé ningú mira",
                    f"Tenen massa poques vistes per vendre's: no hi ha demanda que justifiqui tenir-los al magatzem."))
    if not sel_cat:
        by_c = dead["category"].value_counts()
        share = safe_div(by_c.iloc[0], len(dead))
        if share > 0.2:
            out.append((f"Revisar l'assortiment de {by_c.index[0]}",
                        f"Concentra {num(by_c.iloc[0])} dels {num(len(dead))} productes en alerta ({pct(share, 0)}): "
                        "potser hi ha massa referències per a la demanda que té."))
    else:
        share = safe_div(len(dead), n_active)
        out.append((f"{'Revisar' if share > 0.08 else 'Mantenir'} l'assortiment d'aquesta categoria",
                    f"El {pct(share)} dels seus productes actius estan en alerta"
                    + (", per sobre del que seria raonable." if share > 0.08 else ", una proporció baixa.")))
    return out[:4]


def render_advice(theme: str, scope: str, items: Advice) -> None:
    """Bombeta fixa a baix a la dreta; en clicar-la mostra les accions recomanades."""
    with st.container(key="advice_box"):
        with st.popover("💡", help="Acció recomanada amb les dades i els filtres actuals"):
            st.markdown(f"<div class='adv-title'>Acció recomanada · {theme}</div>"
                        f"<div class='adv-scope'>Àmbit: {scope} · dades de les {now:%H:%M} UTC</div>",
                        unsafe_allow_html=True)
            if not items:
                st.markdown("<div class='adv-item'>Cap acció destacada amb aquests filtres.</div>", unsafe_allow_html=True)
            for i, (act, why) in enumerate(items, start=1):
                st.markdown(f"<div class='adv-item'><div class='adv-act'>{i}. {act}</div>"
                            f"<div class='adv-why'>{why}</div></div>", unsafe_allow_html=True)


ADVICE = None   # (tema, àmbit, accions): cada pantalla de detall l'omple


# ══════════════════════════════════════════════════════════════════════════════
# PORTADA: 4 blocs clicables, sense scroll
# ══════════════════════════════════════════════════════════════════════════════
def tile(col, key: str, eyebrow: str, target: str):
    with col:
        box = st.container(border=True, height=TILE_H, key=f"tile_{key}")
    with box:
        h = st.columns([3, 1.15], vertical_alignment="center")
        h[0].markdown(f"<div class='tile-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
        h[1].button("Veure detall →", key=f"open_{key}", on_click=go_to, args=(target,), width="stretch")
    return box


if view == "home":
    warn_errors(errors, ["live", "feedback", "stock", "pc_events", "pc_sales"])
    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")

    # ── Retorn ──
    with tile(row1[0], "retorn", f"Retorn econòmic · {period}", "retorn"):
        a, b = st.columns([1.25, 1])
        org_share = safe_div(org_rev, total_rev)
        a.markdown(
            f"<div class='big-label'>Facturació total</div><div class='big'>{eur(total_rev)}</div>"
            f"<div class='split'><div style='width:{max(org_share * 100, 2):.1f}%;background:{C['team']}'></div>"
            f"<div style='width:{max((1 - org_share) * 100, 2):.1f}%;background:{C['warning']}'></div></div>"
            f"<div class='sub'><span class='legend-dot' style='background:{C['team']}'></span>Orgànic <b>{eur(org_rev)}</b>"
            f" &nbsp; <span class='legend-dot' style='background:{C['warning']}'></span>Carrusel <b>{eur(car_rev)}</b></div>",
            unsafe_allow_html=True)
        b.markdown(
            f"<div class='big-label'>Ingressos perduts per ruptura</div><div class='big down'>{eur(lost_total)}</div>"
            f"<div class='sub'><b>{num(n_oos_products)}</b> productes afectats · <b class='down'>{num(n_now_oos)}</b> ara sense estoc</div>",
            unsafe_allow_html=True)
        if not org_slots.empty:
            fig = go.Figure(go.Bar(x=org_slots["slot"] + timedelta(minutes=15), y=org_slots["revenue"],
                                   marker_color=C["team"], marker_line_width=0,
                                   hovertemplate="%{x|%H:%M} · %{y:,.0f} €<extra></extra>"))
            style_fig(fig, height=215, legend=False, margin=dict(l=4, r=4, t=22, b=4))
            fig.update_layout(title=dict(text="Facturació orgànica per franja de 30 min", x=0, y=0.98,
                                         font=dict(size=11, color=C["text2"])), bargap=0.25)
            fig.update_xaxes(tickformat="%H:%M", showgrid=False)
            fig.update_yaxes(showticklabels=False, showgrid=False)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="home_rev")

    # ── Gestió d'estoc ──
    with tile(row1[1], "estoc", "Gestió d'estoc · prioritat de reposició", "estoc"):
        a, b = st.columns([1, 1.25])
        with a:
            st.plotly_chart(fig_priority_donut(products, 300), width="stretch", config=PLOT_CFG, key="home_donut")
        with b:
            by_c = products.groupby("category", as_index=False)["lost_revenue"].sum()
            st.markdown("<div class='big-label'>Ingressos perduts per categoria</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_lost_bars(by_c, "category", 285, top=7), width="stretch", config=PLOT_CFG,
                            key="home_lost")

    # ── Rebaixes ──
    with tile(row2[0], "rebaixes", "Rebaixes · efecte mitjà", "rebaixes"):
        a, b = st.columns([1, 2.1])
        if pc_tot:
            a.markdown(
                f"<div class='big-label'>Vendes durant la rebaixa</div>"
                f"<div class='big {'up' if (pc_lift or 0) >= 0 else 'down'}'>{fmt_lift(pc_lift)}</div>"
                f"<div class='sub'>respecte a l'esperat sense rebaixa</div><br>"
                f"<div class='big-label'>Rebaixa mitjana</div><div class='big' style='font-size:1.7rem'>−{pct(pc_tot['discount'])}</div>"
                f"<div class='sub'><b>{pc_tot['windows']}</b> {'rebaixa' if pc_tot['windows'] == 1 else 'rebaixes'} de {pc_dur:.0f} h · "
                f"<b>{num(pc_tot['products'])}</b> productes</div><br>"
                f"<div class='big-label'>Cost màxim per guanyar diners</div>"
                f"<div class='big' style='font-size:1.4rem'>{cost_short(pc_kind, pc_val)}</div>",
                unsafe_allow_html=True)
            with b:
                st.plotly_chart(fig_discount_index(pc_idx, 320, compact=True, dur=pc_dur), width="stretch", config=PLOT_CFG,
                                key="home_idx")
        else:
            st.info(f"Cap rebaixa amb prou dades ha començat {P_IN} amb aquests filtres.")

    # ── Estoc mort ──
    with tile(row2[1], "mort", f"Estoc mort · últims {dead_days} dies", "mort"):
        a, b = st.columns([1, 2.1])
        a.markdown(
            f"<div class='big-label'>Productes en alerta</div><div class='big'>{num(len(dead))}</div>"
            f"<div class='sub'>de <b>{num(n_active)}</b> actius, sense vendes ni intenció</div><br>"
            f"<div class='big-label'>{TYPE_LOOKED}</div>"
            f"<div class='big' style='font-size:1.6rem;color:{C['fallback']}'>{num((dead['dead_type'] == TYPE_LOOKED).sum())}</div>"
            f"<div class='sub'>possible problema de preu</div>",
            unsafe_allow_html=True)
        with b:
            st.plotly_chart(fig_dead_by_category(dead, 320), width="stretch", config=PLOT_CFG, key="home_dead")

# ══════════════════════════════════════════════════════════════════════════════
# DETALL · RETORN ECONÒMIC
# ══════════════════════════════════════════════════════════════════════════════
elif view == "retorn":
    ADVICE = ("Retorn econòmic", scope_label(), advice_retorn())
    warn_errors(errors, ["live", "feedback", "hist", "store", "fb_hourly"])
    tabs = st.tabs(["Vendes i conversió", "Nostre motor vs reserva", "Geografia: categoria × regió"])

    with tabs[0]:
        kpi_row([
            kpi_card("Facturació total", eur(total_rev),
                     f"Orgànic <b>{eur(org_rev)}</b> · Carrusel <b>{eur(car_rev)}</b>"),
            kpi_card("Tiquet mitjà", eur(ticket, 2), delta_html(ticket, h_ticket, lambda v: eur(v, 2), "vs històric 60 d")),
            kpi_card("Taxa de conversió", pct(conv, 2),
                     f"{num(org_pur)} compres / {num(org_views)} vistes<br>"
                     + delta_html(conv * 100, h_conv * 100, lambda v: f"{v:.2f} pp".replace(".", ","), "vs històric")),
            kpi_card("Compres", num(total_pur), f"Orgàniques <b>{num(org_pur)}</b> · Carrusel <b>{num(car_pur)}</b>"),
            kpi_card("Ingressos perduts per ruptura", eur(lost_total),
                     f"<b>{num(n_oos_products)}</b> productes afectats · <b class='down'>{num(n_now_oos)}</b> ara sense estoc",
                     tone="down"),
        ])
        if org_slots.empty:
            st.info("Encara no hi ha franges completes.")
        else:
            c1, c2 = st.columns([1.6, 1])
            fig = go.Figure(go.Bar(x=org_slots["slot"] + timedelta(minutes=15), y=org_slots["revenue"],
                                   marker_color=C["team"], marker_line_width=0,
                                   hovertemplate="%{x|%H:%M} · %{y:,.0f} €<extra></extra>"))
            style_fig(fig, height=540, legend=False, margin=dict(l=8, r=8, t=34, b=8))
            fig.update_layout(title=dict(text="Facturació orgànica per franja de 30 min (€)", x=0,
                                         font=dict(size=13, color=C["text"])), bargap=0.25)
            fig.update_xaxes(tickformat="%H:%M", showgrid=False)
            c1.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            tk = org_slots.assign(t=org_slots["revenue"] / org_slots["purchases"].replace(0, np.nan))
            fig = go.Figure()
            fig.add_hline(y=h_ticket, line=dict(color=C["muted"], dash="dot", width=1),
                          annotation_text=f"històric 60 d: {eur(h_ticket, 2)}", annotation_position="top left",
                          annotation_font=dict(color=C["muted"], size=11))
            fig.add_trace(go.Scatter(x=tk["slot"] + timedelta(minutes=15), y=tk["t"], mode="lines+markers",
                                     line=dict(color=C["team"], width=2), marker=dict(size=5),
                                     hovertemplate="%{x|%H:%M} · %{y:.2f} €<extra></extra>"))
            style_fig(fig, height=540, legend=False, margin=dict(l=8, r=8, t=34, b=8))
            fig.update_layout(title=dict(text="Tiquet mitjà orgànic per franja (€)", x=0, font=dict(size=13, color=C["text"])))
            fig.update_xaxes(tickformat="%H:%M", showgrid=False)
            c2.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    with tabs[1]:
        if fb.empty:
            st.info("Encara no hi ha feedback del carrusel per als filtres seleccionats.")
        else:
            agg = fb.groupby("served_by")[["impressions", "clicks"]].sum()
            agg = agg.join(fb_sales.groupby("served_by")[["purchases", "revenue"]].sum().reindex(agg.index), how="left").fillna(0.0)
            order = [s for s in ["team", "fallback"] if s in agg.index]
            labels = {"team": "Motor", "fallback": "Reserva"}
            colors = {"team": C["team"], "fallback": C["fallback"]}
            metrics = [
                ("Ingressos", lambda r: r["revenue"], lambda v: eur(v)),
                ("€ / 1.000 impressions", lambda r: safe_div(r["revenue"], r["impressions"]) * 1000, lambda v: eur(v, 1)),
                ("CTR", lambda r: safe_div(r["clicks"], r["impressions"]), lambda v: pct(v, 2)),
                ("Conversió", lambda r: safe_div(r["purchases"], r["impressions"]), lambda v: pct(v, 2)),
            ]
            c1, c2 = st.columns([1.15, 1])
            fig = make_subplots(rows=1, cols=4, subplot_titles=[m[0] for m in metrics], horizontal_spacing=0.07)
            for i, (_, getter, fmt) in enumerate(metrics, start=1):
                vals = [getter(agg.loc[s]) for s in order]
                fig.add_trace(go.Bar(
                    x=[labels[s] for s in order], y=vals, marker_color=[colors[s] for s in order], marker_line_width=0,
                    width=0.6, text=[fmt(v) for v in vals], textposition="outside", textfont=dict(color=C["text"]),
                    cliponaxis=False, hovertemplate="%{x}: %{text}<extra></extra>"), row=1, col=i)
                fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, max(vals + [1e-9]) * 1.28], row=1, col=i)
            style_fig(fig, height=470, legend=False, margin=dict(l=8, r=8, t=40, b=8))
            fig.update_xaxes(tickangle=0)
            c1.plotly_chart(fig, width="stretch", config=PLOT_CFG)

            fh = data["fb_hourly"]
            if not fh.empty:
                fh = fh.assign(kind=np.select(
                    [fh["served_by"] == "team", fh["fallback_reason"] == "missing"],
                    ["Motor", "Reserva: sense resposta"], default="Reserva: resposta invàlida"))
                pv = fh.pivot_table(index="hour", columns="kind", values="impressions", aggfunc="sum", fill_value=0)
                fig = go.Figure()
                for k, col in [("Motor", C["team"]), ("Reserva: sense resposta", C["fallback"]),
                               ("Reserva: resposta invàlida", C["warning"])]:
                    if k in pv:
                        fig.add_trace(go.Bar(x=pv.index, y=pv[k], name=k, marker_color=col,
                                             marker_line=dict(color=C["card"], width=1),
                                             hovertemplate="%{x|%H:%M} · %{y:,.0f} peticions<extra>" + k + "</extra>"))
                style_fig(fig, height=470, margin=dict(l=8, r=8, t=40, b=8))
                fig.update_layout(barmode="stack", bargap=0.2,
                                  title=dict(text="Qui respon cada hora", x=0, y=0.99, font=dict(size=13, color=C["text"])),
                                  legend=dict(y=1.0, yanchor="bottom", x=0.35))
                fig.update_xaxes(tickformat="%H:%M", dtick=3600000, showgrid=False)
                c2.plotly_chart(fig, width="stretch", config=PLOT_CFG)

            total_imp = fb["impressions"].sum()
            fb_imp = fb.loc[fb["served_by"] == "fallback", "impressions"].sum()
            team_rpm = safe_div(agg.loc["team", "revenue"], agg.loc["team", "impressions"]) * 1000 if "team" in agg.index else 0
            fb_rpm = safe_div(agg.loc["fallback", "revenue"], agg.loc["fallback", "impressions"]) * 1000 if "fallback" in agg.index else 0
            upside = max((team_rpm - fb_rpm) / 1000 * fb_imp, 0)
            down_h = 0
            if not data["fb_hourly"].empty:
                hh = data["fb_hourly"].groupby("hour").apply(
                    lambda g: safe_div(g.loc[g["fallback_reason"] == "missing", "impressions"].sum(), g["impressions"].sum()),
                    include_groups=False)
                down_h = int((hh > 0.5).sum())
            st.markdown(
                f"<div class='note'>La reserva ha atès <b>{pct(safe_div(fb_imp, total_imp))}</b> de les peticions. "
                f"Cada carrusel del nostre motor factura <b>{eur(team_rpm, 1)}</b> per 1.000 impressions, davant de "
                f"<b>{eur(fb_rpm, 1)}</b> de la reserva: atendre-les nosaltres hauria aportat uns <b>{eur(upside)}</b> més. "
                + (f"Les peticions sense resposta es concentren en <b>{down_h} hores</b> en què el recommender estava aturat: "
                   "és un problema de disponibilitat, no de velocitat." if down_h else "")
                + ("<br>Amb filtre de categoria, impressions i clics inclouen tot el carrusel." if sel_cat else "")
                + "</div>", unsafe_allow_html=True)

    with tabs[2]:
        h1, h2, h3, _ = st.columns([1.6, 1.8, 1.6, 2])
        src = h1.radio("Dades", [f"{period}", "Històric 60 dies"], horizontal=True, key="heat_src")
        metric = h2.radio("Mètrica", ["Compres", "Facturació", "Conversió"], horizontal=True, key="heat_metric")
        normalize = h3.toggle("% sobre el total de cada regió", value=False, key="heat_norm",
                              help="Mostra la barreja de categories de cada regió en lloc dels volums absoluts.")
        heat = data["heat_hist"] if src.startswith("Històric") else data["heat_live"]
        if sel_cat and not heat.empty:
            heat = heat[heat["category"].isin(sel_cat)]
        if heat.empty:
            st.info("No hi ha compres per a aquesta combinació de filtres.")
        else:
            pv = heat.pivot_table(index="category", columns="region", values=["purchases", "revenue", "views"],
                                  aggfunc="sum", fill_value=0)
            if metric == "Compres":
                z, fmt, unit = pv["purchases"], ",.0f", " compres"
            elif metric == "Facturació":
                z, fmt, unit = pv["revenue"], ",.0f", " €"
            else:
                z, fmt, unit = (pv["purchases"] / pv["views"].replace(0, np.nan)).fillna(0.0) * 100, ".2f", " %"
            if normalize and metric != "Conversió":
                z, fmt, unit = (z / z.sum(axis=0).replace(0, np.nan) * 100).fillna(0.0), ".1f", " % de la regió"
            z = z.loc[z.sum(axis=1).sort_values(ascending=True).index]
            z = z[z.sum(axis=0).sort_values(ascending=False).index]
            fig = go.Figure(go.Heatmap(
                z=z.values, x=list(z.columns), y=list(z.index), colorscale=SEQ_BLUE, xgap=2, ygap=2,
                texttemplate="%{z:" + fmt + "}", textfont=dict(size=12),
                colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=C["text2"])),
                hovertemplate="%{y} · %{x}<br>%{z:" + fmt + "}" + unit + "<extra></extra>"))
            style_fig(fig, height=600, legend=False, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_xaxes(showgrid=False, side="top", tickangle=0)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            if metric != "Conversió" and not normalize:
                top3 = z.stack().sort_values(ascending=False).head(3)
                st.markdown("<div class='note'>On es mou més volum: "
                            + " · ".join(f"<b>{c}</b> a {r} ({num(v)}{unit})" for (c, r), v in top3.items())
                            + ". Activa «% sobre el total de cada regió» per veure la barreja de cada magatzem.</div>",
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DETALL · GESTIÓ D'ESTOC
# ══════════════════════════════════════════════════════════════════════════════
elif view == "estoc":
    warn_errors(errors, ["stock", "catalog", "live"])
    cats_avail = sorted(products["category"].dropna().unique().tolist())
    opts = ["Totes"] + cats_avail
    prev = st.session_state.get("estoc_cat_v", "Totes")
    f1, *kcols = st.columns([1.3, 1, 1, 1, 1])
    page_cat = f1.selectbox("Categoria", opts, index=opts.index(prev) if prev in opts else 0, key="estoc_cat_w",
                            on_change=remember, args=("estoc_cat_w", "estoc_cat_v"))
    pv_prod = products if page_cat == "Totes" else products[products["category"] == page_cat]
    oos = pv_prod[pv_prod["oos_count"] > 0]
    ADVICE = ("Gestió d'estoc", scope_label(None if page_cat == "Totes" else page_cat), advice_estoc(pv_prod, oos, page_cat))
    lost = oos["lost_revenue"].sum()
    eur_per_h = safe_div(lost, oos["oos_hours"].sum())
    for col, card in zip(kcols, [
        kpi_card("Ingressos perduts", eur(lost), f"per ruptures {P_OF} · {page_cat.lower() if page_cat != 'Totes' else 'totes les categories'}", tone="down"),
        kpi_card("Cost d'una hora sense estoc", eur(eur_per_h, 0) + "/h",
                 "mitjana del que es deixa de vendre per producte i hora"),
        kpi_card("Productes amb ruptures", num(len(oos)), f"<b class='down'>{num(int(oos['currently_oos'].sum()))}</b> ara sense estoc"),
        kpi_card("Reposar ja (urgència alta)", num(int((oos["urgency_rank"] == 0).sum())),
                 f"{eur(oos.loc[oos['urgency_rank'] == 0, 'lost_revenue'].sum())} perduts entre tots"),
    ]):
        col.markdown(card, unsafe_allow_html=True)

    tabs = st.tabs(["Resum", "Matriu de decisió", "Llista de reposició"])
    with tabs[0]:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("<div class='big-label'>Nivell de prioritat dels productes amb ruptures</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_priority_donut(pv_prod, 520), width="stretch", config=PLOT_CFG)
        with c2:
            if page_cat == "Totes":
                st.markdown("<div class='big-label'>Ingressos perduts per categoria</div>", unsafe_allow_html=True)
                bars = pv_prod.groupby("category", as_index=False)["lost_revenue"].sum()
                st.plotly_chart(fig_lost_bars(bars, "category", 520, top=10), width="stretch", config=PLOT_CFG)
            else:
                st.markdown("<div class='big-label'>Productes que més diners han perdut</div>", unsafe_allow_html=True)
                bars = oos.assign(label=oos["product_id"] + " · " + oos["subcategory"].astype(str))
                st.plotly_chart(fig_lost_bars(bars, "label", 520, top=12), width="stretch", config=PLOT_CFG)
        st.markdown(
            f"<div class='note'><b>Val la pena reposar?</b> Cada hora que un d'aquests productes està sense estoc es deixen de "
            f"facturar de mitjana <b>{eur(eur_per_h)}</b>. Si reposar-lo costa menys que les hores que estarà sense estoc "
            f"multiplicades per aquest import, val la pena. Càlcul: vendes per hora quan hi ha estoc × hores sense estoc × "
            f"preu actual.</div>", unsafe_allow_html=True)

    with tabs[1]:
        if oos.empty:
            st.info("Cap producte amb ruptures d'estoc per a aquests filtres.")
        else:
            log_y = st.toggle("Escala logarítmica", value=True, key="mx_log",
                              help="Separa els productes que venen poc, que en escala lineal queden amuntegats a baix.")
            rng = np.random.default_rng(7)
            jitter = rng.uniform(-0.18, 0.18, len(oos))
            max_price = max(oos["price_eur"].max(), 1.0)
            fig = go.Figure()
            for action in [ACTION_UP, ACTION_KEEP, ACTION_DOWN]:
                d = oos[oos["action"] == action]
                if d.empty:
                    continue
                j = jitter[oos["action"].values == action]
                fig.add_trace(go.Scatter(
                    x=d["oos_count"] + j, y=d["potential_rev_h"].clip(lower=0.1), mode="markers", name=action,
                    marker=dict(size=d["price_eur"], sizemode="area", sizeref=2.0 * max_price / (32 ** 2), sizemin=4,
                                color=ACTION_COLOR[action], opacity=0.8, line=dict(width=1, color=C["bg"])),
                    customdata=np.stack([d["product_id"], d["category"], d["price_eur"], d["lost_revenue"],
                                         d["oos_hours"], d["oos_count"]], axis=-1),
                    hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>Ruptures: %{customdata[5]}"
                                   " (%{customdata[4]:.1f} h)<br>Facturació potencial: %{y:,.0f} €/h<br>"
                                   "Preu: %{customdata[2]:.2f} € · Perdut: %{customdata[3]:,.0f} €<extra>" + action + "</extra>")))
            y_med = float(oos["potential_rev_h"].median())
            fig.add_hline(y=max(y_med, 0.1), line=dict(color=C["muted"], dash="dot", width=1),
                          annotation_text="mediana dels productes amb ruptures", annotation_position="bottom right",
                          annotation_font=dict(color=C["muted"], size=11))
            for r in oos.sort_values("lost_revenue", ascending=False).head(5).itertuples():
                fig.add_annotation(x=r.oos_count, y=np.log10(max(r.potential_rev_h, 0.1)) if log_y else r.potential_rev_h,
                                   text=r.product_id, showarrow=True, arrowhead=0, arrowcolor=C["muted"], ax=34, ay=-18,
                                   font=dict(size=10, color=C["text"]))
            style_fig(fig, height=600, margin=dict(l=8, r=8, t=36, b=8))
            fig.update_xaxes(title=f"Vegades sense estoc {P_IN}", tickmode="linear", dtick=1,
                             range=[0.5, float(oos["oos_count"].max()) + 0.5])
            fig.update_yaxes(title="Facturació potencial (€/h amb estoc)", type="log" if log_y else "linear")
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.caption("A dalt a la dreta, els crítics: molt valuosos i sovint sense estoc. Mida del punt = preu. "
                       "Etiquetats, els 5 que més diners han perdut.")

    with tabs[2]:
        g1, g2 = st.columns([2, 3])
        act_filter = g1.multiselect("Acció", [ACTION_UP, ACTION_KEEP, ACTION_DOWN], default=[ACTION_UP, ACTION_KEEP],
                                    placeholder="Totes les accions", key="stk_act")
        search = g2.text_input("Cerca producte, marca o subcategoria", placeholder="p_… / marca / subcategoria", key="stk_q")
        tv = oos.copy()
        if act_filter:
            tv = tv[tv["action"].isin(act_filter)]
        if search:
            s = search.strip().lower()
            tv = tv[tv["product_id"].str.lower().str.contains(s, regex=False)
                    | tv["brand"].astype(str).str.lower().str.contains(s, regex=False)
                    | tv["subcategory"].astype(str).str.lower().str.contains(s, regex=False)]
        tv = tv.sort_values(["urgency_rank", "lost_revenue"], ascending=[True, False])
        if tv.empty:
            st.info("Cap producte compleix els filtres.")
        else:
            st.dataframe(
                tv[["urgency", "action", "product_id", "category", "subcategory", "brand", "price_eur", "is_active",
                    "oos_count", "oos_hours", "sales_rate_h", "potential_rev_h", "lost_revenue"]],
                width="stretch", hide_index=True, height=560,
                column_config={
                    "urgency": st.column_config.TextColumn("Urgència", width="small"),
                    "action": st.column_config.TextColumn("Acció recomanada"),
                    "product_id": "Producte", "category": "Categoria", "subcategory": "Subcategoria", "brand": "Marca",
                    "price_eur": st.column_config.NumberColumn("Preu", format="%.2f €"),
                    "is_active": st.column_config.CheckboxColumn("En estoc ara"),
                    "oos_count": st.column_config.NumberColumn("Ruptures", format="%d"),
                    "oos_hours": st.column_config.NumberColumn("Hores sense estoc", format="%.1f h"),
                    "sales_rate_h": st.column_config.NumberColumn("Vendes/h", format="%.1f"),
                    "potential_rev_h": st.column_config.NumberColumn("€/h amb estoc", format="%.0f €"),
                    "lost_revenue": st.column_config.ProgressColumn(
                        "Ingressos perduts", format="%.0f €", min_value=0.0,
                        max_value=float(max(oos["lost_revenue"].max(), 1.0))),
                })

# ══════════════════════════════════════════════════════════════════════════════
# DETALL · REBAIXES
# ══════════════════════════════════════════════════════════════════════════════
elif view == "rebaixes":
    ADVICE = ("Rebaixes", scope_label(), advice_rebaixes())
    warn_errors(errors, ["pc_events", "pc_sales", "store"])
    if pc_long.empty:
        st.info(f"Cap rebaixa amb prou dades ha començat {P_IN} amb aquests filtres. "
                "Prova el període «Tot l'esdeveniment» per veure-les totes.")
    else:
        closed = pc_win[~pc_win["open"]]
        dur_h = (closed["end"] - closed["start"]).dt.total_seconds().mean() / 3600 if not closed.empty else np.nan
        kpi_row([
            kpi_card("Rebaixes analitzades", num(pc_tot["windows"]),
                     f"{num(pc_tot['products'])} productes"
                     + (f" · {dur_h:.0f} h cadascuna" if np.isfinite(dur_h) else "")),
            kpi_card("Rebaixa mitjana", "−" + pct(pc_tot["discount"]), "sobre el preu original"),
            kpi_card("Vendes durant la rebaixa", fmt_lift(pc_lift),
                     f"{num(pc_tot['obs_units_durant'])} u reals vs {num(pc_tot['exp_units_durant'])} esperades",
                     tone="up" if (pc_lift or 0) >= 0 else "down"),
            kpi_card("Facturació de més", signed_eur(pc_dR), f"{num(pc_dU)} unitats de més durant les rebaixes",
                     tone="up" if pc_dR >= 0 else "down"),
            kpi_card("Cost unitari màxim per guanyar diners", cost_short(pc_kind, pc_val),
                     (f"{pct(safe_div(pc_val, pc_tot['sale_price']), 0)} del preu mitjà de venda ({eur(pc_tot['sale_price'], 2)})"
                      if pc_kind in ("below", "above") else
                      "Ni amb cost zero surt a compte" if pc_kind == "never" else "Guanya diners sigui quin sigui el cost")),
        ])
        tabs = st.tabs(["Visió general", "Per categoria", "Rendibilitat"])

        with tabs[0]:
            c1, c2 = st.columns([2.2, 1])
            with c1:
                st.plotly_chart(fig_discount_index(pc_idx, 560, dur=pc_dur), width="stretch", config=PLOT_CFG)
            with c2:
                st.markdown(
                    "<div class='note'><b>Com es llegeix.</b> La línia és la mitjana de totes les rebaixes: vendes reals ÷ "
                    "vendes esperades sense rebaixa. <b>100</b> vol dir que la rebaixa no canvia res; <b>150</b>, que es ven "
                    "un 50 % més. Totes les rebaixes s'alineen a la seva hora d'inici (0 h).</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='note'><b>Com es calcula l'esperat.</b> Pes dels productes rebaixats respecte als que no "
                    "s'han rebaixat mai, mesurat a totes les franges a preu normal, aplicat a les vendes d'aquests últims "
                    "a cada franja. Així es neteja l'efecte dels productes nous de l'esdeveniment, que prenen quota al catàleg.</div>",
                    unsafe_allow_html=True)
                if not store.empty and pd.notna(pc_tot.get("first_start")):
                    t0 = pc_tot["first_start"]
                    sl = store.groupby("slot")[["purchases", "new_purchases"]].sum()
                    pre_s, post_s = sl[sl.index < t0], sl[sl.index >= t0]
                    if not pre_s.empty and not post_s.empty:
                        tp, tq = pre_s["purchases"].mean(), post_s["purchases"].mean()
                        ch = safe_div(tq, tp) - 1
                        st.markdown(
                            f"<div class='note'><b>El flux de compradors és constant</b> (3 peticions/s). Compres totals per "
                            f"franja: <b>{num(tp)}</b> abans de la primera rebaixa i <b>{num(tq)}</b> després "
                            f"({'+' if ch >= 0 else ''}{pct(ch)}). "
                            + ("Les rebaixes no porten compres noves: <b>redistribueixen</b> les que ja hi havia."
                               if abs(ch) < 0.03 else "El total de la botiga canvia de manera apreciable.")
                            + "</div>", unsafe_allow_html=True)

        with tabs[1]:
            order = pc_win.groupby("category")["start"].min().sort_values().index.tolist()
            order = [c for c in order if c in set(pc_long["category"])]
            titles = []
            for c in order:
                wc = pc_win[pc_win["category"] == c]
                spans = " i ".join(f"{w.start:%H:%M}–{'ara' if w.open else format(w.end, '%H:%M')}" for w in wc.itertuples())
                titles.append(f"{c} · −{pct(wc['discount'].mean(), 0)} · {spans}")
            fig = make_subplots(rows=len(order), cols=1, shared_xaxes=True,
                                vertical_spacing=0.3 / max(len(order), 1), subplot_titles=titles)
            for i, c in enumerate(order, start=1):
                d = pc_long[pc_long["category"] == c]
                for w in pc_win[pc_win["category"] == c].itertuples():
                    fig.add_vrect(x0=w.start, x1=w.end, fillcolor=C["warning"], opacity=0.14, line_width=0, row=i, col=1)
                fig.add_trace(go.Scatter(
                    x=d["slot"] + timedelta(minutes=15), y=d["exp_units"], name="Esperat sense rebaixa", mode="lines",
                    line=dict(color=C["neutral"], width=2, dash="dash"), legendgroup="exp", showlegend=i == 1,
                    hovertemplate="%{x|%H:%M} · esperat %{y:,.0f} u<extra>" + c + "</extra>"), row=i, col=1)
                fig.add_trace(go.Scatter(
                    x=d["slot"] + timedelta(minutes=15), y=d["obs_units"], name="Real", mode="lines+markers",
                    line=dict(color=C["team"], width=2), marker=dict(size=4), legendgroup="obs", showlegend=i == 1,
                    hovertemplate="%{x|%H:%M} · real %{y:,.0f} u<extra>" + c + "</extra>"), row=i, col=1)
                fig.update_yaxes(rangemode="tozero", row=i, col=1)
            style_fig(fig, height=600, margin=dict(l=8, r=8, t=50, b=8))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(tickformat="%H:%M", dtick=3600000, row=len(order), col=1)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.caption("Unitats venudes per franja de 30 min. Franja groga: rebaixa activa.")

        with tabs[2]:
            tb = pc_by.copy()
            tb["u_dur"] = tb["obs_units_durant"] / tb["exp_units_durant"].replace(0, np.nan) - 1
            tb["dU"] = tb["obs_units_durant"] - tb["exp_units_durant"]
            tb["dR"] = tb["obs_rev_durant"] - tb["exp_rev_durant"]
            th = [cost_threshold(r, u, p) for r, u, p in zip(tb["dR"], tb["dU"], tb["sale_price"])]
            no_data = tb["exp_units_durant"] <= 0
            tb["cost_max"] = [v if k == "below" else np.nan for k, v in th]
            tb["cost_share"] = tb["cost_max"] / tb["sale_price"]
            tb["verdict"] = ["⏳ En curs: encara sense franges completes" if nd else
                             cost_verdict(k, v, p) + (" · en curs" if o else "")
                             for (k, v), p, o, nd in zip(th, tb["sale_price"], tb["open"], no_data)]
            u_aft = tb["obs_units_després"] / tb["exp_units_després"].replace(0, np.nan) - 1
            tb["after"] = [("—" if not np.isfinite(u) else
                            f"✓ Sense efecte ({u:+.0%})" if abs(u) < 0.07 else
                            f"⚠️ Ven {'menys' if u < 0 else 'més'} ({u:+.0%})") for u in u_aft.fillna(np.nan)]
            st.dataframe(
                tb.sort_values("dR", ascending=False)[
                    ["category", "windows", "discount", "sale_price", "u_dur", "dU", "dR", "cost_max", "cost_share",
                     "verdict", "after"]],
                width="stretch", hide_index=True, height=300,
                column_config={
                    "category": "Categoria",
                    "windows": st.column_config.NumberColumn("Rebaixes", format="%d"),
                    "discount": st.column_config.NumberColumn("Rebaixa", format="percent"),
                    "sale_price": st.column_config.NumberColumn("Preu mitjà de venda", format="%.2f €"),
                    "u_dur": st.column_config.NumberColumn("Δ vendes durant", format="percent"),
                    "dU": st.column_config.NumberColumn("Unitats de més", format="%.0f"),
                    "dR": st.column_config.NumberColumn("Facturació de més", format="%.0f €"),
                    "cost_max": st.column_config.NumberColumn("Cost màxim", format="%.2f €"),
                    "cost_share": st.column_config.ProgressColumn("Cost màxim / preu", format="percent",
                                                                  min_value=0.0, max_value=1.0),
                    "verdict": st.column_config.TextColumn("Veredicte", width="large"),
                    "after": st.column_config.TextColumn(f"Comprovació: {HANGOVER_H:.0f} h després"),
                })
            st.markdown(
                "<div class='note'><b>Sense suposar cap marge.</b> Benefici extra de la rebaixa = facturació de més − cost "
                "unitari × unitats de més, comptant només les hores en què la rebaixa és activa. Si es venen més unitats, la "
                "rebaixa dona benefici només si cada unitat costa menys de <b>facturació de més ÷ unitats de més</b>; si la "
                "facturació baixa, cap cost la fa rendible. La columna «" + f"{HANGOVER_H:.0f}" + " h després» comprova si en tornar al preu normal "
                "es ven menys del normal (clients que han avançat la compra): si diu «sense efecte», no hi ha res a "
                "descomptar.</div>", unsafe_allow_html=True)
            if pc_tot["skipped"]:
                st.caption(f"Categories sense prou franges a preu normal per calcular-ne l'esperat: {', '.join(pc_tot['skipped'])}.")

# ══════════════════════════════════════════════════════════════════════════════
# DETALL · ESTOC MORT
# ══════════════════════════════════════════════════════════════════════════════
elif view == "mort":
    ADVICE = ("Estoc mort", scope_label(extra=f"últims {dead_days} dies", with_period=False), advice_mort())
    warn_errors(errors, ["live_all", "hist_all", "clicks_all"])
    s1, s2, *kcols = st.columns([1.2, 1.2, 1, 1, 1])
    s1.slider("Finestra (dies)", 3, 30, dead_days, key="dead_days_w", on_change=remember,
              args=("dead_days_w", "dead_days_v"), help="Període en què el producte no ha de tenir vendes ni intenció.")
    s2.slider("Vendes màximes", 0, 5, dead_max_sales, key="dead_max_w", on_change=remember,
              args=("dead_max_w", "dead_max_v"), help="Fins a quantes vendes a la finestra es considera estoc mort.")
    top_cat = dead["category"].value_counts()
    n_looked = int((dead["dead_type"] == TYPE_LOOKED).sum())
    for col, card in zip(kcols, [
        kpi_card("Productes en alerta", num(len(dead)), f"de {num(n_active)} actius · {pct(safe_div(len(dead), n_active))}"),
        kpi_card(TYPE_LOOKED, num(n_looked),
                 f"amb les seves vistes n'hauria d'haver venut {DEAD_EXPECTED_MIN:.0f} o més: possible problema de preu"),
        kpi_card("Categoria amb més alertes", top_cat.index[0] if not top_cat.empty else "—",
                 f"{num(top_cat.iloc[0])} productes" if not top_cat.empty else ""),
    ]):
        col.markdown(card, unsafe_allow_html=True)
    st.caption("Productes actius sense vendes, carrets ni clics al carrusel a cap regió ni dispositiu. "
               "S'exclouen els donats d'alta dins la finestra.")

    tabs = st.tabs(["Visió general", "Llista per categoria"])
    with tabs[0]:
        if dead.empty:
            st.success("Cap producte actiu sense vendes ni intenció a la finestra seleccionada.")
        else:
            c1, c2 = st.columns([1, 1.25])
            with c1:
                st.markdown("<div class='big-label'>Productes en alerta per categoria</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_dead_by_category(dead, 560), width="stretch", config=PLOT_CFG)
            with c2:
                st.markdown("<div class='big-label'>Visualitzacions i preu de cada producte en alerta</div>",
                            unsafe_allow_html=True)
                fig = go.Figure()
                for t in (TYPE_UNSEEN, TYPE_LOOKED):
                    d = dead[dead["dead_type"] == t]
                    fig.add_trace(go.Scatter(
                        x=d["views_win"].clip(lower=1), y=d["price_eur"], mode="markers", name=t,
                        marker=dict(size=8, color=TYPE_COLOR[t], opacity=0.75, line=dict(width=1, color=C["bg"])),
                        customdata=np.stack([d["product_id"], d["category"], d["brand"], d["exp_sales"]], axis=-1),
                        hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[2]}<br>"
                                       "%{x:,.0f} vistes · %{y:.2f} €<br>Vendes esperades: %{customdata[3]:.1f} · reals: 0"
                                       "<extra>" + t + "</extra>")))
                if overall_conv > 0:
                    fig.add_vline(x=DEAD_EXPECTED_MIN / overall_conv, line=dict(color=C["muted"], dash="dot", width=1),
                                  annotation_text=f"≈ {DEAD_EXPECTED_MIN:.0f} vendes esperades",
                                  annotation_position="top", annotation_font=dict(color=C["muted"], size=10))
                style_fig(fig, height=560, margin=dict(l=8, r=8, t=30, b=8))
                # Escala lineal (les vistes abasten menys d'un ordre de magnitud) i rang ajustat a les dades:
                # de 0 a un 8 % més que el producte amb més vistes; si un dia en tenen més, l'eix s'allarga sol.
                x_max = max(float(dead["views_win"].max()),
                            DEAD_EXPECTED_MIN / overall_conv if overall_conv > 0 else 0.0, 1.0) * 1.08
                fig.update_xaxes(title=f"Vistes als últims {dead_days} dies", type="linear", range=[0, x_max], nticks=8)
                fig.update_yaxes(title="Preu (€)")
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.markdown(
                f"<div class='note'><b>Com es classifiquen.</b> Amb les vistes de cada producte i la conversió típica de la "
                f"seva categoria ({pct(overall_conv)} de mitjana), calculem quantes vendes s'esperarien. "
                f"<span style='color:{C['fallback']}'><b>Es mira però no es compra</b></span>: se n'esperaven {DEAD_EXPECTED_MIN:.0f} "
                f"o més i no se n'ha venut cap; per atzar passaria menys d'un 5 % de les vegades, així que alguna cosa frena la "
                f"compra (sovint el preu): abans de retirar-lo, es pot provar amb rebaixa. "
                f"<span style='color:{C['neutral']}'><b>Gairebé ningú el mira</b></span>: massa poques vistes per jutjar-lo; "
                f"es pot deixar de comprar.</div>",
                unsafe_allow_html=True)

    with tabs[1]:
        if dead.empty:
            st.success("Cap producte en alerta.")
        else:
            tv = dead.sort_values(["category", "dead_type", "exp_sales"], ascending=[True, True, False])
            st.dataframe(
                tv[["category", "dead_type", "product_id", "subcategory", "brand", "price_eur", "views_win",
                    "exp_sales", "h_purchases", "created_at"]],
                width="stretch", hide_index=True, height=600,
                column_config={
                    "category": "Categoria",
                    "dead_type": st.column_config.TextColumn("Tipus"),
                    "product_id": "Producte", "subcategory": "Subcategoria", "brand": "Marca",
                    "price_eur": st.column_config.NumberColumn("Preu", format="%.2f €"),
                    "views_win": st.column_config.ProgressColumn(
                        f"Vistes {dead_days} d", format="%d", min_value=0,
                        max_value=int(max(dead["views_win"].max(), 1))),
                    "exp_sales": st.column_config.NumberColumn("Vendes esperades", format="%.1f",
                                                               help="Amb les seves vistes i la conversió de la categoria"),
                    "h_purchases": st.column_config.NumberColumn("Vendes 60 d", format="%d"),
                    "created_at": st.column_config.DatetimeColumn("Alta", format="DD/MM/YYYY"),
                })

# ── Bombeta de consell (només a les pantalles de detall) ──────────────────────
if ADVICE is not None:
    render_advice(*ADVICE)
