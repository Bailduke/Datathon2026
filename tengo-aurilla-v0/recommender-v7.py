#!/usr/bin/env python3
"""recommender-v7.py · modelo "tengo-aurilla" (jerárquico factorizado + afinidad usuario-producto).

Puntuación del producto i para el usuario u en una petición:

    score(i) = P_jer(i)^theta · A(u, i)

  P_jer        Probabilidad jerárquica P(c | u, sesión) · P(s | c, u, sesión) · P(i | s, u, contexto):
               niveles 1 y 2 Dirichlet-multinomial por usuario (prior poblacional × kappa + recuentos
               fraccionados compra 1 / carrito 0,25 / vista 0,02, sin decaimiento + la sesión como
               pseudo-recuentos); nivel 3 logit condicional dentro de la subcategoría:
               V_i = b_pop·log(pop_i) + z_i·(b_p0 + b_u·r_u + b_dev + b_reg) + b_sale·rebajado_i,
               con pop_i la cuota de compras Gamma-Poisson (histórico + stream en vivo) y z_i el cuantil
               de precio del producto en su subcategoría (rango / nº de productos; 1 = el más barato).
  A(u, i)      Afinidad usuario-producto, FUERA del anidamiento (multiplica, no redistribuye):
               · clases: en el carrito durante el stream (g_lc), en el carrito histórico (g_hc),
                 único producto visto en la sesión (g_vu), primer visto de varios (g_vf), otros vistos (g_vo);
               · gustos latentes, con cuatro "cestas" del usuario (historia, sesión, carrito de hoy y
                 vistas de hoy): perfil de etiquetas (cada descripción de producto lleva 2 de 20
                 etiquetas; Dirichlet-multinomial) y vector de co-compra (16 factores latentes del
                 log-lift producto-producto, una vez descontado lo que ya explica la jerarquía).
  Selección    Solo productos elegibles (en stock y no comprados por el usuario: saciedad); se envían
               los 10 de mayor score, de mayor a menor. Exploración: Thompson Sampling sobre la
               popularidad (Gamma); el slot 1 usa siempre las medias.

Estimación: kappa, lambda y los betas del nivel 3, por máxima verosimilitud con separación temporal
del histórico (build_priors). theta y toda la afinidad, con NUESTRO feedback en vivo: se reproduce el
stream en orden y se ajusta compra_ij ~ Poisson(w_slot · exp(a) · score) (calibrate_live, --calibrate).
Las derivaciones están en modelo.tex / modelo.pdf (esta misma carpeta).

Uso, desde la raíz del repo y con el .env del equipo:
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py --build        # estima y guarda los priors
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py --eval 3000    # evaluación offline (últimos 7 días)
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py --calibrate    # reestima la afinidad con el feedback en vivo
                                                                     # (2-3 min; no lanzarlo durante la final)
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py --smoke        # responde 30 peticiones reales sin enviar
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py --dry-run      # sistema completo sin escribir en RawTree
  .venv/bin/python tengo-aurilla-v0/recommender-v7.py                # EN VIVO
  ... --shard 0/2      # responde solo a la mitad de las peticiones (test A/B sin duplicados)
  ... --set clave=valor  # cambia un parámetro de Config
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import queue
import re
import signal
import sys
import threading
import time
import zlib
from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
LOGS = HERE / "logs"
HIST_T0 = "2026-09-15 07:00:00"          # inicio del histórico (UTC)


# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================
@dataclass
class Config:
    model_version: str = "transformada-neta-v7"
    # --- niveles 1-2: recuentos fraccionados y sesión
    w_purchase: float = 1.0
    w_cart: float = 0.25
    w_view: float = 0.02
    rho_session: float = 0.65            # la vista más reciente pesa 1, la anterior 0,65, ...
    max_session_views: int = 10
    # --- nivel 3: popularidad Gamma-Poisson
    b0: float = 20000.0                  # fuerza del prior, en compras totales de la tienda
    live_weight: float = 1.0             # peso de una compra en vivo frente a una del histórico
    r_shrink: float = 3.0                # encogimiento del perfil de precio: sum(resid) / (n + 3)
    # --- selección y exploración
    thompson: bool = True                # muestrea la popularidad (Gamma) en cada petición
    exploit_slot1: bool = True           # el slot 1 usa siempre las medias
    max_per_sub: int = 0                 # 0 = sin límite por subcategoría (modelo acordado)
    taste: bool = True                   # gustos latentes del usuario: etiquetas de la descripción + co-compra
    taste_k: int = 16                    # nº de factores de co-compra (16 predice mejor que la matriz completa)
    taste_smooth: float = 40.0           # suavizado del lift de co-compra (40 predice mejor que 2: +53 de logver)
    taste_shrink: tuple = (3.0, 1.0, 1.0, 1.0)   # encogimiento de la media por cesta: historia, sesión, carrito hoy, vistas hoy
    tag_alpha: float = 6.0               # fuerza del prior uniforme del perfil de etiquetas (Dirichlet)
    trend: bool = True                   # tendencia por subcategoría: eventos recientes frente a su cuota habitual
    trend_hl: float = 5.0                # vida media (min) del EWMA rápido; 3-5 min predice mejor las compras orgánicas
    trend_hl_slow: float = 240.0         # vida media (min) del EWMA lento, que da la cuota habitual
    trend_a: float = 20.0                # suavizado, en eventos
    trend_scale: float = 1.0             # multiplica el peso calibrado de la tendencia (para probar variantes en vivo)
    session_mix: bool = False            # True: la sesión como mezcla de peso fijo en P(c). Probado el sábado con
                                         # 877 mil impresiones: no mejora (logver -4194,5 frente a -4193,6) -> desactivado
    # --- valores por defecto si un parámetro no se puede estimar
    default_beta_sale: float = 0.10
    default_beta_seen: float = 1.0
    default_lam_cat: float = 5.0
    default_beta_cart: float = 1.5
    # --- estimación offline (días desde HIST_T0)
    day_fit: int = 30                    # < 30: features · [30, 45): ajuste del logit
    day_cal: int = 45                    # >= 45: evaluación / calibración de kappa, lambda, b_seen
    clogit_n: int = 15000
    clogit_eval_n: int = 10000
    grid_kappa: tuple = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    grid_lambda: tuple = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    grid_seen: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    # --- operación
    n_max_products: int = 8000
    ns_spare: int = 16                   # hueco para subcategorías nuevas en vivo
    nc_spare: int = 4
    request_poll_s: float = 0.05
    feedback_poll_s: float = 2.0
    events_poll_s: float = 10.0
    catalog_poll_s: float = 0.75
    stock_reconcile_s: float = 60.0
    metrics_s: float = 60.0
    deadline_s: float = 2.9


CFG = Config()
log = logging.getLogger("v7")

# Afinidad usuario-producto: multiplicadores de la probabilidad jerárquica (fuera del anidamiento).
# Valores estimados el sábado con 877.350 impresiones propias (614 compras); se reestiman con --calibrate.
#   theta : exponente de la probabilidad jerárquica (1 = bien calibrada; estimado 1,03 ± 0,04)
#   g_lc  : producto que el usuario ha metido en el carrito durante el stream (y no ha comprado)
#   (v5: sin la corrección d_lc del carrito de hoy; con los gustos latentes se estimaba en 0,00 ± 0,14)
#   g_hc  : producto en el carrito histórico, sin comprar
#   g_vu / g_vf / g_vo: único producto visto en la sesión / primer visto de varios / los demás vistos
P_REF = 1e-3
# Gustos latentes. Cada producto lleva en su descripción 2 de estas 20 etiquetas; el usuario tiene un perfil de
# etiquetas (Dirichlet-multinomial) y un vector de gustos (factores de co-compra) por cada "cesta":
# historia (compras + carrito), sesión (recent_views), carrito de hoy y vistas de hoy.
TAGS = ("compact and lightweight", "a family-size format", "a luxury feel", "a sporty style", "premium materials",
        "a minimalist design", "smart app control", "bold colours", "organic ingredients",
        "made from recycled materials", "designed for kids", "wireless connectivity", "great value for money",
        "built for professionals", "whisper-quiet operation", "a handmade finish", "a vintage look",
        "a waterproof shell", "extra durability", "a travel-ready build")
K_TAGS = len(TAGS)
N_BASKETS = 4
TASTE_NAMES = ("cf_hist", "cf_sesion", "cf_carrito_hoy", "cf_vistas_hoy", "tag_hist", "tag_sesion", "tag_carrito_hoy",
               "tag_vistas_hoy", "tendencia")
N_TASTE = 2 * N_BASKETS + 1                                # 4 de co-compra, 4 de etiquetas y la tendencia de la subcategoría
PARAMS_V6 = "params_v7.json"                               # calibración propia del v7 (no pisa la de v4/v5/v6)


class Trend:
    """Tendencia por subcategoría. Dos EWMA de eventos (vistas, carrito y compras) por minuto: uno rápido
    (lo reciente) y uno lento (la cuota habitual). tendencia_s = (rápido_s + a) / (rápido_total · cuota_s + a)."""

    def __init__(self, ns: int, hl_fast: float, hl_slow: float, a: float):
        self.fast, self.slow, self.cur = np.zeros(ns), np.zeros(ns), np.zeros(ns)
        self.df, self.ds, self.a = 0.5 ** (1.0 / hl_fast), 0.5 ** (1.0 / hl_slow), a
        self.minute: int | None = None

    def add(self, s: int, ts: float, n: float = 1.0) -> None:
        m = int(ts // 60)
        if self.minute is None:
            self.minute = m
        k = m - self.minute
        if k > 0:                                            # se cierra el minuto en curso y se decae lo transcurrido
            self.fast = (self.fast * self.df + self.cur) * self.df ** (k - 1)
            self.slow = (self.slow * self.ds + self.cur) * self.ds ** (k - 1)
            self.cur = np.zeros_like(self.cur)
            self.minute = m
        self.cur[s] += n

    def log_trend(self) -> np.ndarray:
        tot_slow = self.slow.sum()
        if tot_slow <= 0:
            return np.zeros_like(self.fast)
        f = self.fast * self.df + self.cur
        return np.log((f + self.a) / (f.sum() * self.slow / tot_slow + self.a))
AFF: dict | None = None                  # factores de co-compra y etiquetas (load_affinity)
TILT_DEFAULT = dict(theta=0.55, g_lc=0.9, g_hc=0.7, g_vu=2.3, g_vf=1.1, g_vo=0.8)
TILT_BOUNDS = dict(theta=(0.4, 1.4), g_lc=(0.0, 6.0), g_hc=(0.0, 4.0), g_vu=(0.0, 5.0), g_vf=(0.0, 4.0), g_vo=(0.0, 4.0))


# =============================================================================
# 1. UTILIDADES
# =============================================================================
def setup_logging(verbose: bool = False) -> None:
    LOGS.mkdir(exist_ok=True)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = RotatingFileHandler(LOGS / "recommender-v7.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.handlers[:] = [fh, sh]


def load_env() -> None:
    for p in [HERE / ".env", HERE.parent / ".env", HERE.parent / "datathon-fme-2026" / ".env", Path(".env")]:
        if p.exists():
            for line in p.read_text().splitlines():
                k, _, v = line.partition("=")
                k = k.strip()
                if k and not k.startswith("#"):
                    os.environ.setdefault(k, v.strip())


_FRAC = re.compile(r"\d+")


def parse_ts(s) -> float | None:
    """'2026-11-14 13:18:23.883000000' (UTC) -> epoch en segundos."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace("T", " ")
    if len(s) < 19:
        return None
    try:
        x = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    if len(s) > 20 and s[19] == ".":
        m = _FRAC.match(s, 20)
        if m:
            frac = m.group(0)[:6]
            x += int(frac) / 10 ** len(frac)
    return x


def fmt_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def q_list(xs) -> str:
    return ",".join("'" + str(x).replace("'", "") + "'" for x in xs)


class TTLCache:
    """Diccionario con caducidad y tamaño máximo."""

    def __init__(self, ttl: float, maxsize: int = 300_000):
        self.ttl, self.maxsize = ttl, maxsize
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def add(self, k, v=True) -> None:
        with self.lock:
            self.d[k] = (v, time.time() + self.ttl)
            self.d.move_to_end(k)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)

    def claim(self, k) -> bool:
        """Añade k solo si no está (atómico): True si este hilo se la queda."""
        with self.lock:
            it = self.d.get(k)
            if it is not None and it[1] >= time.time():
                return False
            self.d[k] = (True, time.time() + self.ttl)
            self.d.move_to_end(k)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)
            return True

    def get(self, k, default=None):
        it = self.d.get(k)
        if it is None or it[1] < time.time():
            return default
        return it[0]

    def __contains__(self, k) -> bool:
        return self.get(k, None) is not None

    def purge(self) -> None:
        now = time.time()
        with self.lock:
            for k in [k for k, (_, exp) in self.d.items() if exp < now]:
                self.d.pop(k, None)


def normalize_views(rv) -> list[str]:
    """recent_views -> product_id en orden cronológico, sin duplicados (se queda la última aparición)."""
    if rv is None:
        return []
    if isinstance(rv, str):
        try:
            rv = json.loads(rv)
        except ValueError:
            return []
    if not isinstance(rv, list):
        return []
    out = []
    for x in rv:
        if isinstance(x, dict):
            pid = x.get("product_id")
        elif isinstance(x, (list, tuple)) and x:
            pid = x[0]
        elif isinstance(x, str):
            pid = x
        else:
            continue
        if pid:
            out.append(str(pid))
    last = {pid: k for k, pid in enumerate(out)}
    return [out[k] for k in sorted(last.values())]


def price_positions(price: np.ndarray, sub: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cuantil de precio dentro de la subcategoría, invertido: z_i = rango_i / nº de productos, con el rango
    contado desde el más caro (cerca de 0 = los más caros, 1 = el más barato). Los empates reciben su rango
    medio. Los productos no válidos quedan en 0,5."""
    upos = np.full(price.size, 0.5)
    order = np.lexsort((-price, sub))                  # por subcategoría y precio descendente
    order = order[valid[order]]
    s_sorted = sub[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for idx in np.split(order, bounds):
        if idx.size == 0:
            continue
        p = price[idx]
        ranks = np.arange(1, idx.size + 1, dtype=float)
        _, inv, cnt = np.unique(-p, return_inverse=True, return_counts=True)
        upos[idx] = (np.bincount(inv, weights=ranks) / cnt)[inv] / idx.size
    return upos


def pava_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    blocks = [[float(y[i]), float(max(w[i], 1e-9)), 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            tw = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for v, _, cnt in blocks:
        out += [v] * cnt
    return np.array(out)


# =============================================================================
# 2. RAWTREE HTTP
# =============================================================================
class RawTree:
    def __init__(self, api: str, key: str, team: str):
        self.api, self.key, self.team = api, key, team
        self.local = threading.local()
        self.errors = 0

    def _session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["Authorization"] = f"Bearer {self.key}"
            self.local.s = s
        return s

    def query(self, sql: str, timeout: float = 10.0) -> list[dict]:
        try:
            r = self._session().post(f"{self.api}/v1/query", json={"sql": sql}, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json().get("data", [])
        except Exception:
            self.errors += 1
            raise

    def insert(self, table: str, rows: list[dict], timeout: float = 5.0) -> int:
        if not rows:
            return 0
        r = self._session().post(f"{self.api}/v1/tables/{table}", params={"database": self.team},
                                 json=rows, timeout=timeout)
        if r.status_code >= 400:
            self.errors += 1
            raise RuntimeError(f"insert {table}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            j = r.json()
            return int(j.get("inserted", j.get("written_rows", len(rows))))
        except Exception:  # noqa: BLE001
            return len(rows)


def make_rt() -> RawTree:
    load_env()
    if "RAWTREE_API_KEY" not in os.environ or "TEAM_ID" not in os.environ:
        sys.exit("Faltan TEAM_ID / RAWTREE_API_KEY (fichero .env)")
    return RawTree(os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
                   os.environ["RAWTREE_API_KEY"], os.environ["TEAM_ID"])


# =============================================================================
# 3. ESTIMACIÓN OFFLINE (se guarda en .cache/)
# =============================================================================
def estimate_position_bias(rows: list[dict], index: dict) -> np.ndarray:
    """Atención por slot w_k con el modelo de examen P(clic | i, k) = w_k · r_i (EM) + monotonía (PAVA).
    Solo se usa para calibrar los pesos de sesión y b_seen con las listas del carrusel."""
    n = np.zeros((len(index), 10))
    c = np.zeros((len(index), 10))
    for r in rows:
        i, k = index.get(r["product_id"]), int(r["slot"]) - 1
        if i is not None and 0 <= k < 10:
            n[i, k] += r["imp"]
            c[i, k] += r["clk"]
    ctr = (c.sum(0) + 0.5) / (n.sum(0) + 0.5)
    w = ctr / ctr[0]
    for _ in range(50):
        r_i = c.sum(1) / np.maximum((n * w).sum(1), 1e-9)
        w_new = c.sum(0) / np.maximum((n * r_i[:, None]).sum(0), 1e-12)
        w_new = np.clip(w_new / w_new[0], 1e-3, 1.0)
        done = np.abs(w_new - w).sum() < 1e-6
        w = w_new
        if done:
            break
    w = np.clip(pava_nonincreasing(w, n.sum(0)), 0.03, 1.0)
    w[0] = 1.0
    return w


def gamma_popularity(C: np.ndarray, T: np.ndarray | float, m_sub: np.ndarray, sub: np.ndarray, b0: float):
    """Cuota de compras de cada producto: tasa ~ Gamma(m_s·b0 + C_i, b0 + T_i).
    C_i = compras del producto, T_i = compras totales de la tienda mientras el producto existía,
    m_s = cuota mediana de su subcategoría. Devuelve (forma, tasa); la media es forma / tasa."""
    return m_sub[sub] * b0 + C, b0 + T


def subcat_median_share(C: np.ndarray, T: float, sub: np.ndarray, ns: int) -> np.ndarray:
    share = C / max(T, 1.0)
    floor = 0.5 / max(T, 1.0)
    m = np.full(ns, floor)
    for s in range(ns):
        v = share[sub == s]
        if v.size:
            m[s] = max(float(np.median(v)), floor)
    return m


def clogit_newton(X: np.ndarray, Y: np.ndarray, M: np.ndarray, cols: list[int], iters: int = 30, l2: float = 1e-3):
    """Máxima verosimilitud del logit condicional (Newton). X: (elecciones, opciones, features)."""
    Xs = X[:, :, cols].astype(np.float64)
    d = len(cols)
    beta = np.zeros(d)
    H = np.eye(d)
    for _ in range(iters):
        eta = np.where(M, Xs @ beta, -1e9)
        eta -= eta.max(1, keepdims=True)
        p = np.exp(eta) * M
        p /= p.sum(1, keepdims=True)
        xbar = (p[:, :, None] * Xs).sum(1)
        g = ((Y[:, :, None] * Xs).sum(1) - xbar).sum(0) - l2 * beta
        H = (Xs * p[:, :, None]).reshape(-1, d).T @ Xs.reshape(-1, d) - xbar.T @ xbar + l2 * np.eye(d)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-6:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def clogit_eval(X, Y, M, cols, beta):
    eta = np.where(M, X[:, :, cols].astype(np.float64) @ beta, -1e9)
    eta -= eta.max(1, keepdims=True)
    p = np.exp(eta) * M
    p /= p.sum(1, keepdims=True)
    ll = float(np.log((p * Y).sum(1) + 1e-12).mean())
    top = np.argsort(-p, 1)[:, :3]
    return ll, float(np.take_along_axis(Y, top, 1).sum(1).mean())


SPECIAL_K = 48                              # máximo de productos "especiales" por subcategoría y usuario


def slate_arrays(rows: list[dict], index: dict, urow: dict, N: np.ndarray, ru: np.ndarray, ev0: np.ndarray,
                 pos: np.ndarray, sub: np.ndarray, sub_cat: np.ndarray, pi_c: np.ndarray, pi_s: np.ndarray,
                 prm: dict, cfg: Config, sets_fn) -> dict | None:
    """Para cada compra del carrusel, las piezas del score de cada producto mostrado (hasta 10).
    sets_fn(r) -> (carrito, comprados): productos que el usuario había metido en el carrito / comprado
    ANTES de esa impresión. Los "especiales" (vistos en la sesión, en carrito o comprados) de cada
    subcategoría se guardan aparte para poder recalcular el denominador del logit al variar los betas."""
    n, ns, nc = sub.size, pi_s.size, pi_c.size
    keys = ("nus", "Ss", "nuc", "Sc", "pis", "pic", "base", "seen", "cart", "bought", "Zb", "mask",
            "spb", "sps", "spc", "spt")
    out: dict = {k: [] for k in keys}
    ys = []
    for r in rows:
        shown = [index.get(x, -1) for x in (r["shown"] or [])[:10]]
        y = index.get(r["purchased"], -1)
        if y < 0 or y not in shown:
            continue
        sh = np.array(shown + [-1] * (10 - len(shown)))
        mask = sh >= 0                                     # productos nuevos fuera del catálogo: se ignoran
        if mask.sum() < 2:
            continue
        shc = np.where(mask, sh, 0)
        row = urow.get(r["u"])
        n_us = N[row].astype(np.float64)[:ns] if row is not None else np.zeros(ns)
        g = (prm["b_p0"] + (prm["b_u"] * ru[row] if row is not None else 0.0)
             + prm["b_dev"].get(r.get("d") or "", 0.0) + prm["b_reg"].get(r.get("reg") or "", 0.0))
        base = ev0 * np.exp(pos * g)
        views = [index[v] for v in normalize_views(r["recent_views"]) if v in index][-cfg.max_session_views:]
        S_s = np.zeros(ns)
        for k_, v in enumerate(views):
            S_s[sub[v]] += cfg.rho_session ** (len(views) - 1 - k_)
        carted, bought = sets_fn(r)
        vset, bset = set(views), set(bought)
        cset = set(carted) - bset                          # en el carrito y todavía sin comprar
        by_sub: dict = defaultdict(list)
        for i in vset | cset | bset:
            by_sub[int(sub[i])].append(i)
        sj = sub[shc]
        cj = sub_cat[sj]
        sp = np.zeros((4, 10, SPECIAL_K))
        for j in range(10):
            if mask[j]:
                for k_, i in enumerate(by_sub.get(int(sj[j]), ())[:SPECIAL_K]):
                    sp[:, j, k_] = (base[i], i in vset, i in cset, i in bset)
        n_uc = np.bincount(sub_cat, weights=n_us, minlength=nc)
        S_c = np.bincount(sub_cat, weights=S_s, minlength=nc)
        Zb = np.bincount(sub, weights=base, minlength=ns)
        vals = (n_us[sj], S_s[sj], n_uc[cj], S_c[cj], pi_s[sj], pi_c[cj], base[shc],
                np.array([i in vset for i in shc.tolist()], float), np.array([i in cset for i in shc.tolist()], float),
                np.array([i in bset for i in shc.tolist()], float), Zb[sj], mask.astype(float),
                sp[0], sp[1], sp[2], sp[3])
        for k, v in zip(keys, vals):
            out[k].append(v)
        ys.append(shown.index(y))
    if len(ys) < 300:
        return None
    A = {k: np.array(v) for k, v in out.items()}
    A["y"] = np.array(ys)
    return A


def slate_ll(A: dict, w_pos: np.ndarray, kc: float, ks: float, lc: float, ls: float, bs: float,
             bc: float = 0.0, bb: float = 0.0):
    """Verosimilitud de la compra dentro de la lista mostrada. Score de cada producto:
    w_k · P(c) · P(s|c) · P(i|s); el denominador común de P(c) se cancela al normalizar en la lista.
    bs, bc, bb: coeficientes de "visto en la sesión", "en el carrito sin comprar" y "ya comprado"."""
    boost = np.exp(bs * A["seen"] + bc * A["cart"] + bb * A["bought"])
    Z = A["Zb"] + (A["spb"] * (np.exp(bs * A["sps"] + bc * A["spc"] + bb * A["spt"]) - 1.0)).sum(-1)
    sc = (w_pos * (kc * A["pic"] + A["nuc"] + lc * A["Sc"])
          * (ks * A["pis"] + A["nus"] + ls * A["Ss"]) / (ks + A["nuc"] + ls * A["Sc"])
          * A["base"] * boost / np.maximum(Z, 1e-300)) * A["mask"]
    rows = np.arange(A["y"].size)
    return float(np.log(sc[rows, A["y"]] / sc.sum(1)).mean()), float((sc.argmax(1) == A["y"]).mean())


def coord_ascent(f, params: dict, grids: dict, rounds: int = 4):
    """Maximiza f(**params) recorriendo cada parámetro por su rejilla (ascenso por coordenadas)."""
    best = f(**params)
    for _ in range(rounds):
        changed = False
        for k, grid in grids.items():
            val, v = max((f(**{**params, k: v}), v) for v in grid)
            if val > best + 1e-9:
                best, params[k], changed = val, v, True
        if not changed:
            break
    return params, best


def build_priors(rt: RawTree, cfg: Config, cut_day: int | None = None) -> dict:
    """Descarga el histórico y estima todas las piezas del modelo. Con cut_day, solo usa los
    días anteriores (para la evaluación offline)."""
    t_start = time.time()
    rng = np.random.default_rng(0)
    day_expr = f"dateDiff('day', toDateTime64('{HIST_T0}', 3), ts::DateTime64(3))"
    cut = f" AND {day_expr} < {cut_day}" if cut_day else ""
    log.info("[build] estimando el modelo con el histórico%s", f" (días < {cut_day})" if cut_day else " completo")

    # --- 1. catálogo
    rows = rt.query("""SELECT product_id::String AS p, toString(category) AS cat, toString(subcategory) AS sub,
        price_eur::Float64 AS price FROM shop.catalog ORDER BY p""", 60)
    ids = [r["p"] for r in rows]
    index = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    cats = sorted({r["cat"] for r in rows})
    subs = sorted({r["sub"] for r in rows})
    nc, ns = len(cats), len(subs)
    cat = np.array([cats.index(r["cat"]) for r in rows])
    sub = np.array([subs.index(r["sub"]) for r in rows])
    price = np.array([float(r["price"] or 0) for r in rows])
    sub_cat = np.zeros(ns, int)
    sub_cat[sub] = cat
    upos = price_positions(price, sub, np.ones(n, bool))
    pos = upos                                             # z_i; centrarlo no cambiaría nada (se cancela en el logit)
    log.info("[build] catálogo: %d productos, %d categorías, %d subcategorías", n, nc, ns)

    # --- 2. atención por posición (solo para calibrar con el log del carrusel)
    sc = rt.query(f"""SELECT arrayElement(ids, slot) AS product_id, slot, count() AS imp,
        countIf(clicked = arrayElement(ids, slot)) AS clk
        FROM (SELECT shown::Array(String) AS ids, ifNull(toString(clicked), '') AS clicked
              FROM shop.history_sessions WHERE 1 {cut})
        ARRAY JOIN arrayEnumerate(ids) AS slot GROUP BY product_id, slot""", 120)
    w_pos = estimate_position_bias(sc, index)
    log.info("[build] atención por slot w_k = %s", np.round(w_pos, 3).tolist())

    # --- 3. recuentos fraccionados usuario × subcategoría (todo, pasado y compras futuras)
    subs_lit = "[" + q_list(subs) + "]"
    acc = []
    for k in range(4):
        acc += rt.query(f"""SELECT u, groupArray(s) AS ss, groupArray(w) AS ws, groupArray(wp) AS wps,
              groupArray(pf) AS pfs FROM (
            SELECT e.u AS u, indexOf({subs_lit}, c.sub) AS s, sum(e.w) AS w,
                   sumIf(e.w, e.day < {cfg.day_cal}) AS wp,
                   countIf(e.t = 'purchase' AND e.day >= {cfg.day_cal}) AS pf
            FROM (SELECT user_id::String AS u, product_id::String AS pid, event_type::String AS t,
                         multiIf(event_type::String = 'purchase', {cfg.w_purchase},
                                 event_type::String = 'add_to_cart', {cfg.w_cart}, {cfg.w_view}) AS w,
                         {day_expr} AS day
                  FROM shop.history_events WHERE cityHash64(user_id::String) % 4 = {k} {cut}) AS e
            INNER JOIN (SELECT product_id::String AS pid, subcategory::String AS sub FROM shop.catalog) AS c
              ON c.pid = e.pid
            GROUP BY u, s) GROUP BY u""", 300)
    users = [r["u"] for r in acc]
    urow = {u: i for i, u in enumerate(users)}
    NU = len(users)
    N_all = np.zeros((NU, ns), np.float32)
    N_past = np.zeros((NU, ns), np.float32)
    F_fut = np.zeros((NU, ns), np.float32)
    for i, r in enumerate(acc):
        ss = np.array(r["ss"], int) - 1
        ok = ss >= 0
        N_all[i, ss[ok]] = np.array(r["ws"], float)[ok]
        N_past[i, ss[ok]] = np.array(r["wps"], float)[ok]
        F_fut[i, ss[ok]] = np.array(r["pfs"], float)[ok]
    del acc
    log.info("[build] %d usuarios con actividad en el histórico", NU)

    # --- 4. compras (para popularidad, perfil de precio, productos comprados y el logit)
    rows = rt.query(f"""SELECT user_id::String AS u, product_id::String AS p, device::String AS d,
        region::String AS r, {day_expr} AS day
        FROM shop.history_events WHERE event_type::String = 'purchase' {cut}""", 300)
    rows = [r for r in rows if r["p"] in index and r["u"] in urow]
    up = np.array([urow[r["u"]] for r in rows])
    pi = np.array([index[r["p"]] for r in rows])
    pday = np.array([int(r["day"]) for r in rows])
    devs = sorted({r["d"] for r in rows})
    regs = sorted({r["r"] for r in rows})
    pdev = np.array([devs.index(r["d"]) for r in rows])
    preg = np.array([regs.index(r["r"]) for r in rows])
    del rows
    log.info("[build] %d compras · dispositivos %s · %d regiones", up.size, devs, len(regs))

    def period(mask: np.ndarray) -> dict:
        C = np.bincount(pi[mask], minlength=n).astype(float)
        T = float(mask.sum())
        m_sub = subcat_median_share(C, T, sub, ns)
        shape, rate = gamma_popularity(C, T, m_sub, sub, cfg.b0)
        popm = shape / rate
        E_s = np.bincount(sub, weights=popm * upos, minlength=ns) / np.maximum(np.bincount(sub, weights=popm, minlength=ns), 1e-30)
        resid = upos[pi] - E_s[sub[pi]]
        return dict(C=C, T=T, m_sub=m_sub, lpop=np.log(popm), E_s=E_s,
                    ru_sum=np.bincount(up[mask], weights=resid[mask], minlength=NU),
                    ru_n=np.bincount(up[mask], minlength=NU).astype(float))

    FULL = period(np.ones(up.size, bool))
    A = period(pday < cfg.day_fit)
    B = period(pday < cfg.day_cal)
    bought_full: dict[int, set] = defaultdict(set)
    bought_A: dict[int, set] = defaultdict(set)
    bought_past: dict[int, set] = defaultdict(set)
    for u_, p_, d_ in zip(up.tolist(), pi.tolist(), pday.tolist()):
        bought_full[u_].add(p_)
        if d_ < cfg.day_fit:
            bought_A[u_].add(p_)
        if d_ < cfg.day_cal:
            bought_past[u_].add(p_)

    # productos que cada usuario ha metido en el carrito (afinidad usuario-producto)
    carted_full: dict[int, set] = defaultdict(set)
    carted_past: dict[int, set] = defaultdict(set)
    n_pairs = 0
    for k in range(2):
        for r in rt.query(f"""SELECT u, groupArray(p) AS ps, groupArray(d) AS ds FROM (
                SELECT user_id::String AS u, product_id::String AS p, min({day_expr}) AS d
                FROM shop.history_events WHERE event_type::String = 'add_to_cart'
                  AND cityHash64(user_id::String) % 2 = {k} {cut} GROUP BY u, p) GROUP BY u""", 300):
            row = urow.get(r["u"])
            if row is None:
                continue
            for p_, d_ in zip(r["ps"], r["ds"]):
                i = index.get(p_)
                if i is not None:
                    carted_full[row].add(i)
                    n_pairs += 1
                    if d_ < cfg.day_cal:
                        carted_past[row].add(i)
    n_open = sum(len(v - bought_full.get(u_, set())) for u_, v in carted_full.items())
    log.info("[build] carritos: %d pares usuario-producto, %d sin comprar todavía", n_pairs, n_open)

    # --- 5. priors poblacionales de categoría y subcategoría (fijas)
    c_sub = np.bincount(sub[pi], minlength=ns) + 0.5
    c_cat = np.bincount(sub_cat, weights=c_sub, minlength=nc)
    pi_c = c_cat / c_cat.sum()
    pi_s = c_sub / c_cat[sub_cat]                          # P(s | c): suma 1 dentro de cada categoría

    # --- 6. nivel 3: logit condicional con las compras orgánicas
    #        features con días < day_fit · ajuste [day_fit, day_cal) · evaluación >= day_cal
    opts = [np.flatnonzero(sub == s) for s in range(ns)]
    maxn = max(o.size for o in opts)
    names = ["log_pop", "pos", "pos_x_perfil"] + [f"pos_x_{d}" for d in devs[1:]] + [f"pos_x_{r}" for r in regs[1:]]
    D = len(names)
    ruA = A["ru_sum"] / (A["ru_n"] + cfg.r_shrink)

    def feats(idxs: np.ndarray):
        X = np.zeros((idxs.size, maxn, D), np.float32)
        Y = np.zeros((idxs.size, maxn), np.float32)
        M = np.zeros((idxs.size, maxn), bool)
        keep = np.zeros(idxs.size, bool)
        for row, k in enumerate(idxs.tolist()):
            o = opts[sub[pi[k]]]
            bb = bought_A.get(int(up[k]))
            if bb:
                o = o[~np.isin(o, list(bb))]
            hit = np.flatnonzero(o == pi[k])
            if hit.size == 0 or o.size < 3:
                continue                                   # recompra o subcategoría casi vacía
            m_ = o.size
            ps = pos[o]
            X[row, :m_, 0] = A["lpop"][o]
            X[row, :m_, 1] = ps
            X[row, :m_, 2] = ps * ruA[up[k]]
            if pdev[k] > 0:
                X[row, :m_, 2 + pdev[k]] = ps
            if preg[k] > 0:
                X[row, :m_, 2 + (len(devs) - 1) + preg[k]] = ps
            Y[row, hit[0]] = 1
            M[row, :m_] = True
            keep[row] = True
        return X[keep], Y[keep], M[keep]

    i_fit = np.flatnonzero((pday >= cfg.day_fit) & (pday < cfg.day_cal))
    i_ev = np.flatnonzero(pday >= cfg.day_cal)
    i_fit = rng.choice(i_fit, min(cfg.clogit_n, i_fit.size), replace=False)
    i_ev = rng.choice(i_ev, min(cfg.clogit_eval_n, i_ev.size), replace=False) if i_ev.size else i_ev
    Xf, Yf, Mf = feats(i_fit)
    all_cols = list(range(D))
    beta, se = clogit_newton(Xf, Yf, Mf, all_cols)
    diag: dict = {"clogit": {nm: (round(float(b), 4), round(float(b / s), 1)) for nm, b, s in zip(names, beta, se)}}
    if i_ev.size:
        Xe, Ye, Me = feats(i_ev)
        b_pop_only, _ = clogit_newton(Xf, Yf, Mf, [0])
        ll0, hr0 = clogit_eval(Xe, Ye, Me, [0], b_pop_only)
        ll1, hr1 = clogit_eval(Xe, Ye, Me, all_cols, beta)
        diag["clogit_eval"] = {"n_fit": int(Yf.shape[0]), "n_eval": int(Ye.shape[0]),
                               "opciones_media": float(Me.sum(1).mean()),
                               "LL_azar": float(-np.log(Me.sum(1)).mean()),
                               "LL_pop": ll0, "HR3_pop": hr0, "LL_modelo": ll1, "HR3_modelo": hr1}
        log.info("[build] logit de producto (holdout): solo popularidad LL=%.4f HR@3=%.3f · modelo LL=%.4f HR@3=%.3f",
                 ll0, hr0, ll1, hr1)
        del Xe, Ye, Me
    del Xf, Yf, Mf
    b_pop, b_p0, b_u = float(beta[0]), float(beta[1]), float(beta[2])
    b_dev = {d: (float(beta[2 + k]) if k > 0 else 0.0) for k, d in enumerate(devs)}
    b_reg = {r: (float(beta[2 + (len(devs) - 1) + k]) if k > 0 else 0.0) for k, r in enumerate(regs)}
    log.info("[build] b_pop=%.3f b_p0=%.3f b_u=%.2f b_dev=%s", b_pop, b_p0, b_u,
             {k: round(v, 2) for k, v in b_dev.items()})

    # --- 7. kappa_c y kappa_s: verosimilitud de las compras futuras dado el pasado
    Mcs = np.zeros((ns, nc))
    Mcs[np.arange(ns), sub_cat] = 1.0
    has_f = F_fut.sum(1) > 0
    Wp, Ff = N_past[has_f].astype(np.float64), F_fut[has_f].astype(np.float64)
    Wc, Fc = Wp @ Mcs, Ff @ Mcs
    kappa_c = kappa_s = 5.0
    kdiag = {}
    if has_f.sum() > 100:
        ll_c = {k: float((Fc * np.log((Wc + k * pi_c) / (Wc.sum(1, keepdims=True) + k))).sum() / Fc.sum())
                for k in cfg.grid_kappa}
        ll_s = {k: float((Ff * np.log((Wp + k * pi_s) / (Wc[:, sub_cat] + k))).sum() / Ff.sum())
                for k in cfg.grid_kappa}
        kappa_c, kappa_s = max(ll_c, key=ll_c.get), max(ll_s, key=ll_s.get)
        kdiag = {"usuarios": int(has_f.sum()), "LL_categoria": ll_c, "LL_subcategoria": ll_s}
        log.info("[build] kappa_c=%s (LL %.4f) · kappa_s=%s (LL %.4f) · %d usuarios", kappa_c, ll_c[kappa_c],
                 kappa_s, ll_s[kappa_s], int(has_f.sum()))
    diag["kappa"] = kdiag
    del Wp, Ff, Wc, Fc

    # --- 8. sesión y afinidad usuario-producto: verosimilitud de la compra dentro de la lista mostrada
    #   8a. log del carrusel antiguo (días >= day_cal): lam_sub, b_seen, b_cart y b_bought. Aquella lista
    #       solo tenía productos de la categoría de la sesión: no informa del nivel de categoría.
    #   8b. nuestro feedback en vivo (listas con varias categorías y el simulador que puntúa):
    #       kappa_c, lam_cat y, de nuevo, b_seen, b_cart y b_bought (en vivo lo ya comprado satura).
    prm0 = dict(b_p0=b_p0, b_u=b_u, b_dev=b_dev, b_reg=b_reg)
    lam_sub, b_seen, b_cart, lam_cat = 1.0, cfg.default_beta_seen, cfg.default_beta_cart, cfg.default_lam_cat
    grid_b = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    grid_neg = (-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0)
    cols = """toString(context.device) AS d, toString(context.region) AS reg, recent_views,
        shown::Array(String) AS shown, ifNull(toString(purchased), '') AS purchased"""
    try:
        cal = rt.query(f"""SELECT user_id::String AS u, {cols} FROM shop.history_sessions
            WHERE purchased != '' AND {day_expr} >= {cfg.day_cal} {cut}""", 300)

        def sets_old(r):
            row = urow.get(r["u"])
            return (carted_past.get(row, ()), bought_past.get(row, ())) if row is not None else ((), ())

        A8 = slate_arrays(cal, index, urow, N_past, B["ru_sum"] / (B["ru_n"] + cfg.r_shrink),
                          np.exp(b_pop * B["lpop"]), pos, sub, sub_cat, pi_c, pi_s, prm0, cfg, sets_old)
        if A8 is not None:
            f8 = lambda ls, bs, bc, bb: slate_ll(A8, w_pos, kappa_c, kappa_s, ls, ls, bs, bc, bb)[0]  # noqa: E731
            best, ll_best = coord_ascent(f8, dict(ls=1.0, bs=1.0, bc=1.5, bb=2.0),
                                         dict(ls=cfg.grid_lambda, bs=grid_b, bc=grid_b, bb=grid_b))
            lam_sub, b_seen, b_cart = best["ls"], best["bs"], best["bc"]
            y_ = A8["y"]
            hr = slate_ll(A8, w_pos, kappa_c, kappa_s, best["ls"], best["ls"], best["bs"], best["bc"], best["bb"])[1]
            diag["carrusel_antiguo"] = {
                "n": int(y_.size), "params": dict(best), "LL_modelo": ll_best,
                "LL_solo_posicion": float(np.log(w_pos[y_] / w_pos.sum()).mean()),
                "LL_sin_afinidad": f8(best["ls"], best["bs"], 0.0, 0.0), "LL_sin_sesion": f8(0.0, 0.0, best["bc"], best["bb"]),
                "top1_modelo": hr, "top1_slot1": float((y_ == 0).mean()),
                "pct_compras_ya_comprado": float(A8["bought"][np.arange(y_.size), y_].mean()),
                "pct_compras_en_carrito": float(A8["cart"][np.arange(y_.size), y_].mean())}
            log.info("[build] carrusel antiguo (%d compras): lam_sub=%s b_seen=%s b_cart=%s b_bought=%s · LL solo "
                     "posición %.4f -> sin afinidad %.4f -> modelo %.4f · acierta el comprado %.3f (slot 1: %.3f)",
                     y_.size, best["ls"], best["bs"], best["bc"], best["bb"], diag["carrusel_antiguo"]["LL_solo_posicion"],
                     diag["carrusel_antiguo"]["LL_sin_afinidad"], ll_best, hr, float((y_ == 0).mean()))
    except Exception as e:  # noqa: BLE001
        log.exception("[build] calibración con el carrusel antiguo fallida; valores por defecto: %s", e)
    live_params = CACHE / "priors_live_params.json"
    if cut_day is None:
        try:
            cal = rt.query(f"""SELECT r.user_id::String AS u, toString(r.context.device) AS d,
                toString(r.context.region) AS reg, r.recent_views AS recent_views, f.shown::Array(String) AS shown,
                toString(f.purchased) AS purchased, toString(f.ts::DateTime64(3)) AS fts
                FROM {rt.team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
                WHERE ifNull(toString(f.purchased), '') != ''""", 300)
            us = sorted({r["u"] for r in cal})
            live_ev: dict = defaultdict(list)              # usuario -> [(producto, tipo, primer ts)]
            for k in range(0, len(us), 300):
                for x in rt.query(f"""SELECT user_id::String AS u, product_id::String AS p, event_type::String AS t,
                        toString(min(ts::DateTime64(3))) AS t0 FROM shop.events
                        WHERE event_type::String IN ('add_to_cart', 'purchase')
                          AND user_id::String IN ({q_list(us[k:k + 300])}) GROUP BY u, p, t""", 120):
                    i = index.get(x["p"])
                    if i is not None:
                        live_ev[x["u"]].append((i, x["t"], parse_ts(x["t0"]) or 0.0))

            for r in cal:                                  # compras hechas a través de nuestras propias listas
                i = index.get(r["purchased"])
                if i is not None:
                    live_ev[r["u"]].append((i, "purchase", parse_ts(r["fts"]) or 0.0))

            def sets_live(r):
                row = urow.get(r["u"])
                c = set(carted_full.get(row, ())) if row is not None else set()
                b_ = set(bought_full.get(row, ())) if row is not None else set()
                t_imp = (parse_ts(r["fts"]) or 0.0) - 6.0      # la impresión es unos segundos anterior al feedback
                for i, t, t0 in live_ev.get(r["u"], ()):
                    if t0 < t_imp:
                        (c if t == "add_to_cart" else b_).add(i)
                return c, b_

            A9 = slate_arrays(cal, index, urow, N_all, FULL["ru_sum"] / (FULL["ru_n"] + cfg.r_shrink),
                              np.exp(b_pop * FULL["lpop"]), pos, sub, sub_cat, pi_c, pi_s, prm0, cfg, sets_live)
            if A9 is not None:
                f9 = lambda kc, ks, lc, bs, bc, bb: slate_ll(A9, w_pos, kc, ks, lc, lam_sub, bs, bc, bb)[0]  # noqa: E731
                best, ll_best = coord_ascent(
                    f9, dict(kc=kappa_c, ks=kappa_s, lc=1.0, bs=b_seen, bc=b_cart, bb=-2.0),
                    dict(kc=cfg.grid_kappa, ks=cfg.grid_kappa, lc=cfg.grid_lambda, bs=grid_b, bc=grid_b, bb=grid_neg))
                y_ = A9["y"]
                rr = np.arange(y_.size)
                hr = slate_ll(A9, w_pos, best["kc"], best["ks"], best["lc"], lam_sub, best["bs"], best["bc"], best["bb"])[1]
                diag["carrusel_vivo"] = {
                    "n": int(y_.size), "params": dict(best), "kappa_c_organico": kappa_c, "kappa_s_organico": kappa_s,
                    "LL_modelo": ll_best,
                    "LL_solo_posicion": float(np.log(w_pos[y_] / (w_pos * A9["mask"]).sum(1)).mean()),
                    "LL_sin_carrito": f9(best["kc"], best["ks"], best["lc"], best["bs"], 0.0, best["bb"]),
                    "LL_sin_sesion_cat": f9(best["kc"], best["ks"], 0.0, best["bs"], best["bc"], best["bb"]),
                    "LL_sin_saciedad": f9(best["kc"], best["ks"], best["lc"], best["bs"], best["bc"], 0.0),
                    "LL_kappa_s_organico": f9(best["kc"], kappa_s, best["lc"], best["bs"], best["bc"], best["bb"]),
                    "LL_params_carrusel_antiguo": f9(best["kc"], kappa_s, best["lc"], b_seen, b_cart, 0.0),
                    "top1_modelo": hr, "top1_slot1": float((y_ == 0).mean()),
                    "pct_compras_ya_comprado": float(A9["bought"][rr, y_].mean()),
                    "pct_compras_en_carrito": float(A9["cart"][rr, y_].mean()),
                    "pct_mostrados_en_carrito": float((A9["cart"] * A9["mask"]).sum() / A9["mask"].sum())}
                log.info("[build] carrusel en vivo (%d compras): kappa_c=%s kappa_s=%s lam_cat=%s b_seen=%s b_cart=%s "
                         "b_bought=%s · LL solo posición %.4f -> sin carrito %.4f -> modelo %.4f · acierta el comprado "
                         "%.3f (slot 1: %.3f)", y_.size, best["kc"], best["ks"], best["lc"], best["bs"], best["bc"],
                         best["bb"], diag["carrusel_vivo"]["LL_solo_posicion"], diag["carrusel_vivo"]["LL_sin_carrito"],
                         ll_best, hr, float((y_ == 0).mean()))
                kappa_c, kappa_s, lam_cat, b_seen, b_cart = best["kc"], best["ks"], best["lc"], best["bs"], best["bc"]
            else:
                log.warning("[build] pocas compras en nuestro feedback: se usan los valores del carrusel antiguo")
        except Exception as e:  # noqa: BLE001
            log.exception("[build] calibración con el feedback en vivo fallida: %s", e)
    elif live_params.exists():                             # evaluación offline: se reutiliza lo estimado en vivo
        try:
            lp = json.loads(live_params.read_text())["params"]
            kappa_c, kappa_s, lam_cat = float(lp["kappa_c"]), float(lp["kappa_s"]), float(lp["lam_cat"])
            b_seen, b_cart = float(lp["b_seen"]), float(lp["b_cart"])
            log.info("[build] kappa_c=%s kappa_s=%s lam_cat=%s b_seen=%s b_cart=%s tomados de la estimación en vivo",
                     kappa_c, kappa_s, lam_cat, b_seen, b_cart)
        except Exception as e:  # noqa: BLE001
            log.warning("[build] no se pudo leer %s: %s", live_params.name, e)

    # --- 9. b_sale: efecto de la rebaja, con el stream en vivo (el histórico no tiene cambios de precio).
    #        Las rebajas son temporales (2 h y el precio vuelve): se compara la tasa de compra de cada
    #        producto DURANTE su primera rebaja con su tasa ANTES de ella.
    b_sale = cfg.default_beta_sale
    try:
        if cut_day is None:
            far = "toDateTime64('2100-01-01 00:00:00', 3)"
            chq = f"""SELECT product_id::String AS p, min(ts::DateTime64(3)) AS t0,
                if(count() >= 2, arrayElement(arraySort(groupArray(ts::DateTime64(3))), 2), {far}) AS t1,
                argMin(price_eur::Float64, ts::DateTime64(3)) AS p1
                FROM shop.catalog_updates WHERE change::String = 'price_change' GROUP BY p"""
            ch = rt.query(f"SELECT p, toString(t0) AS t0, toString(t1) AS t1, p1 FROM ({chq})", 60)
            lim = rt.query("SELECT toString(min(ts::DateTime64(3))) AS a, toString(max(ts::DateTime64(3))) AS b FROM shop.events", 60)[0]
            cnt = rt.query(f"""SELECT e.p AS p, countIf(e.t < ch.t0) AS cb, countIf(e.t >= ch.t0 AND e.t < ch.t1) AS cd
                FROM (SELECT product_id::String AS p, ts::DateTime64(3) AS t FROM shop.events
                      WHERE event_type::String = 'purchase') AS e
                INNER JOIN ({chq}) AS ch ON ch.p = e.p GROUP BY p""", 120)
            t_a, t_b = parse_ts(lim["a"]), parse_ts(lim["b"])
            cdict = {r["p"]: (r["cb"], r["cd"]) for r in cnt}
            cb = cd = esperadas = 0.0
            n_prod = 0
            for r in ch:
                i = index.get(r["p"])
                t0, t1 = parse_ts(r["t0"]), min(parse_ts(r["t1"]) or t_b, t_b)
                if i is None or t0 is None or not float(r["p1"]) < price[i] - 1e-9 or t0 - t_a < 600 or t1 <= t0:
                    continue
                x = cdict.get(r["p"], (0, 0))
                # cada producto es su propio control: compras esperadas durante la rebaja con su tasa
                # anterior (juntar productos-hora de lotes distintos da la paradoja de Simpson)
                cb, cd, esperadas = cb + x[0], cd + x[1], esperadas + x[0] * (t1 - t0) / (t0 - t_a)
                n_prod += 1
            if cb >= 200 and cd >= 200 and esperadas > 0:
                ratio = cd / esperadas
                b_sale = float(np.clip(math.log(ratio), 0.0, 0.6))
                diag["rebaja"] = {"productos": n_prod, "compras_antes": cb, "compras_durante": cd,
                                  "compras_esperadas_durante": esperadas, "ratio": ratio}
                log.info("[build] rebajas (stream, %d productos): %d compras durante la rebaja frente a %.0f esperadas "
                         "(x%.3f) -> b_sale=%.3f", n_prod, cd, esperadas, ratio, b_sale)
    except Exception as e:  # noqa: BLE001
        log.warning("[build] b_sale no estimable ahora (%s); se usa %.2f", str(e)[:120], b_sale)

    P = dict(version=4, built_at=time.time(), cut_day=cut_day, ids=ids, cats=cats, subs=subs, cat=cat, sub=sub,
             sub_cat=sub_cat, price=price, pi_c=pi_c, pi_s=pi_s, users=users, N=N_all,
             ru_sum=FULL["ru_sum"], ru_n=FULL["ru_n"], bought={users[u]: frozenset(s) for u, s in bought_full.items()},
             carted={users[u]: frozenset(s) for u, s in carted_full.items()},
             C_hist=FULL["C"], T_hist=FULL["T"], m_sub=FULL["m_sub"], w_pos=w_pos, devs=devs, regs=regs,
             params=dict(kappa_c=float(kappa_c), kappa_s=float(kappa_s), lam_cat=float(lam_cat),
                         lam_sub=float(lam_sub), b_pop=b_pop, b_p0=b_p0, b_u=b_u, b_dev=b_dev, b_reg=b_reg,
                         b_sale=float(b_sale), b_seen=float(b_seen), b_cart=float(b_cart)),
             diag=diag)
    log.info("[build] modelo estimado en %.0f s", time.time() - t_start)
    return P


def load_or_build(rt: RawTree, cfg: Config, cut_day: int | None, force: bool, max_age_h: float = 30.0) -> dict:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / ("priors_live.pkl" if cut_day is None else f"priors_cut{cut_day}.pkl")
    if path.exists() and not force:
        try:
            P = pickle.loads(path.read_bytes())
            if P.get("version") == 4 and time.time() - P["built_at"] < max_age_h * 3600:
                log.info("[build] modelo cargado de %s (%.1f h)", path.name, (time.time() - P["built_at"]) / 3600)
                return P
        except Exception as e:  # noqa: BLE001
            log.warning("[build] no se pudo leer %s: %s", path, e)
    prev: dict = {}
    if path.exists() and cut_day is None:                  # la afinidad calibrada con --calibrate se conserva
        try:
            prev = pickle.loads(path.read_bytes()).get("params", {})
        except Exception:  # noqa: BLE001
            prev = {}
    P = build_priors(rt, cfg, cut_day)
    for k in (*TILT_DEFAULT, "taste"):
        if k in prev and k not in P["params"]:
            P["params"][k] = prev[k]
    path.write_bytes(pickle.dumps(P, protocol=pickle.HIGHEST_PROTOCOL))
    (CACHE / path.name.replace(".pkl", "_params.json")).write_text(
        json.dumps({"params": P["params"], "diag": P["diag"], "w_pos": P["w_pos"].tolist()}, indent=1,
                   ensure_ascii=False, default=str))
    log.info("[build] guardado en %s (%.0f MB)", path, path.stat().st_size / 1e6)
    return P


def apply_params_v6(P: dict) -> None:
    """Si el v6 tiene calibración propia, sustituye la afinidad de la caché compartida."""
    f = CACHE / PARAMS_V6
    if f.exists():
        try:
            P["params"].update(json.loads(f.read_text())["params"])
        except Exception as e:  # noqa: BLE001
            log.warning("[modelo] no se pudo leer %s: %s", PARAMS_V6, e)


def parse_tags(desc) -> list[int]:
    d = str(desc or "").lower()
    return [k for k, t in enumerate(TAGS) if t in d][:2]


def load_affinity(rt: RawTree, P: dict, cfg: Config) -> dict:
    """Etiquetas de cada producto y factores latentes de co-compra (caché en .cache/afinidad.npz).

    Co-compra: con las cestas históricas (compras + carrito) se calcula el log-lift suavizado de cada par de
    productos, se le resta la media de su nivel de parentesco (misma subcategoría / misma categoría / resto,
    que ya recoge la jerarquía) y se queda con sus k direcciones principales: L ~ F · diag(signo) · F'."""
    global AFF
    path = CACHE / "afinidad.npz"
    n = len(P["ids"])
    if path.exists():
        try:
            z = np.load(path)
            if float(z["built_at"]) == float(P["built_at"]) and z["F"].shape == (n, cfg.taste_k) \
                    and float(z["smooth"]) == cfg.taste_smooth:
                AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
                return AFF
        except Exception as e:  # noqa: BLE001
            log.warning("[afinidad] no se pudo leer %s: %s", path.name, e)
    t0 = time.time()
    C = np.zeros((n, n), np.float32)
    cnt = np.zeros(n, np.float32)
    users = set(P["bought"]) | set(P["carted"])
    for u in users:
        hb = set(P["bought"].get(u, ())) | set(P["carted"].get(u, ()))
        b = np.fromiter(hb, np.int64, len(hb))
        b = b[b < n]
        if b.size:
            cnt[b] += 1
            if b.size > 1:
                C[np.ix_(b, b)] += 1
    np.fill_diagonal(C, 0)
    a = cfg.taste_smooth
    L = np.log((C + a) / (np.outer(cnt, cnt) / max(len(users), 1) + a)).astype(np.float64)
    del C
    sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"])
    same_s = sub[:, None] == sub[None, :]
    same_c = cat[:, None] == cat[None, :]
    for m in (same_s, same_c & ~same_s, ~same_c):
        L[m] -= L[m].mean()
    np.fill_diagonal(L, 0)
    w, V = np.linalg.eigh(L)
    o = np.argsort(-np.abs(w))[:cfg.taste_k]
    F = (V[:, o] * np.sqrt(np.abs(w[o]))).astype(np.float32)
    sg = np.sign(w[o]).astype(np.float32)
    tags = np.full((n, 2), -1, np.int16)
    idx = {p: i for i, p in enumerate(P["ids"])}
    for r in rt.query("SELECT product_id::String AS p, toString(description) AS d FROM shop.catalog", 120):
        i = idx.get(r["p"])
        if i is not None:
            ts = parse_tags(r["d"])
            tags[i, :len(ts)] = ts
    np.savez_compressed(path, F=F, sg=sg, tags=tags, eig=w[o], built_at=P["built_at"], smooth=a)
    log.info("[afinidad] %d cestas históricas -> %d factores de co-compra (autovalores %s) · %d productos con "
             "etiquetas · %.0f s", len(users), cfg.taste_k, np.round(w[o][:6], 0).tolist(), int((tags[:, 0] >= 0).sum()),
             time.time() - t0)
    AFF = dict(F=F, sg=sg, tags=tags, eig=w[o])
    return AFF


# =============================================================================
# 4. ESTADO EN VIVO Y SNAPSHOT INMUTABLE
# =============================================================================
class UserStore:
    """Recuentos fraccionados usuario × subcategoría y perfil de precio. Crece con usuarios nuevos."""

    def __init__(self, users: list[str], N: np.ndarray, ru_sum, ru_n, ns_max: int, extra: int = 200_000):
        cap = len(users) + extra
        self.row: dict[str, int] = {u: i for i, u in enumerate(users)}
        self.N = np.zeros((cap, ns_max), np.float32)
        self.N[:N.shape[0], :N.shape[1]] = N
        self.ru_sum = np.zeros(cap)
        self.ru_n = np.zeros(cap)
        self.ru_sum[:len(users)], self.ru_n[:len(users)] = ru_sum, ru_n
        self.n = len(users)

    def get_or_create(self, u: str) -> int:
        r = self.row.get(u)
        if r is not None:
            return r
        if self.n >= self.N.shape[0]:                       # ampliar un 50 %
            extra = self.N.shape[0] // 2
            self.N = np.vstack([self.N, np.zeros((extra, self.N.shape[1]), np.float32)])
            self.ru_sum = np.concatenate([self.ru_sum, np.zeros(extra)])
            self.ru_n = np.concatenate([self.ru_n, np.zeros(extra)])
        r = self.n
        self.n += 1
        self.row[u] = r
        return r


class Snapshot:
    """Lo que lee answer(). No se modifica después de publicarse."""
    __slots__ = ("n", "ids", "index", "sub", "cat", "ok", "pos", "g_shape", "g_rate", "expV0", "sale_mult",
                 "sub_cat", "pi_c", "pi_s", "users", "bought", "carted", "carted_live", "popular", "is_new", "price",
                 "prm", "ns", "nc", "F", "Fs", "selfdot", "tagmat", "bought_hist", "viewed_live", "trend")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class StateHolder:
    snap: Snapshot | None = None


STATE = StateHolder()


class State:
    def __init__(self, P: dict, cfg: Config):
        self.cfg, self.P = cfg, P
        self.lock = threading.RLock()
        NM = cfg.n_max_products
        n = len(P["ids"])
        self.n = self.n_hist = n
        self.ids: list[str] = list(P["ids"])
        self.index: dict[str, int] = {p: i for i, p in enumerate(self.ids)}
        self.cats, self.subs = list(P["cats"]), list(P["subs"])
        self.ns_max, self.nc_max = len(self.subs) + cfg.ns_spare, len(self.cats) + cfg.nc_spare

        def arr(src, dtype, fill=0):
            a = np.full(NM, fill, dtype)
            a[:n] = src
            return a

        self.cat = arr(P["cat"], np.int64)
        self.sub = arr(P["sub"], np.int64)
        self.price = arr(P["price"], np.float64)
        self.price0 = arr(P["price"], np.float64)            # precio inicial: referencia de "rebajado"
        self.active = arr(True, bool, False)
        self.unavailable = np.zeros(NM, bool)
        self.is_new = np.zeros(NM, bool)
        self.C_hist = arr(P["C_hist"], np.float64)
        self.C_live = np.zeros(NM)
        self.T_launch = np.zeros(NM)                         # compras en vivo acumuladas al lanzarse (nuevos)
        self.launch_ts = np.zeros(NM)
        self.T_live = 0.0
        self.sub_cat = np.zeros(self.ns_max, np.int64)
        self.sub_cat[:len(self.subs)] = P["sub_cat"]
        self.pi_c = np.zeros(self.nc_max)
        self.pi_c[:len(self.cats)] = P["pi_c"]
        self.pi_s = np.zeros(self.ns_max)
        self.pi_s[:len(self.subs)] = P["pi_s"]
        self.m_sub = np.full(self.ns_max, float(np.median(P["m_sub"])))
        self.m_sub[:len(self.subs)] = P["m_sub"]
        self.users = UserStore(P["users"], P["N"], P["ru_sum"], P["ru_n"], self.ns_max)
        self.bought: dict[str, frozenset] = dict(P["bought"])
        self.carted: dict[str, frozenset] = dict(P["carted"])   # carrito HISTÓRICO (antes del stream)
        self.carted_live: dict[str, frozenset] = {}             # metido en el carrito durante el stream
        self.viewed_live: dict[str, frozenset] = {}             # visto durante el stream (shop.events)
        self.trend = Trend(self.ns_max, cfg.trend_hl, cfg.trend_hl_slow, cfg.trend_a)
        self.bought_hist: dict[str, frozenset] = P["bought"]    # compras del histórico (no se modifica)
        k = cfg.taste_k
        self.F = np.zeros((NM, k), np.float32)                  # factores de co-compra (productos nuevos: 0)
        self.tagmat = np.zeros((NM, K_TAGS), np.float32)        # etiquetas de cada producto (dos unos por fila)
        sg = np.ones(k, np.float32)
        if AFF is not None and AFF["F"].shape == (n, k):
            self.F[:n], sg = AFF["F"], AFF["sg"]
            for j in range(AFF["tags"].shape[1]):
                ok_ = AFF["tags"][:, j] >= 0
                self.tagmat[np.flatnonzero(ok_), AFF["tags"][ok_, j]] = 1.0
        self.Fs = self.F * sg
        self.selfdot = (self.Fs * self.F).sum(1)
        self.last_change_ts: dict[str, float] = {}
        self.upos = np.full(NM, 0.5)
        self.E_s = np.full(self.ns_max, 0.5)
        self.pos_dirty = True
        self.prm = self.make_params(P["params"], cfg)
        self.refresh()

    @staticmethod
    def make_params(params: dict, cfg: Config, **over) -> dict:
        """Parámetros listos para answer(): los del modelo + los multiplicadores de afinidad ya exponenciados."""
        prm = {**TILT_DEFAULT, "ses_mix": 1.0 if cfg.session_mix else 0.0, **params, **over}
        for k, (lo, hi) in TILT_BOUNDS.items():
            prm[k] = float(min(max(float(prm[k]), lo), hi))
        prm["e_lc"], prm["e_hc"] = math.exp(prm["g_lc"]), math.exp(prm["g_hc"])
        prm["e_vu"], prm["e_vf"], prm["e_vo"] = math.exp(prm["g_vu"]), math.exp(prm["g_vf"]), math.exp(prm["g_vo"])
        t = np.asarray(prm.get("taste") or [0.0] * N_TASTE, float)
        if t.size == 2 * N_BASKETS:                          # calibración anterior a la tendencia: su peso es 0
            t = np.r_[t, 0.0]
        prm["taste"] = t if t.size == N_TASTE else np.zeros(N_TASTE)
        if not cfg.trend:
            prm["taste"] = np.r_[prm["taste"][:-1], 0.0]
        elif cfg.trend_scale != 1.0:
            prm["taste"] = np.r_[prm["taste"][:-1], prm["taste"][-1] * cfg.trend_scale]
        prm["taste_on"] = bool(cfg.taste and AFF is not None and np.any(prm["taste"] != 0))
        return prm

    # --- cantidades derivadas -------------------------------------------------
    def popularity(self):
        cfg, n = self.cfg, self.n
        w = cfg.live_weight
        C = self.C_hist[:n] + w * self.C_live[:n]
        T = np.where(self.is_new[:n], w * np.maximum(self.T_live - self.T_launch[:n], 0.0),
                     self.P["T_hist"] + w * self.T_live)
        return gamma_popularity(C, T, self.m_sub, self.sub[:n], cfg.b0)

    def refresh(self) -> None:
        """Recalcula posiciones de precio (si han cambiado) y la posición esperada por popularidad."""
        n = self.n
        if self.pos_dirty:
            valid = self.price[:n] > 0
            self.upos[:n] = price_positions(self.price[:n], self.sub[:n], valid)
            self.pos_dirty = False
        shape, rate = self.popularity()
        popm = shape / rate
        num = np.bincount(self.sub[:n], weights=popm * self.upos[:n], minlength=self.ns_max)
        den = np.bincount(self.sub[:n], weights=popm, minlength=self.ns_max)
        self.E_s = np.where(den > 0, num / np.maximum(den, 1e-30), 0.5)

    def publish(self, set_global: bool = True) -> Snapshot:
        with self.lock:
            self.refresh()
            n = self.n
            shape, rate = self.popularity()
            sale = (self.price[:n] < self.price0[:n] - 1e-9).astype(float)
            sale_mult = np.exp(self.prm["b_sale"] * sale)
            expV0 = (shape / rate) ** self.prm["b_pop"] * sale_mult
            ok = self.active[:n] & ~self.unavailable[:n] & (self.price[:n] > 0)
            order = np.argsort(-expV0)
            snap = Snapshot(n=n, ids=self.ids, index=self.index, sub=self.sub[:n].copy(), cat=self.cat[:n].copy(),
                            ok=ok, pos=self.upos[:n].copy(), g_shape=shape, g_rate=rate, expV0=expV0,
                            sale_mult=sale_mult, sub_cat=self.sub_cat.copy(), pi_c=self.pi_c.copy(),
                            pi_s=self.pi_s.copy(), users=self.users, bought=self.bought, carted=self.carted,
                            carted_live=self.carted_live, popular=order[ok[order]][:200],
                            F=self.F, Fs=self.Fs, selfdot=self.selfdot, tagmat=self.tagmat,
                            bought_hist=self.bought_hist, viewed_live=self.viewed_live, trend=self.trend.log_trend(),
                            is_new=self.is_new[:n].copy(), price=self.price[:n].copy(), prm=self.prm,
                            ns=self.ns_max, nc=self.nc_max)
            if set_global:
                STATE.snap = snap
            return snap

    # --- actualizaciones en vivo ------------------------------------------------
    def add_bought(self, user: str, i: int) -> None:
        cur = self.bought.get(user)
        if cur is None:
            self.bought[user] = frozenset((i,))
        elif i not in cur:
            self.bought[user] = cur | {i}

    def user_event(self, user: str, i: int, kind: str, organic: bool = True, ts: float | None = None) -> None:
        """Suma el evento a los recuentos del usuario; una compra actualiza además su perfil de precio,
        sus productos comprados y (si es orgánica) la popularidad del producto."""
        if not user or i is None or i < 0 or i >= self.n:
            return
        if ts is not None and organic:                       # evento orgánico con hora: cuenta para la tendencia
            self.trend.add(int(self.sub[i]), ts)
        cfg = self.cfg
        row = self.users.get_or_create(user)
        w = cfg.w_purchase if kind == "purchase" else cfg.w_cart if kind == "add_to_cart" else cfg.w_view
        self.users.N[row, self.sub[i]] += w
        if kind == "add_to_cart":                            # todos los eventos que llegan aquí son del stream
            cur = self.carted_live.get(user)
            if cur is None:
                self.carted_live[user] = frozenset((i,))
            elif i not in cur:
                self.carted_live[user] = cur | {i}
        elif kind == "view":
            cur = self.viewed_live.get(user)
            if cur is None:
                self.viewed_live[user] = frozenset((i,))
            elif i not in cur:
                self.viewed_live[user] = cur | {i}
        if kind == "purchase":
            self.users.ru_sum[row] += self.upos[i] - self.E_s[self.sub[i]]
            self.users.ru_n[row] += 1
            self.add_bought(user, i)
            if organic:
                self.C_live[i] += 1
                self.T_live += 1

    def bulk_views(self, user: str, pids: list, counts: list) -> int:
        """Vistas del stream anteriores al arranque: recuentos por subcategoría y productos vistos hoy."""
        idx = [(self.index.get(p, -1), float(c)) for p, c in zip(pids, counts)]
        idx = [(i, c) for i, c in idx if 0 <= i < self.n]
        if not user or not idx:
            return 0
        row = self.users.get_or_create(user)
        for i, c in idx:
            self.users.N[row, self.sub[i]] += self.cfg.w_view * c
        self.viewed_live[user] = (self.viewed_live.get(user) or frozenset()) | {i for i, _ in idx}
        return len(idx)

    def _code(self, names: list, name: str, cap: int) -> int:
        if name in names:
            return names.index(name)
        if len(names) >= cap:
            return 0
        names.append(name)
        return len(names) - 1

    def add_product(self, r: dict, ts: float) -> int | None:
        pid = r["product_id"]
        if pid in self.index:
            return self.index[pid]
        if self.n >= self.cfg.n_max_products:
            log.error("[catalog] capacidad agotada: no se añade %s", pid)
            return None
        i = self.n
        nc0, ns0 = len(self.cats), len(self.subs)
        c = self._code(self.cats, r.get("category") or "?", self.nc_max)
        s = self._code(self.subs, r.get("subcategory") or "?", self.ns_max)
        if len(self.cats) > nc0:                             # categoría nunca vista: prior pequeña
            self.pi_c[c] = 0.02
            self.pi_c /= self.pi_c.sum()
        if len(self.subs) > ns0:                             # subcategoría nueva dentro de su categoría
            self.sub_cat[s] = c
            sib = np.flatnonzero((self.sub_cat == c) & (self.pi_s > 0))
            self.pi_s[s] = 1.0 / (sib.size + 1)
            self.pi_s[sib] *= sib.size / (sib.size + 1) if sib.size else 1.0
        self.cat[i], self.sub[i] = c, s
        price = float(r.get("price") or 0)
        self.price[i] = self.price0[i] = price
        self.active[i] = str(r.get("active")).lower() in ("true", "1", "none", "")
        self.unavailable[i] = price <= 0
        self.is_new[i] = True
        self.tagmat[i] = 0.0
        self.tagmat[i, parse_tags(r.get("description"))] = 1.0   # el producto nuevo ya tiene etiquetas
        self.C_hist[i] = self.C_live[i] = 0.0
        self.T_launch[i] = self.T_live
        self.launch_ts[i] = ts
        self.ids.append(pid)
        self.index[pid] = i
        self.n += 1
        self.pos_dirty = True
        return i

    def apply_update(self, r: dict, now: float) -> None:
        pid, change = r.get("product_id"), r.get("change") or ""
        ts = parse_ts(r.get("ts")) or now
        if not pid:
            return
        if ts < self.last_change_ts.get(pid, -1e18) and change != "new_product":
            return                                           # llegada fuera de orden
        self.last_change_ts[pid] = max(ts, self.last_change_ts.get(pid, -1e18))
        if change == "new_product" or pid not in self.index:
            if change != "new_product" and not r.get("category"):
                return
            self.add_product(r, ts)
            if change == "new_product":
                return
        i = self.index[pid]
        if change == "price_change":
            newp = float(r.get("price") or 0)
            if newp <= 0:
                self.unavailable[i] = True
            elif abs(newp - self.price[i]) > 1e-9:
                self.price[i] = newp
                self.pos_dirty = True
        elif change == "out_of_stock":
            self.unavailable[i] = True
        elif change == "back_in_stock":
            self.unavailable[i] = False
            self.active[i] = True


# =============================================================================
# 5. INFERENCIA: 10 PRODUCTOS PARA UNA PETICIÓN
# =============================================================================
def answer(S: Snapshot, req: dict, rng, info: dict | None = None) -> list[str]:
    cfg, prm = CFG, S.prm
    user = str(req.get("user_id") or "")
    device = (req.get("device") or "").lower()
    region = req.get("region") or ""
    views = [S.index[p] for p in normalize_views(req.get("recent_views")) if p in S.index and S.index[p] < S.n]
    views = views[-cfg.max_session_views:]

    # --- usuario: recuentos por subcategoría y perfil de precio
    U = S.users
    N, row = U.N, U.row.get(user)
    if row is not None and row < N.shape[0]:
        n_us = N[row].astype(np.float64)
        r_u = float(U.ru_sum[row] / (U.ru_n[row] + cfg.r_shrink))
    else:
        n_us, r_u = np.zeros(S.ns), 0.0

    # --- sesión: pseudo-recuentos temporales, la vista más reciente pesa más
    S_s = np.zeros(S.ns)
    m = len(views)
    for k, i in enumerate(views):
        S_s[S.sub[i]] += cfg.rho_session ** (m - 1 - k)

    # --- elegibles: en stock y no comprados por el usuario
    elig = S.ok.copy()
    b = S.bought.get(user)
    if b:
        elig[[i for i in b if i < S.n]] = False

    # --- nivel 3: utilidad de producto (popularidad, precio, rebaja); lo ya comprado queda fuera aquí
    g = prm["b_p0"] + prm["b_u"] * r_u + prm["b_dev"].get(device, 0.0) + prm["b_reg"].get(region, 0.0)
    mult = np.exp(S.pos * g) * elig

    # --- niveles 1 y 2: Dirichlet (media de la posterior)
    a_s0 = prm["kappa_s"] * S.pi_s + n_us + prm["lam_sub"] * S_s
    a_c0 = prm["kappa_c"] * S.pi_c + np.bincount(S.sub_cat, weights=n_us, minlength=S.nc)
    S_c = np.bincount(S.sub_cat, weights=S_s, minlength=S.nc)
    ses = S_c.sum()
    if ses > 0:
        if prm["ses_mix"]:                                   # mezcla: P(c) = (1-w)·Dirichlet + w·sesión, w = lam/(1+lam)
            a_c0 = a_c0 + prm["lam_cat"] * a_c0.sum() * S_c / ses
        else:                                                # pseudo-recuentos: la sesión se diluye con la historia
            a_c0 = a_c0 + prm["lam_cat"] * S_c

    # --- afinidad usuario-producto: multiplica la probabilidad jerárquica, fuera del anidamiento
    lcs = S.carted_live.get(user) or ()
    hcs = S.carted.get(user) or ()
    lc = [i for i in lcs if i < S.n and elig[i]]             # en el carrito durante el stream, sin comprar
    hc = [i for i in hcs if i < S.n and elig[i] and i not in lcs]   # carrito histórico
    theta = prm["theta"]

    # --- gustos latentes: perfil de etiquetas y vector de co-compra de cada "cesta" del usuario
    #     (historia, sesión, carrito de hoy, vistas de hoy); el propio producto nunca cuenta en su puntuación
    taste, e_taste = None, None
    if prm["taste_on"] or info is not None:
        n = S.n
        taste = np.zeros((N_TASTE, n))
        taste[2 * N_BASKETS] = S.trend[S.sub]                # tendencia de su subcategoría (log)
        hb = set(S.bought_hist.get(user) or ()) | set(hcs)
        for c, src in enumerate((hb, views, lcs, S.viewed_live.get(user) or ())):
            if not src:
                continue
            b = np.fromiter(src, np.int64, len(src))
            b = b[b < n]
            if b.size == 0:
                continue
            num = S.Fs[:n] @ S.F[b].sum(0)                   # co-compra: media encogida de L[i, j], j en la cesta
            den = np.full(n, b.size + cfg.taste_shrink[c])
            num[b] -= S.selfdot[b]
            den[b] -= 1.0
            taste[c] = num / den
            Tb = S.tagmat[b]
            cc = Tb.sum(0)
            tot = float(cc.sum())
            if tot > 0:                                      # etiquetas: suma de log(cuota / (1/K)) de sus dos etiquetas
                al = cfg.tag_alpha
                sc_ = S.tagmat[:n] @ np.log((cc + al / K_TAGS) / (tot + al) * K_TAGS)
                tl = tot - Tb.sum(1)
                sc_[b] = (Tb * np.log((cc[None, :] - Tb + al / K_TAGS) / (tl[:, None] + al) * K_TAGS)).sum(1)
                taste[N_BASKETS + c] = sc_
        if prm["taste_on"]:
            e_taste = np.exp(np.clip(prm["taste"] @ taste, -8.0, 8.0))

    def tilt(Pn: np.ndarray) -> np.ndarray:
        sc = Pn.copy() if theta == 1.0 else Pn ** theta
        if e_taste is not None:
            sc *= e_taste
        if len(views) == 1:                                  # sesión de una sola vista: el producto visto manda
            sc[views[0]] *= prm["e_vu"]
        elif views:                                          # varias vistas: el primero pesa más que el resto
            sc[views[0]] *= prm["e_vf"]
            sc[views[1:]] *= prm["e_vo"]
        if hc:
            sc[hc] *= prm["e_hc"]
        if lc:
            sc[lc] *= prm["e_lc"]
        return sc

    def probs(expV0: np.ndarray) -> np.ndarray:
        base = expV0 * mult
        Z = np.bincount(S.sub, weights=base, minlength=S.ns)
        a_s = np.where(Z > 0, a_s0, 0.0)                    # subcategorías sin productos elegibles: fuera
        A_c = np.bincount(S.sub_cat, weights=a_s, minlength=S.nc)
        a_c = np.where(A_c > 0, a_c0, 0.0)                  # categorías vacías: fuera
        tot = a_c.sum()
        if tot <= 0:
            return np.zeros(S.n)
        p_sub = (a_c / tot)[S.sub_cat] * a_s / np.where(A_c > 0, A_c, 1.0)[S.sub_cat]
        return p_sub[S.sub] * base / np.where(Z > 0, Z, 1.0)[S.sub]

    Pn = probs(S.expV0)                                      # probabilidad jerárquica pura
    P_mean = tilt(Pn)
    if cfg.thompson:                                         # Thompson Sampling sobre la popularidad
        pop = rng.gamma(S.g_shape) / S.g_rate
        P_rank = tilt(probs(pop ** prm["b_pop"] * S.sale_mult))
    else:
        P_rank = P_mean

    chosen: list[int] = []
    per_sub: Counter = Counter()
    if cfg.exploit_slot1 and P_mean.max() > 0:
        chosen.append(int(P_mean.argmax()))
        per_sub[int(S.sub[chosen[0]])] += 1
    k_top = min(60, S.n - 1)
    cand = np.argpartition(-P_rank, k_top)[:k_top]
    for i in cand[np.argsort(-P_rank[cand])].tolist():
        if len(chosen) >= 10:
            break
        if P_rank[i] <= 0 or i in chosen:
            continue
        if cfg.max_per_sub and per_sub[int(S.sub[i])] >= cfg.max_per_sub:
            continue
        chosen.append(i)
        per_sub[int(S.sub[i])] += 1
    if len(chosen) < 10:                                     # relleno: los más populares elegibles
        for i in S.popular.tolist():
            if len(chosen) >= 10:
                break
            if elig[i] and i not in chosen:
                chosen.append(i)
    if info is not None:
        info.update(P=P_mean, Pn=Pn, chosen=chosen, r_u=r_u, g=g, lc=lc, hc=hc, views=views, taste=taste)
    return [S.ids[i] for i in chosen[:10]]


def validate_and_repair(S: Snapshot, items: list[str], bought) -> list[str]:
    """10 IDs distintos, existentes, en stock y no comprados; si falta alguno se rellena."""
    out, seen = [], set()
    for pid in items:
        i = S.index.get(pid)
        if i is None or i >= S.n or not S.ok[i] or (bought and i in bought) or pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
    if len(out) < 10:
        for i in S.popular.tolist():
            if len(out) >= 10:
                break
            if S.ids[i] not in seen and not (bought and i in bought):
                out.append(S.ids[i])
                seen.add(S.ids[i])
    return out[:10]


# =============================================================================
# 6. BUCLES EN VIVO
# =============================================================================
STOP = threading.Event()
OUTBOX: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
REQ_USER = TTLCache(2700)                 # request_id -> user_id (para atribuir el feedback)
SEEN_REQ = TTLCache(900)
PUSH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="push")


class Clock:
    offset = 0.0

    @classmethod
    def now(cls) -> float:
        return time.time() + cls.offset

    @classmethod
    def sync(cls, rt: RawTree) -> None:
        t0 = time.time()
        r = rt.query("SELECT toString(now64(3)) AS t", 5)
        t1 = time.time()
        srv = parse_ts(r[0]["t"])
        if srv:
            cls.offset = srv - (t0 + t1) / 2
            if abs(cls.offset) > 2:
                log.warning("[clock] desfase con el servidor de %.2f s", cls.offset)


class Metrics:
    def __init__(self):
        self.c: Counter = Counter()
        self.lock = threading.Lock()
        self.push_lat: deque = deque(maxlen=2000)
        self.answer_ms: deque = deque(maxlen=2000)
        self.fb: deque = deque(maxlen=50_000)     # (ts, served_by, reason, latency, purchased, revenue)
        self.start = time.time()

    def inc(self, k, v=1):
        with self.lock:
            self.c[k] += v


MET = Metrics()

SQL_REQUESTS = """SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
 user_id::String AS user_id, toString(is_new_user) AS is_new_user, toString(context.device) AS device,
 toString(context.region) AS region, toString(context.page) AS page, recent_views
FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 15 SECOND
ORDER BY ts::DateTime64(3) ASC LIMIT 400"""


def request_loop(rt: RawTree, args, worker: int = 0) -> None:
    """Dos lectores desfasados; SEEN_REQ.claim() garantiza una sola respuesta por petición."""
    rng = np.random.default_rng(20261115 + worker)
    STOP.wait(0.12 * worker)
    backoff = 0.5
    while not STOP.is_set():
        try:
            rows = rt.query(SQL_REQUESTS, timeout=5)
            backoff = 0.5
            fresh = [r for r in rows if r["request_id"] not in SEEN_REQ]
            if len(fresh) > 50:                              # atasco: primero las más nuevas
                fresh = sorted(fresh, key=lambda r: r["ts"], reverse=True)[:50]
            for r in fresh:
                rid = r["request_id"]
                if args.shard and zlib.crc32(rid.encode()) % args.shard[1] != args.shard[0]:
                    SEEN_REQ.add(rid)                        # esta petición es de la otra variante del A/B
                    continue
                if not SEEN_REQ.claim(rid):
                    continue
                ts = parse_ts(r["ts"])
                if Clock.now() - (ts or Clock.now()) > CFG.deadline_s:
                    MET.inc("expired")
                    continue
                snap = STATE.snap
                t0 = time.perf_counter()
                try:
                    items = answer(snap, r, rng)
                except Exception as e:  # noqa: BLE001
                    log.exception("[answer] %s: %s", rid, e)
                    MET.inc("answer_errors")
                    items = [snap.ids[i] for i in snap.popular[:10].tolist()]
                MET.answer_ms.append((time.perf_counter() - t0) * 1000)
                REQ_USER.add(rid, r.get("user_id") or "")
                MET.inc("answered")
                ans = {"request_id": rid, "items": items, "model_version": CFG.model_version, "_ts": ts,
                       "_user": r.get("user_id") or ""}
                try:
                    OUTBOX.put_nowait(ans)
                except queue.Full:
                    try:
                        OUTBOX.get_nowait()
                    except queue.Empty:
                        pass
                    OUTBOX.put_nowait(ans)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "Unknown expression identifier" in msg:
                log.info("[requests] el stream aún no está abierto; reintento en %.1f s", backoff)
            else:
                log.warning("[requests] %s", msg[:200])
            STOP.wait(backoff)
            backoff = min(backoff * 2, 5.0)
            continue
        STOP.wait(CFG.request_poll_s)


def _push_batch(rt: RawTree, args, batch: list[dict]) -> None:
    snap = STATE.snap
    rows, sent = [], []
    for a in batch:
        if Clock.now() - (a["_ts"] or Clock.now()) > CFG.deadline_s:
            MET.inc("dropped_late")
            continue
        items = validate_and_repair(snap, a["items"], snap.bought.get(a["_user"]))
        if items != a["items"]:
            MET.inc("repaired")
        if len(items) == 10 and len(set(items)) == 10:
            rows.append({"request_id": a["request_id"], "items": items, "model_version": a["model_version"]})
            sent.append(a)
    if not rows:
        return
    try:
        n = len(rows) if args.dry_run else rt.insert("submissions", rows, timeout=5)
        done = Clock.now()
        for a in sent:
            if a["_ts"]:
                MET.push_lat.append((done - a["_ts"]) * 1000)
        MET.inc("pushed", n)
    except Exception as e:  # noqa: BLE001
        MET.inc("push_errors")
        log.warning("[push] error: %s; reintento", str(e)[:200])
        for a in sent:
            if Clock.now() - (a["_ts"] or 0) < CFG.deadline_s - 0.3:
                try:
                    OUTBOX.put_nowait(a)
                except queue.Full:
                    break


def push_loop(rt: RawTree, args) -> None:
    """Envía enseguida lo que haya en la cola; hasta 4 inserts a la vez."""
    while not STOP.is_set():
        try:
            first = OUTBOX.get(timeout=0.2)
        except queue.Empty:
            continue
        batch = [first]
        while len(batch) < 50:
            try:
                batch.append(OUTBOX.get_nowait())
            except queue.Empty:
                break
        PUSH_POOL.submit(_push_batch, rt, args, batch)


def feedback_loop(rt: RawTree, st: State, args) -> None:
    """Nuestro feedback: compras del carrusel (usuario -> comprado y recuentos) y métricas."""
    seen = TTLCache(3600, 500_000)
    since = fmt_ts(Clock.now())            # lo anterior ya se cargó en bootstrap_live()
    while not STOP.is_set():
        try:
            rows = rt.query(f"""SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
                served_by::String AS served_by, ifNull(toString(fallback_reason), '') AS reason,
                toFloat64OrZero(ifNull(toString(latency_ms), '0')) AS latency_ms,
                ifNull(toString(purchased), '') AS purchased,
                toFloat64OrZero(ifNull(toString(revenue_eur), '0')) AS revenue
                FROM {rt.team}.feedback WHERE request_id::String != '__init__'
                  AND ts::DateTime64(3) > toDateTime64('{since}', 3)
                ORDER BY ts::DateTime64(3) ASC LIMIT 20000""", 15)
            fresh = [r for r in rows if r["request_id"] not in seen]
            if rows:
                since = fmt_ts(max((parse_ts(rows[-1]["ts"]) or 0) - 60, Clock.now() - 1800))
            need = [r["request_id"] for r in fresh if r["purchased"] and r["request_id"] not in REQ_USER]
            for k in range(0, len(need), 300):
                for x in rt.query(f"""SELECT request_id::String AS request_id, user_id::String AS user_id
                        FROM shop.requests WHERE request_id::String IN ({q_list(need[k:k + 300])})""", 10):
                    REQ_USER.add(x["request_id"], x["user_id"])
            with st.lock:
                for r in fresh:
                    seen.add(r["request_id"])
                    if r["purchased"]:
                        user = REQ_USER.get(r["request_id"]) or ""
                        st.user_event(user, st.index.get(r["purchased"], -1), "purchase", organic=False)
                    MET.fb.append((parse_ts(r["ts"]) or time.time(), r["served_by"], r["reason"],
                                   float(r["latency_ms"] or 0), bool(r["purchased"]), float(r["revenue"] or 0)))
        except Exception as e:  # noqa: BLE001
            log.warning("[feedback] %s", str(e)[:200])
        STOP.wait(CFG.feedback_poll_s)


def events_loop(rt: RawTree, st: State, since: str) -> None:
    """shop.events: recuentos de usuario, popularidad en vivo y compras; publica un snapshot nuevo."""
    seen = TTLCache(600, 300_000)
    first = True
    while not STOP.is_set():
        try:
            total = 0
            while True:
                rows = rt.query(f"""SELECT toString(ts::DateTime64(3)) AS ts, event_type::String AS t,
                    user_id::String AS u, product_id::String AS p FROM shop.events
                    WHERE ts::DateTime64(3) > toDateTime64('{since}', 3)
                    ORDER BY ts::DateTime64(3) ASC LIMIT 10000""", 15)
                total += len(rows)
                with st.lock:
                    for r in rows:
                        key = (r["ts"], r["u"], r["p"], r["t"])
                        if key in seen:
                            continue
                        seen.add(key)
                        st.user_event(r["u"], st.index.get(r["p"], -1), r["t"], ts=parse_ts(r["ts"]))
                if rows:
                    since = fmt_ts((parse_ts(rows[-1]["ts"]) or Clock.now()) - (0 if first else 2.0))
                first = False
                if len(rows) < 10000:
                    break
            st.publish()
            MET.inc("events", total)
        except Exception as e:  # noqa: BLE001
            log.warning("[events] %s", str(e)[:200])
        STOP.wait(CFG.events_poll_s)


SQL_CATALOG_UPDATES = """SELECT product_id::String AS product_id, toString(ts::DateTime64(3)) AS ts,
 change::String AS change, toString(category) AS category, toString(subcategory) AS subcategory,
 price_eur::Float64 AS price, toString(is_active) AS active, ifNull(toString(description), '') AS description
FROM shop.catalog_updates WHERE ts::DateTime64(3) > toDateTime64('{since}', 3)
ORDER BY ts::DateTime64(3) ASC, product_id::String ASC LIMIT {limit}"""


def reconcile_stock(rt: RawTree, st: State) -> None:
    rows = rt.query("""SELECT product_id::String AS product_id FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock') GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""", 15)
    oos = {r["product_id"] for r in rows}
    with st.lock:
        fixed = 0
        for pid, i in list(st.index.items()):
            want = pid in oos
            if i < st.n and st.unavailable[i] != want and st.price[i] > 0:
                st.unavailable[i] = want
                fixed += 1
        if fixed:
            log.warning("[catalog] reconciliación: %d productos corregidos de stock", fixed)
            st.publish()


def catalog_loop(rt: RawTree, st: State, since: str) -> None:
    seen = TTLCache(3600, 100_000)
    last_rec = time.time()
    while not STOP.is_set():
        try:
            rows = rt.query(SQL_CATALOG_UPDATES.format(since=since, limit=5000), 10)
            fresh = [r for r in rows if (r["product_id"], r["ts"], r["change"]) not in seen]
            if fresh:
                with st.lock:
                    for r in fresh:
                        seen.add((r["product_id"], r["ts"], r["change"]))
                        try:
                            st.apply_update(r, Clock.now())
                        except Exception as e:  # noqa: BLE001
                            log.warning("[catalog] %s: %s", r.get("product_id"), e)
                    st.publish()
                log.info("[catalog] %d cambios aplicados %s", len(fresh), dict(Counter(r["change"] for r in fresh)))
            if rows:
                since = fmt_ts((parse_ts(rows[-1]["ts"]) or Clock.now()) - 5.0)
            if time.time() - last_rec > CFG.stock_reconcile_s:
                reconcile_stock(rt, st)
                last_rec = time.time()
        except Exception as e:  # noqa: BLE001
            if "Unknown expression identifier" not in str(e):
                log.warning("[catalog] %s", str(e)[:200])
        STOP.wait(CFG.catalog_poll_s)


def _fb_window(seconds: float):
    t_min = max(Clock.now() - seconds, MET.start + 30)
    fb = [x for x in list(MET.fb) if x[0] >= t_min]
    return fb, [x for x in fb if x[1] == "team"]


def health_loop(rt: RawTree, st: State, args) -> None:
    last_sync = last_metrics = 0.0
    while not STOP.is_set():
        try:
            now = time.time()
            if now - last_sync > 60:
                Clock.sync(rt)
                last_sync = now
            fb, team = _fb_window(300)
            lats = sorted(x[3] for x in team if x[3] > 0)
            am, pl = sorted(MET.answer_ms), sorted(MET.push_lat)
            qf = lambda xs, f: xs[int(f * (len(xs) - 1))] if xs else 0.0  # noqa: E731
            reasons = Counter(x[2] for x in fb if x[1] == "fallback")
            conv = sum(x[4] for x in team) / max(len(team), 1)
            log.info("[status] " + CFG.model_version + " respondidas=%d caducadas=%d enviadas=%d err=%d | answer p50=%.1fms p99=%.1fms | "
                     "edad al enviar p50=%.0fms p95=%.0fms | fb5m nuestras=%d fallback=%d %s conv=%.2f%% lat_p95=%.0fms | "
                     "productos=%d nuevos=%d sin_stock=%d usuarios=%d compras_vivo=%d%s",
                     MET.c["answered"], MET.c["expired"] + MET.c["dropped_late"], MET.c["pushed"], MET.c["push_errors"],
                     qf(am, 0.5), qf(am, 0.99), qf(pl, 0.5), qf(pl, 0.95), len(team), len(fb) - len(team),
                     dict(reasons) if reasons else "", 100 * conv, qf(lats, 0.95), st.n, int(st.is_new[:st.n].sum()),
                     int(st.unavailable[:st.n].sum()), st.users.n, int(st.T_live), " [DRY-RUN]" if args.dry_run else "")
            for c in (REQ_USER, SEEN_REQ):
                c.purge()
            if now - last_metrics > CFG.metrics_s and not args.dry_run:      # métricas a RawTree (scratch)
                last_metrics = now
                rt.insert("scratch", [{
                    "kind": "metrics", "ts": fmt_ts(Clock.now()), "model_version": CFG.model_version,
                    "answered": MET.c["answered"], "pushed": MET.c["pushed"],
                    "expired": MET.c["expired"] + MET.c["dropped_late"], "fb5m_team": len(team),
                    "fb5m_fallback": len(fb) - len(team), "conv5m": conv,
                    "rev5m": sum(x[5] for x in team) / max(len(team), 1), "lat_p95": qf(lats, 0.95),
                    "n_products": st.n, "n_new": int(st.is_new[:st.n].sum()), "live_purchases": int(st.T_live),
                    "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.prm.items()
                               if not isinstance(v, dict)}}], timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("[health] %s", str(e)[:200])
        STOP.wait(10)


# =============================================================================
# 7. ARRANQUE EN VIVO
# =============================================================================
def bootstrap_live(rt: RawTree, st: State) -> tuple[str, str]:
    """Reconstruye el estado en vivo desde RawTree: catálogo, stock, compras, carritos y popularidad."""
    since_cat = "1970-01-01 00:00:00.000"
    wm = fmt_ts(Clock.now())
    try:
        rows = rt.query(SQL_CATALOG_UPDATES.format(since=since_cat, limit=200000), 60)
        with st.lock:
            for r in rows:
                st.apply_update(r, Clock.now())
            st.refresh()
        if rows:
            since_cat = rows[-1]["ts"]
        log.info("[boot] %d cambios de catálogo aplicados %s · %d productos nuevos", len(rows),
                 dict(Counter(r["change"] for r in rows)), int(st.is_new[:st.n].sum()))
        reconcile_stock(rt, st)
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] catalog_updates: %s", str(e)[:200])
    for kind in ("purchase", "add_to_cart"):
        try:
            total = 0
            for k in range(2):
                rows = rt.query(f"""SELECT user_id::String AS u, groupArray(product_id::String) AS ps FROM shop.events
                    WHERE event_type::String = '{kind}' AND ts::DateTime64(3) <= toDateTime64('{wm}', 3)
                      AND cityHash64(user_id::String) % 2 = {k} GROUP BY u""", 120)
                with st.lock:
                    for r in rows:
                        for p in r["ps"]:
                            st.user_event(r["u"], st.index.get(p, -1), kind)
                            total += 1
            log.info("[boot] eventos en vivo '%s': %d", kind, total)
        except Exception as e:  # noqa: BLE001
            log.warning("[boot] eventos %s: %s", kind, str(e)[:200])
    try:                                                   # tendencia: eventos por minuto y subcategoría (últimas 16 h)
        rows = rt.query(f"""WITH cat AS (SELECT product_id::String AS p, any(toString(subcategory)) AS s FROM
                (SELECT product_id, subcategory FROM shop.catalog UNION ALL
                 SELECT product_id, subcategory FROM shop.catalog_updates WHERE change::String = 'new_product') GROUP BY p)
            SELECT cat.s AS s, toUnixTimestamp(toStartOfMinute(e.ts::DateTime64(3))) AS m, count() AS n
            FROM shop.events AS e INNER JOIN cat ON cat.p = e.product_id::String
            WHERE e.ts::DateTime64(3) > toDateTime64('{wm}', 3) - INTERVAL 16 HOUR AND e.ts::DateTime64(3) <= toDateTime64('{wm}', 3)
            GROUP BY s, m ORDER BY m""", 180)
        code = {name: k for k, name in enumerate(st.subs)}
        with st.lock:
            for r in rows:
                if r["s"] in code:
                    st.trend.add(code[r["s"]], float(r["m"]), float(r["n"]))
        tl = st.trend.log_trend()
        top = np.argsort(-tl)[:3]
        log.info("[boot] tendencia: %d minutos-subcategoría cargados · ahora más alta: %s", len(rows),
                 ", ".join(f"{st.subs[k]} x{math.exp(tl[k]):.1f}" for k in top if k < len(st.subs)))
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] tendencia: %s", str(e)[:200])
    try:                                                   # vistas del stream: recuentos y "visto hoy"
        total, parts = 0, 12
        for k in range(parts):
            rows = rt.query(f"""SELECT u, groupArray(p) AS ps, groupArray(c) AS cs FROM (
                    SELECT user_id::String AS u, product_id::String AS p, count() AS c FROM shop.events
                    WHERE event_type::String = 'view' AND ts::DateTime64(3) <= toDateTime64('{wm}', 3)
                      AND cityHash64(user_id::String) % {parts} = {k} GROUP BY u, p) GROUP BY u""", 180)
            with st.lock:
                for r in rows:
                    total += st.bulk_views(r["u"], r["ps"], r["cs"])
        log.info("[boot] vistas en vivo: %d pares usuario-producto", total)
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] vistas en vivo: %s", str(e)[:200])
    try:                                                   # compras totales en vivo antes de cada lanzamiento
        new_idx = np.flatnonzero(st.is_new[:st.n])
        tss = sorted({float(st.launch_ts[i]) for i in new_idx})
        if tss:
            cols = ", ".join(f"countIf(ts::DateTime64(3) < toDateTime64('{fmt_ts(t)}', 3)) AS c{k}"
                             for k, t in enumerate(tss))
            r = rt.query(f"""SELECT {cols} FROM shop.events WHERE event_type::String = 'purchase'
                AND ts::DateTime64(3) <= toDateTime64('{wm}', 3)""", 60)[0]
            cum = {t: float(r[f"c{k}"]) for k, t in enumerate(tss)}
            with st.lock:
                for i in new_idx:
                    st.T_launch[i] = cum[float(st.launch_ts[i])]
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] lanzamientos: %s", str(e)[:200])
    try:
        rows = rt.query(f"""SELECT r.user_id::String AS u, groupArray(f.purchased::String) AS ps
            FROM {rt.team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
            WHERE ifNull(toString(f.purchased), '') != '' GROUP BY u""", 60)
        with st.lock:
            for r in rows:
                for p in r["ps"]:
                    st.user_event(r["u"], st.index.get(p, -1), "purchase", organic=False)
        log.info("[boot] compras del carrusel: %d usuarios", len(rows))
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] compras del carrusel: %s", str(e)[:200])
    return since_cat, wm


def mark_already_submitted(rt: RawTree) -> None:
    try:
        rows = rt.query(f"""SELECT request_id::String AS request_id FROM {rt.team}.submissions
            WHERE request_id::String IN (SELECT request_id::String FROM shop.requests
                                         WHERE ts::DateTime64(3) > now() - INTERVAL 3 MINUTE)""", 15)
        for r in rows:
            SEEN_REQ.add(r["request_id"])
        log.info("[boot] %d peticiones recientes ya respondidas (no se duplican)", len(rows))
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] submissions recientes: %s", str(e)[:200])


def describe(S: Snapshot, st: State, pid: str, user: str = "") -> str:
    i = S.index.get(pid)
    if i is None or i >= S.n:
        return pid
    share = S.g_shape[i] / S.g_rate[i]
    return (f"{pid} · {st.cats[S.cat[i]]} / {st.subs[S.sub[i]]} · {S.price[i]:.2f} € · cuota {1e4 * share:.2f}‱"
            f"{' [NUEVO]' if S.is_new[i] else ''}{' [REBAJADO]' if S.sale_mult[i] > 1 else ''}"
            f"{' [CARRITO DE HOY]' if i in (S.carted_live.get(user) or ()) else ''}"
            f"{' [CARRITO ANTIGUO]' if i in (S.carted.get(user) or ()) else ''}"
            f"{' [YA COMPRADO]' if i in (S.bought.get(user) or ()) else ''}")


def log_params(P: dict) -> None:
    p = P["params"]
    t = {k: p.get(k, TILT_DEFAULT[k]) for k in TILT_DEFAULT}
    log.info("[modelo] %s · kappa_c=%s kappa_s=%s lam_cat=%s lam_sub=%s · b_pop=%.3f b_p0=%.3f b_u=%.2f b_sale=%.3f · "
             "afinidad: theta=%.2f carrito de hoy g=%.2f · carrito antiguo g=%.2f · vista única / primero / otros g=%.2f/%.2f/%.2f",
             CFG.model_version, p["kappa_c"], p["kappa_s"], p["lam_cat"], p["lam_sub"], p["b_pop"], p["b_p0"], p["b_u"],
             p["b_sale"], t["theta"], t["g_lc"], t["g_hc"], t["g_vu"], t["g_vf"], t["g_vo"])
    log.info("[modelo] gustos latentes (%s): %s", "activos" if CFG.taste and AFF is not None and p.get("taste") else
             "sin calibrar", dict(zip(TASTE_NAMES, p.get("taste") or [])))


def run_live(args) -> None:
    rt = make_rt()
    Clock.sync(rt)
    P = load_or_build(rt, CFG, None, args.rebuild)
    apply_params_v6(P)
    load_affinity(rt, P, CFG)
    log_params(P)
    st = State(P, CFG)
    since_cat, since_ev = bootstrap_live(rt, st)
    st.publish()
    if args.smoke:
        run_smoke(rt, st)
        return
    mark_already_submitted(rt)
    MET.start = Clock.now()
    threads = [threading.Thread(target=f, args=a, daemon=True, name=f.__name__) for f, a in (
        (request_loop, (rt, args, 0)), (request_loop, (rt, args, 1)), (push_loop, (rt, args)),
        (feedback_loop, (rt, st, args)), (events_loop, (rt, st, since_ev)), (catalog_loop, (rt, st, since_cat)),
        (health_loop, (rt, st, args)))]
    for t in threads:
        t.start()
    log.info("[main] %s en marcha%s%s (Ctrl+C para parar)", CFG.model_version, " · DRY-RUN" if args.dry_run else "",
             f" · shard {args.shard[0]}/{args.shard[1]}" if args.shard else "")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    died = False
    try:
        while not STOP.is_set():
            time.sleep(1)
            for t in threads:
                if not t.is_alive() and not STOP.is_set():
                    log.error("[main] el hilo %s ha muerto; se para para que el bucle externo lo reinicie", t.name)
                    died = True
                    STOP.set()
    except KeyboardInterrupt:
        STOP.set()
    log.info("[main] parado")
    if died:
        sys.exit(1)


def run_smoke(rt: RawTree, st: State) -> None:
    """Responde las peticiones reales más recientes SIN enviarlas; muestra latencia y ejemplos."""
    rows = rt.query(SQL_REQUESTS.replace("15 SECOND", "120 SECOND"), 10)[-30:]
    rng = np.random.default_rng(1)
    S = STATE.snap
    lat = []
    for k, r in enumerate(rows):
        t0 = time.perf_counter()
        info: dict = {}
        items = answer(S, r, rng, info)
        lat.append((time.perf_counter() - t0) * 1000)
        assert len(items) == 10 and len(set(items)) == 10, "respuesta inválida"
        assert all(S.ok[S.index[x]] for x in items), "producto no elegible"
        if k < 4:
            print(f"\n--- petición {r['request_id']} · page={r.get('page')} device={r.get('device')} "
                  f"region={r.get('region')} is_new_user={r.get('is_new_user')} · perfil de precio r_u={info['r_u']:+.3f}")
            u = r.get("user_id") or ""
            print(f"    usuario: {len(info['lc'])} en el carrito de hoy, {len(info['hc'])} en el carrito antiguo (sin comprar, "
                  f"en stock), {len(S.bought.get(u) or ())} comprados")
            for v in normalize_views(r.get("recent_views"))[-3:]:
                print("    visto:    ", describe(S, st, v, u))
            for j, x in enumerate(items, 1):
                i_ = S.index[x]
                print(f"    slot {j:>2}:   {describe(S, st, x, u)}{' [VISTO]' if i_ in info['views'] else ''} · "
                      f"P jerárquica={info['Pn'][i_]:.4f} · score={info['P'][i_]:.4f}")
    lat = np.array(lat)
    print(f"\n{len(rows)} peticiones · answer() p50={np.median(lat):.2f} ms · p99={np.percentile(lat, 99):.2f} ms · todas válidas")


# =============================================================================
# 7b. CALIBRACIÓN DE LA AFINIDAD USUARIO-PRODUCTO CON NUESTRO FEEDBACK (--calibrate)
# =============================================================================
GLM_COLS = ("const", "usuario_nuevo", "home", "theta", "g_lc", "g_hc", "g_vu", "g_vf", "g_vo", "ya_comprado",
            "sin_P") + TASTE_NAMES
N_BASE = 11                                                # columnas anteriores a los gustos latentes
TILT_MIN_BUYS = dict(theta=200, g_lc=30, g_hc=30, g_vu=15, g_vf=15, g_vo=15)   # compras mínimas para reestimar


def download_feedback(rt: RawTree) -> tuple[list[dict], list[tuple], list[dict]]:
    """Todas las listas mostradas a nuestro equipo con el contexto de su petición, los eventos del stream
    y los cambios de catálogo (para reproducir el stock y los precios de cada instante)."""
    fb: list[dict] = []
    parts = 12
    for k in range(parts):
        fb += rt.query(f"""SELECT f.request_id::String AS rid, r.user_id::String AS user_id,
            toUnixTimestamp64Milli(r.ts::DateTime64(3)) AS t, toString(r.context.page) AS page,
            toString(r.context.device) AS device, toString(r.context.region) AS region,
            toString(r.is_new_user) AS nu, r.recent_views AS recent_views, f.shown::Array(String) AS shown,
            ifNull(toString(f.purchased), '') AS buy, toString(f.model_version) AS mv
            FROM {rt.team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
            WHERE cityHash64(f.request_id::String) % {parts} = {k} AND f.request_id::String != '__init__'""", 600)
    for r in fb:
        r["t"] = int(r["t"])
    events: list[tuple] = []
    parts = 24
    for k in range(parts):
        for x in rt.query(f"""SELECT user_id::String AS u, groupArray(product_id::String) AS p,
                groupArray(event_type::String) AS e, groupArray(toUnixTimestamp64Milli(ts::DateTime64(3))) AS t
                FROM shop.events WHERE cityHash64(user_id::String) % {parts} = {k} GROUP BY u""", 600):
            u = x["u"]
            events.extend((int(t), u, p, e) for t, p, e in zip(x["t"], x["p"], x["e"]))
    events.sort(key=lambda x: x[0])
    cat = rt.query(SQL_CATALOG_UPDATES.format(since="1970-01-01 00:00:00.000", limit=500000), 300)
    return fb, events, cat


def replay_feedback(P: dict, cfg: Config, fb: list[dict], events: list[tuple], cat: list[dict],
                    tilts: list[dict] | None = None, policy_last: int = 0) -> dict:
    """Reproduce el stream en orden cronológico (eventos de usuario, compras del carrusel, stock y precios) y,
    para cada producto de cada lista mostrada, guarda las piezas del modelo de compra: posición, si se compró,
    clase de afinidad y probabilidad jerárquica en ese instante (con el mismo código que responde en vivo).
    Con `tilts`, además, las listas que elegiría answer() con cada juego de multiplicadores en las últimas
    `policy_last` peticiones, para predecir su conversión."""
    import heapq
    st = State(P, cfg)
    for r in cat:                                          # los productos nuevos existen, sin stock hasta su alta
        if r["change"] == "new_product" and r["product_id"] not in st.index:
            i = st.add_product(r, parse_ts(r["ts"]) or 0.0)
            if i is not None:
                st.unavailable[i] = True
    index = st.index
    variants = tilts or [{}]
    snaps: list[Snapshot] = []

    def publish() -> None:
        S0 = st.publish(set_global=False)
        base = {k: getattr(S0, k) for k in Snapshot.__slots__}
        snaps[:] = [Snapshot(**{**base, "prm": State.make_params(P["params"], cfg, **tl)}) for tl in variants]

    publish()
    order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
    first_policy = len(order) - policy_last if tilts else len(order)
    rank = {k: n for n, k in enumerate(order)}
    stream = heapq.merge(
        ((int((parse_ts(r["ts"]) or 0.0) * 1000), 0, k) for k, r in enumerate(cat)),
        ((e[0], 1, k) for k, e in enumerate(events)),
        ((fb[k]["t"], 2, k) for k in order),
        sorted((fb[k]["t"] + 5000, 3, k) for k in order if fb[k]["buy"]))
    M = sum(len((r["shown"] or [])[:10]) for r in fb)
    X = np.zeros((M, 10))                                  # slot, compra, nuevo, home, LC, HC, VU, VF, VO, ya comprado
    G = np.zeros(M, np.int64)                              # lista a la que pertenece cada impresión
    PN = np.zeros(M)
    AF = np.zeros((M, N_TASTE), np.float32)          # puntuaciones de gustos de cada producto mostrado
    n_pol = max(len(order) - first_policy, 0)
    pol = np.zeros((n_pol, len(variants), 10, 6 + N_TASTE), np.float32)   # lp, LC, HC, VU, VF, VO, gustos
    pol_ctx = np.zeros((n_pol, 3))                         # nuevo, home, compró (con la lista real)
    rng = np.random.default_rng(0)
    m, dirty = 0, False
    for t, typ, k in stream:
        if typ == 1:
            _, u, p, e = events[k]
            i = index.get(p)
            if i is not None:
                st.user_event(u, i, e, ts=t / 1000)
        elif typ == 0:
            r = cat[k]
            i = index.get(r["product_id"])
            if r["change"] == "new_product":
                if i is not None:
                    st.unavailable[i] = st.price[i] <= 0
                    st.T_launch[i], st.launch_ts[i] = st.T_live, t / 1000
            else:
                st.apply_update(r, t / 1000)
            dirty = True
        elif typ == 3:
            st.user_event(fb[k]["user_id"], index.get(fb[k]["buy"], -1), "purchase", organic=False)
        else:
            r = fb[k]
            if dirty:
                publish()
                dirty = False
            tl = st.trend.log_trend()                       # la tendencia, en el instante de la petición
            for S_ in snaps:
                S_.trend = tl
            u = r["user_id"]
            bs, lcs, hcs = st.bought.get(u) or (), st.carted_live.get(u) or (), st.carted.get(u) or ()
            nuevo, home = float(r["nu"] == "true"), float(r["page"] == "home")
            n_k = rank[k] - first_policy
            info: dict = {}
            for v, S in enumerate(snaps if n_k >= 0 else snaps[:1]):
                info = {}
                answer(S, r, rng, info)
                if n_k >= 0:
                    Pn_, lc_, hc_, vs_ = info["Pn"], set(info["lc"]), set(info["hc"]), info["views"]
                    for j, i in enumerate(info["chosen"][:10]):
                        pol[n_k, v, j, :6] = (math.log(max(float(Pn_[i]), 1e-12) / P_REF), i in lc_, i in hc_,
                                              len(vs_) == 1 and i == vs_[0], len(vs_) > 1 and i == vs_[0],
                                              len(vs_) > 1 and i in vs_[1:])
                        pol[n_k, v, j, 6:] = info["taste"][:, i]
            if n_k >= 0:
                pol_ctx[n_k] = (nuevo, home, float(bool(r["buy"])))
            Pn, vs, ts_ = info["Pn"], info["views"], info["taste"]
            for j, pid in enumerate((r["shown"] or [])[:10]):
                i = index.get(pid, -1)
                b = float(i in bs)
                X[m] = (j, float(pid == r["buy"]), nuevo, home, float(i in lcs) * (1 - b),
                        float(i in hcs and i not in lcs) * (1 - b), float(len(vs) == 1 and i == vs[0]) * (1 - b),
                        float(len(vs) > 1 and i == vs[0]) * (1 - b), float(len(vs) > 1 and i in vs[1:]) * (1 - b), b)
                G[m] = rank[k]
                if 0 <= i < Pn.size:
                    PN[m] = float(Pn[i])
                    AF[m] = ts_[:, i]
                m += 1
    return dict(X=X[:m], PN=PN[:m], AF=AF[:m], G=G[:m], pol=pol, pol_ctx=pol_ctx)


def glm_design(X: np.ndarray, PN: np.ndarray, AF: np.ndarray) -> np.ndarray:
    ok = PN > 0
    lp = np.where(ok, np.log(np.maximum(PN, 1e-12) / P_REF), 0.0)
    nop = (~ok & (X[:, 9] == 0)).astype(float)             # fuera de catálogo o sin stock según nuestro estado
    return np.column_stack([np.ones(len(X)), X[:, 2], X[:, 3], lp, X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9],
                            nop, AF.astype(np.float64)])


def cond_fit(Mx: np.ndarray, y: np.ndarray, off: np.ndarray, gid: np.ndarray, l2: float = 1e-3):
    """Logit condicional dentro de cada lista con compra: P(comprado = j | lista) = softmax(off_j + x_j·b).
    El usuario es el mismo para los 10 productos, así que su propensión a comprar no contamina los pesos."""
    has = np.bincount(gid, weights=y, minlength=int(gid.max()) + 1)[gid] > 0
    Mx, y, off, gid = Mx[has], y[has], off[has], gid[has]
    _, inv = np.unique(gid, return_inverse=True)
    ng, d = int(inv.max()) + 1, Mx.shape[1]
    b = np.zeros(d)
    H = np.eye(d)
    p = np.zeros(len(y))
    for _ in range(200):
        e = Mx @ b + off
        mx = np.full(ng, -np.inf)
        np.maximum.at(mx, inv, e)
        ex = np.exp(e - mx[inv])
        p = ex / np.bincount(inv, weights=ex, minlength=ng)[inv]
        g = Mx.T @ (y - p) - l2 * b
        xbar = np.zeros((ng, d))
        np.add.at(xbar, inv, Mx * p[:, None])
        H = (Mx * p[:, None]).T @ Mx - xbar.T @ xbar + l2 * np.eye(d)
        step = np.linalg.solve(H, g)
        mxs = float(np.abs(step).max())
        b = b + (step / mxs if mxs > 1 else step)
        if mxs < 1e-8:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(H))), float((y * np.log(np.maximum(p, 1e-300))).sum()), ng


def poisson_fit(A: np.ndarray, y: np.ndarray, off: np.ndarray, l2: float = 1e-4):
    """Regresión de Poisson con enlace log (Newton amortiguado). Devuelve coeficientes, errores estándar y logver."""
    b = np.zeros(A.shape[1])
    b[0] = math.log(max(float(y.sum()), 1.0) / float(np.exp(off).sum()))
    H = np.eye(len(b))
    for _ in range(300):
        mu = np.exp(np.clip(A @ b + off, -50, 20))
        g = A.T @ (y - mu) - l2 * b
        H = (A * mu[:, None]).T @ A + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        mx = float(np.abs(step).max())
        b = b + (step / mx if mx > 1 else step)
        if mx < 1e-8:
            break
    mu = np.exp(np.clip(A @ b + off, -50, 20))
    return b, np.sqrt(np.diag(np.linalg.inv(H))), float((y * np.log(mu) - mu).sum())


def calibrate_live(rt: RawTree, cfg: Config, policy_last: int = 30000, args_dry: bool = False) -> None:
    """Reestima los multiplicadores de afinidad usuario-producto con todo nuestro feedback:
        compra_ij ~ Poisson( w_slot · exp(a + a_nuevo + a_home) · P_j^theta · exp(g_clase + gustos) )
    con P_j la probabilidad jerárquica del producto en el instante de la petición. Guarda los parámetros en la
    caché del modelo (hay que reiniciar el proceso en vivo para que los cargue)."""
    path = CACHE / "priors_live.pkl"
    if not path.exists():
        sys.exit("No hay modelo en caché: ejecuta antes --build")
    P = pickle.loads(path.read_bytes())
    apply_params_v6(P)
    load_affinity(rt, P, cfg)
    cfg.thompson = False
    t0 = time.time()
    fb, events, cat = download_feedback(rt)
    log.info("[calib] %d listas (%d compras), %d eventos del stream, %d cambios de catálogo · %.0f s", len(fb),
             sum(1 for r in fb if r["buy"]), len(events), len(cat), time.time() - t0)
    old = {k: float(P["params"].get(k, TILT_DEFAULT[k])) for k in TILT_DEFAULT}
    old["taste"] = [float(v) for v in State.make_params(P["params"], cfg)["taste"]]
    R = replay_feedback(P, cfg, fb, events, cat)
    X, PN = R["X"], R["PN"]
    w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]
    y = X[:, 1]
    A = glm_design(X, PN, R["AF"])
    use_taste = cfg.taste and AFF is not None and y.sum() >= 300
    if not use_taste:
        A = A[:, :N_BASE]
    b, se, ll = poisson_fit(A, y, np.log(w))
    _, _, ll0 = poisson_fit(A[:, [0, 1, 2, 3, 9, 10]], y, np.log(w))
    _, _, ll1 = poisson_fit(A[:, :N_BASE], y, np.log(w))
    log.info("[calib] regresión de Poisson · %d impresiones, %d compras · logver %.1f (solo jerárquica: %.1f · "
             "+ carrito y vistos: %.1f) · %.0f s", len(y), int(y.sum()), ll, ll0, ll1, time.time() - t0)
    for name, bi, si in zip(GLM_COLS, b, se):
        log.info("[calib]   %-14s %+.3f ± %.3f   (x%.2f)", name, bi, si, math.exp(min(bi, 20)))
    buys = dict(theta=int(y.sum()), g_lc=int(y[X[:, 4] > 0].sum()), g_hc=int(y[X[:, 5] > 0].sum()),
                g_vu=int(y[X[:, 6] > 0].sum()), g_vf=int(y[X[:, 7] > 0].sum()), g_vo=int(y[X[:, 8] > 0].sum()))
    # los pesos que ordenan la lista se estiman DENTRO de cada lista (logit condicional); la Poisson se queda
    # para el nivel (usuario nuevo, home) y para la previsión de conversión
    cols = [k for k in range(A.shape[1]) if k > 2]
    bc_, sec_, llc, n_lists = cond_fit(A[:, cols], y, np.log(w), R["G"])
    bc = np.zeros(A.shape[1])
    bc[cols] = bc_
    log.info("[calib] logit condicional dentro de la lista · %d listas con compra · logver %.1f", n_lists, llc)
    for k_, name in enumerate(GLM_COLS[:A.shape[1]]):
        if k_ > 2:
            log.info("[calib]   %-14s Poisson %+.3f · dentro de la lista %+.3f ± %.3f", name, b[k_], bc[k_],
                     sec_[cols.index(k_)])
    est = dict(theta=bc[3], g_lc=bc[4], g_hc=bc[5], g_vu=bc[6], g_vf=bc[7], g_vo=bc[8])
    new = dict(old)
    for k, v in est.items():
        if buys[k] >= TILT_MIN_BUYS[k] and math.isfinite(v):
            lo, hi = TILT_BOUNDS[k]
            new[k] = round(float(min(max(v, lo), hi)), 3)
        else:
            log.warning("[calib] %s: solo %d compras; se mantiene %.3f", k, buys[k], old[k])
    log.info("[calib] compras por clase: carrito de hoy %d · carrito antiguo %d · vista única %d · primer visto %d · "
             "otros vistos %d", buys["g_lc"], buys["g_hc"], buys["g_vu"], buys["g_vf"], buys["g_vo"])
    if use_taste:
        new["taste"] = [round(float(min(max(v, -0.5), 4.0)), 3) for v in bc[N_BASE:N_BASE + N_TASTE]]
    log.info("[calib] antes: %s", old)
    log.info("[calib] ahora: %s", new)
    # conversión prevista de la lista que elegiría el modelo, con los multiplicadores de antes y los de ahora
    zero = dict(theta=1.0, g_lc=0.0, g_hc=0.0, g_vu=0.0, g_vf=0.0, g_vo=0.0, taste=[0.0] * N_TASTE)
    R2 = replay_feedback(P, cfg, fb, events, cat, tilts=[zero, old, new], policy_last=min(policy_last, len(fb)))
    pol, ctx = R2["pol"], R2["pol_ctx"]
    lin = (b[0] + b[1] * ctx[:, 0] + b[2] * ctx[:, 1])[:, None, None] + b[3] * pol[..., 0] + b[4] * pol[..., 1] \
        + b[5] * pol[..., 2] + b[6] * pol[..., 3] + b[7] * pol[..., 4] + b[8] * pol[..., 5]
    if use_taste:
        lin = lin + pol[..., 6:] @ b[N_BASE:N_BASE + N_TASTE]
    pred = (np.asarray(P["w_pos"])[None, None, :] * np.exp(np.clip(lin, -50, 5))).sum(-1)      # (peticiones, variantes)
    comp = pol[..., 1:6].sum(2).mean(0)                    # productos de cada clase por lista
    names = ("sin afinidad", "parámetros anteriores", "parámetros nuevos")
    diag = {"n_listas": len(fb), "n_compras": int(y.sum()), "logver": ll, "logver_solo_jerarquica": ll0,
            "logver_con_carrito_y_vistos": ll1,
            "coef": {n: [float(bi), float(si)] for n, bi, si in zip(GLM_COLS, b, se)}, "compras_por_clase": buys,
            "antes": old, "ahora": new, "calibrado": fmt_ts(time.time()), "prevision": {}}
    log.info("[calib] últimas %d peticiones: conversión real %.2f %% (con las listas que se mostraron)",
             len(ctx), 100 * ctx[:, 2].mean())
    for v, name in enumerate(names):
        nu = ctx[:, 0] > 0
        diag["prevision"][name] = {"todas": float(100 * pred[:, v].mean()), "nuevos": float(100 * pred[nu, v].mean()),
                                   "con_historia": float(100 * pred[~nu, v].mean())}
        log.info("[calib]   %-22s conversión prevista %.2f %% (usuarios nuevos %.2f %% · con historia %.2f %%) · por "
                 "lista: %.2f carrito de hoy, %.2f carrito antiguo, %.2f vistos", name, 100 * pred[:, v].mean(),
                 100 * pred[nu, v].mean(), 100 * pred[~nu, v].mean(), comp[v, 0], comp[v, 1], comp[v, 2] + comp[v, 3] + comp[v, 4])
    log.info("[calib] (la previsión extrapola el modelo de compra a listas que no se han mostrado: orienta, "
             "no sustituye a la prueba en vivo)")
    if args_dry:
        log.info("[calib] --dry-run: no se guarda nada")
        return
    (CACHE / PARAMS_V6).write_text(json.dumps({"params": new, "diag": diag}, indent=1, ensure_ascii=False, default=str))
    log.info("[calib] parámetros guardados en %s (no toca la calibración de v4/v5) · %.0f s en total · reinicia el "
             "proceso en vivo para cargarlos", PARAMS_V6, time.time() - t0)


# =============================================================================
# 8. EVALUACIÓN OFFLINE (mismas métricas que recommender-v2.py --eval)
# =============================================================================
def run_eval(args) -> None:
    """Evaluación offline con los últimos 7 días del histórico (el modelo se estima con los anteriores).

    Ojo: el carrusel antiguo y el que puntúa en vivo se comportan al revés con lo YA COMPRADO (en el
    histórico era el 60 % de las compras; en vivo casi no se compra). El modelo lo excluye a propósito,
    así que las métricas se separan por el estado previo del producto comprado."""
    rt = make_rt()
    cut_day = 53
    P = load_or_build(rt, CFG, cut_day, args.rebuild)
    log_params(P)
    st = State(P, CFG)
    S = st.publish(set_global=False)
    n_test = int(args.eval)
    day_expr = f"dateDiff('day', toDateTime64('{HIST_T0}', 3), ts::DateTime64(3))"
    rows = rt.query(f"""SELECT user_id::String AS user_id, toString(context.device) AS device,
        toString(context.region) AS region, toString(context.page) AS page, recent_views,
        ifNull(toString(purchased), '') AS purchased
        FROM shop.history_sessions WHERE purchased != '' AND {day_expr} >= {cut_day}
        ORDER BY cityHash64(request_id::String) LIMIT {n_test}""", 120)
    best = [S.ids[i] for i in np.argsort(-P["C_hist"])[:10]]
    cat_best = {c: [S.ids[i] for i in np.flatnonzero(P["cat"] == c)[np.argsort(-P["C_hist"][P["cat"] == c])][:10]]
                for c in range(len(st.cats))}
    rng = np.random.default_rng(3)
    res: dict = defaultdict(Counter)
    lat = []
    for r in rows:
        y = S.index.get(r["purchased"])
        if y is None:
            continue
        u = r["user_id"]
        estado = ("ya comprado (excluido)" if y in (S.bought.get(u) or ()) else
                  "en su carrito" if y in (S.carted.get(u) or ()) else "sin compra ni carrito previos")
        info: dict = {}
        t0 = time.perf_counter()
        items = answer(S, r, rng, info)
        lat.append((time.perf_counter() - t0) * 1000)
        views = [v for v in normalize_views(r["recent_views"]) if v in S.index]
        cb = cat_best.get(int(S.cat[S.index[views[-1]]]), best) if views else best
        for seg in ("todas las compras", estado):
            c = res[seg]
            c["n"] += 1
            c["hr10"] += r["purchased"] in items
            c["hr3"] += r["purchased"] in items[:3]
            c["rank"] += int((info["P"] > info["P"][y]).sum()) < 100
            c["base"] += r["purchased"] in best
            c["cat"] += r["purchased"] in cb
    lat = np.array(lat)
    print(f"\nEvaluación offline · modelo estimado con días < {cut_day} · {len(rows)} compras del carrusel antiguo (holdout)")
    print(f"answer(): p50 {np.median(lat):.2f} ms · p99 {np.percentile(lat, 99):.2f} ms\n")
    print(f"{'producto comprado':<32}{'n':>6}{'HR@10':>8}{'HR@3':>8}{'top-100':>9}{'best sellers':>14}{'best-cat':>10}")
    for seg in ("todas las compras", "ya comprado (excluido)", "en su carrito", "sin compra ni carrito previos"):
        c = res[seg]
        n = max(c["n"], 1)
        print(f"{seg:<32}{c['n']:>6}{c['hr10'] / n:>8.3f}{c['hr3'] / n:>8.3f}{c['rank'] / n:>9.3f}"
              f"{c['base'] / n:>14.3f}{c['cat'] / n:>10.3f}")
    print("best sellers = los 10 más vendidos; best-cat = los 10 más vendidos de la categoría de la última vista.")
    print("El log antiguo solo mostraba esa categoría, así que 'best-cat' parte con ventaja.")


# =============================================================================
# 9. CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Recomendador tengo-aurilla (jerárquico factorizado + afinidad usuario-producto)")
    ap.add_argument("--build", action="store_true", help="estima el modelo, lo guarda y sale")
    ap.add_argument("--rebuild", action="store_true", help="ignora la caché del modelo")
    ap.add_argument("--calibrate", action="store_true",
                    help="reestima la afinidad usuario-producto con nuestro feedback en vivo y sale")
    ap.add_argument("--eval", metavar="N", help="evaluación offline con N impresiones del holdout")
    ap.add_argument("--smoke", action="store_true", help="responde 30 peticiones reales sin enviarlas")
    ap.add_argument("--dry-run", action="store_true", help="todo el sistema, sin escribir en RawTree")
    ap.add_argument("--shard", metavar="K/N", help="responde solo las peticiones con hash(request_id) %% N == K")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR", help="cambia un parámetro de Config")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    for kv in args.set:
        k, _, v = kv.partition("=")
        if not hasattr(CFG, k):
            sys.exit(f"--set: Config no tiene '{k}'")
        old = getattr(CFG, k)
        setattr(CFG, k, v.lower() in ("1", "true", "yes", "si") if isinstance(old, bool) else type(old)(v))
        log.info("[config] %s = %s (antes %s)", k, getattr(CFG, k), old)
    if args.model_version:
        CFG.model_version = args.model_version
    if args.shard:
        k, _, n = args.shard.partition("/")
        args.shard = (int(k), int(n))
        if not 0 <= args.shard[0] < args.shard[1]:
            sys.exit("--shard K/N: hace falta 0 <= K < N")
    if args.eval:
        run_eval(args)
    elif args.calibrate:
        calibrate_live(make_rt(), CFG, args_dry=args.dry_run)
    elif args.build:
        P = load_or_build(make_rt(), CFG, None, True)
        log_params(P)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
