#!/usr/bin/env python3
"""recommender-v2.py · "Bayes-TS": implementació de model.txt (especificació v3).

Re-ranking bayesià en streaming per al repte RawTree (Datathon FME 2026):

  FASE A · priors offline (secció 2 de model.txt)
      catàleg · biaix de posició w_k (EM + PAVA) · IPS amb comptes efectius (ESS)
      · prior jeràrquic Beta (global -> categoria -> subcategoria -> producte)
      · senyals orgànics de history_events · embeddings TF-IDF amb hashing
      · transicions Dirichlet P(i|j) (co-visita / co-compra) amb prior estructural
      · regressió logística bayesiana contextual (Laplace, case-control)
  FASE B · inferència per petició (secció 3)
      filtre dur d'estoc -> ~100 candidats per quotes -> Thompson Sampling
      (Beta per producte + beta ~ N(mu, Sigma)) -> utilitat (compra + preu)
      -> ordenació (slot 1 explotació, 2-8 MMR + quotes, 9-10 exploració)
  BUCLE ASÍNCRON (secció 4)
      feedback (Beta amb biaix de posició + ADF contextual) · tendències amb
      shop.events · catàleg (nous productes, preus, estoc) · canvi de règim
      (diumenge 09:00 UTC) · circuit breaker · checkpoint local + mètriques a scratch

Ús (des de l'arrel del repo, amb el .env de l'equip):
  .venv/bin/python recommender-v2.py --build      # només construeix i desa els priors
  .venv/bin/python recommender-v2.py --eval 5000  # avaluació offline (últims 7 dies)
  .venv/bin/python recommender-v2.py --smoke      # respon 30 peticions reals sense enviar-les
  .venv/bin/python recommender-v2.py --dry-run    # sistema complet però sense escriure a RawTree
  .venv/bin/python recommender-v2.py              # EN VIU: respon a team_NN.submissions

Desviacions respecte de model.txt, totes per evidència de les dades (configurables amb --set):
  - prior_estimator="exam": theta_i = compres / examinacions esperades (model d'examen 2.7.1) en lloc del
    SNIPS amb propensitats (2.2). Amb SNIPS + sòl de prior 0.001, la mu sortia correlacionada negativament
    amb les vendes (Spearman -0.78); amb "exam", +0.17 i millor HR a les dues avaluacions offline.
  - delta_same=1.0: no es penalitza el producte ja vist (3.4.2). Al carrusel antic, un producte vist a la
    sessió convertia x9-x14 més que un de no vist.
  - lambda_same_cat=2.0 i quota_cat=7 (no 5): el log antic només mostrava la categoria de l'última vista i
    no permet aprendre el cost del producte d'una altra categoria; 82% de compres orgàniques hi són dins.
  - Font de candidats "seguretat": best sellers de les categories de la sessió (recall@100 0.52 -> 0.58).

Simplificacions respecte de model.txt (documentades al codi on apliquen):
  - Persistència: checkpoint local (.cache/) + mètriques a team_NN.scratch, en
    lloc del snapshot complet en chunks (5.1-5.2).
  - Sense refit complet periòdic del contextual (4.3.5, opcional) ni pes IPS
    dels slots d'exploració (4.2.3, opcional).
  - Features d'hora i dia de la setmana (14-17) fora: el stream en viu té ritme
    constant i la final dura 2 h, extrapolar el cicle diari de l'històric és risc.
  - Senyals orgànics reescalats a l'escala del carrusel (q * taxa carrusel /
    taxa orgànica) per no barrejar targets (2.4.2 avisa que són targets diferents).
  - Temperatura d'exploració (T_eff) aplicada al soroll de Thompson: dividir
    tota la eta per T no canvia l'ordre i no exploraria.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
LOGS = ROOT / "logs"
LN2 = math.log(2.0)


# =============================================================================
# 0. CONFIGURACIÓ (model.txt 6.4 i apèndix B)
# =============================================================================
@dataclass
class Config:
    model_version: str = "v2-bayes-ts"
    # --- candidats (3.3)
    n_candidates: int = 100
    q_trans: int = 35
    q_content: int = 20
    q_trend: int = 10
    q_prior: int = 15
    q_explore: int = 10
    q_safe: int = 10
    # --- scoring (3.4.2)
    lambda_T_view_home: float = 0.8      # home / category: substituts
    lambda_T_buy_home: float = 0.3
    lambda_T_view_product: float = 0.4   # product: complements
    lambda_T_buy_product: float = 0.9
    lambda_C: float = 0.5
    # EXTENSIÓ (no és a model.txt): el log antic només mostrava la categoria de l'última vista, de manera
    # que ni l'IPS ni el contextual poden aprendre que el producte d'una altra categoria converteix menys.
    # Evidència: 82% de compres orgàniques dins la categoria vista; best sellers globals 0,9% vs 1,7%.
    lambda_same_cat: float = 2.0
    lambda_price: float = -0.25
    rho: float = 0.06                    # tie-breaker d'ingressos (3.5)
    # model.txt (3.4.2) penalitza el producte ja vist a la pàgina de producte (delta_same=0.6). Les dades diuen el
    # contrari: al carrusel antic, un producte vist a la sessió convertia x9-x14 més (3,9% vs 0,29% als slots 1-3).
    delta_same: float = 1.0              # 1.0 = sense penalització
    pen_click: float = math.log(1 / 0.8)
    temperature: float = 1.0
    calib_a: float = 0.0
    # --- ordenació (3.6)
    lambda_div: float = 0.20
    lambda_div_new_user: float = 0.25
    quota_sub: int = 3
    quota_cat: int = 7                   # model.txt diu 5; 7 per l'evidència de dalt (lambda_same_cat)
    quota_brand_top5: int = 2
    exploit_slot1: bool = True
    epsilon_explore: float = 0.5
    c_ucb: float = 0.15
    c_new: float = 0.6
    explore_ess: float = 50.0
    explore_top: int = 20
    t_eff_new_user: float = 1.2
    cold_reserve_ratio: float = 0.7
    # --- priors (2.2-2.4)
    e_min: float = 0.01
    w_min: float = 0.05
    omega_max: float = 50.0
    ess_max: float = 5000.0
    ess_new: float = 8.0
    kappa_1: float = 0.5
    kappa_2: float = 0.5
    n_max_product_prior: float = 300.0
    k_org: float = 0.01
    org_atc_weight: float = 5.0
    org_smooth: float = 20.0
    half_life_hist_days: float = 1e6      # sense decaïment: no hi ha deriva a l'històric (model.txt: 21 dies; eval empatada)
    prior_estimator: str = "exam"        # "exam" (model d'examen, per defecte) o "ips" (2.2 tal qual)
    prior_mean_floor: float = 1e-5
    # --- transicions (2.5)
    trans_days: int = 30
    coview_sample_mod: int = 2
    top_k_trans: int = 200
    alpha_pop: float = 0.5
    lambda_a: float = 0.6
    lambda_p: float = 1.0
    M0: float = 50.0
    lam_sub_s: float = 1.0
    lam_brand_s: float = 0.6
    lam_cat_s: float = 0.25
    lam_sem_s: float = 0.8
    rho_recency: float = 0.65
    tau_sess: float = 900.0
    max_trans_views: int = 5
    max_centroid_views: int = 10
    atc_trans_weight: float = 2.0
    emb_dim: int = 512
    # --- contextual (2.6, 4.3)
    ctx_train_days: int = 21
    ctx_cases: int = 25000
    ctx_control_sessions: int = 25000
    ctx_controls_per_session: int = 3
    ctx_prior_sd: float = 1.0
    ctx_prior_sd_price: float = 0.5
    ctx_prior_sd_bias: float = 10.0
    newton_iters: int = 10
    adf_weight_click: float = 0.5
    adf_mode: str = "full"               # "full" (tots els mostrats), "positives" (4.3.1 tal qual), "off"
    alpha_min_new: float = 1.0           # Beta del producte nou amb alpha >= 1 (veure add_new_product)
    lambda_sigma: float = 0.97           # reinjecció de variància cada 5 min (4.3.4)
    lambda_mu: float = 0.995
    # --- online (4.2)
    half_life_slow_days: float = 7.0
    half_life_fast_min: float = 20.0
    c_neg_click: float = 0.05
    c_neg_purchase: float = 0.02
    # --- tendències (4.5)
    eta_trend: float = 0.7
    k_trend: float = 0.005
    trend_window_min: int = 15
    trend_baseline_min: int = 180
    ring_minutes: int = 240
    # --- catàleg (4.6)
    n_max_products: int = 8000
    price_eps: float = 0.5
    price_bonus_cap: float = 0.4
    price_half_life_s: float = 7200.0
    restock_bonus: float = 0.15
    restock_half_life_s: float = 3600.0
    knn_k: int = 25
    # --- règim final (4.7)
    final_start_utc: str = "2026-11-15 09:00:00"
    mu_live_normal: float = 0.0
    soft_trigger_cooldown_s: float = 7200.0
    # --- operació (4.8)
    request_poll_s: float = 0.05
    feedback_poll_s: float = 1.5
    events_poll_s: float = 10.0
    catalog_poll_s: float = 0.75
    live_trans_s: float = 300.0
    persist_s: float = 60.0
    deadline_s: float = 2.9
    stock_reconcile_s: float = 60.0
    user_bought_days: int = 30


CFG = Config()
POSITION_DEFAULT = np.array([1.00, 0.72, 0.52, 0.39, 0.30, 0.23, 0.18, 0.14, 0.11, 0.08])
FEATURES = ["bias", "ios_x_pp", "android_x_pp", "product_x_taub", "category_x_tauv",
            "reg0_x_pp", "reg1_x_pp", "reg2_x_pp", "reg3_x_pp", "newuser_x_fview",
            "m_x_content", "same_sub_last", "same_brand_last", "pp_within_sub",
            "upurch_x_fview", "atc_x_taub", "f_view", "f_atc", "f_q", "user_cat_aff",
            "in_session"]
PRICE_FEATURES = {1, 2, 5, 6, 7, 8, 13}
D_FEAT = len(FEATURES)

log = logging.getLogger("v2")


# =============================================================================
# 1. UTILITATS
# =============================================================================
def setup_logging(verbose: bool = False) -> None:
    LOGS.mkdir(exist_ok=True)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = RotatingFileHandler(LOGS / "recommender-v2.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.handlers[:] = [fh, sh]


def load_env() -> None:
    for p in [ROOT / ".env", ROOT / "datathon-fme-2026" / ".env", Path(".env")]:
        if p.exists():
            for line in p.read_text().splitlines():
                k, _, v = line.partition("=")
                k = k.strip()
                if k and not k.startswith("#"):
                    os.environ.setdefault(k, v.strip())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


_FRAC = re.compile(r"\d+")


def parse_ts(s) -> float | None:
    """'2026-11-14 13:18:23.883000000' (UTC) -> epoch en segons."""
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


class TTLCache:
    """Diccionari amb caducitat i mida màxima (LRU per inserció)."""

    def __init__(self, ttl: float, maxsize: int = 200_000):
        self.ttl, self.maxsize = ttl, maxsize
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def add(self, k, v=True, ttl: float | None = None) -> None:
        with self.lock:
            self.d[k] = (v, time.time() + (ttl or self.ttl))
            self.d.move_to_end(k)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)

    def get(self, k, default=None):
        it = self.d.get(k)
        if it is None or it[1] < time.time():
            return default
        return it[0]

    def claim(self, k) -> bool:
        """Afegeix k només si no hi és, de forma atòmica: True si aquest fil l'ha reclamat."""
        with self.lock:
            it = self.d.get(k)
            if it is not None and it[1] >= time.time():
                return False
            self.d[k] = (True, time.time() + self.ttl)
            self.d.move_to_end(k)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)
            return True

    def __contains__(self, k) -> bool:
        return self.get(k, None) is not None

    def __len__(self) -> int:
        return len(self.d)

    def purge(self) -> None:
        now = time.time()
        with self.lock:
            dead = [k for k, (_, exp) in self.d.items() if exp < now]
            for k in dead:
                self.d.pop(k, None)


def pava_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators per a una seqüència no creixent (2.7.2)."""
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
    out = []
    for v, _, cnt in blocks:
        out += [v] * cnt
    return np.array(out)


# =============================================================================
# 2. RAWTREE HTTP (5.9)
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

    def query(self, sql: str, timeout: float = 10.0, retries: int = 0) -> list[dict]:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self._session().post(f"{self.api}/v1/query", json={"sql": sql}, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("retry-after", 1)))
                    raise RuntimeError("429 rate limited")
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                return r.json().get("data", [])
            except Exception as e:  # noqa: BLE001
                last = e
                self.errors += 1
                if attempt < retries:
                    time.sleep(min(0.5 * 2 ** attempt, 5.0))
        raise last  # type: ignore[misc]

    def insert(self, table: str, rows: list[dict], timeout: float = 5.0) -> int:
        if not rows:
            return 0
        r = self._session().post(f"{self.api}/v1/tables/{table}", params={"database": self.team},
                                 json=rows, timeout=timeout)
        if r.status_code >= 400:
            self.errors += 1
            raise RuntimeError(f"insert {table}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            return int(r.json().get("written_rows", len(rows)))
        except Exception:  # noqa: BLE001
            return len(rows)


# =============================================================================
# 3. TEXT: EMBEDDINGS TF-IDF AMB HASHING (2.5.5)
# =============================================================================
TOKEN_RE = re.compile(r"[a-z0-9àèéíòóúç]{2,}")


def embed_product(desc: str, cat: str, sub: str, brand: str, D: int) -> np.ndarray:
    v = np.zeros(D, np.float32)
    toks = TOKEN_RE.findall((desc or "").lower())
    grams = Counter(toks + [a + "_" + b for a, b in zip(toks, toks[1:])])

    def add(tok: str, weight: float) -> None:
        h = zlib.crc32(tok.encode("utf-8")) & 0xFFFFFFFF   # estable entre processos
        v[h % D] += (1.0 if (h >> 31) == 0 else -1.0) * weight

    for tok, tf in grams.items():
        add(tok, 1.0 + math.log1p(tf))
    for tok in (f"cat:{cat}", f"sub:{sub}", f"brand:{brand}"):
        add(tok.lower(), 2.0 * (1.0 + math.log1p(1)))
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 0 else v


# =============================================================================
# 4. FASE A · PRIORS OFFLINE (secció 2)
# =============================================================================
def _cut_filter(cut: str | None) -> str:
    return f" AND ts::DateTime64(3) < toDateTime64('{cut}', 3)" if cut else ""


def estimate_position_bias(rows: list[dict], index: dict) -> tuple[np.ndarray, dict]:
    """EM del model d'examen P(clic|i,k) = w_k r_i + PAVA (2.7)."""
    P = len(index)
    n = np.zeros((P, 10))
    c = np.zeros((P, 10))
    for r in rows:
        i = index.get(r["product_id"])
        k = int(r["slot"]) - 1
        if i is None or not 0 <= k < 10:
            continue
        n[i, k] += r["imp"]
        c[i, k] += r["clk"]
    ctr = (c.sum(0) + 0.5) / (n.sum(0) + 0.5)
    w = ctr / ctr[0]
    it = 0
    for it in range(50):
        r_i = c.sum(1) / np.maximum((n * w).sum(1), 1e-9)
        w_new = c.sum(0) / np.maximum((n * r_i[:, None]).sum(0), 1e-12)
        w_new = np.clip(w_new / w_new[0], 1e-3, 1.0)
        done = np.abs(w_new - w).sum() < 1e-6
        w = w_new
        if done:
            break
    w = np.clip(pava_nonincreasing(w, n.sum(0)), 0.03, 1.0)
    w[0] = 1.0
    info = {"em_iters": it + 1, "ctr_slot": (c.sum(0) / np.maximum(n.sum(0), 1)).round(5).tolist()}
    if w[1] / w[0] > 0.9:  # 2.7.4: log sense prou randomització
        info["fallback_default"] = True
        w = POSITION_DEFAULT.copy()
    return w, info


def hierarchical_prior(theta, n_eff, cat, sub, ncat, nsub, cfg: Config):
    """Empirical Bayes per moments i propagació global -> cat -> subcat -> producte (2.3)."""
    has = n_eff > 0

    def moments(mask):
        wts, th = n_eff[mask], theta[mask]
        if wts.size < 2 or wts.sum() <= 0:
            return None
        m = float((wts * th).sum() / wts.sum())
        v = float((wts * (th - m) ** 2).sum() / wts.sum())
        m = float(np.clip(m, cfg.prior_mean_floor, 0.25))   # model.txt: 0.001; massa alt per a aquest log
        if v <= 1e-12:
            return m, 0.0
        return m, float(np.clip(m * (1 - m) / v - 1, 0.5, 500))

    g = moments(has) or (0.004, 1.0)
    mG, nuG = g[0], max(g[1], 0.5)
    mc, nuc = np.full(ncat, mG), np.full(ncat, nuG)
    for k in range(ncat):
        r = moments(has & (cat == k))
        nu_own, m_own = (r[1], r[0]) if r else (0.0, mG)
        mc[k] = (nu_own * m_own + cfg.kappa_1 * nuG * mG) / (nu_own + cfg.kappa_1 * nuG)
        nuc[k] = np.clip(nu_own + cfg.kappa_1 * nuG, 0.5, 500)
    sub_cat = np.zeros(nsub, int)
    for s in range(nsub):
        cs = cat[sub == s]
        sub_cat[s] = Counter(cs.tolist()).most_common(1)[0][0] if cs.size else 0
    ms, nus = np.zeros(nsub), np.zeros(nsub)
    for s in range(nsub):
        pc = sub_cat[s]
        r = moments(has & (sub == s))
        nu_own, m_own = (r[1], r[0]) if r else (0.0, mc[pc])
        ms[s] = (nu_own * m_own + cfg.kappa_2 * nuc[pc] * mc[pc]) / (nu_own + cfg.kappa_2 * nuc[pc])
        nus[s] = np.clip(nu_own + cfg.kappa_2 * nuc[pc], 0.5, 500)
    nu_i = np.clip(nus[sub], 1, cfg.n_max_product_prior)
    return ms[sub] * nu_i, (1 - ms[sub]) * nu_i, {"mG": mG, "nuG": nuG, "m_cat": mc, "nu_cat": nuc,
                                                   "m_sub": ms, "nu_sub": nus}


def build_transitions(cfg: Config, n, cat, sub, brand, E, pairs: tuple):
    """Dirichlet P(i|j) = (n~_ji + m_ji) / (sum n~_j + M0), top-K per fila en CSR (2.5)."""
    pj, pi, cnt = pairs
    n_in = np.bincount(pi, weights=cnt, minlength=n)
    nt = cnt / np.maximum(n_in[pi], 1.0) ** cfg.alpha_pop          # correcció de popularitat (20b)
    order = np.argsort(pj, kind="stable")
    pj, pi, nt = pj[order], pi[order], nt[order]
    starts = np.searchsorted(pj, np.arange(n + 1))
    rowsum = np.bincount(pj, weights=nt, minlength=n)
    K = min(cfg.top_k_trans, n - 1)
    indptr = np.zeros(n + 1, np.int64)
    indices = np.zeros(n * K, np.int32)
    data = np.zeros(n * K, np.float32)
    Zs = np.zeros(n)
    for b0 in range(0, n, 400):
        rows = np.arange(b0, min(n, b0 + 400))
        raw = (cfg.lam_sub_s * (sub[rows, None] == sub[None, :n])
               + cfg.lam_brand_s * (brand[rows, None] == brand[None, :n])
               + cfg.lam_cat_s * (cat[rows, None] == cat[None, :n])).astype(np.float32)
        raw += cfg.lam_sem_s * np.clip(E[rows] @ E[:n].T, 0, None) ** 2
        raw[np.arange(rows.size), rows] = 0.0
        Z = raw.sum(1)
        Zs[rows] = Z
        m = cfg.M0 * raw / np.maximum(Z, 1e-9)[:, None]
        for r_i, j in enumerate(rows):
            s, e = starts[j], starts[j + 1]
            if e > s:
                np.add.at(m[r_i], pi[s:e], nt[s:e])     # parells repetits (compra + cistella) se sumen
        Pm = m / (rowsum[rows] + cfg.M0)[:, None]
        top = np.argpartition(-Pm, K, axis=1)[:, :K]
        vals = np.take_along_axis(Pm, top, axis=1)
        vals = vals / np.maximum(vals.sum(1, keepdims=True), 1e-12)
        for r_i, j in enumerate(rows):
            indices[j * K:(j + 1) * K] = top[r_i]
            data[j * K:(j + 1) * K] = vals[r_i]
    indptr[:] = np.arange(n + 1) * K
    return {"indptr": indptr, "indices": indices, "data": data, "Z": Zs,
            "denom": rowsum + cfg.M0, "norm": float(np.percentile(data[data > 0], 95))}


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size)
    r[order] = np.arange(a.size)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]          # rangs mitjans per als empats


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1]) if a.size > 2 else 0.0


def _pairs(rows: list[dict], index: dict, w: float = 1.0) -> tuple:
    pj, pi, c = [], [], []
    for r in rows:
        a, b = index.get(r["p"]), index.get(r["q"])
        if a is not None and b is not None and a != b:
            pj.append(a)
            pi.append(b)
            c.append(float(r["n"]) * w)
    return np.array(pj, np.int64), np.array(pi, np.int64), np.array(c, np.float64)


def build_priors(rt: RawTree, cfg: Config, cut: str | None = None) -> dict:
    t0 = time.time()
    cs = _cut_filter(cut)
    hist_end = rt.query("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.history_sessions", 60)[0]["t"]
    ref = cut or hist_end[:19]
    log.info("[build] priors amb dades fins a %s (cut=%s)", ref, cut)

    # --- A1 catàleg (shop.catalog: els nous productes i canvis s'apliquen en viu)
    rows = rt.query("""SELECT product_id::String AS product_id, toString(category) AS category,
        toString(subcategory) AS subcategory, toString(brand) AS brand, price_eur::Float64 AS price,
        toString(description) AS description, toString(is_active) AS active
        FROM shop.catalog ORDER BY product_id""", 60)
    ids = [r["product_id"] for r in rows]
    index = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    cats = sorted({r["category"] for r in rows})
    subs = sorted({r["subcategory"] for r in rows})
    brands = sorted({r["brand"] for r in rows})
    cat = np.array([cats.index(r["category"]) for r in rows], np.int16)
    sub = np.array([subs.index(r["subcategory"]) for r in rows], np.int16)
    brand = np.array([brands.index(r["brand"]) for r in rows], np.int16)
    price = np.array([float(r["price"] or 0) for r in rows], np.float32)
    active = np.array([str(r["active"]).lower() in ("true", "1") for r in rows])
    desc = [r.get("description") or "" for r in rows]
    log.info("[build] catàleg: %d productes, %d categories, %d subcategories, %d marques",
             n, len(cats), len(subs), len(brands))

    # --- A6 biaix de posició (2.7)
    sc = rt.query(f"""SELECT arrayElement(ids, slot) AS product_id, slot, count() AS imp,
        countIf(clicked = arrayElement(ids, slot)) AS clk
        FROM (SELECT shown::Array(String) AS ids, ifNull(toString(clicked), '') AS clicked
              FROM shop.history_sessions WHERE 1 {cs})
        ARRAY JOIN arrayEnumerate(ids) AS slot GROUP BY product_id, slot""", 120)
    w, winfo = estimate_position_bias(sc, index)
    log.info("[build] biaix de posició w_k = %s %s", np.round(w, 3).tolist(),
             "(per defecte)" if winfo.get("fallback_default") else "")

    # --- A2 IPS amb correcció de posició i decaïment (2.2, 5.4.2)
    wlit = "[" + ",".join(f"{x:.6f}" for x in w) + "]"
    hl = cfg.half_life_hist_days
    ips = rt.query(f"""SELECT product_id, count() AS imp, sum(om) AS sw, sum(om * y) AS sy, sum(om * om) AS sw2,
            sum(dw * wk) AS exam, sum(dw * y) AS buys
        FROM (SELECT arrayElement(ids, slot) AS product_id, toFloat64(arrayElement(ids, slot) = purchased) AS y,
                arrayElement({wlit}, slot) AS wk,
                exp(-{LN2} * dateDiff('second', t, toDateTime64('{ref}', 3)) / 86400.0 / {hl}) AS dw,
                least(1.0 / (greatest(arrayElement(props, slot), {cfg.e_min})
                             * greatest(arrayElement({wlit}, slot), {cfg.w_min})), {cfg.omega_max}) * dw AS om
              FROM (SELECT shown::Array(String) AS ids, propensity::Array(Float64) AS props,
                           ifNull(toString(purchased), '') AS purchased, ts::DateTime64(3) AS t
                    FROM shop.history_sessions WHERE 1 {cs})
              ARRAY JOIN arrayEnumerate(ids) AS slot)
        GROUP BY product_id""", 120)
    sw, sy, sw2, imp, exam, buys = (np.zeros(n) for _ in range(6))
    for r in ips:
        i = index.get(r["product_id"])
        if i is not None:
            sw[i], sy[i], sw2[i], imp[i], exam[i], buys[i] = r["sw"], r["sy"], r["sw2"], r["imp"], r["exam"], r["buys"]
    ess_ips = np.minimum(np.where(sw2 > 0, sw ** 2 / np.maximum(sw2, 1e-12), 0.0), cfg.ess_max)
    theta_ips = np.where(sw > 0, sy / np.maximum(sw, 1e-12), 0.0)
    # Estimador per defecte: MLE del model d'examen P(compra|i,k) = w_k * theta_i (2.7.1):
    #   theta_i = compres_i / sum_k n_ik w_k,  ESS_i = examinacions esperades.
    # El SNIPS amb propensitats (2.2) queda com a opció: sobre aquest log dona taxes ~0 a la cua
    # i, amb el sòl del prior, una mu correlacionada NEGATIVAMENT amb les vendes (rho = -0.78).
    if cfg.prior_estimator == "ips":
        ess, theta, has = ess_ips, theta_ips, sw > 0
    else:
        ess = np.minimum(exam, cfg.ess_max)
        theta = np.where(exam > 0, buys / np.maximum(exam, 1e-12), 0.0)
        has = exam > 0

    # --- A3 prior jeràrquic (2.3)
    alpha0, beta0, hinfo = hierarchical_prior(theta, ess, cat, sub, len(cats), len(subs), cfg)
    m_prior = alpha0 / (alpha0 + beta0)
    theta = np.where(has, theta, m_prior)
    s_slow, f_slow = ess * theta, ess * (1 - theta)
    log.info("[build] prior (%s): %d productes amb impressions · ESS mitjà %.1f · m_global %.5f · nu_global %.1f",
             cfg.prior_estimator, int((imp > 0).sum()), float(ess[imp > 0].mean()), hinfo["mG"], hinfo["nuG"])

    # --- A3b senyals orgànics (2.4)
    ce = _cut_filter(cut)
    org = rt.query(f"""SELECT product_id, sumIf(dw, t = 'view') AS v, sumIf(dw, t = 'add_to_cart') AS a,
        sumIf(dw, t = 'purchase') AS p, countIf(t = 'purchase') AS p_raw
        FROM (SELECT product_id::String AS product_id, event_type::String AS t,
                exp(-{LN2} * dateDiff('second', ts::DateTime64(3), toDateTime64('{ref}', 3)) / 86400.0 / {hl}) AS dw
              FROM shop.history_events WHERE 1 {ce})
        GROUP BY product_id""", 120)
    V, A, Pp, Praw = (np.zeros(n) for _ in range(4))
    for r in org:
        i = index.get(r["product_id"])
        if i is not None:
            V[i], A[i], Pp[i], Praw[i] = r["v"], r["a"], r["p"], r["p_raw"]
    nsub = len(subs)
    Vs = np.bincount(sub, weights=V, minlength=nsub)
    Ps = np.bincount(sub, weights=Pp, minlength=nsub)
    As = np.bincount(sub, weights=A, minlength=nsub)
    cnt_s = np.maximum(np.bincount(sub, minlength=nsub), 1)
    m_s_org = (Ps + 1) / (Vs + 20)
    q = np.clip((Pp + cfg.org_smooth * m_s_org[sub]) / (V + cfg.org_smooth), 0, 1)
    r_scale = hinfo["mG"] / max(Pp.sum() / max(V.sum(), 1e-9), 1e-9)   # escala orgànic -> carrusel
    n_org = cfg.k_org * (V + cfg.org_atc_weight * A)
    s_org = n_org * np.clip(q * r_scale, 0, 1)
    f_org = np.maximum(n_org - s_org, 0)

    def stdz(x):
        sd = x.std()
        return ((x - x.mean()) / sd).astype(np.float32) if sd > 0 else np.zeros_like(x, np.float32)

    f_view = stdz(np.log1p(V))
    f_atc = stdz(np.log1p(A) - np.log1p((As / cnt_s)[sub]))
    f_q = stdz(logit(np.clip(q, 0.001, 0.5)))
    mu_post = (alpha0 + s_slow + s_org) / (alpha0 + beta0 + s_slow + f_slow + s_org + f_org)
    log.info("[build] sanitat (2.3.5): Spearman(mu, compres orgàniques)=%.2f · Spearman(theta_IPS, mu)=%.2f",
             _spearman(mu_post, Praw), _spearman(theta_ips, mu_post))

    # --- A4 embeddings i transicions (2.5)
    E = np.stack([embed_product(desc[i], cats[cat[i]], subs[sub[i]], brands[brand[i]], cfg.emb_dim)
                  for i in range(n)])
    win = f"ts::DateTime64(3) >= toDateTime64('{ref}', 3) - INTERVAL {cfg.trans_days} DAY {ce}"
    cv = rt.query(f"""WITH v AS (SELECT session_id::String AS sid, product_id::String AS pid FROM shop.history_events
        WHERE event_type::String = 'view' AND {win} AND cityHash64(session_id::String) % {cfg.coview_sample_mod} = 0)
        SELECT a.pid AS p, b.pid AS q, count() AS n FROM v AS a INNER JOIN v AS b ON a.sid = b.sid
        WHERE a.pid != b.pid GROUP BY p, q ORDER BY n DESC LIMIT {cfg.top_k_trans} BY p""", 300)
    buy_rows = []
    for kind, lam in (("purchase", cfg.lambda_p), ("add_to_cart", cfg.lambda_a)):
        rr = rt.query(f"""WITH v AS (SELECT session_id::String AS sid, product_id::String AS pid, ts::DateTime64(3) AS t
                FROM shop.history_events WHERE event_type::String = 'view' AND {win}),
            b AS (SELECT session_id::String AS sid, product_id::String AS pid, ts::DateTime64(3) AS t
                FROM shop.history_events WHERE event_type::String = '{kind}' AND {win})
            SELECT v.pid AS p, b.pid AS q, count() AS n FROM v INNER JOIN b ON v.sid = b.sid
            WHERE v.pid != b.pid AND v.t <= b.t GROUP BY p, q ORDER BY n DESC LIMIT {cfg.top_k_trans} BY p""", 300)
        buy_rows.append(_pairs(rr, index, lam))
    pv = _pairs(cv, index)
    pb = tuple(np.concatenate([a[k] for a in buy_rows]) for k in range(3))
    tv = build_transitions(cfg, n, cat, sub, brand, E, pv)
    tb = build_transitions(cfg, n, cat, sub, brand, E, pb)
    log.info("[build] transicions: %d parells co-visita, %d parells co-compra/cistella",
             pv[0].size, pb[0].size)

    # --- usuaris: compres 30 dies, preu mitjà i afinitat per categoria (4.4, feature 29)
    ub = rt.query(f"""SELECT user_id::String AS u, groupUniqArray(product_id::String) AS ps, count() AS n,
        avg(log1p(price_eur::Float64)) AS lp FROM shop.history_events
        WHERE event_type::String = 'purchase'
          AND ts::DateTime64(3) >= toDateTime64('{ref}', 3) - INTERVAL {cfg.user_bought_days} DAY {ce}
        GROUP BY u""", 120)
    user_bought, user_npurch, user_logp = {}, {}, {}
    for r in ub:
        s = frozenset(index[p] for p in r["ps"] if p in index)
        if s:
            user_bought[r["u"]] = s
        user_npurch[r["u"]] = int(r["n"])
        user_logp[r["u"]] = float(r["lp"])
    aff_raw: dict[str, np.ndarray] = {}
    for part in range(4):
        rr = rt.query(f"""SELECT e.u AS u, c.cat AS cat, sumIf(e.dw, e.t = 'view') AS v,
            sumIf(e.dw, e.t = 'add_to_cart') AS a, sumIf(e.dw, e.t = 'purchase') AS p
            FROM (SELECT user_id::String AS u, product_id::String AS pid, event_type::String AS t,
                    exp(-{LN2} * dateDiff('second', ts::DateTime64(3), toDateTime64('{ref}', 3)) / 86400.0 / {hl}) AS dw
                  FROM shop.history_events WHERE cityHash64(user_id::String) % 4 = {part} {ce}) AS e
            INNER JOIN (SELECT product_id::String AS pid, category::String AS cat FROM shop.catalog) AS c
              ON c.pid = e.pid GROUP BY u, cat""", 120)
        for r in rr:
            k = cats.index(r["cat"]) if r["cat"] in cats else None
            if k is None:
                continue
            arr = aff_raw.setdefault(r["u"], np.zeros(len(cats)))
            arr[k] += r["p"] + 0.25 * r["a"] + 0.02 * r["v"]
    tot = np.sum([a for a in aff_raw.values()], axis=0) if aff_raw else np.ones(len(cats))
    m_cat_aff = tot / max(tot.sum(), 1e-9)
    user_aff = {u: ((a + 20 * m_cat_aff) / (a.sum() + 20)).astype(np.float32) for u, a in aff_raw.items()}
    log.info("[build] usuaris: %d amb compres (30 dies), %d amb afinitat", len(user_bought), len(user_aff))

    # --- preu
    order_p = np.argsort(price)
    price_pct = np.empty(n, np.float32)
    price_pct[order_p] = (np.arange(n) + 0.5) / n
    price_pct_sub = np.zeros(n, np.float32)
    sub_sorted_prices = {}
    for s in range(nsub):
        idx = np.flatnonzero(sub == s)
        if idx.size:
            o = idx[np.argsort(price[idx])]
            price_pct_sub[o] = (np.arange(idx.size) + 0.5) / idx.size
            sub_sorted_prices[s] = np.sort(price[idx])

    P = dict(version=2, built_at=time.time(), cut=cut, ref=ref, ids=ids, cats=cats, subs=subs, brands=brands,
             cat=cat, sub=sub, brand=brand, price=price, active=active, desc=desc, w=w, winfo=winfo,
             alpha0=alpha0, beta0=beta0, s_slow=s_slow, f_slow=f_slow, s_org=s_org, f_org=f_org,
             ess_ips=ess, imp=imp, hinfo=hinfo, f_view=f_view, f_atc=f_atc, f_q=f_q, p_raw=Praw,
             r_scale=r_scale, E=E, tv=tv, tb=tb, user_bought=user_bought, user_npurch=user_npurch,
             user_logp=user_logp, user_aff=user_aff, m_cat_aff=m_cat_aff.astype(np.float32),
             price_pct=price_pct, price_pct_sub=price_pct_sub, sub_sorted_prices=sub_sorted_prices,
             p95_price=float(np.percentile(price, 95)), mean_logp=float(np.log1p(price).mean()),
             mu_beta=np.zeros(D_FEAT), Sigma_beta=np.eye(D_FEAT), feat_mean=np.zeros(D_FEAT),
             feat_std=np.ones(D_FEAT))

    # --- A5 contextual (2.6) amb les mateixes funcions que s'usen en viu
    try:
        fit_contextual(rt, cfg, P, ref, cs)
    except Exception as e:  # noqa: BLE001
        log.exception("[build] contextual ha fallat; es continua amb beta = 0: %s", e)
    log.info("[build] priors llestos en %.0f s", time.time() - t0)
    return P


def fit_contextual(rt: RawTree, cfg: Config, P: dict, ref: str, cs: str) -> None:
    """Logística bayesiana Bernoulli amb offset logit(mu_i), case-control + Laplace (2.6.3-2.6.4)."""
    t0 = time.time()
    win = f"ts::DateTime64(3) >= toDateTime64('{ref}', 3) - INTERVAL {cfg.ctx_train_days} DAY {cs}"
    cols = """toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id, toString(context.device) AS device,
        toString(context.region) AS region, toString(context.page) AS page, recent_views,
        shown::Array(String) AS shown, ifNull(toString(clicked), '') AS clicked,
        ifNull(toString(purchased), '') AS purchased"""
    # Ull: l'àlies 'ifNull(toString(purchased), '') AS purchased' amaga la columna al WHERE (ClickHouse).
    cases = rt.query(f"""SELECT {cols} FROM shop.history_sessions WHERE purchased != '' AND {win}
        ORDER BY cityHash64(request_id::String) LIMIT {cfg.ctx_cases}""", 120)
    ctrls = rt.query(f"""SELECT {cols} FROM shop.history_sessions WHERE {win}
        ORDER BY cityHash64(toString(request_id) || 'c') LIMIT {cfg.ctx_control_sessions}""", 120)
    n_sess = int(rt.query(f"SELECT count() AS n FROM shop.history_sessions WHERE {win}", 60)[0]["n"])
    ms = MutableState(P, cfg)
    snap = ms.publish(set_global=False)
    w = P["w"]
    rng = np.random.default_rng(7)
    X, y, om, off = [], [], [], []

    def add(row, picks):
        req = {"user_id": row["user_id"], "is_new_user": "false", "device": row["device"], "region": row["region"],
               "page": row["page"], "recent_views": row["recent_views"], "ts": row["ts"]}
        rc = make_ctx(snap, req)
        cand = np.array([i for i, _, _ in picks], np.int64)
        Z = raw_features(snap, rc, cand)
        X.append(Z)
        y.extend(yy for _, yy, _ in picks)
        om.extend(ww for _, _, ww in picks)
        off.append(logit(snap.mu[cand]))

    for row in cases:
        idx = snap.index.get(row["purchased"])
        if idx is None:
            continue
        add(row, [(idx, 1.0, 1.0)])
    w_ctrl = 0.0
    for row in ctrls:
        shown = row["shown"] or []
        opts = [(k, p) for k, p in enumerate(shown[:10]) if p != row["purchased"] and p in snap.index]
        if not opts:
            continue
        sel = rng.choice(len(opts), size=min(cfg.ctx_controls_per_session, len(opts)), replace=False)
        picks = []
        for s_ in sel:
            k, p = opts[int(s_)]
            ww = 1.0 if p == row["clicked"] else float(w[k])
            w_ctrl += ww
            picks.append((snap.index[p], 0.0, ww))
        add(row, picks)
    X = np.vstack(X).astype(np.float64)
    y, om, off = np.array(y), np.array(om), np.concatenate(off)
    mean = X.mean(0)
    sd = X.std(0)
    mean[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-9] = 0.0
    Xs = np.where(sd > 0, (X - mean) / np.where(sd > 0, sd, 1.0), 0.0)
    Xs[:, 0] = 1.0
    tau = np.full(D_FEAT, cfg.ctx_prior_sd)
    tau[list(PRICE_FEATURES)] = cfg.ctx_prior_sd_price
    tau[0] = cfg.ctx_prior_sd_bias
    Lam0 = np.diag(1 / tau ** 2)
    beta = np.zeros(D_FEAT)

    def objective(b):
        eta = off + Xs @ b
        return float((om * (y * eta - np.logaddexp(0, eta))).sum() - 0.5 * b @ Lam0 @ b)

    obj = objective(beta)
    for _ in range(cfg.newton_iters):
        p = expit(off + Xs @ beta)
        g = Xs.T @ (om * (y - p)) - Lam0 @ beta
        H = (Xs * (om * p * (1 - p))[:, None]).T @ Xs + Lam0
        step = np.linalg.solve(H + 1e-9 * np.eye(D_FEAT), g)
        t = 1.0
        while t > 1e-3:
            nb = beta + t * step
            no = objective(nb)
            if no >= obj - 1e-9:
                beta, obj = nb, no
                break
            t /= 2
        if np.abs(t * step).max() < 1e-6:
            break
    p = expit(off + Xs @ beta)
    H = (Xs * (om * p * (1 - p))[:, None]).T @ Xs + Lam0
    Sigma = np.linalg.inv(H + 1e-9 * np.eye(D_FEAT))
    # correcció de l'intercept del case-control: controls mostrejats amb taxa r
    w_pop = n_sess * float(np.sum(w))
    r = max(w_ctrl / max(w_pop, 1e-9), 1e-9)
    beta[0] += math.log(r)
    P.update(mu_beta=beta, Sigma_beta=(Sigma + Sigma.T) / 2, feat_mean=mean, feat_std=sd,
             ctx_info={"rows": int(len(y)), "cases": int(y.sum()), "n_sess": n_sess, "r_ctrl": r,
                       "coef": {FEATURES[k]: round(float(beta[k]), 4) for k in range(D_FEAT)}})
    log.info("[build] contextual: %d files (%d compres), %.0f s · coeficients: %s", len(y), int(y.sum()),
             time.time() - t0, {k: v for k, v in P["ctx_info"]["coef"].items() if abs(v) > 0.05})


def load_or_build_priors(rt: RawTree, cfg: Config, cut: str | None, force: bool, max_age_h: float = 12) -> dict:
    CACHE.mkdir(exist_ok=True)
    tag = "live" if cut is None else "cut_" + re.sub(r"[^0-9]", "", cut)[:12]
    path = CACHE / f"priors_{tag}.pkl"
    if path.exists() and not force:
        try:
            P = pickle.loads(path.read_bytes())
            if P.get("version") == 2 and time.time() - P["built_at"] < max_age_h * 3600:
                log.info("[build] priors carregats de %s (%.1f h)", path.name, (time.time() - P["built_at"]) / 3600)
                return P
        except Exception as e:  # noqa: BLE001
            log.warning("[build] no s'ha pogut llegir %s: %s", path, e)
    P = build_priors(rt, cfg, cut)
    path.write_bytes(pickle.dumps(P, protocol=pickle.HIGHEST_PROTOCOL))
    log.info("[build] priors desats a %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return P


# =============================================================================
# 5. ESTAT: MUTABLE (workers) I SNAPSHOT IMMUTABLE (inferència) (1.4, 4.6.8)
# =============================================================================
class Snapshot:
    """Estat que llegeix answer(). Mai es modifica després de publicar-lo."""
    __slots__ = ("n", "n_off", "ids", "index", "cat", "sub", "brand", "price", "log_price", "price_pct",
                 "price_pct_sub", "price_q", "ok", "is_new", "alpha", "beta", "mu", "ess_data", "E",
                 "tv", "tb", "live", "struct_cat", "f_view", "f_atc", "f_q", "mu_beta", "L_beta",
                 "feat_mean", "feat_std", "trend", "new_bonus", "restock_bonus", "price_bonus",
                 "best_safe", "new_idx", "new_set", "new_subcats", "user_bought", "user_npurch", "user_logp",
                 "user_aff", "user_clicked", "live_atc", "m_cat_aff", "mean_logp", "params", "cb", "w",
                 "ts", "trend_new_hot", "pop")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class MutableState:
    def __init__(self, P: dict, cfg: Config):
        self.cfg, self.P = cfg, P
        N = cfg.n_max_products
        n = len(P["ids"])
        self.lock = threading.RLock()
        self.n = self.n_off = n
        self.ids: list[str] = list(P["ids"])
        self.index: dict[str, int] = {p: i for i, p in enumerate(self.ids)}
        self.cats, self.subs, self.brands = list(P["cats"]), list(P["subs"]), list(P["brands"])

        def arr(src, dtype, fill=0):
            a = np.full(N, fill, dtype)
            a[:n] = src
            return a

        self.cat = arr(P["cat"], np.int16)
        self.sub = arr(P["sub"], np.int16)
        self.brand = arr(P["brand"], np.int16)
        self.price = arr(P["price"], np.float64)
        self.price_pct = arr(P["price_pct"], np.float32)
        self.price_pct_sub = arr(P["price_pct_sub"], np.float32)
        self.active = arr(P["active"], bool, False)
        self.unavailable = np.zeros(N, bool)
        self.is_new = np.zeros(N, bool)
        self.alpha0 = arr(P["alpha0"], np.float64, 1.0)
        self.beta0 = arr(P["beta0"], np.float64, 100.0)
        self.s_slow = arr(P["s_slow"], np.float64)
        self.f_slow = arr(P["f_slow"], np.float64)
        self.s_fast, self.f_fast = np.zeros(N), np.zeros(N)
        self.s_click, self.f_click = np.zeros(N), np.zeros(N)
        self.s_org = arr(P["s_org"], np.float64)
        self.f_org = arr(P["f_org"], np.float64)
        self.f_view = arr(P["f_view"], np.float32)
        self.f_atc = arr(P["f_atc"], np.float32)
        self.f_q = arr(P["f_q"], np.float32)
        self.p_raw = arr(P["p_raw"], np.float64)
        self.E = np.zeros((N, cfg.emb_dim), np.float32)
        self.E[:n] = P["E"]
        self.tv, self.tb = P["tv"], P["tb"]
        self.live = None
        self.mu_beta = np.array(P["mu_beta"], float)
        self.Sigma_beta = np.array(P["Sigma_beta"], float)
        tau = np.full(D_FEAT, cfg.ctx_prior_sd)
        tau[list(PRICE_FEATURES)] = cfg.ctx_prior_sd_price
        tau[0] = 1.0                     # online: la incertesa de l'intercept es reinjecta amb sd 1, no 10
        self.Sigma0 = np.diag(tau ** 2)
        self.feat_mean = np.array(P["feat_mean"], float)
        self.feat_std = np.array(P["feat_std"], float)
        self.trend = np.zeros(N)
        self.price_b0, self.price_ts = np.zeros(N), np.zeros(N)
        self.restock_ts = np.full(N, -1e18)
        self.new_set: set[str] = set()
        self.new_subcats: set[int] = set()
        self.last_change_ts: dict[str, float] = {}
        self.user_bought: dict[str, frozenset] = dict(P["user_bought"])
        self.user_npurch: dict[str, int] = dict(P["user_npurch"])
        self.user_logp: dict[str, float] = dict(P["user_logp"])
        self.user_aff: dict[str, np.ndarray] = P["user_aff"]
        self.user_clicked = TTLCache(3600, 100_000)
        self.live_atc = TTLCache(1800, 100_000)
        self.w = np.array(P["w"], float)
        hi = P["hinfo"]
        self.m_sub = np.array(hi["m_sub"], float)
        self.mG = float(hi["mG"])
        self.sub_sorted_prices = P["sub_sorted_prices"]
        self.sorted_prices = np.sort(P["price"])
        self.p95_price = P["p95_price"]
        # règim, circuit breaker, mètriques
        self.final_trigger_ts: float | None = None
        self.hard_fired = False
        self.last_soft_trigger = -1e18
        self.cb = "NORMAL"
        self.last_decay = time.time()
        self.last_sigma_reinject = time.time()
        # tendències: ring buffer per minut (4.5.1)
        R = cfg.ring_minutes
        self.ring_v = np.zeros((R, N), np.int32)
        self.ring_a = np.zeros((R, N), np.int32)
        self.ring_p = np.zeros((R, N), np.int32)
        self.ring_min = np.full(R, -1, np.int64)
        self.trend_hot = np.zeros(N, np.int16)
        self.trend_last = time.time()
        self.trend_new_hot: frozenset = frozenset()
        self.lam_now = np.zeros(N)
        self.sub_tv = 0.0

    # --- paràmetres segons el règim (4.7.2-4.7.3)
    def params(self, now: float | None = None) -> dict:
        cfg = self.cfg
        now = now or time.time()
        base = dict(rho_fast=0.0, ess_max=cfg.ess_max, t_eff=1.0, eps=cfg.epsilon_explore, c_ucb=cfg.c_ucb,
                    c_new=cfg.c_new, eta_T=cfg.eta_trend, K_T=cfg.k_trend, ess_new=cfg.ess_new,
                    mu_live=cfg.mu_live_normal, lambda_sigma=cfg.lambda_sigma, lambda_mu=cfg.lambda_mu,
                    regime="normal")
        t0 = self.final_trigger_ts
        if t0 is None or now < t0:
            return base
        e = (now - t0) / 60.0
        shift = dict(rho_fast=0.6, ess_max=500.0, t_eff=1.25, eps=0.7, c_ucb=0.25, c_new=0.8, eta_T=1.0,
                     K_T=0.01, ess_new=6.0, lambda_sigma=0.90, lambda_mu=0.99)
        mu_live = 0.3 + 0.4 * min(e / 30.0, 1.0)
        if e < 30:
            out = dict(shift, regime="final-shift")
        elif e < 60:
            f = (e - 30) / 30
            out = dict(shift, regime="final-anneal")
            for k in ("t_eff", "eps", "c_ucb", "c_new"):
                out[k] = shift[k] + f * (base[k] - shift[k])
            out["rho_fast"] = 0.6 - 0.3 * f
        else:
            out = dict(base, rho_fast=0.3, ess_max=2000.0, eta_T=1.0, K_T=0.01, ess_new=6.0,
                       lambda_sigma=0.90, lambda_mu=0.99, regime="final")
        out["mu_live"] = mu_live
        return out

    # --- publicació atòmica d'un snapshot nou
    def publish(self, set_global: bool = True) -> Snapshot:
        with self.lock:
            cfg, n, now = self.cfg, self.n, time.time()
            prm = self.params(now)
            alpha = self.alpha0[:n] + self.s_slow[:n] + self.s_fast[:n] + self.s_org[:n]
            beta = self.beta0[:n] + self.f_slow[:n] + self.f_fast[:n] + self.f_org[:n]
            alpha = np.maximum(alpha, 1e-3)
            beta = np.maximum(beta, 1e-3)
            mu = alpha / (alpha + beta)
            ess_data = self.s_slow[:n] + self.f_slow[:n] + self.s_fast[:n] + self.f_fast[:n]
            is_new = self.is_new[:n].copy()
            # (69) amb l'evidència pròpia (no la força del prior): 0,2 al principi i baixa amb les dades
            new_bonus = np.where(is_new, prm["c_new"] / np.sqrt(ess_data + prm["ess_new"] + 1), 0.0)
            price_bonus = self.price_b0[:n] * 0.5 ** ((now - self.price_ts[:n]) / cfg.price_half_life_s)
            restock = cfg.restock_bonus * 0.5 ** ((now - self.restock_ts[:n]) / cfg.restock_half_life_s)
            restock[self.restock_ts[:n] < 0] = 0.0
            price = self.price[:n].copy()
            ok = self.active[:n] & ~self.unavailable[:n] & (price > 0)
            trend = self.trend[:n].copy() if self.cb == "NORMAL" else np.zeros(n)
            order = np.lexsort((-mu, -self.p_raw[:n]))           # best sellers, desempat per mu
            best_safe = order[ok[order]][:80]
            try:
                L = np.linalg.cholesky(self.Sigma_beta + 1e-8 * np.eye(D_FEAT))
            except np.linalg.LinAlgError:
                log.error("Sigma_beta no és definida positiva: es reinicia a Sigma_0")
                self.Sigma_beta = self.Sigma0.copy()
                L = np.linalg.cholesky(self.Sigma_beta)
            new_idx = np.flatnonzero(is_new)
            snap = Snapshot(
                n=n, n_off=self.n_off, ids=self.ids, index=self.index, cat=self.cat[:n].copy(),
                sub=self.sub[:n].copy(), brand=self.brand[:n].copy(), price=price,
                log_price=np.log1p(price), price_pct=self.price_pct[:n].copy(),
                price_pct_sub=self.price_pct_sub[:n].copy(),
                price_q=np.clip(np.log1p(price) / math.log1p(self.p95_price), 0, 1),
                ok=ok, is_new=is_new, alpha=alpha, beta=beta, mu=mu, ess_data=ess_data, E=self.E[:n],
                tv=self.tv, tb=self.tb, live=self.live, struct_cat=None, f_view=self.f_view[:n].copy(),
                f_atc=self.f_atc[:n].copy(), f_q=self.f_q[:n].copy(), mu_beta=self.mu_beta.copy(), L_beta=L,
                feat_mean=self.feat_mean, feat_std=self.feat_std, trend=trend, new_bonus=new_bonus,
                restock_bonus=restock, price_bonus=price_bonus, best_safe=best_safe, new_idx=new_idx,
                new_set=frozenset(self.new_set), new_subcats=frozenset(self.new_subcats),
                user_bought=self.user_bought, user_npurch=self.user_npurch, user_logp=self.user_logp,
                user_aff=self.user_aff, user_clicked=self.user_clicked, live_atc=self.live_atc,
                m_cat_aff=self.P["m_cat_aff"], mean_logp=self.P["mean_logp"], params=prm, cb=self.cb,
                w=self.w, ts=now, trend_new_hot=self.trend_new_hot, pop=self.p_raw[:n].copy())
            if set_global:
                STATE.snap = snap
            return snap

    # --- usuaris (copy-on-write: answer() només llegeix frozensets)
    def add_bought(self, user: str, idx: int) -> None:
        if not user or idx is None or idx < 0:
            return
        cur = self.user_bought.get(user, frozenset())
        if idx not in cur:
            self.user_bought[user] = cur | {idx}

    # --- decaïment periòdic dels comptes (4.2.4, 4.3.4)
    def maybe_decay(self) -> None:
        now = time.time()
        dt_min = (now - self.last_decay) / 60.0
        if dt_min < 1.0:
            return
        cfg = self.cfg
        prm = self.params(now)
        g_slow = 0.5 ** (dt_min / (cfg.half_life_slow_days * 1440))
        g_fast = 0.5 ** (dt_min / cfg.half_life_fast_min)
        n = self.n
        self.s_slow[:n] *= g_slow
        self.f_slow[:n] *= g_slow
        self.s_fast[:n] *= g_fast
        self.f_fast[:n] *= g_fast
        tot = self.s_slow[:n] + self.f_slow[:n] + self.s_fast[:n] + self.f_fast[:n]
        over = tot > prm["ess_max"]
        if over.any():
            sc = prm["ess_max"] / tot[over]
            for a in (self.s_slow, self.f_slow, self.s_fast, self.f_fast):
                a[:n][over] *= sc
        self.last_decay = now
        if now - self.last_sigma_reinject >= 300:
            ls, lm = prm["lambda_sigma"], prm["lambda_mu"]
            self.Sigma_beta = ls * self.Sigma_beta + (1 - ls) * self.Sigma0
            self.mu_beta = lm * self.mu_beta
            self.last_sigma_reinject = now


class StateHolder:
    snap: Snapshot | None = None


STATE = StateHolder()


# =============================================================================
# 6. FASE B · INFERÈNCIA (secció 3)
# =============================================================================
@dataclass
class ReqCtx:
    user: str
    is_new_user: bool
    device: str
    region: str
    page: str
    now: float
    views: list            # [(idx, ts|None)] en ordre cronològic, sense duplicats (última aparició)
    idx_views: np.ndarray
    last_idx: int
    c_t: np.ndarray | None
    tau_view: np.ndarray
    tau_buy: np.ndarray
    has_atc: bool
    bought: frozenset
    clicked: frozenset
    upurch: int
    uaff: np.ndarray
    ref_logp: float
    region_bucket: int
    cold: bool
    cold_reserve: bool
    new_views: list = field(default_factory=list)
    cat_share: np.ndarray = field(default_factory=lambda: np.zeros(1))   # pes de cada categoria a la sessió


def normalize_views(rv) -> list[tuple[str, float | None]]:
    """Parsing defensiu de recent_views (3.1)."""
    if rv is None:
        return []
    if isinstance(rv, str):
        try:
            rv = json.loads(rv)
        except ValueError:
            return []
    out = []
    if not isinstance(rv, list):
        return []
    for x in rv:
        if isinstance(x, dict):
            pid, ts = x.get("product_id"), x.get("ts")
        elif isinstance(x, (list, tuple)) and x:
            pid, ts = x[0], (x[1] if len(x) > 1 else None)
        elif isinstance(x, str):
            pid, ts = x, None
        else:
            continue
        if pid:
            out.append((str(pid), parse_ts(ts)))
    last = {}
    for k, (pid, ts) in enumerate(out):     # duplicats: es queda l'última aparició
        last[pid] = k
    return [out[k] for k in sorted(last.values())]


def recency_weights(views: list, now: float, cfg: Config) -> np.ndarray:
    m = len(views)
    w = np.array([cfg.rho_recency ** (m - 1 - r) for r in range(m)])
    for r, (_, ts) in enumerate(views):
        if ts is not None:
            w[r] *= math.exp(-max(now - ts, 0.0) / cfg.tau_sess)
    return w


def structural_row(snap: Snapshot, j: int, targets: np.ndarray | None = None) -> np.ndarray:
    """Prior estructural normalitzat P_struct(i|j) (2.5.3), per a productes nous (4.6.2 pas 7-8)."""
    cfg = CFG
    t = np.arange(snap.n) if targets is None else targets
    raw = (cfg.lam_sub_s * (snap.sub[t] == snap.sub[j]) + cfg.lam_brand_s * (snap.brand[t] == snap.brand[j])
           + cfg.lam_cat_s * (snap.cat[t] == snap.cat[j])).astype(np.float32)
    raw += cfg.lam_sem_s * np.clip(snap.E[t] @ snap.E[j], 0, None) ** 2
    raw[t == j] = 0.0
    if targets is not None and j < snap.n_off:
        return raw / max(snap.tv["Z"][j], 1e-9) * (cfg.M0 / snap.tv["denom"][j])
    return raw / max(float(raw.sum()), 1e-9)


def session_tau(snap: Snapshot, views: list, wts: np.ndarray, atc: frozenset) -> tuple[np.ndarray, np.ndarray]:
    """tau^view, tau^buy de la sessió (2.5.6), amb fallback per a productes nous i blend en viu."""
    cfg = CFG
    n = snap.n
    tv = np.zeros(n, np.float32)
    tb = np.zeros(n, np.float32)
    mu_live = snap.params["mu_live"]
    tl = np.zeros(n, np.float32) if mu_live > 0 and snap.live is not None else None
    sel = list(range(max(0, len(views) - cfg.max_trans_views), len(views)))
    for r in sel:
        j, wr = views[r][0], float(wts[r])
        if wr <= 0:
            continue
        if j < snap.n_off:
            for M, out in ((snap.tv, tv), (snap.tb, tb)):
                s, e = M["indptr"][j], M["indptr"][j + 1]
                out[M["indices"][s:e]] += wr * M["data"][s:e]
            if snap.new_idx.size:
                pn = wr * structural_row(snap, j, snap.new_idx)
                tv[snap.new_idx] += pn
                tb[snap.new_idx] += pn
        else:
            pr = wr * structural_row(snap, j)
            tv += pr
            tb += pr
        if tl is not None:
            L = snap.live
            if j < L["n"]:
                s, e = L["indptr"][j], L["indptr"][j + 1]
                tl[L["indices"][s:e]] += wr * L["data"][s:e]
    for j in atc:
        if 0 <= j < snap.n_off:
            M = snap.tb
            s, e = M["indptr"][j], M["indptr"][j + 1]
            tb[M["indices"][s:e]] += cfg.atc_trans_weight * M["data"][s:e]
    if tl is not None and tl.sum() > 0:
        tv = (1 - mu_live) * tv + mu_live * tl
        tb = (1 - mu_live) * tb + mu_live * tl
    return tv / snap.tv["norm"], tb / snap.tb["norm"]


def make_ctx(snap: Snapshot, req: dict) -> ReqCtx:
    cfg = CFG
    now = parse_ts(req.get("ts")) or time.time()
    user = str(req.get("user_id") or "?")
    is_new = str(req.get("is_new_user")).lower() in ("true", "1")
    device = (req.get("device") or "web").lower()
    region = req.get("region") or "?"
    page = (req.get("page") or "home").lower()
    raw = normalize_views(req.get("recent_views"))
    views = [(snap.index[p], ts) for p, ts in raw if p in snap.index and snap.index[p] < snap.n]
    new_views = [p for p, _ in raw if p in snap.new_set]
    idx_views = np.array([i for i, _ in views], np.int64)
    c_t = None
    tau_view = tau_buy = None
    ncat = int(snap.cat.max()) + 1
    cat_share = np.zeros(ncat)
    if views:
        cw = views[-cfg.max_centroid_views:]
        wts_c = recency_weights(cw, now, cfg)
        v = (snap.E[[i for i, _ in cw]] * wts_c[:, None]).sum(0)
        nv = float(np.linalg.norm(v))
        c_t = v / nv if nv > 0 else None
        wn = wts_c / wts_c.sum() if wts_c.sum() > 0 else np.full(len(cw), 1.0 / len(cw))
        np.add.at(cat_share, [int(snap.cat[i]) for i, _ in cw], wn)
    atc_entry = snap.live_atc.get(user)
    atc = frozenset(i for i in (atc_entry or ()) if i < snap.n)
    if views or atc:
        wts = recency_weights(views, now, cfg) if views else np.zeros(0)
        tau_view, tau_buy = session_tau(snap, views, wts, atc)
    else:
        tau_view = np.zeros(snap.n, np.float32)
        tau_buy = np.zeros(snap.n, np.float32)
    uaff = snap.user_aff.get(user)
    if uaff is None:
        uaff = snap.m_cat_aff
    if uaff.size < int(snap.cat.max()) + 1:          # categoria nova vista en viu
        uaff = np.concatenate([uaff, np.full(int(snap.cat.max()) + 1 - uaff.size, 1.0 / (uaff.size + 1))])
    if idx_views.size:
        ref_logp = float(snap.log_price[idx_views].mean())
    else:
        ref_logp = snap.user_logp.get(user, snap.mean_logp)
    cold = is_new or bool(new_views)
    cold_reserve = bool(new_views) or any(int(snap.sub[i]) in snap.new_subcats for i in idx_views.tolist())
    clicked = snap.user_clicked.get(user) or frozenset()
    return ReqCtx(user=user, is_new_user=is_new, device=device, region=region, page=page, now=now, views=views,
                  idx_views=idx_views, last_idx=int(idx_views[-1]) if idx_views.size else -1, c_t=c_t,
                  tau_view=tau_view, tau_buy=tau_buy, has_atc=bool(atc), bought=snap.user_bought.get(user, frozenset()),
                  clicked=clicked, upurch=int(snap.user_npurch.get(user, 0)), uaff=uaff, ref_logp=ref_logp,
                  region_bucket=zlib.crc32(region.encode()) % 4, cold=cold, cold_reserve=cold_reserve,
                  new_views=new_views, cat_share=cat_share)


def raw_features(snap: Snapshot, rc: ReqCtx, cand: np.ndarray) -> np.ndarray:
    """z_{t,i} sense estandarditzar (taula 2.6.2, sense els termes fixos de (38))."""
    k = cand.size
    Z = np.zeros((k, D_FEAT), np.float32)
    pp = snap.price_pct[cand]
    ltv = np.log1p(rc.tau_view[cand])
    ltb = np.log1p(rc.tau_buy[cand])
    csim = snap.E[cand] @ rc.c_t if rc.c_t is not None else np.zeros(k, np.float32)
    fv = snap.f_view[cand]
    Z[:, 0] = 1.0
    Z[:, 1] = pp * (rc.device == "ios")
    Z[:, 2] = pp * (rc.device == "android")
    Z[:, 3] = ltb * (rc.page == "product")
    Z[:, 4] = ltv * (rc.page == "category")
    Z[:, 5 + rc.region_bucket] = pp
    Z[:, 9] = fv * rc.is_new_user
    Z[:, 10] = min(len(rc.views), 10) / 10.0 * csim
    if rc.last_idx >= 0:
        Z[:, 11] = snap.sub[cand] == snap.sub[rc.last_idx]
        Z[:, 12] = snap.brand[cand] == snap.brand[rc.last_idx]
    Z[:, 13] = snap.price_pct_sub[cand]
    Z[:, 14] = math.log1p(rc.upurch) * fv
    Z[:, 15] = ltb * rc.has_atc
    Z[:, 16] = fv
    Z[:, 17] = snap.f_atc[cand]
    Z[:, 18] = snap.f_q[cand]
    cats = snap.cat[cand].astype(np.int64)
    Z[:, 19] = np.log1p(10.0 * rc.uaff[np.minimum(cats, rc.uaff.size - 1)])
    if rc.idx_views.size:
        Z[:, 20] = np.isin(cand, rc.idx_views)
    return Z


def standardize(snap: Snapshot, Z: np.ndarray) -> np.ndarray:
    sd = snap.feat_std
    Zs = np.where(sd > 0, (Z - snap.feat_mean) / np.where(sd > 0, sd, 1.0), 0.0)
    Zs[:, 0] = 1.0
    return Zs


def deterministic_scores(snap: Snapshot, rc: ReqCtx, cand: np.ndarray):
    """Part determinista de (38): una sola vegada per petició (3.8)."""
    cfg = CFG
    Z = standardize(snap, raw_features(snap, rc, cand))
    mu = snap.mu[cand]
    if rc.page == "product":
        lv, lb = cfg.lambda_T_view_product, cfg.lambda_T_buy_product
    else:
        lv, lb = cfg.lambda_T_view_home, cfg.lambda_T_buy_home
    csim = snap.E[cand] @ rc.c_t if rc.c_t is not None else np.zeros(cand.size, np.float32)
    eta = (logit(mu) + Z @ snap.mu_beta
           + lv * np.log1p(rc.tau_view[cand]) + lb * np.log1p(rc.tau_buy[cand])
           + cfg.lambda_C * csim
           + cfg.lambda_same_cat * rc.cat_share[np.minimum(snap.cat[cand], rc.cat_share.size - 1)]
           + cfg.lambda_price * (snap.log_price[cand] - rc.ref_logp)
           + snap.trend[cand] + snap.new_bonus[cand] + snap.restock_bonus[cand] + snap.price_bonus[cand])
    if rc.page == "product" and rc.idx_views.size:
        same = np.isin(cand, rc.idx_views) & ~snap.is_new[cand]
        eta -= same * math.log(1 / cfg.delta_same)
    if rc.clicked:
        eta -= np.isin(cand, np.fromiter(rc.clicked, np.int64)) * cfg.pen_click
    eta = cfg.calib_a + eta / cfg.temperature
    p = expit(eta)
    U = p * (1.0 + cfg.rho * snap.price_q[cand])
    return Z, eta, p, U


def sample_scores(snap: Snapshot, rc: ReqCtx, cand: np.ndarray, Z, eta_det, rng, noise_scale: float):
    """Thompson Sampling: una mostra Beta per producte i una beta per petició (3.4.1-3.4.2)."""
    cfg = CFG
    a, b = snap.alpha[cand], snap.beta[cand]
    theta = rng.beta(a, b)
    beta_t = snap.mu_beta + snap.L_beta @ rng.standard_normal(D_FEAT)
    mu = a / (a + b)
    delta = (logit(theta) - logit(mu) + Z @ (beta_t - snap.mu_beta)) * noise_scale / cfg.temperature
    p = expit(eta_det + delta)
    return p, p * (1.0 + cfg.rho * snap.price_q[cand])


def topk(scores: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0 or k <= 0:
        return idx[:0]
    if idx.size > k:
        part = np.argpartition(-scores[idx], k)[:k]
        idx = idx[part]
    return idx[np.argsort(-scores[idx])]


def build_candidates(snap: Snapshot, rc: ReqCtx, ok: np.ndarray, rng, n_cand: int) -> np.ndarray:
    """~100 candidats amb quotes per font (3.3)."""
    cfg = CFG
    chosen: dict[int, None] = {}
    lists = []
    tau = rc.tau_view + rc.tau_buy
    lists.append(topk(tau, ok & (tau > 0), cfg.q_trans * 2))
    content = None
    if rc.c_t is not None:
        content = snap.E @ rc.c_t
        lists.append(topk(content, ok, cfg.q_content * 2))
    else:
        lists.append(np.zeros(0, np.int64))
    lists.append(topk(snap.trend, ok & (snap.trend > 0), cfg.q_trend * 2) if snap.cb == "NORMAL"
                 else np.zeros(0, np.int64))
    if rc.idx_views.size:
        cats = np.unique(snap.cat[rc.idx_views])
    elif rc.user in snap.user_aff:
        aff = rc.uaff[:len(snap.m_cat_aff)]
        lift = aff / np.maximum(snap.m_cat_aff, 1e-9)
        cats = np.argsort(-lift)[:2]
    else:
        cats = np.zeros(0, np.int64)
    pmask = ok & np.isin(snap.cat, cats) if cats.size else ok
    lists.append(topk(snap.mu, pmask, cfg.q_prior * 2))
    exp_mask = ok & ((snap.ess_data < cfg.explore_ess) | snap.is_new)
    if snap.cb == "NORMAL" and exp_mask.any():
        idx = np.flatnonzero(exp_mask)
        draw = np.full(snap.n, -1.0)
        draw[idx] = rng.beta(snap.alpha[idx], snap.beta[idx])
        lists.append(topk(draw, exp_mask, cfg.q_explore * 2))
    else:
        lists.append(np.zeros(0, np.int64))
    if cats.size:                                     # best sellers de les categories de la sessió
        lists.append(topk(snap.pop, ok & np.isin(snap.cat, cats), cfg.q_safe * 2))
    else:
        safe = [i for i in snap.best_safe.tolist() if ok[i]]
        lists.append(np.array(safe[:cfg.q_safe * 2], np.int64))
    quotas = [cfg.q_trans, cfg.q_content, cfg.q_trend, cfg.q_prior, cfg.q_explore, cfg.q_safe]
    scale = n_cand / max(sum(quotas), 1)
    # cold start: veïns de contingut i de subcategoria del producte nou (3.7.2)
    if rc.cold_reserve:
        extra = []
        for pid in rc.new_views:
            j = snap.index.get(pid)
            if j is None or j >= snap.n:
                continue
            if ok[j]:
                extra.append(j)
            sim = snap.E @ snap.E[j]
            extra += topk(sim, ok, 25).tolist()
            extra += topk(snap.mu, ok & ((snap.sub == snap.sub[j]) | (snap.brand == snap.brand[j])), 10).tolist()
        for i in extra:
            chosen.setdefault(int(i))
    for i in snap.trend_new_hot:                      # productes nous amb tendència (4.5.8)
        if i < snap.n and ok[i]:
            chosen.setdefault(int(i))
    ptr = [0] * len(lists)
    for s, L in enumerate(lists):                     # primera passada: quotes
        q = int(round(quotas[s] * scale))
        taken = 0
        while ptr[s] < L.size and taken < q:
            i = int(L[ptr[s]])
            ptr[s] += 1
            if i not in chosen:
                chosen[i] = None
                taken += 1
    progress = True
    while len(chosen) < n_cand and progress:          # segona: les fonts que sobren omplen el buit
        progress = False
        for s, L in enumerate(lists):
            while ptr[s] < L.size:
                i = int(L[ptr[s]])
                ptr[s] += 1
                if i not in chosen:
                    chosen[i] = None
                    progress = True
                    break
            if len(chosen) >= n_cand:
                break
    return np.fromiter(chosen.keys(), np.int64)


def order_slate(snap: Snapshot, rc: ReqCtx, cand: np.ndarray, U: np.ndarray, U_mean: np.ndarray, rng,
                explore: bool) -> list[int]:
    """Slot 1 explotació; 2-8 MMR amb quotes toves; 9-10 exploració (3.6.2)."""
    cfg = CFG
    prm = snap.params
    k = cand.size
    lam_div = cfg.lambda_div_new_user if rc.is_new_user else cfg.lambda_div
    Ec = snap.E[cand]
    S = Ec @ Ec.T
    subs, cats, brands = snap.sub[cand], snap.cat[cand], snap.brand[cand]
    ess = snap.ess_data[cand]
    first = int(np.argmax(U_mean if cfg.exploit_slot1 else U))
    chosen = [first]
    avail = np.ones(k, bool)
    avail[first] = False
    maxsim = S[first].copy()
    n_sub, n_cat, n_brand5 = Counter([subs[first]]), Counter([cats[first]]), Counter([brands[first]])
    for slot in range(2, 11):
        if not avail.any():
            break
        pick = -1
        if slot >= 9 and explore and rng.random() < prm["eps"]:
            E_mask = avail & ((ess < cfg.explore_ess) | snap.is_new[cand])
            if E_mask.any():
                ej = np.flatnonzero(E_mask)
                sc = U[ej] * (1 + prm["c_ucb"] / np.sqrt(ess[ej] + 1))
                top = ej[np.argsort(-sc)[:cfg.explore_top]]
                pick = int(rng.choice(top))
        if pick < 0:
            score = U * (1 - lam_div * np.maximum(maxsim, 0)) if slot <= 8 else U.copy()
            viol = np.array([n_sub[subs[j]] >= cfg.quota_sub or n_cat[cats[j]] >= cfg.quota_cat
                             or (slot <= 5 and n_brand5[brands[j]] >= cfg.quota_brand_top5) for j in range(k)])
            sc = np.where(avail & ~viol, score, -np.inf)
            if np.isfinite(sc).any():
                pick = int(np.argmax(sc))
            else:                                       # quotes relaxades: mai un slot buit
                pick = int(np.argmax(np.where(avail, U, -np.inf)))
        chosen.append(pick)
        avail[pick] = False
        maxsim = np.maximum(maxsim, S[pick])
        n_sub[subs[pick]] += 1
        n_cat[cats[pick]] += 1
        if slot <= 5:
            n_brand5[brands[pick]] += 1
    return [int(cand[j]) for j in chosen]


def apply_cold_reservation(snap: Snapshot, rc: ReqCtx, chosen: list[int], cand: np.ndarray, p: np.ndarray) -> list[int]:
    """Reserva un slot (3-6) per al producte nou o un de molt similar si p >= 0.7 p(slot 10) (3.7.2)."""
    cfg = CFG
    pos = {int(c): j for j, c in enumerate(cand)}
    p_last = p[pos[chosen[-1]]] if chosen and chosen[-1] in pos else 0.0
    targets = [snap.index[x] for x in rc.new_views if x in snap.index and snap.index[x] < snap.n]
    best, best_p = None, -1.0
    for j in range(cand.size):
        i = int(cand[j])
        if i in chosen:
            continue
        related = snap.is_new[i] or any(snap.sub[i] == snap.sub[t] for t in targets)
        if related and p[j] >= cfg.cold_reserve_ratio * p_last and p[j] > best_p:
            best, best_p = i, p[j]
    if best is None or any(snap.is_new[i] for i in chosen[2:6]):
        return chosen
    out = chosen[:9]
    out.insert(3, best)
    return out


def fill_safe(snap: Snapshot, bought: frozenset, items: list[int], ok: np.ndarray) -> list[int]:
    seen = set(items)
    for i in snap.best_safe.tolist():
        if len(items) >= 10:
            break
        if i not in seen and ok[i] and i not in bought:
            items.append(i)
            seen.add(i)
    if len(items) < 10:
        for i in np.flatnonzero(ok).tolist():
            if len(items) >= 10:
                break
            if i not in seen and i not in bought:
                items.append(i)
                seen.add(i)
    return items


def answer(snap: Snapshot, req: dict, rng, info: dict | None = None) -> list[str]:
    """10 product_id per a una petició (3.8)."""
    cfg = CFG
    rc = make_ctx(snap, req)
    ok = snap.ok.copy()
    if rc.bought:
        b = np.fromiter((i for i in rc.bought if i < snap.n), np.int64)
        ok[b] = False
    if snap.cb == "SAFE":
        items = fill_safe(snap, rc.bought, [], ok)
        return [snap.ids[i] for i in items[:10]]
    prm = snap.params
    n_cand = cfg.n_candidates if snap.cb == "NORMAL" else 50
    cand = build_candidates(snap, rc, ok, rng, n_cand)
    if cand.size < 10:
        cand = np.unique(np.concatenate([cand, np.array(fill_safe(snap, rc.bought, [], ok), np.int64)]))
    Z, eta_det, p_det, U_det = deterministic_scores(snap, rc, cand)
    noise = prm["t_eff"] * (cfg.t_eff_new_user if rc.is_new_user else 1.0)
    p, U = sample_scores(snap, rc, cand, Z, eta_det, rng, noise)
    chosen = order_slate(snap, rc, cand, U, U_det, rng, explore=snap.cb == "NORMAL")
    if rc.cold_reserve:
        chosen = apply_cold_reservation(snap, rc, chosen, cand, p)
    items, seen = [], set()
    for i in chosen:
        if ok[i] and i not in seen:
            items.append(i)
            seen.add(i)
    if len(items) < 10:
        items = fill_safe(snap, rc.bought, items, ok)
    if info is not None:
        info.update(cand=cand, rc=rc, p_det=p_det, U_det=U_det, items=items[:10])
    return [snap.ids[i] for i in items[:10]]


def validate_and_repair(snap: Snapshot, items: list[str], bought: frozenset) -> list[str]:
    """Contracte de validació abans del push (4.8.1)."""
    out, seen = [], set()
    for pid in items:
        i = snap.index.get(pid)
        if i is None or i >= snap.n or not snap.ok[i] or i in bought or pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
    if len(out) < 10:
        idx = fill_safe(snap, bought, [snap.index[x] for x in out], snap.ok)
        out = [snap.ids[i] for i in idx]
    return out[:10]


# =============================================================================
# 7. BUCLE ASÍNCRON (secció 4)
# =============================================================================
STOP = threading.Event()
OUTBOX: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
REQ_META = TTLCache(2700, 300_000)
SEEN_REQ = TTLCache(900, 300_000)


class Clock:
    """Offset del rellotge del servidor (4.8.6)."""
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
                log.warning("[clock] desfasament amb el servidor de %.2f s", cls.offset)


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.c = Counter()
        self.push_lat = deque(maxlen=2000)      # ms des del ts de la petició fins al push confirmat
        self.answer_ms = deque(maxlen=2000)
        self.fb = deque(maxlen=50_000)          # (ts, served_by, reason, latency, purchased, revenue, team)
        self.conv_windows = deque(maxlen=500)
        self.cold_hist = deque(maxlen=200)
        self.start = time.time()

    def inc(self, k, v=1):
        with self.lock:
            self.c[k] += v


MET = Metrics()


def q_list(xs) -> str:
    return ",".join("'" + str(x).replace("'", "") + "'" for x in xs)


# --- 7.1 peticions i push --------------------------------------------------------
SQL_REQUESTS = """SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
 user_id::String AS user_id, toString(is_new_user) AS is_new_user, toString(context.device) AS device,
 toString(context.region) AS region, toString(context.page) AS page, recent_views
FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 15 SECOND
ORDER BY ts::DateTime64(3) ASC LIMIT 400"""


def request_loop(rt: RawTree, args, worker: int = 0) -> None:
    """Dos lectors desfasats: cada consulta triga ~0,2 s, així que la petició espera la meitat.
    SEEN_REQ.claim() garanteix que cada request_id només el respon un lector."""
    rng = np.random.default_rng(20261115 + worker)
    STOP.wait(0.12 * worker)
    backoff = 0.5
    while not STOP.is_set():
        try:
            rows = rt.query(SQL_REQUESTS, timeout=5)
            backoff = 0.5
            fresh = [r for r in rows if r["request_id"] not in SEEN_REQ]
            if len(fresh) > 50:                              # backlog: primer les més noves (4.8.2)
                fresh = sorted(fresh, key=lambda r: r["ts"], reverse=True)[:50]
            for r in fresh:
                rid = r["request_id"]
                if not SEEN_REQ.claim(rid):                  # l'altre lector ja la respon
                    continue
                ts = parse_ts(r["ts"])
                age = Clock.now() - (ts or Clock.now())
                if age > CFG.deadline_s:
                    MET.inc("expired")
                    continue
                snap = STATE.snap
                t0 = time.perf_counter()
                try:
                    items = answer(snap, r, rng)
                except Exception as e:  # noqa: BLE001
                    log.exception("[answer] %s: %s", rid, e)
                    MET.inc("answer_errors")
                    ok = snap.ok
                    items = [snap.ids[i] for i in fill_safe(snap, frozenset(), [], ok)]
                MET.answer_ms.append((time.perf_counter() - t0) * 1000)
                REQ_META.add(rid, {"user_id": r.get("user_id"), "is_new_user": r.get("is_new_user"),
                                   "device": r.get("device"), "region": r.get("region"), "page": r.get("page"),
                                   "recent_views": r.get("recent_views"), "ts": r.get("ts")})
                MET.inc("answered")
                MET.cold_hist.append((time.time(), str(r.get("is_new_user")).lower() in ("true", "1")))
                ans = {"request_id": rid, "items": items, "model_version": CFG.model_version, "_ts": ts,
                       "_user": r.get("user_id")}
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
            if "Unknown expression identifier" in msg or "UNKNOWN_IDENTIFIER" in msg:
                log.info("[requests] el stream encara no és obert; reintent en %.1f s", backoff)
            else:
                log.warning("[requests] %s", msg[:200])
            MET.inc("request_query_errors")
            STOP.wait(backoff)
            backoff = min(backoff * 2, 5.0)
            continue
        STOP.wait(CFG.request_poll_s)


PUSH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="push")


def _push_batch(rt: RawTree, args, batch: list[dict]) -> None:
    """Valida contra l'snapshot actual (l'estoc pot haver canviat) i insereix; en paral·lel."""
    snap = STATE.snap
    rows, sent = [], []
    for a in batch:
        age = Clock.now() - (a["_ts"] or Clock.now())
        if age > CFG.deadline_s:                         # arribaria tard: no ocupa el lot
            MET.inc("dropped_late")
            continue
        bought = snap.user_bought.get(a["_user"] or "", frozenset())
        items = validate_and_repair(snap, a["items"], bought)
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
        if n != len(rows):
            log.warning("[push] written_rows=%s de %d", n, len(rows))
    except Exception as e:  # noqa: BLE001
        MET.inc("push_errors")
        log.warning("[push] error: %s; reintent", str(e)[:200])
        for a in sent:
            if Clock.now() - (a["_ts"] or 0) < CFG.deadline_s - 0.3:
                try:
                    OUTBOX.put_nowait(a)
                except queue.Full:
                    break


def push_loop(rt: RawTree, args) -> None:
    """Envia de seguida el que hi ha a la cua; fins a 4 inserts en vol perquè un insert lent
    (picos del servidor de 1-2 s) no retardi les respostes següents."""
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


# --- 7.2 feedback (4.1-4.4) -----------------------------------------------------
def fetch_request_meta(rt: RawTree, ids: list[str]) -> None:
    for k in range(0, len(ids), 300):
        chunk = ids[k:k + 300]
        rows = rt.query(f"""SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
            user_id::String AS user_id, toString(is_new_user) AS is_new_user, toString(context.device) AS device,
            toString(context.region) AS region, toString(context.page) AS page, recent_views
            FROM shop.requests WHERE request_id::String IN ({q_list(chunk)})""", 10)
        for r in rows:
            REQ_META.add(r["request_id"], {k2: r.get(k2) for k2 in
                                           ("user_id", "is_new_user", "device", "region", "page", "recent_views", "ts")})


def on_feedback_one(ms: MutableState, snap: Snapshot, r: dict, prm: dict) -> None:
    cfg = CFG
    rid = r["request_id"]
    shown = r.get("shown") or []
    clicked = r.get("clicked") or ""
    purchased = r.get("purchased") or ""
    served = r.get("served_by") or ""
    meta = REQ_META.get(rid)
    user = (meta or {}).get("user_id") or ""
    if purchased:
        ms.add_bought(user, ms.index.get(purchased, -1))
        if user:
            ms.user_npurch[user] = ms.user_npurch.get(user, 0) + 1
    if served != "team":
        return
    idx = [ms.index.get(p, -1) for p in shown[:10]]
    w = ms.w
    ds, df = 1.0 - prm["rho_fast"], prm["rho_fast"]
    neg = cfg.c_neg_purchase
    pi = ms.index.get(purchased, -1) if purchased else -1
    ci = ms.index.get(clicked, -1) if clicked else -1
    for k, i in enumerate(idx):                               # 4.2.2 model de compra
        if i < 0:
            continue
        if i == pi:
            ms.s_slow[i] += ds
            ms.s_fast[i] += df
        elif i == ci and pi < 0:
            ms.f_slow[i] += ds
            ms.f_fast[i] += df
        else:
            ms.f_slow[i] += ds * neg * w[k]
            ms.f_fast[i] += df * neg * w[k]
        if i == ci:                                           # 4.2.1 model de clic
            ms.s_click[i] += 1
        else:
            ms.f_click[i] += cfg.c_neg_click * w[k]
    if ci >= 0 and pi < 0 and user:
        cur = ms.user_clicked.get(user) or frozenset()
        ms.user_clicked.add(user, cur | {ci})
    # 4.3 ADF contextual. model.txt (4.3.1) només actualitza amb el producte comprat o clicat: sense
    # negatius l'intercept deriva sense límit (+4,4 logits en 13 min en viu). Per defecte ("full") s'usen
    # els 10 mostrats amb els mateixos pesos que l'ajust offline (2.6.1): comprat y=1, clicat sense
    # compra y=0 amb pes 1, ignorat y=0 amb pes w_k (probabilitat que l'hagi examinat).
    if cfg.adf_mode != "off" and meta:
        try:
            if cfg.adf_mode == "positives":
                obs = [(pi, 1.0, 1.0)] if pi >= 0 else ([(ci, 0.0, cfg.adf_weight_click)] if ci >= 0 else [])
            else:
                obs = [(i, 1.0 if i == pi else 0.0, 1.0 if i in (pi, ci) else float(w[k]))
                       for k, i in enumerate(idx) if 0 <= i < snap.n]
            if obs:
                rc = make_ctx(snap, meta)
                cand = np.array([o[0] for o in obs], np.int64)
                Zs = standardize(snap, raw_features(snap, rc, cand))
                offs = logit(snap.mu[cand])
                mu_b, Sig = ms.mu_beta.copy(), ms.Sigma_beta.copy()
                for (_, y, om), z, o in zip(obs, Zs, offs):
                    p = float(np.clip(expit(o + z @ mu_b), 1e-6, 1 - 1e-6))
                    s = om * p * (1 - p)
                    Sz = Sig @ z
                    Sig = Sig - s * np.outer(Sz, Sz) / (1 + s * z @ Sz)
                    mu_b = mu_b + Sig @ z * om * (y - p)
                Sig = (Sig + Sig.T) / 2
                if np.min(np.diag(Sig)) < 1e-10:
                    Sig += 1e-8 * np.eye(D_FEAT)
                if np.all(np.isfinite(mu_b)) and np.all(np.isfinite(Sig)):
                    ms.mu_beta, ms.Sigma_beta = mu_b, Sig
        except Exception as e:  # noqa: BLE001
            log.debug("[adf] %s", e)


FB_STATE: dict = {"last": None}          # ts del darrer feedback processat (es desa al checkpoint)


def feedback_loop(rt: RawTree, ms: MutableState, args) -> None:
    team = rt.team
    seen = TTLCache(3600, 500_000)
    last_done = FB_STATE.get("last")
    since = fmt_ts(max(last_done, Clock.now() - 1800)) if last_done else fmt_ts(Clock.now() - 1800)
    while not STOP.is_set():
        try:
            rows = rt.query(f"""SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
                shown::Array(String) AS shown, served_by::String AS served_by,
                ifNull(toString(fallback_reason), '') AS fallback_reason,
                toFloat64OrZero(ifNull(toString(latency_ms), '0')) AS latency_ms,
                ifNull(toString(clicked), '') AS clicked, ifNull(toString(purchased), '') AS purchased,
                toFloat64OrZero(ifNull(toString(revenue_eur), '0')) AS revenue_eur
                FROM {team}.feedback WHERE request_id::String != '__init__'
                  AND ts::DateTime64(3) > toDateTime64('{since}', 3)
                ORDER BY ts::DateTime64(3) ASC LIMIT 20000""", 15)
            fresh = [r for r in rows if r["request_id"] not in seen]
            if rows:
                last = max(parse_ts(r["ts"]) or 0 for r in rows)
                since = fmt_ts(max(last - 60, Clock.now() - 1800))
            missing = [r["request_id"] for r in fresh if r["request_id"] not in REQ_META
                       and (r.get("purchased") or r.get("clicked"))]
            if missing:
                try:
                    fetch_request_meta(rt, missing[:900])
                except Exception as e:  # noqa: BLE001
                    log.debug("[feedback] meta: %s", e)
            if fresh:
                snap = STATE.snap
                with ms.lock:
                    prm = ms.params()
                    for r in fresh:
                        seen.add(r["request_id"])
                        try:
                            on_feedback_one(ms, snap, r, prm)
                        except Exception as e:  # noqa: BLE001
                            log.debug("[feedback] fila: %s", e)
                        MET.fb.append((parse_ts(r["ts"]) or time.time(), r.get("served_by"), r.get("fallback_reason"),
                                       float(r.get("latency_ms") or 0), bool(r.get("purchased")),
                                       float(r.get("revenue_eur") or 0)))
                    FB_STATE["last"] = max([FB_STATE.get("last") or 0.0] + [parse_ts(r["ts"]) or 0.0 for r in fresh])
                    ms.maybe_decay()
                    ms.publish()
                MET.inc("feedback_rows", len(fresh))
            else:
                with ms.lock:
                    ms.maybe_decay()
        except Exception as e:  # noqa: BLE001
            log.warning("[feedback] %s", str(e)[:200])
        STOP.wait(CFG.feedback_poll_s)


# --- 7.3 tendències amb shop.events (4.5) -----------------------------------------
def ring_add(ms: MutableState, i: int, minute: int, kind: str, cnt: int = 1) -> None:
    R = ms.cfg.ring_minutes
    b = minute % R
    if ms.ring_min[b] != minute:
        if ms.ring_min[b] > minute:
            return                                            # massa vell
        ms.ring_v[b] = 0
        ms.ring_a[b] = 0
        ms.ring_p[b] = 0
        ms.ring_min[b] = minute
    arr = ms.ring_v if kind == "view" else ms.ring_a if kind == "add_to_cart" else ms.ring_p
    arr[b, i] += cnt


def recompute_trends(ms: MutableState) -> None:
    cfg = ms.cfg
    n = ms.n
    prm = ms.params()
    M = int(Clock.now() // 60)
    age = M - ms.ring_min
    w15 = (age >= 0) & (age < cfg.trend_window_min)
    w180 = (age >= cfg.trend_window_min) & (age < cfg.trend_window_min + cfg.trend_baseline_min)
    V15, A15, P15 = (r[w15, :n].sum(0).astype(float) for r in (ms.ring_v, ms.ring_a, ms.ring_p))
    V180, A180, P180 = (r[w180, :n].sum(0).astype(float) for r in (ms.ring_v, ms.ring_a, ms.ring_p))
    sub = ms.sub[:n].astype(np.int64)
    ns = int(sub.max()) + 1
    Vs = np.bincount(sub, weights=V180, minlength=ns)
    Ps = np.bincount(sub, weights=P180, minlength=ns)
    m_s = (Ps + 1) / (Vs + 50)                                               # (52)
    kt = 50.0
    lam_base = (P180 + kt * m_s[sub]) / (V180 + kt)                          # (53)
    lam_now = (P15 + kt * lam_base) / (V15 + kt)                             # (54)
    s_i = P15 + 0.25 * A15
    e_i = (P180 + 0.25 * A180) / 12.0                                        # (56)
    L_comb = (s_i + 1) / (e_i + 1)                                           # (57)
    z = (P15 - lam_base * V15) / np.sqrt(lam_base * V15 + 1)                 # (58)
    s_sub = np.bincount(sub, weights=s_i, minlength=ns)
    e_sub = np.bincount(sub, weights=e_i, minlength=ns)
    L_sub = (s_sub + 1) / (e_sub + 1)                                        # (59)
    n_i = s_i + e_i
    L_shr = (n_i * L_comb + 10.0 * L_sub[sub]) / (n_i + 10.0)                # (60)
    gate = np.clip((P15 + 0.25 * A15) / 3.0, 0, 1)                           # (61)
    b = prm["eta_T"] * np.tanh(np.log(np.clip(L_shr, 0.25, 8))) * gate       # (62)
    b = np.where((z > 2) & (P15 >= 3), b * 1.3, b)
    new = ms.is_new[:n]
    hot_new = new & ((A15 >= 3) | (P15 >= 1))                                # (4.5.8)
    b = np.where(hot_new, b * 1.5, b)
    b = np.clip(b, -0.8, 0.8)
    now = time.time()
    dt = now - ms.trend_last
    prev = ms.trend[:n] * 0.5 ** (dt / 600.0)                                # (65)
    ms.trend[:n] = np.where(gate > 0, b, prev)
    ms.trend_last = now
    persistent = (L_shr > 1.5) & (gate >= 1)                                 # (63)-(64)
    ms.trend_hot[:n] = np.where(persistent, ms.trend_hot[:n] + 1, 0)
    inj = (ms.trend_hot[:n] >= 2) & ms.active[:n] & ~ms.unavailable[:n]
    if inj.any():
        n_tr = prm["K_T"] * (V15[inj] + 4 * A15[inj])
        p_tr = np.clip(lam_now[inj] * ms.P["r_scale"], 0.0005, 0.3)
        ms.s_fast[:n][inj] += n_tr * p_tr
        ms.f_fast[:n][inj] += n_tr * (1 - p_tr)
    ms.trend_new_hot = frozenset(np.flatnonzero(hot_new).tolist())
    ms.lam_now[:n] = lam_now
    p15 = np.bincount(sub, weights=V15, minlength=ns)                         # mix de trànsit (4.5.7)
    p180 = np.bincount(sub, weights=V180, minlength=ns)
    if p15.sum() > 500 and p180.sum() > 500:
        ms.sub_tv = 0.5 * float(np.abs(p15 / p15.sum() - p180 / p180.sum()).sum())


def bootstrap_trends(rt: RawTree, ms: MutableState) -> str:
    t_end = fmt_ts(Clock.now())
    rows = rt.query(f"""SELECT product_id::String AS product_id,
        toUnixTimestamp(toStartOfMinute(ts::DateTime64(3))) AS m,
        countIf(event_type::String = 'view') AS v, countIf(event_type::String = 'add_to_cart') AS a,
        countIf(event_type::String = 'purchase') AS p
        FROM shop.events
        WHERE ts::DateTime64(3) > now() - INTERVAL {ms.cfg.trend_window_min + ms.cfg.trend_baseline_min + 1} MINUTE
          AND ts::DateTime64(3) <= toDateTime64('{t_end}', 3)
        GROUP BY product_id, m ORDER BY m ASC""", 60)
    with ms.lock:
        for r in rows:
            i = ms.index.get(r["product_id"])
            if i is None:
                continue
            minute = int(r["m"]) // 60
            for kind, key in (("view", "v"), ("add_to_cart", "a"), ("purchase", "p")):
                if r[key]:
                    ring_add(ms, i, minute, kind, int(r[key]))
        recompute_trends(ms)
    log.info("[events] ring de tendències inicialitzat amb %d files (producte x minut)", len(rows))
    return t_end


def events_loop(rt: RawTree, ms: MutableState, since: str) -> None:
    seen = TTLCache(600, 200_000)
    first = True
    while not STOP.is_set():
        try:
            n_rows = 0
            while True:
                cond = f"ts::DateTime64(3) > toDateTime64('{since}', 3)"
                rows = rt.query(f"""SELECT toString(ts::DateTime64(3)) AS ts, event_type::String AS t,
                    user_id::String AS u, session_id::String AS s, product_id::String AS p
                    FROM shop.events WHERE {cond} ORDER BY ts::DateTime64(3) ASC LIMIT 10000""", 15)
                n_rows += len(rows)
                with ms.lock:
                    for r in rows:
                        key = (r["ts"], r["s"], r["p"], r["t"])
                        if key in seen:
                            continue
                        seen.add(key)
                        i = ms.index.get(r["p"])
                        ts = parse_ts(r["ts"]) or Clock.now()
                        if i is None:
                            continue
                        ring_add(ms, i, int(ts // 60), r["t"])
                        if r["t"] == "add_to_cart" and r["u"]:
                            cur = ms.live_atc.get(r["u"]) or frozenset()
                            ms.live_atc.add(r["u"], cur | {i})
                        elif r["t"] == "purchase":
                            ms.add_bought(r["u"], i)
                if rows:
                    last = parse_ts(rows[-1]["ts"]) or Clock.now()
                    since = fmt_ts(last - (0 if first else 2.0))
                first = False
                if len(rows) < 10000:
                    break
            with ms.lock:
                recompute_trends(ms)
                ms.publish()
            MET.inc("events_rows", n_rows)
        except Exception as e:  # noqa: BLE001
            log.warning("[events] %s", str(e)[:200])
        STOP.wait(CFG.events_poll_s)


def live_transitions_loop(rt: RawTree, ms: MutableState) -> None:
    """Matriu de co-visita en viu (5.6.4), per al blend del règim final (4.7.2)."""
    while not STOP.is_set():
        try:
            rows = rt.query("""WITH v AS (SELECT session_id::String AS sid, product_id::String AS pid
                    FROM shop.events WHERE event_type::String = 'view'
                      AND ts::DateTime64(3) > now() - INTERVAL 60 MINUTE)
                SELECT a.pid AS p, b.pid AS q, count() AS n FROM v AS a INNER JOIN v AS b ON a.sid = b.sid
                WHERE a.pid != b.pid GROUP BY p, q ORDER BY n DESC LIMIT 50 BY p""", 60)
            with ms.lock:
                n = ms.n
                pj, pi, c = _pairs(rows, ms.index)
                order = np.lexsort((pi, pj))
                pj, pi, c = pj[order], pi[order], c[order]
                rowsum = np.bincount(pj, weights=c, minlength=n)
                indptr = np.searchsorted(pj, np.arange(n + 1)).astype(np.int64)
                data = (c / np.maximum(rowsum[pj], 1e-9)).astype(np.float32)
                ms.live = {"n": n, "indptr": indptr, "indices": pi.astype(np.int32), "data": data}
                ms.publish()
            log.info("[live-trans] matriu en viu: %d parells", len(rows))
        except Exception as e:  # noqa: BLE001
            log.warning("[live-trans] %s", str(e)[:200])
        STOP.wait(CFG.live_trans_s)


# --- 7.4 catàleg (4.6) -----------------------------------------------------------
def _code(lst: list, name: str) -> int:
    if name not in lst:
        lst.append(name)
    return lst.index(name)


def add_new_product(ms: MutableState, r: dict) -> int | None:
    """new_product: inicialització amb kNN + subcategoria + marca + global (4.6.2)."""
    cfg = ms.cfg
    pid = r["product_id"]
    if pid in ms.index:
        i = ms.index[pid]
        ms.is_new[i] = True
        return i
    if ms.n >= cfg.n_max_products:
        log.error("[catalog] N_MAX assolit: no s'afegeix %s", pid)
        return None
    i = ms.n
    c = _code(ms.cats, r.get("category") or "?")
    s = _code(ms.subs, r.get("subcategory") or "?")
    b = _code(ms.brands, r.get("brand") or "?")
    ms.cat[i], ms.sub[i], ms.brand[i] = c, s, b
    price = float(r.get("price") or 0)
    ms.price[i] = price
    ms.active[i] = str(r.get("active")).lower() in ("true", "1", "none", "")
    ms.unavailable[i] = price <= 0
    ms.is_new[i] = True
    e = embed_product(r.get("description") or "", r.get("category") or "", r.get("subcategory") or "",
                      r.get("brand") or "", cfg.emb_dim)
    ms.E[i] = e
    ms.price_pct[i] = np.searchsorted(ms.sorted_prices, price) / max(len(ms.sorted_prices), 1)
    sp = ms.sub_sorted_prices.get(s)
    ms.price_pct_sub[i] = (np.searchsorted(sp, price) / len(sp)) if sp is not None and len(sp) else 0.5
    n = ms.n
    alpha = ms.alpha0[:n] + ms.s_slow[:n] + ms.s_fast[:n] + ms.s_org[:n]
    beta = ms.beta0[:n] + ms.f_slow[:n] + ms.f_fast[:n] + ms.f_org[:n]
    mu_all = alpha / (alpha + beta)
    sim = ms.E[:n] @ e
    nb = np.argsort(-sim)[:cfg.knn_k]
    nb = nb[sim[nb] > 0.05]
    mu_sub = float(ms.m_sub[s]) if s < len(ms.m_sub) else ms.mG
    if nb.size:
        wk = np.maximum(sim[nb], 0) ** 2
        mu_knn = float((wk * mu_all[nb]).sum() / max(wk.sum(), 1e-6))
    else:
        mu_knn = mu_sub
    in_brand = np.flatnonzero(ms.brand[:n] == b)
    mu_brand = float(mu_all[in_brand].mean()) if in_brand.size >= 30 else mu_sub
    mu_prior = 0.45 * mu_knn + 0.30 * mu_sub + 0.15 * mu_brand + 0.10 * ms.mG          # (67)
    ess = ms.params()["ess_new"]                                                     # (68)
    # Amb mu ~ 0,002, Beta(mu*8, (1-mu)*8) té alpha ~ 0,016: sd de logit(theta) ~ 8 i Thompson omple les
    # llistes de productes nous a l'atzar. Es manté la mitjana mu_prior però amb alpha >= 1 (sd ~ 1 logit).
    a0 = max(mu_prior * ess, cfg.alpha_min_new)
    ms.alpha0[i], ms.beta0[i] = a0, a0 * (1 - mu_prior) / max(mu_prior, 1e-6)
    for a in (ms.s_slow, ms.f_slow, ms.s_fast, ms.f_fast, ms.s_org, ms.f_org, ms.s_click, ms.f_click):
        a[i] = 0.0
    ms.f_view[i] = ms.f_atc[i] = ms.f_q[i] = 0.0
    ms.p_raw[i] = 0.0
    ms.ids.append(pid)
    ms.index[pid] = i
    ms.n += 1
    ms.new_set.add(pid)
    ms.new_subcats.add(s)
    return i


def apply_catalog_update(ms: MutableState, r: dict, now: float) -> None:
    cfg = ms.cfg
    pid = r.get("product_id")
    change = r.get("change") or ""
    ts = parse_ts(r.get("ts")) or now
    if not pid:
        return
    if ts < ms.last_change_ts.get(pid, -1e18) and change != "new_product":
        return                                                    # fora d'ordre (4.6.1)
    ms.last_change_ts[pid] = max(ts, ms.last_change_ts.get(pid, -1e18))
    if change == "new_product" or pid not in ms.index:
        if change != "new_product" and not r.get("category"):
            return
        add_new_product(ms, r)
        if change == "new_product":
            return
    i = ms.index[pid]
    if change == "price_change":
        newp = float(r.get("price") or 0)
        old = float(ms.price[i])
        if newp <= 0:
            ms.unavailable[i] = True
            log.warning("[catalog] preu no vàlid per a %s", pid)
            return
        if abs(newp - old) > 1e-9:
            ratio = newp / max(old, 1e-6)
            ms.price_b0[i] = float(np.clip(-cfg.price_eps * math.log(ratio), -cfg.price_bonus_cap,
                                           cfg.price_bonus_cap))          # (71)
            ms.price_ts[i] = ts
            ms.price[i] = newp
    elif change == "out_of_stock":
        ms.unavailable[i] = True
    elif change == "back_in_stock":
        ms.unavailable[i] = False
        ms.active[i] = True
        ms.restock_ts[i] = ts                                       # (72)
    if "active" in r and r.get("active") is not None and change not in ("out_of_stock", "back_in_stock"):
        ms.active[i] = str(r["active"]).lower() in ("true", "1")


SQL_CATALOG_UPDATES = """SELECT product_id::String AS product_id, toString(ts::DateTime64(3)) AS ts,
 change::String AS change, toString(category) AS category, toString(subcategory) AS subcategory,
 toString(brand) AS brand, price_eur::Float64 AS price, toString(description) AS description,
 toString(is_active) AS active
FROM shop.catalog_updates WHERE ts::DateTime64(3) > toDateTime64('{since}', 3)
ORDER BY ts::DateTime64(3) ASC, product_id::String ASC LIMIT 5000"""


def reconcile_stock(rt: RawTree, ms: MutableState) -> None:
    rows = rt.query("""SELECT product_id::String AS product_id FROM shop.catalog_updates
        WHERE change::String IN ('out_of_stock', 'back_in_stock') GROUP BY product_id
        HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'""", 15)
    oos = {r["product_id"] for r in rows}
    with ms.lock:
        fixed = 0
        for pid, i in list(ms.index.items()):
            if i >= ms.n:
                continue
            want = pid in oos
            if ms.unavailable[i] != want and ms.price[i] > 0:
                ms.unavailable[i] = want
                fixed += 1
        if fixed:
            log.warning("[catalog] reconciliació: %d productes corregits d'estoc", fixed)
            ms.publish()


def catalog_loop(rt: RawTree, ms: MutableState, since: str) -> None:
    seen = TTLCache(3600, 100_000)
    last_rec = time.time()
    while not STOP.is_set():
        try:
            rows = rt.query(SQL_CATALOG_UPDATES.format(since=since), 10)
            fresh = [r for r in rows if (r["product_id"], r["ts"], r["change"]) not in seen]
            if fresh:
                now = Clock.now()
                with ms.lock:
                    for r in fresh:
                        seen.add((r["product_id"], r["ts"], r["change"]))
                        try:
                            apply_catalog_update(ms, r, now)
                        except Exception as e:  # noqa: BLE001
                            log.warning("[catalog] %s: %s", r.get("product_id"), e)
                    ms.publish()
                log.info("[catalog] %d canvis aplicats (%s)", len(fresh),
                         dict(Counter(r["change"] for r in fresh)))
            if rows:
                since = fmt_ts((parse_ts(rows[-1]["ts"]) or Clock.now()) - 5.0)
            if time.time() - last_rec > CFG.stock_reconcile_s:
                reconcile_stock(rt, ms)
                last_rec = time.time()
        except Exception as e:  # noqa: BLE001
            if "Unknown expression identifier" not in str(e):
                log.warning("[catalog] %s", str(e)[:200])
        STOP.wait(CFG.catalog_poll_s)


# --- 7.5 salut, règim i circuit breaker (4.5.7, 4.7, 4.8) ------------------------------
def trigger_regime(ms: MutableState, reason: str, hard: bool = False) -> None:
    """regime_shift() (4.7.2). Idempotent: s'executa una sola vegada (4.7.1).

    Els disparadors suaus (CUSUM, TV, cold ratio) només actuen des d'1 h abans de la final:
    durant la pràctica una falsa alarma retallaria l'evidència històrica abans d'hora.
    """
    now = time.time()
    final_ts = parse_ts(CFG.final_start_utc) or 0.0
    with ms.lock:
        if hard:
            if ms.hard_fired:
                return
            ms.hard_fired = True
            if ms.final_trigger_ts is not None and now - ms.final_trigger_ts < 1800:
                return                                  # ja s'està adaptant per un disparador suau
        else:
            if ms.final_trigger_ts is not None:
                return
            if Clock.now() < final_ts - 3600:
                if now - ms.last_soft_trigger > 600:
                    log.warning("[regime] alarma (%s) ignorada: encara som a la pràctica", reason)
                    ms.last_soft_trigger = now
                return
        ms.final_trigger_ts = now
        n = ms.n
        ms.s_slow[:n] *= 0.25                       # 4.7.2 accions immediates
        ms.f_slow[:n] *= 0.25
        ms.Sigma_beta = 0.5 * ms.Sigma_beta + 0.5 * ms.Sigma0
        ms.mu_beta = 0.8 * ms.mu_beta
        ms.publish()
    log.warning("[regime] CANVI DE RÈGIM activat (%s)", reason)


def health_loop(rt: RawTree, ms: MutableState, args) -> None:
    final_ts = parse_ts(CFG.final_start_utc)
    last_sync = 0.0
    last_cusum = time.time()
    g = 0.0
    r_ref = None
    cb_good_since = None
    last_lat_warn = 0.0
    while not STOP.is_set():
        try:
            now = time.time()
            if now - last_sync > 60:
                Clock.sync(rt)
                last_sync = now
            if not ms.hard_fired and Clock.now() >= final_ts:
                trigger_regime(ms, "rellotge: diumenge 09:00 UTC", hard=True)
            # mètriques de feedback dels últims 5 min (només de peticions posteriors a l'arrencada)
            t_min = max(Clock.now() - 300, MET.start + 30)
            fb = [x for x in list(MET.fb) if x[0] >= t_min]
            team = [x for x in fb if x[1] == "team"]
            fallback = [x for x in fb if x[1] == "fallback"]
            fb_rate = len(fallback) / max(len(fb), 1)
            inv_rate = sum(1 for x in fallback if x[2] == "invalid") / max(len(fb), 1)
            lats = sorted(x[3] for x in team if x[3] > 0)
            p99 = lats[int(0.99 * (len(lats) - 1))] if lats else 0.0
            # Circuit breaker (4.8.4), adaptat a les mesures: el càlcul triga 1-20 ms i la latència
            # és de xarxa (~1 s), així que servir best sellers no l'arregla. Només les respostes
            # INVÀLIDES (un error del model o de l'estoc) fan baixar de nivell.
            prev = ms.cb
            if len(fb) >= 30 and not args.dry_run:
                if inv_rate > 0.02:
                    ms.cb, cb_good_since = "SAFE", None
                elif inv_rate > 0.005:
                    if ms.cb == "NORMAL":
                        ms.cb = "DEGRADED"
                    cb_good_since = None
                elif inv_rate == 0:
                    cb_good_since = cb_good_since or now
                    if ms.cb != "NORMAL" and now - cb_good_since > 120:
                        ms.cb = "NORMAL"
                if (fb_rate > 0.03 or p99 > 2500) and now - last_lat_warn > 60:
                    log.warning("[latència] fallback %.1f%% %s · p99 %.0f ms (xarxa/servidor)", 100 * fb_rate,
                                dict(Counter(x[2] for x in fallback)), p99)
                    last_lat_warn = now
            if ms.cb != prev:
                log.warning("[cb] %s -> %s (invàlides %.1f%%, fallback %.1f%%, p99 %.0f ms)", prev, ms.cb,
                            100 * inv_rate, 100 * fb_rate, p99)
                ms.publish()
            # CUSUM de conversió (4.5.7), cada minut amb finestres de 5 min
            if now - last_cusum >= 60 and len(team) >= 100:
                r_t = sum(x[4] for x in team) / len(team)
                r_ref = r_t if r_ref is None else r_ref + (1 - 0.5 ** (1 / 60)) * (r_t - r_ref)
                g = max(0.0, g + (r_ref - r_t) - 0.005)
                if g > max(5 * r_ref, 0.05):
                    trigger_regime(ms, f"CUSUM conversió (g={g:.3f})")
                    g, r_ref = 0.0, r_t
                last_cusum = now
            if ms.sub_tv > 0.25:
                trigger_regime(ms, f"mix de subcategories TV={ms.sub_tv:.2f}")
            REQ_META.purge()
            SEEN_REQ.purge()
            ms.user_clicked.purge()
            ms.live_atc.purge()
            # línia d'estat
            pl = sorted(MET.push_lat)[-2000:]
            am = sorted(MET.answer_ms)
            q = lambda xs, f: xs[int(f * (len(xs) - 1))] if xs else 0.0  # noqa: E731
            reasons = Counter(x[2] for x in fallback)
            conv = sum(x[4] for x in team) / max(len(team), 1)
            snap = STATE.snap
            prm = snap.params if snap else {}
            log.info("[status] req answered=%d expired=%d pushed=%d err=%d | answer p50=%.0fms p99=%.0fms | "
                     "edat al push p50=%.0fms p95=%.0fms | fb5m team=%d fallback=%d %s conv=%.2f%% | cb=%s "
                     "regime=%s n=%d nous=%d sense_estoc=%d | beta0=%.2f sd_beta=%.3f%s",
                     MET.c["answered"], MET.c["expired"] + MET.c["dropped_late"], MET.c["pushed"],
                     MET.c["push_errors"], q(am, 0.5), q(am, 0.99), q(pl, 0.5), q(pl, 0.95), len(team),
                     len(fallback), dict(reasons) if reasons else "", 100 * conv, ms.cb, prm.get("regime"),
                     ms.n, int(ms.is_new[:ms.n].sum()), int(ms.unavailable[:ms.n].sum()),
                     float(ms.mu_beta[0]), float(np.sqrt(np.diag(ms.Sigma_beta)[1:]).mean()),
                     " [DRY-RUN]" if args.dry_run else "")
        except Exception as e:  # noqa: BLE001
            log.warning("[health] %s", e)
        STOP.wait(10)


# --- 7.6 persistència (4.8.7, 5.x simplificat) -------------------------------------
def checkpoint_path() -> Path:
    return CACHE / "online_state.pkl"


def save_checkpoint(ms: MutableState) -> None:
    with ms.lock:
        n = ms.n
        nz = np.flatnonzero((ms.s_fast[:n] + ms.f_fast[:n] + ms.s_click[:n] + ms.f_click[:n]) > 0)
        st = {"ts": time.time(), "built_at": ms.P["built_at"], "mu_beta": ms.mu_beta, "Sigma_beta": ms.Sigma_beta,
              "s_slow": {ms.ids[i]: (ms.s_slow[i], ms.f_slow[i]) for i in range(n)},
              "fast": {ms.ids[i]: (ms.s_fast[i], ms.f_fast[i], ms.s_click[i], ms.f_click[i]) for i in nz},
              "final_trigger_ts": ms.final_trigger_ts, "hard_fired": ms.hard_fired,
              "adf_version": 2, "fb_last": FB_STATE.get("last")}
    tmp = checkpoint_path().with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(checkpoint_path())


def restore_checkpoint(ms: MutableState, max_age_h: float = 6.0) -> None:
    p = checkpoint_path()
    if not p.exists():
        return
    try:
        st = pickle.loads(p.read_bytes())
        if st.get("built_at") != ms.P["built_at"] or time.time() - st["ts"] > max_age_h * 3600:
            log.info("[checkpoint] ignorat (priors diferents o massa vell)")
            return
        with ms.lock:
            for pid, (s, f) in st["s_slow"].items():
                i = ms.index.get(pid)
                if i is not None:
                    ms.s_slow[i], ms.f_slow[i] = s, f
            for pid, (s, f, sc, fc) in st["fast"].items():
                i = ms.index.get(pid)
                if i is not None:
                    ms.s_fast[i], ms.f_fast[i], ms.s_click[i], ms.f_click[i] = s, f, sc, fc
            if st.get("adf_version") == 2:       # la beta de l'ADF antic (només positius) havia derivat
                ms.mu_beta, ms.Sigma_beta = st["mu_beta"], st["Sigma_beta"]
            # el feedback ja comptat al checkpoint no es torna a comptar
            FB_STATE["last"] = st.get("fb_last") or (st["ts"] - 3.0)
            ms.final_trigger_ts, ms.hard_fired = st.get("final_trigger_ts"), st.get("hard_fired", False)
        log.info("[checkpoint] estat online restaurat (%.0f min)", (time.time() - st["ts"]) / 60)
    except Exception as e:  # noqa: BLE001
        log.warning("[checkpoint] no s'ha pogut restaurar: %s", e)


def persist_loop(rt: RawTree, ms: MutableState, args) -> None:
    while not STOP.wait(CFG.persist_s):
        try:
            if args.dry_run:                     # no trepitja el checkpoint del procés en viu
                continue
            save_checkpoint(ms)
            t_min = Clock.now() - 300
            fb = [x for x in list(MET.fb) if x[0] >= t_min]
            team = [x for x in fb if x[1] == "team"]
            snap = STATE.snap
            row = {"kind": "metrics", "ts": fmt_ts(Clock.now()), "model_version": CFG.model_version,
                   "answered": MET.c["answered"], "pushed": MET.c["pushed"],
                   "expired": MET.c["expired"] + MET.c["dropped_late"], "push_errors": MET.c["push_errors"],
                   "fb5m_team": len(team), "fb5m_fallback": len(fb) - len(team),
                   "conv5m": sum(x[4] for x in team) / max(len(team), 1),
                   "rev5m": sum(x[5] for x in team) / max(len(team), 1),
                   "cb": ms.cb, "regime": snap.params["regime"] if snap else "",
                   "n_products": ms.n, "n_new": int(ms.is_new[:ms.n].sum()),
                   "beta": {FEATURES[k]: round(float(ms.mu_beta[k]), 4) for k in range(D_FEAT)}}
            rt.insert("scratch", [row], timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("[persist] %s", str(e)[:200])


# =============================================================================
# 8. ARRENCADA EN VIU
# =============================================================================
def bootstrap_live(rt: RawTree, ms: MutableState) -> tuple[str, str]:
    """Aplica tots els catalog_updates, compres en viu i estoc; inicialitza tendències."""
    since_cat = "1970-01-01 00:00:00.000"
    try:
        rows = rt.query(SQL_CATALOG_UPDATES.format(since=since_cat).replace("LIMIT 5000", "LIMIT 100000"), 30)
        now = Clock.now()
        with ms.lock:
            for r in rows:
                apply_catalog_update(ms, r, now)
        if rows:
            since_cat = rows[-1]["ts"]
        log.info("[boot] %d catalog_updates aplicats: %s · %d productes nous", len(rows),
                 dict(Counter(r["change"] for r in rows)), int(ms.is_new[:ms.n].sum()))
        reconcile_stock(rt, ms)
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] catalog_updates: %s", str(e)[:200])
    for sql, label in (
        ("""SELECT user_id::String AS u, groupUniqArray(product_id::String) AS ps FROM shop.events
            WHERE event_type::String = 'purchase' GROUP BY u""", "compres orgàniques en viu"),
        (f"""SELECT r.user_id::String AS u, groupUniqArray(f.purchased::String) AS ps
            FROM {rt.team}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
            WHERE ifNull(toString(f.purchased), '') != '' GROUP BY u""", "compres al carrusel")):
        try:
            rows = rt.query(sql, 60)
            with ms.lock:
                for r in rows:
                    for p in r["ps"]:
                        ms.add_bought(r["u"], ms.index.get(p, -1))
            log.info("[boot] %s: %d usuaris", label, len(rows))
        except Exception as e:  # noqa: BLE001
            log.warning("[boot] %s: %s", label, str(e)[:200])
    try:
        rows = rt.query("""SELECT user_id::String AS u, groupUniqArray(product_id::String) AS ps FROM shop.events
            WHERE event_type::String = 'add_to_cart' AND ts::DateTime64(3) > now() - INTERVAL 30 MINUTE
            GROUP BY u""", 30)
        for r in rows:
            ms.live_atc.add(r["u"], frozenset(ms.index[p] for p in r["ps"] if p in ms.index))
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] live_atc: %s", str(e)[:200])
    since_ev = fmt_ts(Clock.now())
    try:
        since_ev = bootstrap_trends(rt, ms)
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] tendències: %s", str(e)[:200])
    return since_cat, since_ev


def mark_already_submitted(rt: RawTree) -> None:
    try:
        rows = rt.query(f"""SELECT request_id::String AS request_id FROM {rt.team}.submissions
            WHERE request_id::String IN (SELECT request_id::String FROM shop.requests
                                         WHERE ts::DateTime64(3) > now() - INTERVAL 3 MINUTE)""", 15)
        for r in rows:
            SEEN_REQ.add(r["request_id"])
        log.info("[boot] %d peticions recents ja respostes (no es duplicaran)", len(rows))
    except Exception as e:  # noqa: BLE001
        log.warning("[boot] submissions recents: %s", str(e)[:200])


def make_rt() -> RawTree:
    load_env()
    if "RAWTREE_API_KEY" not in os.environ or "TEAM_ID" not in os.environ:
        sys.exit("Falten TEAM_ID / RAWTREE_API_KEY (fitxer .env)")
    return RawTree(os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com"),
                   os.environ["RAWTREE_API_KEY"], os.environ["TEAM_ID"])


def describe(snap: Snapshot, pid: str) -> str:
    i = snap.index.get(pid)
    if i is None:
        return pid
    ms_cats = STATE_NAMES.get("cats", [])
    ms_subs = STATE_NAMES.get("subs", [])
    c = ms_cats[snap.cat[i]] if snap.cat[i] < len(ms_cats) else "?"
    s = ms_subs[snap.sub[i]] if snap.sub[i] < len(ms_subs) else "?"
    flag = " [NOU]" if snap.is_new[i] else ""
    return f"{pid} · {c} / {s} · {snap.price[i]:.2f} € · mu={snap.mu[i]:.4f}{flag}"


STATE_NAMES: dict = {}


def run_live(args) -> None:
    rt = make_rt()
    Clock.sync(rt)
    P = load_or_build_priors(rt, CFG, None, args.rebuild)
    ms = MutableState(P, CFG)
    STATE_NAMES.update(cats=ms.cats, subs=ms.subs)
    since_cat, since_ev = bootstrap_live(rt, ms)
    restore_checkpoint(ms)
    ms.publish()
    if args.smoke:
        run_smoke(rt, args)
        return
    mark_already_submitted(rt)
    MET.start = Clock.now()
    threads = [threading.Thread(target=f, args=a, daemon=True, name=f.__name__) for f, a in (
        (request_loop, (rt, args, 0)), (request_loop, (rt, args, 1)), (push_loop, (rt, args)),
        (feedback_loop, (rt, ms, args)),
        (events_loop, (rt, ms, since_ev)), (catalog_loop, (rt, ms, since_cat)),
        (live_transitions_loop, (rt, ms)), (persist_loop, (rt, ms, args)), (health_loop, (rt, ms, args)))]
    for t in threads:
        t.start()
    log.info("[main] %s en marxa%s (Ctrl+C per aturar)", CFG.model_version, " · DRY-RUN" if args.dry_run else "")

    def stop(*_):
        STOP.set()

    signal.signal(signal.SIGTERM, stop)
    died = False
    try:
        while not STOP.is_set():
            time.sleep(1)
            for t in threads:
                if not t.is_alive() and not STOP.is_set():
                    log.error("[main] el fil %s ha mort; s'atura perquè el supervisor el reiniciï", t.name)
                    died = True
                    STOP.set()
    except KeyboardInterrupt:
        STOP.set()
    try:
        if not args.dry_run:
            save_checkpoint(ms)
    except Exception as e:  # noqa: BLE001
        log.warning("[main] checkpoint final: %s", e)
    log.info("[main] aturat")
    if died:
        sys.exit(1)


def run_smoke(rt: RawTree, args) -> None:
    """Respon les peticions reals més recents SENSE enviar-les; mostra latència i exemples."""
    rows = rt.query(SQL_REQUESTS.replace("60 SECOND", "120 SECOND"), 10)[-30:]
    rng = np.random.default_rng(1)
    snap = STATE.snap
    lat = []
    for k, r in enumerate(rows):
        t0 = time.perf_counter()
        items = answer(snap, r, rng)
        lat.append((time.perf_counter() - t0) * 1000)
        assert len(items) == 10 and len(set(items)) == 10, "resposta invàlida"
        assert all(snap.ok[snap.index[x]] for x in items), "producte no elegible"
        if k < 3:
            print(f"\n--- petició {r['request_id']} · page={r.get('page')} device={r.get('device')} "
                  f"region={r.get('region')} is_new_user={r.get('is_new_user')}")
            for v in normalize_views(r.get("recent_views"))[-3:]:
                print("    vist:      ", describe(snap, v[0]))
            for j, x in enumerate(items, 1):
                print(f"    slot {j:>2}:    ", describe(snap, x))
    lat = np.array(lat)
    print(f"\n{len(rows)} peticions · answer() p50={np.median(lat):.1f} ms · p99={np.percentile(lat, 99):.1f} ms"
          f" · màx={lat.max():.1f} ms · totes vàlides")


# =============================================================================
# 9. AVALUACIÓ OFFLINE (secció 7)
# =============================================================================
def run_eval(args) -> None:
    rt = make_rt()
    hist_end = parse_ts(rt.query("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.history_sessions")[0]["t"])
    cut = fmt_ts(hist_end - 7 * 86400)[:19]
    P = load_or_build_priors(rt, CFG, cut, args.rebuild)
    ms = MutableState(P, CFG)
    STATE_NAMES.update(cats=ms.cats, subs=ms.subs)
    snap = ms.publish(set_global=False)
    n_test = int(args.eval)
    rows = rt.query(f"""SELECT toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id,
        toString(context.device) AS device, toString(context.region) AS region, toString(context.page) AS page,
        recent_views, shown::Array(String) AS shown, ifNull(toString(clicked), '') AS clicked,
        ifNull(toString(purchased), '') AS purchased
        FROM shop.history_sessions WHERE clicked != '' AND ts::DateTime64(3) >= toDateTime64('{cut}', 3)
        ORDER BY cityHash64(request_id::String) LIMIT {n_test}""", 120)
    best = [snap.ids[i] for i in np.argsort(-P["p_raw"])[:10]]
    cat_best = {}
    for c in range(len(ms.cats)):
        idx = np.flatnonzero(P["cat"] == c)
        cat_best[c] = [snap.ids[i] for i in idx[np.argsort(-P["p_raw"][idx])][:10]]
    rng = np.random.default_rng(3)
    res = defaultdict(lambda: Counter())
    lat = []
    for r in rows:
        req = dict(r, is_new_user="false")
        info: dict = {}
        t0 = time.perf_counter()
        items = answer(snap, req, rng, info)
        lat.append((time.perf_counter() - t0) * 1000)
        cand_ids = {snap.ids[i] for i in info["cand"].tolist()}
        views = normalize_views(r["recent_views"])
        last_cat = snap.cat[snap.index[views[-1][0]]] if views and views[-1][0] in snap.index else None
        cb = cat_best.get(int(last_cat), best) if last_cat is not None else best
        for seg in ("tots", f"page={r['page']}"):
            for tgt_name, tgt in (("clic", r["clicked"]), ("compra", r["purchased"])):
                if not tgt:
                    continue
                c = res[(seg, tgt_name)]
                c["n"] += 1
                c["v2"] += tgt in items
                c["v2_top3"] += tgt in items[:3]
                c["recall100"] += tgt in cand_ids
                c["baseline"] += tgt in best
                c["cat_bestsellers"] += tgt in cb
    lat = np.array(lat)
    print(f"\nAvaluació offline · tall {cut} · {len(rows)} impressions amb clic del holdout (7 dies)")
    print(f"answer(): p50 {np.median(lat):.1f} ms · p99 {np.percentile(lat, 99):.1f} ms\n")
    print(f"{'segment':<18}{'target':<8}{'n':>7}{'HR@10 v2':>10}{'HR@3 v2':>9}{'recall@100':>12}"
          f"{'HR@10 baseline':>16}{'HR@10 best-cat':>16}")
    for (seg, tg), c in sorted(res.items()):
        n = max(c["n"], 1)
        print(f"{seg:<18}{tg:<8}{c['n']:>7}{c['v2'] / n:>10.3f}{c['v2_top3'] / n:>9.3f}{c['recall100'] / n:>12.3f}"
              f"{c['baseline'] / n:>16.3f}{c['cat_bestsellers'] / n:>16.3f}")
    print("\nNota: el log històric només mostrava productes de la categoria de l'última vista, així que")
    print("el clic sempre és d'aquesta categoria: el HR@10 està esbiaixat a favor de 'best-cat' (7.2.1).")
    eval_organic(rt, snap, P, cut, n_test, best, cat_best)


def eval_organic(rt: RawTree, snap: Snapshot, P: dict, cut: str, n_test: int, best: list, cat_best: dict) -> None:
    """Avaluació no esbiaixada pel carrusel antic: sessions orgàniques del holdout amb compra.

    Per a cada sessió, la petició es construeix amb les vistes ANTERIORS a la primera compra i
    es mira si el producte comprat és al top-10. Mesura si recomanem el que la gent tria lliurement.
    """
    sids = [r["sid"] for r in rt.query(f"""SELECT session_id::String AS sid FROM shop.history_events
        WHERE event_type::String = 'purchase' AND ts::DateTime64(3) >= toDateTime64('{cut}', 3)
        GROUP BY sid ORDER BY cityHash64(sid) LIMIT {n_test}""", 120)]
    ev = []
    for k in range(0, len(sids), 1000):
        ev += rt.query(f"""SELECT session_id::String AS sid, user_id::String AS u, device::String AS d,
            region::String AS r, product_id::String AS p, event_type::String AS t, toString(ts::DateTime64(3)) AS ts
            FROM shop.history_events WHERE session_id::String IN ({q_list(sids[k:k + 1000])})
            ORDER BY ts::DateTime64(3) ASC""", 120)
    by = defaultdict(list)
    for e in ev:
        by[e["sid"]].append(e)
    rng = np.random.default_rng(5)
    res = defaultdict(Counter)
    for sid, es in by.items():
        k_buy = next((k for k, e in enumerate(es) if e["t"] == "purchase"), None)
        if k_buy is None:
            continue
        target = es[k_buy]["p"]
        views = [{"product_id": e["p"], "ts": e["ts"]} for e in es[:k_buy] if e["t"] == "view"][-10:]
        if not views:
            continue
        tp = parse_ts(es[k_buy]["ts"]) or 0
        req = {"user_id": es[0]["u"], "is_new_user": "false", "device": es[0]["d"], "region": es[0]["r"],
               "page": "product", "recent_views": views, "ts": fmt_ts(tp - 1)}
        info: dict = {}
        items = answer(snap, req, rng, info)
        cand_ids = {snap.ids[i] for i in info["cand"].tolist()}
        seen_before = target in {v["product_id"] for v in views}
        last = snap.index.get(views[-1]["product_id"])
        cb = cat_best.get(int(snap.cat[last]), best) if last is not None else best
        for seg in ("tots", "comprat ja vist" if seen_before else "comprat no vist"):
            c = res[seg]
            c["n"] += 1
            c["v2"] += target in items
            c["v2_top3"] += target in items[:3]
            c["recall100"] += target in cand_ids
            c["baseline"] += target in best
            c["cat_bestsellers"] += target in cb
    print(f"\nAvaluació orgànica · {sum(1 for _ in by)} sessions del holdout amb compra (vistes prèvies -> compra)")
    print(f"{'segment':<20}{'n':>7}{'HR@10 v2':>10}{'HR@3 v2':>9}{'recall@100':>12}{'HR@10 baseline':>16}"
          f"{'HR@10 best-cat':>16}")
    for seg, c in sorted(res.items()):
        n = max(c["n"], 1)
        print(f"{seg:<20}{c['n']:>7}{c['v2'] / n:>10.3f}{c['v2_top3'] / n:>9.3f}{c['recall100'] / n:>12.3f}"
              f"{c['baseline'] / n:>16.3f}{c['cat_bestsellers'] / n:>16.3f}")


# =============================================================================
# 10. CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="construeix els priors i surt")
    ap.add_argument("--rebuild", action="store_true", help="ignora la memòria cau de priors")
    ap.add_argument("--eval", metavar="N", help="avaluació offline amb N impressions del holdout")
    ap.add_argument("--smoke", action="store_true", help="respon 30 peticions reals sense enviar-les")
    ap.add_argument("--dry-run", action="store_true", help="tot el sistema però sense escriure a RawTree")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="CLAU=VALOR",
                    help="canvia un hiperparàmetre de Config, p. ex. --set lambda_same_cat=2.0 (repetible)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    for kv in args.set:
        k, _, v = kv.partition("=")
        if not hasattr(CFG, k):
            sys.exit(f"--set: Config no té '{k}'")
        old = getattr(CFG, k)
        setattr(CFG, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes"))
        log.info("[config] %s = %s (abans %s)", k, getattr(CFG, k), old)
    if args.model_version:
        CFG.model_version = args.model_version
    if args.eval:
        run_eval(args)
        return
    if args.build:
        rt = make_rt()
        load_or_build_priors(rt, CFG, None, True)
        return
    run_live(args)


if __name__ == "__main__":
    main()
