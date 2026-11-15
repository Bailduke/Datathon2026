"""Paso 1: ¿qué vida media del EWMA predice mejor la cuota de compras orgánicas de cada subcategoría en los
5 minutos siguientes?  tendencia_s = (EWMA eventos de s + a) / (EWMA total × cuota habitual de s + a).
Se ajusta el exponente beta con la primera mitad del stream y se evalúa en la segunda."""
import os, sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rt import q
f = os.path.join(HERE, ".cache", "tend_min.pkl")
if os.path.exists(f): d = pd.read_pickle(f)
else:
    r = q("""WITH cat AS (SELECT product_id::String AS p, any(toString(subcategory)) AS s FROM
                          (SELECT product_id, subcategory FROM shop.catalog UNION ALL
                           SELECT product_id, subcategory FROM shop.catalog_updates WHERE change::String = 'new_product') GROUP BY p)
             SELECT cat.s AS s, toStartOfMinute(e.ts::DateTime64(3)) AS m, countIf(e.event_type::String = 'purchase') AS b,
                    countIf(e.event_type::String = 'view') AS v, countIf(e.event_type::String = 'add_to_cart') AS c
             FROM shop.events AS e INNER JOIN cat ON cat.p = e.product_id::String GROUP BY s, m""", 300)
    d = pd.DataFrame(r["data"]); d["m"] = pd.to_datetime(d.m); d.to_pickle(f)
for c in "bvc": d[c] = d[c].astype(float)
mins = pd.date_range(d.m.min(), d.m.max() - pd.Timedelta(minutes=1), freq="1min")
piv = lambda c: d.pivot(index="m", columns="s", values=c).reindex(mins).fillna(0).values
B, V, Cc = piv("b"), piv("v"), piv("c"); subs = sorted(d.s.unique()); T, S = B.shape
print(f"{T} minutos ({mins[0]:%H:%M}–{mins[-1]:%H:%M} UTC) · {S} subcategorías · {int(B.sum())} compras · {int(V.sum())} vistas")
H = 5
fut = np.zeros_like(B)
for k in range(1, H + 1): fut[:-k] += B[k:]
fut[-H:] = 0
def ewma(X, hl):
    dec = 0.5 ** (1.0 / hl); E = np.zeros_like(X); e = np.zeros(X.shape[1])
    for t in range(X.shape[0]): e = e * dec + X[t]; E[t] = e
    return E
qB = ewma(B, 240.0); qB = qB / np.maximum(qB.sum(1, keepdims=True), 1e-9)          # cuota habitual de compras (vida media 4 h)
warm = 60; half = T // 2; tr = np.zeros(T, bool); tr[warm:half] = True; te = np.zeros(T, bool); te[half:T - H] = True
def ll(beta, trend, mask):
    lp = np.log(qB[mask] + 1e-6) + beta * np.log(trend[mask]); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    return float((fut[mask] * lp).sum())
base_te = ll(0.0, np.ones_like(B), te); n_te = fut[te].sum()
print(f"evaluación: {int(n_te)} compras en la segunda mitad · logver con cuotas fijas {base_te:.0f} ({base_te/n_te:.4f} por compra)\n")
res = []
for name, X in (("vistas", V), ("compras", B), ("vistas+carrito+compras", V + Cc + B)):
    qX = ewma(X, 240.0); qX = qX / np.maximum(qX.sum(1, keepdims=True), 1e-9)
    for hl in (1, 2, 3, 5, 8, 12, 20, 30):
        E = ewma(X, hl); Et = E.sum(1, keepdims=True)
        for a in (5.0, 20.0, 60.0):
            trend = (E + a) / (Et * qX + a)
            betas = np.linspace(0, 1.6, 17); b = betas[np.argmax([ll(x, trend, tr) for x in betas])]
            res.append((name, hl, a, b, ll(b, trend, te) - base_te))
R = pd.DataFrame(res, columns=["señal", "vida_media_min", "a", "beta", "mejora"])
best = R.loc[R.groupby(["señal", "vida_media_min"]).mejora.idxmax()]
print("mejora de log-verosimilitud en la segunda mitad (mejor suavizado 'a' de cada caso):")
print(best.pivot(index="vida_media_min", columns="señal", values="mejora").round(0).to_string())
top = R.sort_values("mejora", ascending=False).head(5); print("\nlas 5 mejores combinaciones:"); print(top.round(2).to_string(index=False))
# retraso al empezar y cola al terminar, con la mejor combinación de 'vistas+carrito+compras'
bb = R[R["señal"] == "vistas+carrito+compras"].sort_values("mejora", ascending=False).iloc[0]
X = V + Cc + B; qX = ewma(X, 240.0); qX = qX / np.maximum(qX.sum(1, keepdims=True), 1e-9)
share = B / np.maximum(B.sum(1, keepdims=True), 1); med = np.median(share, 0)
sm = pd.DataFrame(share).rolling(5, min_periods=1).mean().values
print(f"\nepisodios (umbral de aviso: tendencia >= 2) con vida media {bb.vida_media_min:.0f} min y a = {bb.a:.0f}; entre paréntesis, con vida media 2 y 12:")
for s in range(S):
    hot = (sm[:, s] > 4 * max(med[s], 1e-4)) & (pd.DataFrame(B[:, s]).rolling(5, min_periods=1).sum().values[:, 0] >= 12)
    idx = np.flatnonzero(hot)
    if idx.size < 15: continue
    t0, t1 = idx[0], idx[-1]; out = []
    for hl in (bb.vida_media_min, 2, 12):
        E = ewma(X, hl); tr_ = (E + bb.a) / (E.sum(1, keepdims=True) * qX + bb.a)
        on = np.flatnonzero(tr_[max(t0 - 10, 0):t1, s] >= 2.0); delay = (on[0] - 10) if on.size else np.nan
        off = np.flatnonzero(tr_[t1:t1 + 90, s] < 2.0); tail = off[0] if off.size else np.nan
        out.append((delay, tail, tr_[t0:t1, s].max()))
    print(f"  {subs[s][:20]:20} {mins[t0]:%H:%M}–{mins[t1]:%H:%M} · se detecta a los {out[0][0]:.0f} min, se apaga {out[0][1]:.0f} min después de acabar, máximo x{out[0][2]:.1f}"
          f"   (vm 2: {out[1][0]:.0f}/{out[1][1]:.0f} · vm 12: {out[2][0]:.0f}/{out[2][1]:.0f})")
