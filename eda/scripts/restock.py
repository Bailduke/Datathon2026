"""¿Se compra distinto un producto justo después de volver a tener stock? (1) compras orgánicas del stream,
(2) nuestras impresiones del carrusel. Tasa por tramo de tiempo desde la reposición, frente a la tasa
normal de esos mismos productos."""
import os, sys, pickle, bisect, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rt import q
fb, events, cat = pickle.load(open(os.path.join(HERE, ".cache", "calib_dl.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
w_pos = np.asarray(P["w_pos"])
ts = lambda s: pd.Timestamp(s).value // 10**6
T_END = max(ts(r["ts"]) for r in cat); T0 = min(e[0] for e in events)
out, back = {}, {}
for r in cat:
    if r["change"] == "out_of_stock": out.setdefault(r["product_id"], []).append(ts(r["ts"]))
    elif r["change"] == "back_in_stock": back.setdefault(r["product_id"], []).append(ts(r["ts"]))
prods = sorted(back)
dur = [b - max([o for o in out.get(p, []) if o < b], default=np.nan) for p in prods for b in back[p]]
print(f"{len(prods)} productos con alguna reposición · {sum(len(v) for v in back.values())} reposiciones · duración de la rotura (min): mediana {np.nanmedian(dur)/6e4:.0f}, p10 {np.nanpercentile(dur,10)/6e4:.0f}, p90 {np.nanpercentile(dur,90)/6e4:.0f}")
EDG = [0, 10, 30, 60, 120, 1e9]; LAB = ["0–10 min", "10–30 min", "30–60 min", "1–2 h", "más de 2 h o nunca agotado"]
def state(p, t):
    """(en stock, minutos desde la última reposición o inf)"""
    o = out.get(p, []); b = back.get(p, [])
    lo = max([x for x in o if x <= t], default=None); lb = max([x for x in b if x <= t], default=None)
    if lo is not None and (lb is None or lb < lo): return False, np.inf
    return True, (t - lb) / 6e4 if lb is not None else np.inf
# (1) orgánico: compras y minutos de exposición en stock por tramo, solo productos con reposición
buy = {}
for t, u, p, e in events:
    if e == "purchase" and p in back and t <= T_END: buy.setdefault(p, []).append(t)
cnt = np.zeros(5); expo = np.zeros(5)
step = 60_000
for p in prods:
    grid = np.arange(T0, T_END, step)
    st = [state(p, t) for t in grid]
    for (ok, m) in st:
        if ok: expo[min(np.searchsorted(EDG, m, side="right") - 1, 4)] += 1
    for t in buy.get(p, []):
        ok, m = state(p, t)
        if ok: cnt[min(np.searchsorted(EDG, m, side="right") - 1, 4)] += 1
rate = cnt / expo
print("\n(1) COMPRAS ORGÁNICAS de esos productos, por tiempo desde la reposición")
for k in range(5): print(f"   {LAB[k]:28s} compras {int(cnt[k]):6d} · minutos-producto en stock {int(expo[k]):8d} · compras/hora por producto {60*rate[k]:.2f} · x{rate[k]/rate[4]:.2f} ± {rate[k]/rate[4]/np.sqrt(max(cnt[k],1)):.2f}")
# (2) carrusel: impresiones de productos con reposición
rows = []
for r in fb:
    for j, p in enumerate((r["shown"] or [])[:10]):
        if p in back:
            ok, m = state(p, r["t"])
            rows.append((p, j, int(p == r["buy"]), ok, m, r["nu"] == "true"))
d = pd.DataFrame(rows, columns=["p", "slot", "y", "ok", "m", "nu"]); d["w"] = w_pos[d.slot.values]
print(f"\n(2) CARRUSEL: {len(d)} impresiones de productos con reposición, {int(d.y.sum())} compras · mostradas con el producto agotado: {int((~d.ok).sum())} ({int(d[~d.ok].y.sum())} compras)")
d = d[d.ok].copy(); d["tr"] = pd.cut(d.m, EDG, labels=LAB, right=False)
base = d.groupby("p").apply(lambda x: x.y.sum() / x.w.sum(), include_groups=False)       # tasa media de cada producto
d["esp"] = d.w * d.p.map(base)
g = d.groupby("tr", observed=True).agg(imp=("y", "size"), compras=("y", "sum"), exam=("w", "sum"), esperadas=("esp", "sum"))
g["conv_x_atencion_%"] = (100 * g.compras / g.exam).round(3); g["obs/esp"] = (g.compras / g.esperadas).round(2); g["±"] = (g["obs/esp"] / np.sqrt(g.compras.clip(lower=1))).round(2)
print(g.round(1).to_string())
