"""H11 · Exposición de la popularidad solo con el producto en stock: ¿de qué tamaño es el efecto?"""
import numpy as np
from collections import defaultdict
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
index = {p: i for i, p in enumerate(P["ids"])}
allp = np.array([te for (te, u, p, e) in events if e == "purchase"]); t_end = max(e[0] for e in events[-1000:]); t_start = events[0][0]
Tl = len(allp); print(f"stream: {(t_end-t_start)/3.6e6:.1f} h · compras orgánicas de hoy T_live = {Tl} · T_hist = {float(P['T_hist']):.0f} · b0 = 20000")
oos = defaultdict(list); open_ = {}; launch = {}; newsub = {}
for r in cat:
    ts = int(rv.parse_ts(r["ts"]) * 1000); p = r["product_id"]
    if r["change"] == "new_product": launch[p] = ts
    elif r["change"] == "out_of_stock": open_.setdefault(p, ts)
    elif r["change"] == "back_in_stock" and p in open_: oos[p].append((open_.pop(p), ts))
for p, a in open_.items(): oos[p].append((a, t_end))
cnt = lambda a, b: int(np.searchsorted(allp, b) - np.searchsorted(allp, a))
rows = []
for p in set(list(index) + list(launch)):
    t0 = launch.get(p, t_start); T = cnt(t0, t_end); To = sum(cnt(max(a, t0), b) for a, b in oos.get(p, ()) if b > t0)
    rows.append((p, p in launch, T, To, len(oos.get(p, ()))))
C = np.asarray(P["C_hist"]); top = set(np.array(P["ids"])[np.argsort(-C)[:100]]); top500 = set(np.array(P["ids"])[np.argsort(-C)[:500]])
def summ(name, sel):
    f = np.array([To / max(T, 1) for (p, nw, T, To, k) in rows if sel(p, nw)]); k = np.array([k for (p, nw, T, To, k_) in rows if sel(p, nw) for k in [k_]])
    print(f"  {name:28s} n={f.size:5d} · alguna rotura {100*(k>0).mean():5.1f} % · fracción de exposición sin stock: media {100*f.mean():5.1f} % · mediana {100*np.median(f):5.1f} % · p90 {100*np.quantile(f,.9):5.1f} % · máx {100*f.max():5.1f} %")
    return f
print("fracción de las compras de la tienda (desde el alta) que ocurrieron con el producto SIN stock:")
summ("catálogo (5.000)", lambda p, nw: not nw); summ("100 más vendidos", lambda p, nw: p in top); summ("500 más vendidos", lambda p, nw: p in top500); summ("productos nuevos", lambda p, nw: nw)
# cambio en log q_i (media de la posterior) al final de los datos
m_sub = np.asarray(P["m_sub"]); sub = np.asarray(P["sub"]); b0 = 20000.0; Th = float(P["T_hist"])
live_c = defaultdict(int)
for (te, u, p, e) in events:
    if e == "purchase": live_c[p] += 1
d_cat, d_top = [], []
for (p, nw, T, To, k) in rows:
    if nw: continue
    i = index[p]; a = m_sub[sub[i]] * b0 + C[i] + live_c[p]; d = np.log((b0 + Th + T) / (b0 + Th + T - To)); d_cat.append(d)
    if p in top: d_top.append(d)
print(f"cambio en log q_i · catálogo: medio {np.mean(d_cat):.4f} · máx {np.max(d_cat):.4f} | 100 más vendidos: medio {np.mean(d_top):.4f} · máx {np.max(d_top):.4f}")
d_new = [np.log((b0 + T) / (b0 + T - To)) for (p, nw, T, To, k) in rows if nw]
print(f"cambio en log q_i · productos nuevos: medio {np.mean(d_new):.4f} · mediana {np.median(d_new):.4f} · p90 {np.quantile(d_new,.9):.4f} · máx {np.max(d_new):.4f}  (n={len(d_new)})")
print("  para comparar: desv. típica de log q_i entre productos del catálogo = %.2f" % np.std(np.log((C + 1) / C.sum())))
# el domingo a las 10:00 habrá ~2,6 veces más stream: mismo cálculo escalando T y To
sc = 2.6; d2 = [np.log((b0 + Th + sc * T) / (b0 + Th + sc * (T - To))) for (p, nw, T, To, k) in rows if p in top]
dn2 = [np.log((b0 + sc * T) / (b0 + sc * (T - To))) for (p, nw, T, To, k) in rows if nw]
print(f"proyección al domingo 10:00 (stream ×2,6): 100 más vendidos medio {np.mean(d2):.4f} máx {np.max(d2):.4f} · nuevos medio {np.mean(dn2):.4f} p90 {np.quantile(dn2,.9):.4f} máx {np.max(dn2):.4f}")
