"""¿Cada cuánto hay tendencias por subcategoría en el stream y de qué tamaño? Compras orgánicas por subcategoría en cubos de 15 min."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rt import q
r = q("""WITH cat AS (SELECT product_id::String AS p, any(toString(subcategory)) AS s, any(toString(category)) AS c FROM
                      (SELECT product_id, subcategory, category FROM shop.catalog UNION ALL
                       SELECT product_id, subcategory, category FROM shop.catalog_updates WHERE change::String = 'new_product') GROUP BY p)
         SELECT cat.s AS s, any(cat.c) AS c, toStartOfInterval(e.ts::DateTime64(3), INTERVAL 15 MINUTE) AS b,
                countIf(e.event_type::String = 'purchase') AS compras, countIf(e.event_type::String = 'view') AS vistas
         FROM shop.events AS e INNER JOIN cat ON cat.p = e.product_id::String GROUP BY s, b ORDER BY s, b""", 300)
d = pd.DataFrame(r["data"]); d["b"] = pd.to_datetime(d.b); d["compras"] = d.compras.astype(float); d["vistas"] = d.vistas.astype(float)
last = d.b.max(); d = d[d.b < last]                      # el último cubo está incompleto
P = d.pivot(index="b", columns="s", values="compras").fillna(0); V = d.pivot(index="b", columns="s", values="vistas").fillna(0)
tot = P.sum(1); share = P.div(tot, axis=0); base = share.median(0); lift = share.div(base + 1e-9, axis=1)
print(f"stream: {P.index.min():%H:%M}–{P.index.max():%H:%M} UTC · {len(P)} cubos de 15 min · {P.shape[1]} subcategorías · {int(tot.sum())} compras orgánicas")
ep = []
for s in P.columns:
    hot = (lift[s] >= 2.5) & (P[s] >= 30)
    if not hot.any(): continue
    idx = np.flatnonzero(hot.values); groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for g in groups:
        ep.append((P.index[g[0]], s, d[d.s == s].c.iloc[0], 15 * len(g), float(lift[s].iloc[g].max()), float(share[s].iloc[g].mean()), float(base[s]),
                   float((P[s].iloc[g] - base[s] * tot.iloc[g]).sum())))
E = pd.DataFrame(ep, columns=["inicio", "subcategoría", "categoría", "min", "lift_max", "cuota_durante", "cuota_normal", "compras_extra"]).sort_values("inicio")
print(f"\nepisodios con cuota >= 2,5 veces la habitual: {len(E)} en {len(P)*15/60:.1f} h")
for _, x in E.iterrows():
    print(f"  {x.inicio:%H:%M} UTC · {x['subcategoría'][:22]:22} ({x['categoría'][:18]:18}) · {int(x['min']):3d} min · hasta x{x.lift_max:.1f} · cuota {100*x.cuota_durante:.1f} % (normal {100*x.cuota_normal:.1f} %) · {int(x.compras_extra)} compras extra")
print(f"\ncompras 'extra' de todos los episodios: {int(E.compras_extra.sum())} = {100*E.compras_extra.sum()/tot.sum():.1f} % de las compras orgánicas del stream")
hot_time = sum(E["min"]) / (len(P) * 15)
print(f"tiempo con alguna tendencia activa: {100*hot_time:.0f} % · durante un episodio, su subcategoría se lleva de media el {100*E.cuota_durante.mean():.0f} % de las compras")
# ¿se anticipa con las vistas? cuota de vistas en el cubo anterior al inicio
vs = V.div(V.sum(1), axis=0); vb = vs.median(0)
for _, x in E.iterrows():
    i = P.index.get_loc(x.inicio)
    pre = vs[x["subcategoría"]].iloc[i - 1] / (vb[x["subcategoría"]] + 1e-9) if i > 0 else float("nan"); now = vs[x["subcategoría"]].iloc[i] / (vb[x["subcategoría"]] + 1e-9)
    print(f"  {x.inicio:%H:%M} {x['subcategoría'][:22]:22} lift de VISTAS: cubo anterior x{pre:.1f} · primer cubo x{now:.1f}")
