"""¿La demanda de un producto nuevo depende del tiempo desde su lanzamiento? Por tanda de lanzamiento (100 productos cada 4 h) y por edad:
razón = compras (o vistas) por producto nuevo / por producto antiguo medio de su subcategoría, en la misma franja de reloj. No consulta RawTree."""
import os, pickle, numpy as np, calendar, time
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); P = pickle.loads(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb").read())
ids = list(P["ids"]); sub = list(P["sub"]); sidx = {s: k for k, s in enumerate(P["subs"])}; L = [-1.0] * 5000
ms = lambda s: calendar.timegm(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) * 1000
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in set(ids[5000:]): ids.append(r["product_id"]); sub.append(sidx[r["subcategory"]]); L.append(ms(r["ts"]))
index = {p: i for i, p in enumerate(ids)}; n = len(ids); sub = np.array(sub); L = np.array(L); n0 = 5000
T = np.array([e[0] for e in events], float); I = np.array([index.get(e[2], -1) for e in events]); K = np.array([{"view": 0, "add_to_cart": 1, "purchase": 2}.get(e[3], 3) for e in events])
t0, t1 = T.min(), T.max(); H = 3600e3; edges = [0, 1, 2, 4, 8, 12, 16]; batches = np.unique(L[n0:])
for kind, name in ((2, "COMPRAS"), (0, "VISTAS")):
    m = (K == kind) & (I >= 0); Tm, Im = T[m], I[m]; tot = np.zeros((len(batches) + 1, len(edges) - 1, 2))
    print(f"\n{name}: razón nuevo / antiguo medio de la subcategoría (± error)   filas = tanda de lanzamiento (hora desde el inicio), columnas = edad")
    print("            " + "".join(f"{a:>3d}–{b:<2d} h      " for a, b in zip(edges[:-1], edges[1:])))
    for bi, Lg in enumerate(batches):
        g = np.flatnonzero(L == Lg); row = []
        for ei, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            lo, hi = Lg + a * H, Lg + b * H
            if hi > t1 + 1: row.append("      ·       "); continue
            wd = (Tm >= lo) & (Tm < hi); cnt = np.bincount(Im[wd], minlength=n).astype(float); o = e = 0.0
            for s in np.unique(sub[g]): o += cnt[g[sub[g] == s]].sum(); e += (sub[g] == s).sum() * cnt[:n0][sub[:n0] == s].mean()
            tot[bi, ei] = (o, e); tot[-1, ei] += (o, e); row.append(f"{o/e:5.2f} ± {np.sqrt(max(o,1))/e:4.2f}  ")
        print(f"   +{(Lg-t0)/H:4.0f} h   " + "".join(row))
    print("   todas     " + "".join(f"{o/e:5.2f} ± {np.sqrt(max(o,1))/e:4.2f}  " if e > 0 else "      ·       " for o, e in tot[-1]))
# tasa de compra por vista según la edad (nuevos) frente a los antiguos
mv = (K == 0) & (I >= n0); mp = (K == 2) & (I >= n0); av = (T[mv] - L[I[mv]]) / H; ap = (T[mp] - L[I[mp]]) / H
print("\nCompras por cada 100 vistas, productos nuevos según su edad: " + " · ".join(f"{a}–{b} h {100*((ap>=a)&(ap<b)).sum()/max(((av>=a)&(av<b)).sum(),1):.1f}" for a, b in zip(edges[:-1], edges[1:])))
print(f"   productos antiguos: {100*((K==2)&(I>=0)&(I<n0)).sum()/((K==0)&(I>=0)&(I<n0)).sum():.1f}")
