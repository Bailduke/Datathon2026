"""H13 · ¿Qué compras se nos escapan? Para cada compra del carrusel (en listas de cualquier política), posición que
ocupaba el producto comprado en NUESTRO orden (v2) entre todos los elegibles en ese instante."""
import numpy as np
from lab import *
P = load_P(); fb, events, cat = pickle.load(open(os.path.join(C, "dl_2040.pkl"), "rb"))
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); rv.AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
rows = []
def extra(st, S, r, info, sh, t):
    out = np.zeros((len(sh), 1), np.float32)
    if r["buy"]:
        i = S.index.get(r["buy"], -1)
        if 0 <= i < S.n:
            Pf, Pn, ts = info["P"], info["Pn"], info["taste"]
            rk = int((Pf > Pf[i]).sum()) if Pf[i] > 0 else 99999; rkn = int((Pn > Pn[i]).sum()) if Pn[i] > 0 else 99999
            vs = info["views"]; cls = "carrito hoy" if i in info["lc"] else "carrito antiguo" if i in info["hc"] else "visto" if i in vs else "sin clase"
            top10 = np.sort(Pf)[-10]
            rows.append((r["nu"] == "true", r["mv"], rk, rkn, cls, float(Pf[i] / max(top10, 1e-300)), float(st.cfg and (S.prm["taste"] @ ts[:, i])), r["page"],
                         int(sh.index(i)) if i in sh else -1, bool(S.is_new[i])))
    return out
rv.CFG.thompson = False
R = rv.replay_feedback(P, rv.CFG, fb, events, cat, extra_fn=extra, n_extra=1); R.pop("state")
import pandas as pd
d = pd.DataFrame(rows, columns=["nuevo", "mv", "rk", "rkn", "clase", "ratio10", "gusto", "page", "slot", "prod_nuevo"])
d.to_pickle(os.path.join(C, "h13.pkl"))
d["grupo"] = np.where(d.mv == "tengo-aurilla-v2", "listas v2", np.where(d.mv == "tengo-aurilla-v0", "listas v0", "otras listas"))
print("compras:", len(d))
def tab(g):
    return pd.Series({"n": len(g), "top10 %": 100 * (g.rk < 10).mean(), "top30 %": 100 * (g.rk < 30).mean(), "top100 %": 100 * (g.rk < 100).mean(), "mediana": g.rk.median(),
                      "top10 solo jerárquica %": 100 * (g.rkn < 10).mean()})
print("posición del producto comprado en el orden del v2 (0 = primero):"); print(d.groupby(["grupo", "nuevo"]).apply(tab).round(1).to_string())
o = d[(d.grupo == "otras listas")]
print("\ncompras en listas ajenas al v0/v2 que el v2 NO habría puesto en su top-10, por clase y página:")
miss = o[o.rk >= 10]; print(miss.groupby(["nuevo", "clase"]).size().unstack(fill_value=0).to_string()); print(miss.groupby(["nuevo", "page"]).size().unstack(fill_value=0).to_string())
print("   de ellas, producto nuevo de catálogo: %d · mediana de su posición: %.0f · mediana del cociente score/score del 10º: %.2f · con gusto<=0: %.0f %%" % (miss.prod_nuevo.sum(), miss.rk.median(), miss.ratio10.median(), 100 * (miss.gusto <= 0).mean()))
print("   posición en la lista mostrada (slot) de esas compras:", miss.slot.value_counts().sort_index().to_dict())
