"""¿El producto visto rinde de forma distinta en la posición 1 que en las demás? Interacción clase de visto × posición 1."""
import os, sys, pickle, importlib.util, logging, numpy as np, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes())
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); z = np.load(os.path.join(C, "assaig.npz"))
X, PN, AF, gid, tt = z["Xf"], z["PNf"], z["AFf"], z["gid"], z["tt"]
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
K = np.zeros((len(X), 3)); m = 0                                  # única, primero de 2+, otros vistos
for k in order:
    r = fb[k]; vs = rv.normalize_views(r["recent_views"]); n = len(vs)
    for pid in (r["shown"] or [])[:10]:
        if pid in vs: K[m, 0 if n == 1 else 1 if vs.index(pid) == 0 else 2] = 1
        m += 1
K = K * (1 - X[:, 8])[:, None]
slot = X[:, 0].astype(int); w = np.asarray(P["w_pos"])[slot]; y = X[:, 1]; off = np.log(w); nou = X[:, 2] > 0; s1 = (slot == 0).astype(float)
A = rv.glm_design(X, PN, AF); A0 = np.column_stack([np.delete(A, [6, 7], axis=1), K])
A1 = np.column_stack([A0, K * s1[:, None], s1 * (K.sum(1) == 0)])             # + interacciones con posición 1 y posición 1 para no vistos
names = ["vista única", "primero de 2+", "otros vistos"]
def order_ll(M, b, mask):
    e = (M @ b + off)[mask]; yy = y[mask]; gg = gid[mask]; o = np.argsort(gg, kind="stable"); e, yy, gg = e[o], yy[o], gg[o]; bnd = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1], True]); tot = n = 0
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); n += 1
    return tot, n
print("conversión por impresión (%), por clase y posición:")
for c, nm in enumerate(names + ["no visto"]):
    msk = (K[:, c] > 0) if c < 3 else (K.sum(1) == 0) & (X[:, 8] == 0)
    print(f"  {nm:14s} " + " · ".join(f"pos {lab}: {100*y[msk & sel].sum()/max((msk & sel).sum(),1):.2f} ({int(y[msk & sel].sum())}/{int((msk & sel).sum())})" for lab, sel in (("1", slot == 0), ("2–3", (slot >= 1) & (slot <= 2)), ("4–10", slot >= 3))))
b0, s0, l0 = rv.poisson_fit(A0, y, off); b1, s1_, l1 = rv.poisson_fit(A1, y, off)
print(f"\nsin interacción: logver {l0:.1f} · con interacción: {l1:.1f} (mejora {l1-l0:+.1f}, 4 parámetros)")
print("  efecto base (posiciones 2–10): " + " · ".join(f"{nm} {v:+.2f}±{s:.2f}" for nm, v, s in zip(names, b1[-7:-4], s1_[-7:-4])))
print("  extra en posición 1:           " + " · ".join(f"{nm} {v:+.2f}±{s:.2f}" for nm, v, s in zip(names, b1[-4:-1], s1_[-4:-1])) + f" · no vistos {b1[-1]:+.2f}±{s1_[-1]:.2f}")
cut = np.quantile(tt, 0.7); tr, te = tt < cut, tt >= cut
bt0, _, _ = rv.poisson_fit(A0[tr], y[tr], off[tr]); bt1, _, _ = rv.poisson_fit(A1[tr], y[tr], off[tr])
for nm, msk in (("todos", te), ("nuevos", te & nou), ("con historia", te & ~nou)):
    o0 = order_ll(A0, bt0, msk); o1 = order_ll(A1, bt1, msk); print(f"  orden fuera de muestra, {nm:13s} ({o0[1]} compras): sin {o0[0]:.1f} · con {o1[0]:.1f} · diferencia {o1[0]-o0[0]:+.1f}")
# tasa prevista en posición 1 de un producto de 'vista única' frente al que el modelo puso en posición 1 en esas mismas listas
eta = A1 @ b1
single_any = np.zeros(len(X), bool)
