"""¿Pesa más el PRIMER producto visto de la sesión que el último? Se sustituyen las clases «último visto / visto antes»
por «vista única / primero de 2+ / intermedio / último de 2+» en el modelo de compra, con el resto igual."""
import os, sys, pickle, importlib.util, logging, numpy as np, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); cfg = rv.CFG
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); z = np.load(os.path.join(C, "assaig.npz"))
X, PN, AF, gid, tt = z["Xf"], z["PNf"], z["AFf"], z["gid"], z["tt"]
st = rv.State(P, cfg)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
K = np.zeros((len(X), 4)); nv = np.zeros(len(X)); m = 0          # única, primero, intermedio, último
for k in order:
    r = fb[k]; vs = rv.normalize_views(r["recent_views"]); n = len(vs)
    for pid in (r["shown"] or [])[:10]:
        nv[m] = n
        if pid in vs:
            j = vs.index(pid); K[m, 0 if n == 1 else 1 if j == 0 else 3 if j == n - 1 else 2] = 1
        m += 1
assert m == len(X)
notb = 1 - X[:, 8]; K = K * notb[:, None]
w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]; off = np.log(w); nou = X[:, 2] > 0
A = rv.glm_design(X, PN, AF)                                    # columnas 6 y 7 = último visto / visto antes
A2 = np.column_stack([np.delete(A, [6, 7], axis=1), K])
names = ["vista única", "primero de 2+", "intermedio", "último de 2+"]
print("impresiones y compras por clase (todas las listas):")
for c, nm in enumerate(names):
    msk = K[:, c] > 0; print(f"  {nm:14s} {int(msk.sum()):6d} impresiones · {int(y[msk].sum()):3d} compras · por unidad de atención {100*y[msk].sum()/w[msk].sum():.2f} % | nuevos {int(y[msk & nou].sum())}/{int((msk & nou).sum())} · con historia {int(y[msk & ~nou].sum())}/{int((msk & ~nou).sum())}")
def fit(M, mask=None):
    Mm, ym, om = (M, y, off) if mask is None else (M[mask], y[mask], off[mask]); return rv.poisson_fit(Mm, ym, om)
def order_ll(M, b, mask):
    e = (M @ b + off)[mask]; yy = y[mask]; gg = gid[mask]; o = np.argsort(gg, kind="stable"); e, yy, gg = e[o], yy[o], gg[o]; bnd = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1], True]); tot = n = 0
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); n += 1
    return tot, n
b1, s1, l1 = fit(A); b2, s2, l2 = fit(A2)
print(f"\nmodelo actual (último visto {b1[6]:+.2f}±{s1[6]:.2f}, visto antes {b1[7]:+.2f}±{s1[7]:.2f}) logver {l1:.1f}")
print(f"con cuatro clases: logver {l2:.1f} (mejora {l2-l1:+.1f}) · " + " · ".join(f"{nm} {v:+.2f}±{s:.2f}" for nm, v, s in zip(names, b2[-4:], s2[-4:])))
cut = np.quantile(tt, 0.7); tr, te = tt < cut, tt >= cut
bt1, _, _ = fit(A, tr); bt2, _, _ = fit(A2, tr)
for nm, msk in (("todos", te), ("usuarios nuevos", te & nou), ("con historia", te & ~nou)):
    o1 = order_ll(A, bt1, msk); o2 = order_ll(A2, bt2, msk); print(f"  orden fuera de muestra, {nm:16s} ({o1[1]} compras): actual {o1[0]:.1f} · cuatro clases {o2[0]:.1f} · diferencia {o2[0]-o1[0]:+.1f}")
for nm, msk in (("SOLO USUARIOS NUEVOS", nou), ("SOLO CON HISTORIA", ~nou)):
    keep = [k for k in range(A2.shape[1]) if np.abs(A2[msk][:, k]).sum() > 0]; bb, ss, _ = fit(A2[:, keep], msk)
    print(f"{nm}: " + " · ".join(f"{n_} {bb[keep.index(A2.shape[1]-4+c)]:+.2f}±{ss[keep.index(A2.shape[1]-4+c)]:.2f}" for c, n_ in enumerate(names)))
# ¿en qué posición los ponía el modelo? (solo para ver cuánto margen hay)
for c, nm in enumerate(names):
    msk = (K[:, c] > 0); print(f"  {nm:14s} posición media a la que se mostró: {1 + X[msk, 0].mean():.1f} · mostrado en posición 1: {100*(X[msk, 0] == 0).mean():.0f} %")
