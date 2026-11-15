"""(T1) ¿Aportan los priors por dispositivo (dispositivo × categoría y dispositivo × banda de precio)?
(T2) ¿Cambian los pesos si se estiman dentro de cada lista (logit condicional) en vez de con la Poisson agrupada?"""
import os, sys, pickle, importlib.util, logging, numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache"); sys.path.insert(0, HERE)
from rt import q
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); cfg = rv.CFG
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); z = np.load(os.path.join(C, "assaig.npz"))
X, PN, AF, gid, tt = z["Xf"], z["PNf"], z["AFf"], z["gid"], z["tt"]
st = rv.State(P, cfg)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
n = st.n; price = st.price[:n]; sub = st.sub[:n]; catg = st.cat[:n]
med = pd.Series(price).groupby(sub).transform("median").values; band = np.digitize(price / np.maximum(med, 1e-9), [0.6, 1.0, 1.6])
h = pd.DataFrame(q("""WITH c AS (SELECT product_id::String AS p, any(toString(category)) AS cat, any(toString(subcategory)) AS s, any(price_eur::Float64) AS pr FROM shop.catalog GROUP BY p),
       m AS (SELECT s, median(pr) AS md FROM c GROUP BY s)
       SELECT toString(e.device) AS dev, c.cat AS cat, multiIf(c.pr < 0.6 * m.md, 0, c.pr < m.md, 1, c.pr < 1.6 * m.md, 2, 3) AS band, count() AS n
       FROM shop.history_events AS e INNER JOIN c ON c.p = e.product_id::String INNER JOIN m ON m.s = c.s WHERE e.event_type::String = 'purchase' GROUP BY dev, cat, band""", 300)["data"])
h["n"] = h.n.astype(float)
def lift(col):
    t = h.groupby(["dev", col]).n.sum().unstack().fillna(0); return np.log((t.div(t.sum(1), axis=0)).div(t.sum(0) / t.values.sum(), axis=1))
Lc, Lb = lift("cat"), lift("band")
print("log-lift dispositivo × categoría (extremos):", {d: (Lc.loc[d].idxmin()[:12], round(Lc.loc[d].min(), 2), Lc.loc[d].idxmax()[:12], round(Lc.loc[d].max(), 2)) for d in Lc.index})
print("log-lift dispositivo × banda de precio (barato → caro):", {d: Lb.loc[d].round(2).tolist() for d in Lb.index})
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"]); cats = st.cats
fc = np.zeros(len(X)); fbnd = np.zeros(len(X)); dev = np.empty(len(X), object); m = 0
for k in order:
    r = fb[k]; d = (r.get("device") or "").lower()
    for pid in (r["shown"] or [])[:10]:
        i = st.index.get(pid, -1); dev[m] = d
        if i >= 0 and d in Lc.index:
            cn = cats[catg[i]]
            if cn in Lc.columns: fc[m] = Lc.loc[d, cn]
            fbnd[m] = Lb.loc[d, int(band[i])]
        m += 1
w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]; off = np.log(w); nou = X[:, 2] > 0
A = rv.glm_design(X, PN, AF)
def order_ll(M, b, mask, has_list_cols=True):
    e = (M @ b + off)[mask]; yy = y[mask]; gg = gid[mask]; o = np.argsort(gg, kind="stable"); e, yy, gg = e[o], yy[o], gg[o]; bnd = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1], True]); tot = n_ = hit = 0
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); hit += int(ee.argmax() == k); n_ += 1
    return tot, n_, hit
cut = np.quantile(tt, 0.7); tr, te = tt < cut, tt >= cut
print("\n(T1) PRIORS POR DISPOSITIVO")
b0, s0, l0 = rv.poisson_fit(A, y, off); bt0, _, _ = rv.poisson_fit(A[tr], y[tr], off[tr])
for nm, extra in (("dispositivo × categoría", [fc]), ("dispositivo × banda de precio", [fbnd]), ("ambas", [fc, fbnd])):
    A2 = np.column_stack([A] + extra); b2, s2, l2 = rv.poisson_fit(A2, y, off); bt2, _, _ = rv.poisson_fit(A2[tr], y[tr], off[tr])
    r_ = " · ".join(f"{v:+.2f}±{s:.2f}" for v, s in zip(b2[-len(extra):], s2[-len(extra):]))
    o = {k_: order_ll(A2, bt2, msk)[0] - order_ll(A, bt0, msk)[0] for k_, msk in (("todos", te), ("nuevos", te & nou), ("con historia", te & ~nou))}
    print(f"  + {nm:30s} peso {r_} · logver {l2-l0:+.1f} · orden fuera de muestra: todos {o['todos']:+.1f} · nuevos {o['nuevos']:+.1f} · con historia {o['con historia']:+.1f}")
for nm, msk in (("solo usuarios nuevos", nou), ("solo con historia", ~nou)):
    A2 = np.column_stack([A, fc, fbnd]); keep = [k for k in range(A2.shape[1]) if np.abs(A2[msk][:, k]).sum() > 0]; bb, ss, _ = rv.poisson_fit(A2[msk][:, keep], y[msk], off[msk])
    print(f"  {nm}: dispositivo × categoría {bb[-2]:+.2f}±{ss[-2]:.2f} · dispositivo × precio {bb[-1]:+.2f}±{ss[-1]:.2f}")
print("\n(T2) POISSON AGRUPADA FRENTE A LOGIT CONDICIONAL DENTRO DE LA LISTA")
cols = [k for k in range(A.shape[1]) if k not in (0, 1, 2)]; names = [rv.GLM_COLS[k] for k in cols]
has = pd.Series(y).groupby(gid).transform("sum").values > 0
def cl_fit(mask):
    Mx = A[mask][:, cols]; of = off[mask]; yy = y[mask]; gg = gid[mask]; o = np.argsort(gg, kind="stable"); Mx, of, yy, gg = Mx[o], of[o], yy[o], gg[o]
    _, inv = np.unique(gg, return_inverse=True); G = inv.max() + 1
    def nll(b):
        e = Mx @ b + of; mx = np.full(G, -np.inf); np.maximum.at(mx, inv, e); ex = np.exp(e - mx[inv]); Z = np.bincount(inv, ex, G); p = ex / Z[inv]
        return -float((yy * (e - mx[inv] - np.log(Z[inv]))).sum()) + 0.5e-3 * b @ b, -(Mx.T @ (yy - p)) + 1e-3 * b
    return minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
bc = cl_fit(has); bct = cl_fit(has & tr)
print("  peso              Poisson   condicional")
for k_, nm in enumerate(names):
    if nm in ("sin_P", "ya_comprado"): continue
    print(f"  {nm:16s} {b0[cols[k_]]:+.2f}     {bc[k_]:+.2f}")
bfull = np.zeros(A.shape[1]); bfull[cols] = bct
for nm, msk in (("todos", te), ("nuevos", te & nou), ("con historia", te & ~nou)):
    o1 = order_ll(A, bt0, msk); o2 = order_ll(A, bfull, msk)
    print(f"  orden fuera de muestra, {nm:13s} ({o1[1]} compras): Poisson {o1[0]:.1f} (acierta {100*o1[2]/o1[1]:.0f} %) · condicional {o2[0]:.1f} (acierta {100*o2[2]/o2[1]:.0f} %) · diferencia {o2[0]-o1[0]:+.1f}")
