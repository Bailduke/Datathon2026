"""Tres ideas del tercer agente, sobre el modelo de compra actual (con gustos y tendencia):
 (i) la tendencia solo vale si la sesión ya contiene esa subcategoría; (ii) categoría de la sesión; (iii) compras históricas del usuario en la subcategoría."""
import os, sys, pickle, importlib.util, logging, numpy as np, warnings
from collections import Counter
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); cfg = rv.CFG
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); z = np.load(os.path.join(C, "assaig.npz"))
X, PN, AF, gid, tt = z["Xf"], z["PNf"], z["AFf"], z["gid"], z["tt"]
st = rv.State(P, cfg)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
sub, catg, index = st.sub, st.cat, st.index
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
M = len(X); ss = np.zeros(M); sc = np.zeros(M); hs = np.zeros(M); hc = np.zeros(M); m = 0
for k in order:
    r = fb[k]; vs = [index[x] for x in rv.normalize_views(r["recent_views"]) if x in index]
    S, Cs = {int(sub[i]) for i in vs}, {int(catg[i]) for i in vs}
    hb = P["bought"].get(r["user_id"], ()); hsub = Counter(int(sub[i]) for i in hb); hcat = Counter(int(catg[i]) for i in hb)
    for pid in (r["shown"] or [])[:10]:
        i = index.get(pid, -1)
        if i >= 0:
            ss[m] = int(sub[i]) in S; sc[m] = int(catg[i]) in Cs; hs[m] = np.log1p(hsub.get(int(sub[i]), 0)); hc[m] = np.log1p(hcat.get(int(catg[i]), 0))
        m += 1
w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]; off = np.log(w); nou = X[:, 2] > 0; home = X[:, 3] > 0
A = rv.glm_design(X, PN, AF); tr_col = A.shape[1] - 1; trend = A[:, tr_col]
def order_ll(Mx, b, mask):
    e = (Mx @ b + off)[mask]; yy = y[mask]; gg = gid[mask]; o = np.argsort(gg, kind="stable"); e, yy, gg = e[o], yy[o], gg[o]; bnd = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1], True]); tot = 0.0
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum()))
    return tot
cut = np.quantile(tt, 0.7); trn, te = tt < cut, tt >= cut
b0, s0, l0 = rv.poisson_fit(A, y, off); bt0, _, _ = rv.poisson_fit(A[trn], y[trn], off[trn])
def test(title, A2, k):
    b2, s2, l2 = rv.poisson_fit(A2, y, off); bt2, _, _ = rv.poisson_fit(A2[trn], y[trn], off[trn])
    d = {nm: order_ll(A2, bt2, msk) - order_ll(A, bt0, msk) for nm, msk in (("todos", te), ("nuevos", te & nou), ("hist", te & ~nou))}
    print(f"{title}\n   pesos {' · '.join(f'{v:+.2f}±{s:.2f}' for v, s in zip(b2[-k:], s2[-k:]))} · logver {l2-l0:+.1f} · orden fuera de muestra: todos {d['todos']:+.1f} · nuevos {d['nuevos']:+.1f} · con historia {d['hist']:+.1f}")
hot = trend > np.log(2)
print(f"impresiones con tendencia >= x2: {int(hot.sum())} · con la subcategoría en la sesión {int((hot & (ss > 0)).sum())} ({int(y[hot & (ss > 0)].sum())} compras) · en home {int((hot & home).sum())} ({int(y[hot & home].sum())}) · resto {int((hot & (ss == 0) & ~home).sum())} ({int(y[hot & (ss == 0) & ~home].sum())})")
for nm, msk in (("subcategoría en la sesión", hot & (ss > 0)), ("home", hot & home), ("resto", hot & (ss == 0) & ~home)):
    print(f"   tendencia activa, {nm:26s}: conversión por unidad de atención {100*y[msk].sum()/max(w[msk].sum(),1e-9):.2f} % · mismo grupo sin tendencia: {100*y[~hot & ((ss > 0) if nm.startswith('sub') else home if nm == 'home' else (ss == 0) & ~home)].sum()/w[~hot & ((ss > 0) if nm.startswith('sub') else home if nm == 'home' else (ss == 0) & ~home)].sum():.2f} %")
A2 = np.column_stack([np.delete(A, tr_col, axis=1), trend * (ss > 0), trend * home, trend * ((ss == 0) & ~home)])
test("(i) tendencia según la sesión [en la sesión · home · resto] (antes: un solo peso %+.2f)" % b0[tr_col], A2, 3)
test("(ii) + categoría del producto entre las de la sesión", np.column_stack([A, sc]), 1)
test("(ii') + subcategoría del producto entre las de la sesión", np.column_stack([A, ss]), 1)
test("(iii) + log(1 + compras históricas del usuario en la subcategoría)", np.column_stack([A, hs]), 1)
test("(iii') + ídem en la categoría", np.column_stack([A, hc]), 1)
test("(ii)+(iii) categoría de sesión y compras históricas en subcategoría", np.column_stack([A, sc, hs]), 2)
