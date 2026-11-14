"""Variante 'recuentos': N = 0·compras + 1·carrito + 0,1·vistas, kappa_c = 5, lam_cat = 0,5. Crea eda/lab/.cache_counts/
con su propio priors_live.pkl y lanza la calibración en seco para comparar la previsión con la variante actual."""
import shutil, sys, numpy as np
from pathlib import Path
from lab import *
W = (0.0, 1.0, 0.1); D = Path(HERE) / ".cache_counts"; D.mkdir(exist_ok=True)
P = load_P(); index = {p: i for i, p in enumerate(P["ids"])}; sub = np.asarray(P["sub"]); urow = {u: k for k, u in enumerate(P["users"])}
baskets = pickle.load(open(os.path.join(HERE, "../scripts/.cache/baskets.pkl"), "rb")); hv = pickle.load(open(os.path.join(C, "hist_views.pkl"), "rb"))
N = np.zeros(P["N"].shape, np.float64)
for u, (ps, es) in baskets.items():
    r = urow.get(u)
    if r is None: continue
    for p_, e in zip(ps, es):
        i = index.get(p_)
        if i is not None: N[r, sub[i]] += W[0] if e == "purchase" else W[1]
for u, (ii, cc) in hv.items():
    r = urow.get(u)
    if r is not None: np.add.at(N[r], sub[ii.astype(int)], W[2] * cc.astype(float))
print("recuento medio por usuario: antes %.1f · ahora %.1f" % (P["N"].sum(1).mean(), N.sum(1).mean()))
P["N"] = N.astype(P["N"].dtype); P["params"]["kappa_c"] = 5.0; P["params"]["lam_cat"] = 0.5
(D / "priors_live.pkl").write_bytes(pickle.dumps(P, protocol=pickle.HIGHEST_PROTOCOL))
shutil.copy(os.path.join(C, "afinidad.npz"), D / "afinidad.npz")
if not (D / "dl.pkl").exists(): os.symlink(os.path.join(C, "dl.pkl"), D / "dl.pkl")
rv.CACHE = D; rv.CFG.w_purchase, rv.CFG.w_cart, rv.CFG.w_view = W
rv.setup_logging(False); rv.calibrate_live(rv.make_rt(), rv.CFG, args_dry="--save" not in sys.argv)
