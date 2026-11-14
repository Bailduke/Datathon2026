"""Valores fijados a mano en los niveles 1-2 y en la popularidad: rho de la sesión (con lam_cat), pesos de recuento
(compra / carrito / vista) y b0. Una pasada de replay; para cada variante se recalcula la probabilidad jerárquica
de lo mostrado y se reajusta TODO el modelo de compra (clases + gustos + controles de lista)."""
import zlib, json, numpy as np
from lab import *
P = load_P(); rt = None; fb, events, cat = get_data(max_age_h=99)
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); rv.AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
RHO = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0); LAM = (0.5, 1.0, 2.0)
VARS = [dict(rho=r, lam_cat=l) for r in RHO for l in LAM if not (r == 0.65 and l == 1.0)]
VARS += [dict(w_counts=w) for w in ((1, .25, 0.0), (1, .25, .05), (1, .25, .1), (1, .1, .02), (1, .5, .02), (1, 1.0, .02), (.5, .25, .02), (2, .25, .02), (1, .25, .02))]
VARS += [dict(b0=2000.0), dict(b0=200000.0)]
baskets = pickle.load(open(os.path.join(HERE, "../scripts/.cache/baskets.pkl"), "rb")); hv = pickle.load(open(os.path.join(C, "hist_views.pkl"), "rb"))
def hook(st):
    cap, ns = st.users.N.shape; idx = st.index; sub = st.sub
    NKs = [np.zeros((cap, ns), np.float32) for _ in range(3)]
    for u, (ps, es) in baskets.items():
        row = st.users.row.get(u)
        if row is None: continue
        for p_, e in zip(ps, es):
            i = idx.get(p_)
            if i is not None: NKs[0 if e == "purchase" else 1][row, sub[i]] += 1
    for u, (ii, cc) in hv.items():
        row = st.users.row.get(u)
        if row is not None: np.add.at(NKs[2][row], sub[ii.astype(int)], cc.astype(np.float32))
    rec = NKs[0] + 0.25 * NKs[1] + 0.02 * NKs[2]; n0 = len(P["users"])
    d = np.abs(rec[:n0, :P["N"].shape[1]] - P["N"]); print(f"recuentos por tipo reconstruidos: diferencia media con N {d.mean():.4f} · máx {d.max():.2f} · N medio {P['N'].mean():.3f}", flush=True)
    rv.NK = NKs
cache = {}; rng = np.random.default_rng(0)
def extra(st, S, r, info, sh, t):
    out = np.full((len(sh), len(VARS)), np.nan, np.float32)
    if not (r["buy"] or zlib.crc32(r["rid"].encode()) % 5 == 0): return out
    if cache.get("S") is not S:
        base = {k: getattr(S, k) for k in rv.Snapshot.__slots__}; sn = []
        for v in VARS:
            d = dict(base); prm = dict(S.prm); prm.update({k: x for k, x in v.items() if k != "b0"}); prm["taste_on"] = False; d["prm"] = prm
            if "b0" in v:
                old = rv.CFG.b0; rv.CFG.b0 = v["b0"]; sh_, rt_ = st.popularity(); rv.CFG.b0 = old
                d["g_shape"], d["g_rate"] = sh_, rt_; d["expV0"] = (sh_ / rt_) ** S.prm["b_pop"] * S.sale_mult
            sn.append(rv.Snapshot(**d))
        cache["S"], cache["sn"] = S, sn
    ii = [i for i in sh]
    for v, Sv in enumerate(cache["sn"]):
        inf = {"no_taste": True}; rv.answer(Sv, r, rng, inf); Pn = inf["Pn"]
        out[:, v] = [Pn[i] if 0 <= i < Pn.size else 0.0 for i in ii]
    return out
rv.CFG.thompson = False; t0 = time.time()
R = rv.replay_feedback(P, rv.CFG, fb, events, cat, extra_fn=extra, n_extra=len(VARS), state_hook=hook); R.pop("state")
print(f"replay con {len(VARS)} variantes: {time.time()-t0:.0f} s", flush=True)
pickle.dump({"R": {k: R[k] for k in ("X", "PN", "AF", "KI", "EX")}, "VARS": VARS}, open(os.path.join(C, "h_rho.pkl"), "wb"), protocol=4)
