"""Comparación cruzada de políticas: cada variante (parámetros de los niveles 1-2) se calibra entera, elige sus listas
para las últimas N peticiones, y TODAS las listas se valoran con el modelo de compra de cada variante (juez).
Una variante es mejor si su lista gana también según el juez de la otra. Uso: xeval.py '<json lista de variantes>' [N]"""
import sys, json, numpy as np
from lab import *
VARS = [dict()] + json.loads(sys.argv[1]); LAST = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
P = load_P(); fb, events, cat = get_data(max_age_h=99)
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); rv.AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
baskets = pickle.load(open(os.path.join(HERE, "../scripts/.cache/baskets.pkl"), "rb")); hv = pickle.load(open(os.path.join(C, "hist_views.pkl"), "rb"))
def hook(st):
    cap, ns = st.users.N.shape; NKs = [np.zeros((cap, ns), np.float32) for _ in range(3)]
    for u, (ps, es) in baskets.items():
        row = st.users.row.get(u)
        if row is None: continue
        for p_, e in zip(ps, es):
            i = st.index.get(p_)
            if i is not None: NKs[0 if e == "purchase" else 1][row, st.sub[i]] += 1
    for u, (ii, cc) in hv.items():
        row = st.users.row.get(u)
        if row is not None: np.add.at(NKs[2][row], st.sub[ii.astype(int)], cc.astype(np.float32))
    rv.NK = NKs
cache = {}; rng = np.random.default_rng(0)
def extra(st, S, r, info, sh, t):
    out = np.zeros((len(sh), len(VARS) - 1), np.float32)
    if cache.get("S") is not S:
        base = {k: getattr(S, k) for k in rv.Snapshot.__slots__}; sn = []
        for v in VARS[1:]:
            prm = dict(S.prm); prm.update(v); prm["taste_on"] = False; sn.append(rv.Snapshot(**{**base, "prm": prm}))
        cache["S"], cache["sn"] = S, sn
    for v, Sv in enumerate(cache["sn"]):
        inf = {"no_taste": True}; rv.answer(Sv, r, rng, inf); Pn = inf["Pn"]; out[:, v] = [Pn[i] if 0 <= i < Pn.size else 0.0 for i in sh]
    return out
rv.CFG.thompson = False; t0 = time.time()
R = rv.replay_feedback(P, rv.CFG, fb, events, cat, extra_fn=extra, n_extra=len(VARS) - 1, state_hook=hook); R.pop("state")
X, AF, CT = R["X"], R["AF"], R["CT"].astype(float); y = X[:, 1]; wp = np.asarray(P["w_pos"]); off = np.log(wp[X[:, 0].astype(int)])
B, tilts = [], []; NB = rv.N_BASE
for v, var in enumerate(VARS):
    pn = R["PN"] if v == 0 else R["EX"][:, v - 1].astype(float)
    A = np.column_stack([rv.glm_design(X, pn, AF), CT]); b, se, l = fit(A, y, off); B.append(b)
    tl = dict(theta=b[3], g_lc=b[4], d_lc=-b[9], g_hc=b[5], g_v0=b[6], g_v1=b[7], taste=[float(min(max(x, -0.5), 4.0)) for x in b[NB:NB + 8]])
    tilts.append({**var, **tl}); print(f"variante {v} {json.dumps(var)}: logver {l:.1f} · theta {b[3]:.2f}", flush=True)
R2 = rv.replay_feedback(P, rv.CFG, fb, events, cat, tilts=tilts, policy_last=LAST, state_hook=hook); R2.pop("state")
pol, polx, ctx, ctl = R2["pol"].astype(float), R2["polx"].astype(float), R2["pol_ctx"], R2["pol_ctl"]
nu = ctx[:, 0] > 0
def value(v_list, v_judge):
    b = B[v_judge]; f = pol[:, v_list]; lp = polx[:, v_list, v_judge]
    lin = (b[0] + b[1] * ctx[:, 0] + b[2] * ctx[:, 1] + ctl @ b[NB + 8:NB + 11])[:, None] + b[3] * lp + b[4] * f[..., 1] + b[5] * f[..., 2] \
        + b[6] * f[..., 3] + b[7] * f[..., 4] + b[9] * f[..., 1] * lp + f[..., 5:] @ b[NB:NB + 8]
    return (wp[None, :] * np.exp(np.clip(lin, -50, 5))).sum(1)
print(f"\núltimas {len(ctx)} peticiones · conversión real con lo que se mostró {100*ctx[:,2].mean():.2f} % · {time.time()-t0:.0f} s")
for j in range(len(VARS)):
    print(f"según el juez {j}:")
    for l_ in range(len(VARS)):
        v = value(l_, j); print(f"   lista de la variante {l_}: {100*v.mean():.3f} % (nuevos {100*v[nu].mean():.3f} · historia {100*v[~nu].mean():.3f})")
ov = [np.mean([len(set()) for _ in ()]) if False else 0]
