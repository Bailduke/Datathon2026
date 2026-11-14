"""PRECIO · ¿Se puede reducir la pendiente de precio del nivel 3 a un solo coeficiente (beta_u · r_u)?
Hoy: pos_i · (b_p0 + b_u·r_u + b_dev[dispositivo] + b_reg[región]). Un replay con la región y el dispositivo reales."""
import json, numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); rv.AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
VARS = [("solo beta_u·r_u (propuesta)", dict(b_p0=0.0, b_dev={}, b_reg={})), ("beta_0 + beta_u·r_u (sin dispositivo ni región)", dict(b_dev={}, b_reg={})),
        ("sin región", dict(b_reg={})), ("sin dispositivo", dict(b_dev={})), ("sin beta_0", dict(b_p0=0.0)), ("sin precio (tampoco r_u)", dict(b_p0=0.0, b_u=0.0, b_dev={}, b_reg={})),
        ("sin r_u (resto igual)", dict(b_u=0.0))]
cache = {}; rng = np.random.default_rng(0)
def extra(st, S, r, info, sh, t):
    out = np.zeros((len(sh), len(VARS)), np.float32)
    if cache.get("S") is not S:
        base = {k: getattr(S, k) for k in rv.Snapshot.__slots__}; sn = []
        for _, v in VARS:
            prm = dict(S.prm); prm.update(v); prm["taste_on"] = False; sn.append(rv.Snapshot(**{**base, "prm": prm}))
        cache["S"], cache["sn"] = S, sn
    for v, Sv in enumerate(cache["sn"]):
        inf = {"no_taste": True}; rv.answer(Sv, r, rng, inf); Pn = inf["Pn"]; out[:, v] = [Pn[i] if 0 <= i < Pn.size else 0.0 for i in sh]
    return out
rv.CFG.thompson = False
R = rv.replay_feedback(P, rv.CFG, fb, events, cat, extra_fn=extra, n_extra=len(VARS)); R.pop("state")
X, PN, AF, KI, CT = R["X"], R["PN"], R["AF"], R["KI"], R["CT"].astype(float)
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
mv = np.array([fb[j]["mv"] for j in rid]); v2 = mv == "tengo-aurilla-v2"
des = lambda pn: np.column_stack([rv.glm_design(X, pn, AF), CT]); A = des(PN); s0 = score(A, y, off, t, rid, nu)
b0, _, _ = fit(A[~v2], y[~v2], off[~v2]); v0 = slate(A[v2], y[v2], off[v2], b0, rid[v2])[0]
print(f"datos hasta {time.strftime('%H:%M', time.localtime(t.max()/1000))} · {int(y.sum())} compras (v2: {int(y[v2].sum())}) · regiones presentes en las peticiones: {len({f['region'] for f in fb})} · dispositivos: {sorted({f['device'] for f in fb})}")
print(f"actual: Poisson {s0[0]:.1f} · orden temporal {s0[1]:.1f} · orden CV {s0[2]:.1f} · orden en listas v2 {v0:.1f}")
print(f"{'variante':50s} {'Poisson':>8} {'ord.temp':>9} {'ord.CV':>8} {'nuevos':>8} {'historia':>9} {'solo v2':>8}")
for v, (nm, _) in enumerate(VARS):
    Av = des(R["EX"][:, v].astype(float)); s = score(Av, y, off, t, rid, nu) - s0; bv, _, _ = fit(Av[~v2], y[~v2], off[~v2])
    print(f"{nm:50s} {s[0]:+8.1f} {s[1]:+9.1f} {s[2]:+8.1f} {s[3]:+8.1f} {s[4]:+9.1f} {slate(Av[v2], y[v2], off[v2], bv, rid[v2])[0]-v0:+8.1f}", flush=True)
pickle.dump({k: R[k] for k in ("X", "PN", "AF", "KI", "CT", "EX")}, open(os.path.join(C, "price.pkl"), "wb"), protocol=4)
