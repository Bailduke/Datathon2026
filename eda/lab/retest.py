"""Repaso de las candidatas con datos frescos (una descarga): nivel del v2, pesos de recuento (W), usuario nuevo ×
co-compra con el carrito de hoy (H5b) y producto ya mostrado 4+ veces (H8c). Autocontenido: un replay."""
import sys, json, numpy as np
from collections import defaultdict
from lab import *
P = load_P(); fb, events, cat = get_data(refresh="--refresh" in sys.argv, max_age_h=99)
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); rv.AFF = dict(F=z["F"], sg=z["sg"], tags=z["tags"], eig=z["eig"])
baskets = pickle.load(open(os.path.join(HERE, "../scripts/.cache/baskets.pkl"), "rb")); hv = pickle.load(open(os.path.join(C, "hist_views.pkl"), "rb"))
WV = dict(w_counts=(0, 1, .1), kappa_c=5.0, lam_cat=0.5)
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
    if cache.get("S") is not S:
        prm = dict(S.prm); prm.update(WV); prm["taste_on"] = False
        cache["S"], cache["Sv"] = S, rv.Snapshot(**{**{k: getattr(S, k) for k in rv.Snapshot.__slots__}, "prm": prm})
    inf = {"no_taste": True}; rv.answer(cache["Sv"], r, rng, inf); Pn = inf["Pn"]
    return np.array([[Pn[i] if 0 <= i < Pn.size else 0.0] for i in sh], np.float32)
rv.CFG.thompson = False
R = rv.replay_feedback(P, rv.CFG, fb, events, cat, extra_fn=extra, n_extra=1, state_hook=hook); R.pop("state")
X, PN, AF, KI, CT = R["X"], R["PN"], R["AF"], R["KI"], R["CT"].astype(float)
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
mv = np.array([fb[j]["mv"] for j in rid]); v2 = mv == "tengo-aurilla-v2"
G = rv.glm_design(X, PN, AF); A = np.column_stack([G, CT]); NB = rv.N_BASE
print(f"datos hasta {time.strftime('%H:%M', time.localtime(t.max()/1000))} · listas {len(fb)} · compras {int(y.sum())} (nuevos {int(y[nu].sum())} · historia {int(y[~nu].sum())}) · v2: {v2.sum()//10} listas, {int(y[v2].sum())} compras")
pre = t < V2_START; bp, _, _ = fit(A[pre], y[pre], off[pre]); mu = np.exp(np.clip(A @ bp + off, -50, 20))
for nm, m in (("todas", v2), ("nuevos", v2 & nu), ("historia", v2 & ~nu)):
    n_ = m.sum() / 10; print(f"   nivel del v2 ({nm:8s}): real {100*y[m].sum()/n_:.2f} % ± {100*np.sqrt(y[m].sum())/n_:.2f} · previsto con el modelo anterior al v2 {100*mu[m].sum()/n_:.2f} % · obs/esp {y[m].sum()/mu[m].sum():.2f}")
hrs = (t - min(e[0] for e in events[:5])) / 3.6e6; A4 = np.column_stack([A, hrs]); b4, _, _ = fit(A4[pre], y[pre], off[pre]); mu4 = np.exp(np.clip(A4 @ b4 + off, -50, 20))
print("   con la hora como cuarto control: obs/esp del v2 " + " · ".join(f"{nm} {y[m].sum()/mu4[m].sum():.2f}" for nm, m in (("todas", v2), ("nuevos", v2 & nu), ("historia", v2 & ~nu))) + f" · coef hora {fit(A4, y, off)[0][-1]:+.3f}")
for lo, hi in ((0, 1), (1, 2), (2, 99)):
    mm = v2 & (t >= V2_START + lo * 3.6e6) & (t < V2_START + hi * 3.6e6)
    if mm.sum(): print(f"   v2, hora {lo}-{hi} desde su arranque: listas {mm.sum()//10} · real {1000*y[mm].sum()/mm.sum():.2f} % · obs/esp (3 controles) {y[mm].sum()/mu[mm].sum():.2f} · (4 controles) {y[mm].sum()/mu4[mm].sum():.2f}")
# W
Aw = np.column_stack([rv.glm_design(X, R["EX"][:, 0].astype(float), AF), CT])
s = score(Aw, y, off, t, rid, nu) - score(A, y, off, t, rid, nu)
print(f"W  pesos de recuento 0/1/0,1 + kappa_c 5 + lam_cat 0,5: Poisson {s[0]:+.1f} · orden temporal {s[1]:+.1f} · CV {s[2]:+.1f} (nuevos {s[3]:+.1f} · historia {s[4]:+.1f})")
# solo en las listas del v2: ¿cuál explica mejor qué se compró?
bA, _, _ = fit(A[~v2], y[~v2], off[~v2]); bW, _, _ = fit(Aw[~v2], y[~v2], off[~v2])
print(f"   ajustando sin las listas del v2 y evaluando el orden solo en ellas: actual {slate(A[v2], y[v2], off[v2], bA, rid[v2])[0]:.1f} · W {slate(Aw[v2], y[v2], off[v2], bW, rid[v2])[0]:.1f} ({int(y[v2].sum())} compras)")
evaluate2(A, np.column_stack([A, nu * G[:, NB + 2]]), y, off, t, rid, nu, "H5b nuevo × co-compra con carrito de hoy")
bH, _, _ = fit(np.column_stack([A, nu * G[:, NB + 2]])[~v2], y[~v2], off[~v2])
print(f"   idem, orden solo en las listas del v2: actual {slate(A[v2], y[v2], off[v2], bA, rid[v2])[0]:.1f} · H5b {slate(np.column_stack([A, nu * G[:, NB + 2]])[v2], y[v2], off[v2], bH, rid[v2])[0]:.1f}")
order = np.argsort(np.array([f["t"] for f in fb]), kind="stable"); rows_of = defaultdict(list)
for m_, kk in enumerate(rid): rows_of[kk].append(m_)
seen = np.zeros(len(y)); uexp = defaultdict(dict)
for kk in order:
    ex = uexp[fb[kk]["user_id"]]
    for m_ in rows_of.get(kk, ()): seen[m_] = ex.get(KI[m_, 1], 0)
    for m_ in rows_of.get(kk, ()): ex[KI[m_, 1]] = ex.get(KI[m_, 1], 0) + 1
evaluate2(A, np.column_stack([A, seen >= 4]), y, off, t, rid, nu, "H8c ya mostrado 4+ veces a ese usuario")
evaluate2(A, np.column_stack([A, np.log1p(seen)]), y, off, t, rid, nu, "H8c log(1+veces ya mostrado)")
b, _, _ = fit(A, y, off); mu = np.exp(np.clip(A @ b + off, -50, 20))
print("   obs/esp por veces ya mostrado:", " · ".join(f"{a}-{c}: {int(y[(seen>=a)&(seen<=c)].sum())}/{mu[(seen>=a)&(seen<=c)].sum():.1f}" for a, c in ((0, 0), (1, 1), (2, 3), (4, 9), (10, 999))),
      "| solo v2:", " · ".join(f"{a}-{c}: {int(y[v2&(seen>=a)&(seen<=c)].sum())}/{mu[v2&(seen>=a)&(seen<=c)].sum():.1f}" for a, c in ((0, 0), (1, 1), (2, 3), (4, 999))))
print("   v2: fracción de impresiones ya mostradas antes al mismo usuario: %.1f %% (4+ veces: %.1f %%)" % (100 * (seen[v2] > 0).mean(), 100 * (seen[v2] >= 4).mean()))
