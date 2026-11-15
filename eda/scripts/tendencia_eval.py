"""¿Cuánto cambia la lista y la conversión prevista con el peso de tendencia, cuarto de hora a cuarto de hora?
Reproduce el stream con el v6 (datos hasta las 20:43 UTC, incluye el episodio de Game Consoles)."""
import os, sys, pickle, importlib.util, logging, math, numpy as np, pandas as pd
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); cfg = rv.CFG; cfg.thompson = False
rv.load_affinity(None, P, cfg)
fb, events, cat = pickle.load(open(os.path.join(HERE, "../lab/.cache/dl.pkl"), "rb"))
R = rv.replay_feedback(P, cfg, fb, events, cat)
X, PN = R["X"], R["PN"]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]
A = rv.glm_design(X, PN, R["AF"]); b, se, ll = rv.poisson_fit(A, y, np.log(w)); NB, NT = rv.N_BASE, rv.N_TASTE
b0, _, ll0 = rv.poisson_fit(A[:, :NB + NT - 1], y, np.log(w))
print(f"{len(fb)} listas, {int(y.sum())} compras · peso de la tendencia {b[NB+NT-1]:+.2f} ± {se[NB+NT-1]:.2f} · logver {ll0:.1f} -> {ll:.1f} (+{ll-ll0:.1f})")
tr = R["AF"][:, NT - 1]; hot = tr > math.log(2)
print(f"impresiones con tendencia >= x2: {int(hot.sum())} ({100*hot.mean():.2f} %) · compras en ellas: {int(y[hot].sum())} · conversión por unidad de atención: {100*y[hot].sum()/w[hot].sum():.2f} % frente a {100*y[~hot].sum()/w[~hot].sum():.2f} % en el resto")
taste = [float(v) for v in b[NB:NB + NT]]
t_on = dict(theta=float(b[3]), g_lc=float(b[4]), g_hc=float(b[5]), g_v0=float(b[6]), g_v1=float(b[7]), taste=taste)
t_off = dict(t_on, taste=taste[:-1] + [0.0])
N = 60000
R2 = rv.replay_feedback(P, cfg, fb, events, cat, tilts=[t_off, t_on], policy_last=N)
pol, ctx = R2["pol"], R2["pol_ctx"]
lin = (b[0] + b[1] * ctx[:, 0] + b[2] * ctx[:, 1])[:, None, None] + b[3] * pol[..., 0] + b[4] * pol[..., 1] + b[5] * pol[..., 2] + b[6] * pol[..., 3] + b[7] * pol[..., 4] + pol[..., 5:] @ b[NB:NB + NT]
pred = (np.asarray(P["w_pos"])[None, None, :] * np.exp(np.clip(lin, -50, 5))).sum(-1)
ts = np.sort(np.array([r["t"] for r in fb]))[-N:]
d = pd.DataFrame({"t": pd.to_datetime(ts, unit="ms"), "sin": pred[:, 0], "con": pred[:, 1],
                  "n_tend_sin": (pol[:, 0, :, 5 + NT - 1] > math.log(2)).sum(1), "n_tend_con": (pol[:, 1, :, 5 + NT - 1] > math.log(2)).sum(1)})
d["b"] = d.t.dt.floor("15min")
g = d.groupby("b").agg(peticiones=("sin", "size"), sin=("sin", "mean"), con=("con", "mean"), prod_tend_sin=("n_tend_sin", "mean"), prod_tend_con=("n_tend_con", "mean"))
g["sin"] *= 100; g["con"] *= 100; g["mejora_%"] = 100 * (g.con / g.sin - 1)
print("\nconversión prevista por cuarto de hora (UTC), sin y con el peso de tendencia; productos 'en tendencia' por lista:")
print(g.round(2).to_string())
print(f"\ntotal: sin {100*d.sin.mean():.3f} % · con {100*d.con.mean():.3f} % · mejora {100*(d.con.mean()/d.sin.mean()-1):.2f} %")
