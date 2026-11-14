"""Una pasada cronológica por los eventos en vivo; en cada lista mostrada calcula la probabilidad jerárquica
de cada producto con VARIAS configuraciones de parámetros, y ajusta para cada una la regresión de Poisson
de la compra (posición como offset). Sirve para elegir cómo entra la sesión en P(c)."""
import os, sys, pickle, time, importlib.util, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv4", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v4.py"))
rv4 = importlib.util.module_from_spec(spec); sys.modules["rv4"] = rv4; spec.loader.exec_module(rv4)
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb")); ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
P = pickle.loads((rv4.CACHE / "priors_live.pkl").read_bytes())
cfg = rv4.CFG; cfg.thompson = False
VARS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
    dict(ses_mix=0, lam_cat=1.0), dict(ses_mix=0, lam_cat=3.0), dict(ses_mix=0, lam_cat=10.0),
    dict(ses_mix=1, lam_cat=0.1), dict(ses_mix=1, lam_cat=0.25), dict(ses_mix=1, lam_cat=0.5), dict(ses_mix=1, lam_cat=1.0),
    dict(ses_mix=1, lam_cat=2.0), dict(ses_mix=1, lam_cat=4.0)]
st = rv4.State(P, cfg)
rt = rv4.make_rt()
rows = rt.query(rv4.SQL_CATALOG_UPDATES.format(since="1970-01-01 00:00:00.000", limit=200000), 60)
for r in rows:
    if r["change"] == "new_product":
        st.apply_update(r, time.time())
S0 = st.publish(set_global=False)
snaps = []
for v in VARS:
    d = {k: getattr(S0, k) for k in S0.__slots__}; d["prm"] = rv4.State.make_params(P["params"], cfg, **v); d["bought"] = {}
    snaps.append(rv4.Snapshot(**d))
index = S0.index
t0 = time.time(); stream = []
for u, (ps, es, ts) in ev.items():
    for p, e, t in zip(ps, es, ts):
        i = index.get(p)
        if i is not None: stream.append((int(t), 0, u, i, e))
for r in fb:
    if r["buy"] and r["buy"] in index: stream.append((int(r["t"]) + 5000, 0, r["u"], index[r["buy"]], "fb"))
for k, r in enumerate(fb): stream.append((int(r["t"]), 1, k, 0, ""))
stream.sort(key=lambda x: (x[0], x[1]))
rng = np.random.default_rng(0)
M = len(fb) * 10
X = np.zeros((M, 9)); PN = np.full((M, len(VARS)), np.nan); m = 0     # slot y nuevo home LC HC V0 V1 B
for (t, typ, a, i, e) in stream:
    if typ == 0:
        st.user_event(a, i, "purchase" if e == "fb" else e, organic=(e != "fb")); continue
    r = fb[a]; u = r["u"]
    req = {"user_id": u, "device": r["dev"], "region": "", "recent_views": r["rv"]}
    shown = [index.get(p, -1) for p in (r["shown"] or [])[:10]]
    lcs = st.carted_live.get(u) or (); hcs = st.carted.get(u) or (); bs = st.bought.get(u) or ()
    vs = [index[p] for p in rv4.normalize_views(r["rv"]) if p in index]
    for j, i in enumerate(shown):
        b = float(i in bs)
        X[m + j] = (j, float((r["shown"] or [])[j] == r["buy"]), float(r["nu"] == "true"), float(r["page"] == "home"),
                    float(i in lcs) * (1 - b), float(i in hcs and i not in lcs) * (1 - b),
                    float(bool(vs) and i == vs[-1]) * (1 - b), float(i in vs[:-1]) * (1 - b), b)
    for k, S in enumerate(snaps):
        info = {}; rv4.answer(S, req, rng, info)
        PN[m:m + len(shown), k] = [info["Pn"][i] if i >= 0 else np.nan for i in shown]
    m += 10
print(f"pasada hecha en {time.time()-t0:.0f} s · {m} impresiones · {int(X[:,1].sum())} compras", flush=True)
np.savez_compressed(os.path.join(C, "grid.npz"), X=X, PN=PN, V=json.dumps(VARS))
w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]

def fit(pn, extra=()):
    ok = np.isfinite(pn) & (pn > 0)
    lp = np.where(ok, np.log(np.maximum(pn, 1e-12)), np.log(1e-3)) - np.log(1e-3)
    cols = [np.ones(m), X[:, 2], X[:, 3], lp, X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 4] * lp, (~ok).astype(float)] + list(extra)
    A = np.column_stack(cols); b = np.zeros(A.shape[1]); b[0] = np.log(y.sum() / w.sum()); off = np.log(w)
    for _ in range(300):
        mu = np.exp(np.clip(A @ b + off, -50, 20)); g = A.T @ (y - mu) - 1e-4 * b
        H = (A * mu[:, None]).T @ A + 1e-4 * np.eye(len(b)); s = np.linalg.solve(H, g); mx = np.abs(s).max()
        b = b + (s / mx if mx > 1 else s)
        if mx < 1e-8: break
    mu = np.exp(np.clip(A @ b + off, -50, 20))
    return float((y * np.log(mu) - mu).sum()), b, np.sqrt(np.diag(np.linalg.inv(H)))

print("coeficientes: const nuevo home theta | LC HC V0 V1 B | LCxlp sinP")
for k, v in enumerate(VARS):
    ll, b, se = fit(PN[:, k])
    print(f"{json.dumps(v):44s} logver {ll:9.1f} · " + " ".join(f"{x:+.2f}" for x in b))
