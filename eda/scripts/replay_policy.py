"""Estimación offline (basada en modelo) de la conversión de la política nueva: para cada petición ya servida,
qué lista daría hoy answer() con el estado de ese instante, y cuánta compra predice para ella la regresión de
Poisson ajustada con lo realmente observado. Es una extrapolación: orienta, no sustituye a la prueba en vivo."""
import os, sys, pickle, time, importlib.util, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv4", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v4.py"))
rv4 = importlib.util.module_from_spec(spec); sys.modules["rv4"] = rv4; spec.loader.exec_module(rv4)
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb")); ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
P = pickle.loads((rv4.CACHE / "priors_live.pkl").read_bytes())
cfg = rv4.CFG; cfg.thompson = False
G = np.load(os.path.join(C, "grid.npz")); X, PN0 = G["X"], G["PN"][:, 0]
w_pos = np.asarray(P["w_pos"]); m = len(X); w = w_pos[X[:, 0].astype(int)]; y = X[:, 1]
ok = np.isfinite(PN0) & (PN0 > 0); lp = np.where(ok, np.log(np.maximum(PN0, 1e-12)), np.log(1e-3)) - np.log(1e-3)
A = np.column_stack([np.ones(m), X[:, 2], X[:, 3], lp, X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 4] * lp])
b = np.zeros(A.shape[1]); b[0] = np.log(y.sum() / w.sum())
for _ in range(300):
    mu = np.exp(np.clip(A @ b + np.log(w), -50, 20)); g = A.T @ (y - mu) - 1e-4 * b
    H = (A * mu[:, None]).T @ A + 1e-4 * np.eye(len(b)); s = np.linalg.solve(H, g); mx = np.abs(s).max(); b = b + (s / mx if mx > 1 else s)
    if mx < 1e-8: break
print("coef (const nuevo home theta LC HC V0 V1 B LCxlp):", " ".join(f"{x:+.2f}" for x in b))
VARS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [dict()]
st = rv4.State(P, cfg); rt = rv4.make_rt()
cat_rows = rt.query(rv4.SQL_CATALOG_UPDATES.format(since="1970-01-01 00:00:00.000", limit=200000), 60)
for r in cat_rows:                                        # los productos nuevos existen desde el principio (sin stock hasta su alta)
    if r["change"] == "new_product":
        st.apply_update(r, rv4.parse_ts(r["ts"])); st.unavailable[st.index[r["product_id"]]] = True
snaps = []
def rebuild():
    S0 = st.publish(set_global=False)
    snaps.clear()
    for v in VARS:
        d = {k: getattr(S0, k) for k in S0.__slots__}; d["prm"] = rv4.State.make_params(P["params"], cfg, **v); snaps.append(rv4.Snapshot(**d))
    return S0
S0 = rebuild()
index = S0.index
stream = []
for k, r in enumerate(cat_rows): stream.append((int(rv4.parse_ts(r["ts"]) * 1000), -1, k, 0, ""))
dirty = False
for u, (ps, es, ts) in ev.items():
    for p, e, t in zip(ps, es, ts):
        i = index.get(p)
        if i is not None: stream.append((int(t), 0, u, i, e))
for r in fb:
    if r["buy"] and r["buy"] in index: stream.append((int(r["t"]) + 5000, 0, r["u"], index[r["buy"]], "fb"))
for k, r in enumerate(fb): stream.append((int(r["t"]), 1, k, 0, ""))
stream.sort(key=lambda x: (x[0], x[1]))
rng = np.random.default_rng(0)
pred = np.zeros((len(fb), len(VARS))); comp = np.zeros((len(fb), len(VARS), 4)); top3 = np.zeros((len(fb), len(VARS), 4))
for (t, typ, a, i, e) in stream:
    if typ == -1:
        r = cat_rows[a]
        if r["change"] == "new_product": st.unavailable[st.index[r["product_id"]]] = False
        else: st.apply_update(r, t / 1000)
        dirty = True; continue
    if typ == 0:
        st.user_event(a, i, "purchase" if e == "fb" else e, organic=(e != "fb")); continue
    if dirty:
        rebuild(); dirty = False
    r = fb[a]
    req = {"user_id": r["u"], "device": r["dev"], "region": "", "recent_views": r["rv"]}
    nuevo, home = float(r["nu"] == "true"), float(r["page"] == "home")
    for k, S in enumerate(snaps):
        info = {}; rv4.answer(S, req, rng, info)
        ch = info["chosen"][:10]; Pn = info["Pn"]; lc, hc, vs = set(info["lc"]), set(info["hc"]), info["views"]
        tot = 0.0
        for j, i in enumerate(ch):
            l = np.log(max(Pn[i], 1e-12)) - np.log(1e-3)
            LC, HC = float(i in lc), float(i in hc); V0 = float(bool(vs) and i == vs[-1]); V1 = float(i in vs[:-1])
            tot += w_pos[j] * np.exp(b[0] + b[1] * nuevo + b[2] * home + b[3] * l + b[4] * LC + b[5] * HC + b[6] * V0 + b[7] * V1 + b[9] * LC * l)
            comp[a, k] += (LC, HC, V0 + V1, 1 - max(LC, HC, min(V0 + V1, 1)))
            if j < 3: top3[a, k] += (LC, HC, V0 + V1, 1 - max(LC, HC, min(V0 + V1, 1)))
        pred[a, k] = tot
mu_log = mu.reshape(-1, 10).sum(1); y_log = y.reshape(-1, 10).sum(1)
nu = np.array([r["nu"] == "true" for r in fb]); mv = np.array([r["mv"] for r in fb]); sb = np.array([r["sb"] for r in fb])
def line(name, mask):
    s = f"{name:34s} n={int(mask.sum()):6d} · real {100*y_log[mask].mean():.2f} % · modelo sobre lo mostrado {100*mu_log[mask].mean():.2f} %"
    for k, v in enumerate(VARS):
        s += f" · política {k}: {100*pred[mask, k].mean():.2f} %"
    print(s)
line("todas las peticiones", np.ones(len(fb), bool)); line("usuarios nuevos", nu); line("usuarios con historia", ~nu)
line("servidas por el baseline", sb == "fallback"); line("servidas por tengo-aurilla-v0", mv == "tengo-aurilla-v0")
for k, v in enumerate(VARS):
    print(f"política {k} {json.dumps(v)}: por lista -> carrito hoy {comp[:, k, 0].mean():.2f} · carrito antiguo {comp[:, k, 1].mean():.2f} · vistos {comp[:, k, 2].mean():.2f} · resto {comp[:, k, 3].mean():.2f}"
          f" | slots 1-3 -> {top3[:, k, 0].mean():.2f} / {top3[:, k, 1].mean():.2f} / {top3[:, k, 2].mean():.2f} / {top3[:, k, 3].mean():.2f}")
