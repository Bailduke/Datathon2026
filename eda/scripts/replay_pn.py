"""Reproduce en orden cronológico los eventos en vivo y, para cada lista mostrada, calcula con el código
de producción la probabilidad jerárquica pura P(c)·P(s|c)·P(i|s) de cada producto (sin los términos de
visto/carrito y sin excluir lo ya comprado), con el estado del usuario en el instante de la petición.
Salida: .cache/imp_pn.pkl  (rid -> vector de 10)"""
import os, sys, pickle, time, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv4", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v4.py"))
rv4 = importlib.util.module_from_spec(spec); sys.modules["rv4"] = rv4; spec.loader.exec_module(rv4)
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
P = pickle.loads((rv4.CACHE / "priors_live.pkl").read_bytes())
cfg = rv4.CFG; cfg.thompson = False
for kv in sys.argv[1:]:                                   # p. ej. kappa_s=5
    k, v = kv.split("="); P["params"][k] = float(v)
st = rv4.State(P, cfg)
st.prm["e_seen"] = st.prm["e_cart"] = 1.0
S = st.publish(set_global=False)
S.bought, S.carted = {}, {}
index = S.index
t0 = time.time()
stream = []
for u, (ps, es, ts) in ev.items():
    for p, e, t in zip(ps, es, ts):
        i = index.get(p)
        if i is not None:
            stream.append((int(t), 0, u, i, e))
for r in fb:
    if r["buy"] and r["buy"] in index:
        stream.append((int(r["t"]) + 5000, 0, r["u"], index[r["buy"]], "fbpurchase"))
for k, r in enumerate(fb):
    stream.append((int(r["t"]), 1, k, 0, ""))
stream.sort(key=lambda x: (x[0], x[1]))
print(f"flujo: {len(stream)} elementos ({time.time()-t0:.0f} s)", flush=True)
rng = np.random.default_rng(0)
out = {}
nreq = 0
for (t, typ, a, i, e) in stream:
    if typ == 0:
        if e == "fbpurchase":
            st.user_event(a, i, "purchase", organic=False)
        else:
            st.user_event(a, i, e)
        continue
    r = fb[a]
    info = {}
    rv4.answer(S, {"user_id": r["u"], "device": r["dev"], "region": "", "recent_views": r["rv"]}, rng, info)
    Pm = info["P"]
    out[r["rid"]] = np.array([Pm[index[p]] if p in index else np.nan for p in (r["shown"] or [])[:10]] + [np.nan] * (10 - len((r["shown"] or [])[:10])))
    nreq += 1
    if nreq % 20000 == 0:
        print(f"   {nreq} peticiones ({time.time()-t0:.0f} s)", flush=True)
tag = ("_" + "_".join(sys.argv[1:])) if len(sys.argv) > 1 else ""
pickle.dump(out, open(os.path.join(C, f"imp_pn{tag}.pkl"), "wb"), protocol=4)
print("hecho:", len(out), "listas en", round(time.time() - t0), "s")
