import os, sys, pickle, time, importlib.util, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv4", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v4.py"))
rv4 = importlib.util.module_from_spec(spec); sys.modules["rv4"] = rv4; spec.loader.exec_module(rv4)
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb")); ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
P = pickle.loads((rv4.CACHE / "priors_live.pkl").read_bytes())
cfg = rv4.CFG; cfg.thompson = False
st = rv4.State(P, cfg); rt = rv4.make_rt()
for r in rt.query(rv4.SQL_CATALOG_UPDATES.format(since="1970-01-01 00:00:00.000", limit=200000), 60):
    if r["change"] == "new_product": st.apply_update(r, time.time())
S = st.publish(set_global=False)
d = {k: getattr(S, k) for k in S.__slots__}; d["prm"] = rv4.State.make_params(P["params"], cfg, g_lc=0, d_lc=0, g_hc=0, g_v0=0, g_v1=0); S = rv4.Snapshot(**d)
index = S.index
stream = []
for u, (ps, es, ts) in ev.items():
    for p, e, t in zip(ps, es, ts):
        i = index.get(p)
        if i is not None: stream.append((int(t), 0, u, i, e))
for r in fb:
    if r["buy"] and r["buy"] in index: stream.append((int(r["t"]) + 5000, 0, r["u"], index[r["buy"]], "fb"))
for k, r in enumerate(fb): stream.append((int(r["t"]), 1, k, 0, ""))
stream.sort(key=lambda x: (x[0], x[1]))
rng = np.random.default_rng(0)
stats = {}; shownn = 0
for (t, typ, a, i, e) in stream:
    if typ == 0:
        st.user_event(a, i, "purchase" if e == "fb" else e, organic=(e != "fb")); continue
    r = fb[a]
    key = r["mv"] if r["sb"] == "team" else "baseline"
    info = {}; rv4.answer(S, {"user_id": r["u"], "device": r["dev"], "region": "", "recent_views": r["rv"]}, rng, info)
    Pn = info["Pn"]; ch = info["chosen"][:10]
    sh = [index.get(p, -1) for p in (r["shown"] or [])[:10]]
    bs = st.bought.get(r["u"]) or ()
    s = stats.setdefault(key, [0, 0.0, 0.0, 0.0, 0, 0.0])
    s[0] += 1; s[1] += sum(Pn[i] for i in sh if i >= 0); s[2] += sum(Pn[i] for i in ch); s[3] += len(set(sh) & set(ch)) / 10
    s[4] += sum(1 for i in sh if i in bs); s[5] += sum(1 for i in sh if i >= 0 and Pn[i] < 1e-4 and i not in bs) / 10
    if key == "tengo-aurilla-v0" and shownn < 3 and a % 7 == 0:
        shownn += 1
        print(f"\n--- {r['rid']} page={r['page']} nuevo={r['nu']} vistos={[S.ids.index(x) if False else x for x in rv4.normalize_views(r['rv'])][-3:]}")
        for j in range(10):
            i, c = sh[j], ch[j]
            print(f"   slot {j+1:2d}  enviado {rv4.describe(S, st, r['shown'][j], r['u'])[:95]:95s} Pn={Pn[i] if i>=0 else float('nan'):.4f} | ahora {S.ids[c]} Pn={Pn[c]:.4f}")
print()
for k, s in stats.items():
    print(f"{k or '(sin versión)':20s} listas {s[0]:6d} · suma de P jerárquica: lo enviado {s[1]/s[0]:.4f} vs top-10 de ahora {s[2]/s[0]:.4f} · coincidencia {100*s[3]/s[0]:.0f} % · "
          f"ya comprados por lista {s[4]/s[0]:.2f} · productos con P<1e-4 {100*s[5]/s[0]:.0f} %")
