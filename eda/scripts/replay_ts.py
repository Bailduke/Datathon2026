"""Coste de la exploración (Thompson Sampling sobre la popularidad) y variantes de la política, con el modelo
de compra calibrado: se reproduce el stream una vez y se predice la conversión de cada variante."""
import os, sys, pickle, time, importlib.util, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv4", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v4.py"))
rv4 = importlib.util.module_from_spec(spec); sys.modules["rv4"] = rv4; spec.loader.exec_module(rv4)
P = pickle.loads((rv4.CACHE / "priors_live.pkl").read_bytes())
cfg = rv4.CFG
rt = rv4.make_rt()
rv4.load_affinity(rt, P, cfg)
dl = os.path.join(C, "calib_dl.pkl")
if os.path.exists(dl) and time.time() - os.path.getmtime(dl) < 6 * 3600:
    fb, events, cat = pickle.load(open(dl, "rb"))
else:
    fb, events, cat = rv4.download_feedback(rt); pickle.dump((fb, events, cat), open(dl, "wb"), protocol=4)
coef = P["diag"]["afinidad"]["coef"]; b = np.array([coef[n][0] for n in rv4.GLM_COLS])
w_pos = np.asarray(P["w_pos"])
def value(pol, ctx):
    lin = (b[0] + b[1] * ctx[:, 0] + b[2] * ctx[:, 1])[:, None, None] + b[3] * pol[..., 0] + b[4] * pol[..., 1] + b[5] * pol[..., 2] \
        + b[6] * pol[..., 3] + b[7] * pol[..., 4] + b[9] * pol[..., 1] * pol[..., 0] + pol[..., 5:] @ b[rv4.N_BASE:rv4.N_BASE + 8]
    return (w_pos[None, None, :] * np.exp(np.clip(lin, -50, 5))).sum(-1)
LAST = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
for label, th in (("sin exploración (medias)", False), ("con Thompson Sampling", True)):
    cfg.thompson = th
    R = rv4.replay_feedback(P, cfg, fb, events, cat, tilts=[{}], policy_last=LAST)
    v = value(R["pol"], R["pol_ctx"]); nu = R["pol_ctx"][:, 0] > 0
    print(f"{label:28s} conversión prevista {100*v.mean():.2f} % (nuevos {100*v[nu].mean():.2f} % · con historia {100*v[~nu].mean():.2f} %)", flush=True)
