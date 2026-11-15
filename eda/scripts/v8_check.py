"""Comprobación del recommender-v8 con el stream guardado (sin RawTree):
 (1) con impute_new=false reproduce el v7 (misma logver dentro de la lista: -3424,3);
 (2) con impute_new=true reproduce la variante 'tag' del backtest (-3398,9) y los factores imputados coinciden con el cálculo independiente;
 (3) ningún NaN ni excepción al pasar por add_product 400 veces."""
import os, sys, pickle, importlib.util, logging, time, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C_ = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv8", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v8.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv8"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); rv.apply_params_v6(P); cfg = rv.CFG; cfg.thompson = False
print("modelo por defecto:", cfg.model_version, "· calibración:", rv.PARAMS_V6, "· pesos:", {k: P["params"].get(k) for k in ("theta", "g_vu", "taste")})
fb, events, cat = pickle.load(open(os.path.join(C_, "dl_latest.pkl"), "rb")); rv.load_affinity(None, P, cfg)
st = rv.State(P, cfg); n0 = st.n
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
n = st.n; sub = st.sub[:n]; TG = st.tagmat[:n]; F0 = np.asarray(rv.AFF["F"]); ref = np.zeros((n - n0, F0.shape[1]), np.float32)
for i in range(n0, n):
    c = np.flatnonzero(sub[:n0] == sub[i]); ct = c[(TG[c] @ TG[i]) >= 1]; ref[i - n0] = F0[ct if ct.size else c].mean(0)
print(f"productos: {n0} con factores + {n-n0} nuevos · n_fact {st.n_fact} · diferencia máxima con el cálculo independiente {np.abs(st.F[n0:n]-ref).max():.2e} · "
      f"Fs y selfdot coherentes: {np.allclose(st.Fs[n0:n], st.F[n0:n]*st.sg)} {np.allclose(st.selfdot[n0:n], (st.Fs[n0:n]*st.F[n0:n]).sum(1))} · "
      f"nuevos con vector nulo: {int((np.abs(st.F[n0:n]).sum(1)==0).sum())} · NaN: {int(np.isnan(st.F[:n]).sum())} · factores antiguos intactos: {np.array_equal(st.F[:n0], F0)}")
for flag, expect in ((False, -3424.3), (True, -3398.9)):
    cfg.impute_new = flag; t = time.time(); R = rv.replay_feedback(P, cfg, fb, events, cat)
    X = R["X"]; y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; A = rv.glm_design(X, R["PN"], R["AF"]); cols = list(range(3, A.shape[1]))
    b, se, ll, ng = rv.cond_fit(A[:, cols], y, np.log(w), R["G"])
    print(f"impute_new={flag!s:5s} logver dentro de la lista {ll:.1f} (esperado {expect}) · {ng} listas con compra · NaN en gustos: {int(np.isnan(R['AF']).sum())} · "
          f"pesos de co-compra {' '.join(f'{v:.2f}' for v in b[rv.N_BASE-3:rv.N_BASE+1])} · {time.time()-t:.0f} s", flush=True)
