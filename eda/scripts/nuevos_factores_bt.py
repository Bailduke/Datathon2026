"""Backtest: imputar los factores de co-compra a los productos nuevos (hoy valen 0 → su puntuación de co-compra es 0).
Variantes del vector de factores de un producto nuevo:
  base   ceros (lo que corre)
  sub    media de los factores de los productos antiguos de su subcategoría
  tag    media de los de su subcategoría que comparten al menos una etiqueta (si no hay, la de la subcategoría)
  subn   'sub' reescalado a la norma mediana de los productos de la subcategoría
  tagn   'tag' reescalado igual
Se reproduce el stream una vez con el código del v7 y cinco fotos del estado. No consulta RawTree."""
import os, sys, pickle, importlib.util, logging, heapq, math, time, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C_ = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv7", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v7.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv7"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); rv.apply_params_v6(P); cfg = rv.CFG; cfg.thompson = False
fb, events, cat = pickle.load(open(os.path.join(C_, "dl_latest.pkl"), "rb")); rv.load_affinity(None, P, cfg)
MODES = ["base", "sub", "tag", "subn", "tagn"]; NM = len(MODES)
st = rv.State(P, cfg); n0 = st.n
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index:
        i = st.add_product(r, rv.parse_ts(r["ts"]) or 0.0)
        if i is not None: st.unavailable[i] = True
index = st.index; n = st.n; sg = np.asarray(rv.AFF["sg"], np.float32); sub = st.sub[:n]; TG = st.tagmat[:n]
FV = [st.F.copy() for _ in MODES]; nrm = np.linalg.norm(st.F[:n0], axis=1); used = []
for i in range(n0, n):
    c = np.flatnonzero(sub[:n0] == sub[i])
    if c.size == 0: continue
    ct = c[(TG[c] @ TG[i]) >= 1]; ct = ct if ct.size else c; used.append(ct.size)
    cs, ctg = st.F[c].mean(0), st.F[ct].mean(0); tn = np.median(nrm[c])
    FV[1][i] = cs; FV[2][i] = ctg; FV[3][i] = cs * tn / max(np.linalg.norm(cs), 1e-9); FV[4][i] = ctg * tn / max(np.linalg.norm(ctg), 1e-9)
print(f"{n-n0} productos nuevos · norma de los factores: antiguos mediana {np.median(nrm):.2f} · imputados sub {np.median(np.linalg.norm(FV[1][n0:n],axis=1)):.2f} · tag {np.median(np.linalg.norm(FV[2][n0:n],axis=1)):.2f} · "
      f"vecinos con etiqueta común: mediana {int(np.median(used))}", flush=True)
snaps = []; SL = rv.Snapshot.__slots__
def publish():
    S0 = st.publish(set_global=False); base = {k: getattr(S0, k) for k in SL}; out = []
    for q in range(NM):
        Fs = FV[q] * sg; out.append(rv.Snapshot(**{**base, "F": FV[q], "Fs": Fs, "selfdot": (Fs * FV[q]).sum(1)}))
    snaps[:] = out
FsV = None; publish()
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"]); rank = {k: q for q, k in enumerate(order)}
stream = heapq.merge(((int((rv.parse_ts(r["ts"]) or 0.0) * 1000), 0, k) for k, r in enumerate(cat)), ((e[0], 1, k) for k, e in enumerate(events)),
                     ((fb[k]["t"], 2, k) for k in order), sorted((fb[k]["t"] + 5000, 3, k) for k in order if fb[k]["buy"]))
M = sum(len((r["shown"] or [])[:10]) for r in fb); NT = rv.N_TASTE; NR = len(order); NPOL = 60000; first = NR - NPOL
X = np.zeros((M, 10)); G = np.zeros(M, np.int64); PN = np.zeros(M); AF = np.zeros((M, NM, NT), np.float32); NEWr = np.zeros(M)
POL = np.zeros((NPOL, NM, 10, 6 + NT), np.float32)          # log h, clases, gustos (bajo 'base') de la lista de cada variante
PCF = np.zeros((NPOL, NM, 10, NM, 4), np.float32)           # co-compra de esos productos bajo cada variante
PNEW = np.zeros((NPOL, NM, 10), np.float32); CTX = np.zeros((NPOL, 3))
rng = np.random.default_rng(0); m = 0; dirty = False; t_start = time.time()
for t, typ, k in stream:
    if typ == 1:
        _, u, p, e = events[k]; i = index.get(p)
        if i is not None: st.user_event(u, i, e, ts=t / 1000)
    elif typ == 0:
        r = cat[k]; i = index.get(r["product_id"])
        if r["change"] == "new_product":
            if i is not None: st.unavailable[i] = st.price[i] <= 0; st.T_launch[i], st.launch_ts[i] = st.T_live, t / 1000
        else: st.apply_update(r, t / 1000)
        dirty = True
    elif typ == 3: st.user_event(fb[k]["user_id"], index.get(fb[k]["buy"], -1), "purchase", organic=False)
    else:
        r = fb[k]
        if dirty: publish(); dirty = False
        tl = st.trend.log_trend()
        for S_ in snaps: S_.trend = tl
        u = r["user_id"]; bs, lcs, hcs = st.bought.get(u) or (), st.carted_live.get(u) or (), st.carted.get(u) or ()
        nuevo, home = float(r["nu"] == "true"), float(r["page"] == "home"); n_k = rank[k]; infos = []
        for S in snaps:
            info = {}; rv.answer(S, r, rng, info); infos.append(info)
        if n_k >= first:
            a = n_k - first; Pn_ = infos[0]["Pn"]
            for v, inf in enumerate(infos):
                lc_, hc_, vs_ = set(inf["lc"]), set(inf["hc"]), inf["views"]
                for j, i in enumerate(inf["chosen"][:10]):
                    POL[a, v, j, :6] = (math.log(max(float(Pn_[i]), 1e-12) / rv.P_REF), i in lc_, i in hc_, len(vs_) == 1 and i == vs_[0], len(vs_) > 1 and i == vs_[0], len(vs_) > 1 and i in vs_[1:])
                    POL[a, v, j, 6:] = infos[0]["taste"][:, i]; PNEW[a, v, j] = float(i >= n0)
                    for q in range(NM): PCF[a, v, j, q] = infos[q]["taste"][:4, i]
            CTX[a] = (nuevo, home, float(bool(r["buy"])))
        vs = infos[0]["views"]
        for j, pid in enumerate((r["shown"] or [])[:10]):
            i = index.get(pid, -1); b = float(i in bs)
            X[m] = (j, float(pid == r["buy"]), nuevo, home, float(i in lcs) * (1 - b), float(i in hcs and i not in lcs) * (1 - b), float(len(vs) == 1 and i == vs[0]) * (1 - b),
                    float(len(vs) > 1 and i == vs[0]) * (1 - b), float(len(vs) > 1 and i in vs[1:]) * (1 - b), b)
            G[m] = n_k
            if 0 <= i < infos[0]["Pn"].size:
                PN[m] = float(infos[0]["Pn"][i]); NEWr[m] = float(i >= n0)
                for q in range(NM): AF[m, q] = infos[q]["taste"][:, i]
            m += 1
        if n_k % 40000 == 0: print(f"  {n_k}/{NR} peticiones · {time.time()-t_start:.0f} s", flush=True)
X, G, PN, AF, NEWr = X[:m], G[:m], PN[:m], AF[:m], NEWr[:m]
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; NB = rv.N_BASE; wp = np.asarray(P["w_pos"])
print(f"\n{len(y)} impresiones · {int(y.sum())} compras · de productos nuevos: {int(NEWr.sum())} impresiones, {int(y[NEWr>0].sum())} compras")
print("\n(1) Ajuste dentro de la lista (logit condicional; mismos parámetros en todas las variantes)")
B = {}
for q, name in enumerate(MODES):
    A = rv.glm_design(X, PN, AF[:, q]); cols = list(range(3, A.shape[1]))
    bc, sec, llc, ng = rv.cond_fit(A[:, cols], y, np.log(w), G)
    bc2, sec2, llc2, _ = rv.cond_fit(np.column_stack([A[:, cols], NEWr]), y, np.log(w), G)
    bp2, _, _ = rv.poisson_fit(np.column_stack([A, NEWr]), y, np.log(w)); B[name] = bp2
    print(f"   {name:5s} logver {llc:9.1f} · con indicador 'producto nuevo': {llc2:9.1f}, coeficiente {bc2[-1]:+.2f} ± {sec2[-1]:.2f} (x{math.exp(bc2[-1]):.2f}) · pesos de co-compra " + " ".join(f"{v:.2f}" for v in bc[NB-3:NB-3+4]))
print(f"\n(2) Conversión prevista de la lista que elegiría cada variante (últimas {NPOL} peticiones). 'Verdad' = modelo de la variante + indicador residual de producto nuevo")
nu = CTX[:, 0] > 0
print("   productos nuevos por lista de 10: " + " · ".join(f"{n_} {10*s:.2f}" for n_, s in zip(MODES, PNEW.mean((0, 2)))))
for q, tn in enumerate(MODES):
    bp = B[tn]
    lin = (bp[0] + bp[1] * CTX[:, 0] + bp[2] * CTX[:, 1])[:, None, None] + bp[3] * POL[..., 0] + POL[..., 1:6] @ bp[4:9] + PCF[:, :, :, q, :] @ bp[NB:NB + 4] + POL[..., 10:] @ bp[NB + 4:NB + NT] + bp[-1] * PNEW
    val = (wp[None, None, :] * np.exp(np.clip(lin, -50, 5))).sum(-1); d = val - val[:, [0]]; se = d.std(0) / np.sqrt(len(d))
    print(f"   verdad = '{tn}' (indicador residual {bp[-1]:+.2f})")
    for v, n_ in enumerate(MODES):
        print(f"      lista de '{n_:5s}' {100*val[:, v].mean():.3f} %  ({100*d[:, v].mean()/val[:, 0].mean():+.2f} % ± {100*se[v]/val[:, 0].mean():.2f}) · usuarios nuevos {100*val[nu, v].mean():.3f} % · con historia {100*val[~nu, v].mean():.3f} %")
print(f"\n{time.time()-t_start:.0f} s")
