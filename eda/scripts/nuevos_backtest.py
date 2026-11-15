"""Backtest: ¿arreglar la popularidad inicial de los productos nuevos mejora el modelo (v7)?
Hoy un producto nuevo arranca con la cuota MEDIANA de su subcategoría y fuerza b0; los datos dicen que vende ~3 veces eso.
Variantes (solo cambian los productos nuevos):
  base      Gamma(m_s·b0 + C, b0 + T)                      (lo que corre)
  media     prior centrado en la cuota MEDIA de la subcategoría, misma fuerza b0
  debil     media + prior débil (0,5 compras ficticias): los datos mandan antes
  carritos  debil + los carritos cuentan como 0,4 compras (exposición corregida)
Se reproduce el stream una vez con el código del v7 y cuatro fotos del estado. No consulta RawTree."""
import os, sys, pickle, importlib.util, logging, heapq, math, time, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C_ = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv7", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v7.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv7"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); rv.apply_params_v6(P); cfg = rv.CFG; cfg.thompson = False
fb, events, cat = pickle.load(open(os.path.join(C_, "dl_latest.pkl"), "rb")); rv.load_affinity(None, P, cfg)
MODES = ["base", "media", "debil", "carritos"]; NM = len(MODES); MODE = {"m": "base"}; A_WEAK, CW = 0.5, 0.4
KAPPA = sum(1 for e in events if e[3] == "add_to_cart") / sum(1 for e in events if e[3] == "purchase")
sub0 = np.asarray(P["sub"]); nsub = np.bincount(sub0, minlength=len(P["subs"])); MEAN_S = np.bincount(sub0, weights=P["C_hist"], minlength=len(P["subs"])) / P["T_hist"] / np.maximum(nsub, 1)
print(f"cuota media / cuota mediana por subcategoría: mediana de la razón {np.median(MEAN_S / P['m_sub'][:len(MEAN_S)]):.2f} · carritos por compra {KAPPA:.2f}", flush=True)
def popularity(self):
    n = self.n; w = self.cfg.live_weight; C = self.C_hist[:n] + w * self.C_live[:n]; new = self.is_new[:n]
    T = np.where(new, w * np.maximum(self.T_live - self.T_launch[:n], 0.0), self.P["T_hist"] + w * self.T_live)
    shape, rate = rv.gamma_popularity(C, T, self.m_sub, self.sub[:n], self.cfg.b0); m = MODE["m"]
    if m != "base" and new.any():
        s = self.sub[:n]; mu0 = np.where(s < len(MEAN_S), MEAN_S[np.minimum(s, len(MEAN_S) - 1)], 2.8 * self.m_sub[s]); mu0 = np.maximum(mu0, 1e-9)
        a = mu0 * self.cfg.b0 if m == "media" else A_WEAK
        k = CW if m == "carritos" else 0.0
        shape = np.where(new, a + C + k * self.K_live[:n], shape); rate = np.where(new, a / mu0 + T * (1 + k * KAPPA), rate)
    return shape, rate
rv.State.popularity = popularity
st = rv.State(P, cfg); st.K_live = np.zeros(cfg.n_max_products)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index:
        i = st.add_product(r, rv.parse_ts(r["ts"]) or 0.0)
        if i is not None: st.unavailable[i] = True
index = st.index; snaps = []
def publish():
    out = []
    for m in MODES[::-1]:                                   # 'base' al final: el estado interno queda como en producción
        MODE["m"] = m; out.append(st.publish(set_global=False))
    snaps[:] = out[::-1]
publish()
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"]); rank = {k: n for n, k in enumerate(order)}
stream = heapq.merge(((int((rv.parse_ts(r["ts"]) or 0.0) * 1000), 0, k) for k, r in enumerate(cat)), ((e[0], 1, k) for k, e in enumerate(events)),
                     ((fb[k]["t"], 2, k) for k in order), sorted((fb[k]["t"] + 5000, 3, k) for k in order if fb[k]["buy"]))
M = sum(len((r["shown"] or [])[:10]) for r in fb); NT = rv.N_TASTE; NR = len(order)
X = np.zeros((M, 10)); G = np.zeros(M, np.int64); PN = np.zeros((M, NM)); AF = np.zeros((M, NT), np.float32); NEWr = np.zeros(M)
POL = np.zeros((NR, NM, 10, 6 + NT), np.float32)            # clases y gustos de la lista que elegiría cada variante
PLP = np.zeros((NR, NM, 10, NM), np.float32)               # log h de esos productos bajo cada variante (para valorar con cualquier 'verdad')
PNEW = np.zeros((NR, NM, 10), np.float32); CTX = np.zeros((NR, 3))
rng = np.random.default_rng(0); m = 0; dirty = False; t_start = time.time()
for t, typ, k in stream:
    if typ == 1:
        _, u, p, e = events[k]; i = index.get(p)
        if i is not None:
            st.user_event(u, i, e, ts=t / 1000)
            if e == "add_to_cart": st.K_live[i] += 1
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
        LP = [np.log(np.maximum(inf["Pn"], 1e-12) / rv.P_REF) for inf in infos]
        for v, inf in enumerate(infos):
            lc_, hc_, vs_ = set(inf["lc"]), set(inf["hc"]), inf["views"]
            for j, i in enumerate(inf["chosen"][:10]):
                POL[n_k, v, j, :6] = (0.0, i in lc_, i in hc_, len(vs_) == 1 and i == vs_[0], len(vs_) > 1 and i == vs_[0], len(vs_) > 1 and i in vs_[1:])
                POL[n_k, v, j, 6:] = inf["taste"][:, i]; PNEW[n_k, v, j] = float(snaps[0].is_new[i])
                for q in range(NM): PLP[n_k, v, j, q] = LP[q][i]
        CTX[n_k] = (nuevo, home, float(bool(r["buy"])))
        vs, ts_ = infos[0]["views"], infos[0]["taste"]
        for j, pid in enumerate((r["shown"] or [])[:10]):
            i = index.get(pid, -1); b = float(i in bs)
            X[m] = (j, float(pid == r["buy"]), nuevo, home, float(i in lcs) * (1 - b), float(i in hcs and i not in lcs) * (1 - b), float(len(vs) == 1 and i == vs[0]) * (1 - b),
                    float(len(vs) > 1 and i == vs[0]) * (1 - b), float(len(vs) > 1 and i in vs[1:]) * (1 - b), b)
            G[m] = n_k
            if 0 <= i < infos[0]["Pn"].size:
                for q in range(NM): PN[m, q] = float(infos[q]["Pn"][i])
                AF[m] = ts_[:, i]; NEWr[m] = float(snaps[0].is_new[i])
            m += 1
        if n_k % 40000 == 0: print(f"  {n_k}/{NR} peticiones · {time.time()-t_start:.0f} s", flush=True)
X, G, PN, AF, NEWr = X[:m], G[:m], PN[:m], AF[:m], NEWr[:m]
np.savez_compressed(os.path.join(C_, "nuevos_bt.npz"), X=X, G=G, PN=PN, AF=AF, NEWr=NEWr, POL=POL, PLP=PLP, PNEW=PNEW, CTX=CTX)
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; NB = rv.N_BASE
print(f"\n{len(y)} impresiones · {int(y.sum())} compras · productos nuevos: {int(NEWr.sum())} impresiones ({100*NEWr.mean():.1f} %), {int(y[NEWr>0].sum())} compras ({100*y[NEWr>0].sum()/y.sum():.1f} %)")
print("\n(1) Ajuste dentro de la lista (logit condicional, mismos parámetros en todas las variantes)")
B = {}
for q, name in enumerate(MODES):
    A = rv.glm_design(X, PN[:, q], AF); cols = list(range(3, A.shape[1]))
    bc, sec, llc, ng = rv.cond_fit(A[:, cols], y, np.log(w), G)
    A2 = np.column_stack([A[:, cols], NEWr]); bc2, sec2, llc2, _ = rv.cond_fit(A2, y, np.log(w), G)
    bp, sep, llp = rv.poisson_fit(A, y, np.log(w)); bp2, sep2, llp2 = rv.poisson_fit(np.column_stack([A, NEWr]), y, np.log(w))
    B[name] = (bp, bp2, bc)
    print(f"   {name:9s} logver {llc:9.1f} · theta {bc[0]:.3f} · con indicador 'producto nuevo': logver {llc2:9.1f}, coeficiente {bc2[-1]:+.2f} ± {sec2[-1]:.2f} (x{math.exp(bc2[-1]):.2f}) · Poisson {llp:9.1f}")
print("\n(2) Conversión prevista de la lista que elegiría cada variante (todas las peticiones)")
def value(bp, q_truth, c_new=0.0):
    lin = (bp[0] + bp[1] * CTX[:, 0] + bp[2] * CTX[:, 1])[:, None, None] + bp[3] * PLP[..., q_truth] + POL[..., 1:6] @ bp[4:9] + POL[..., 6:] @ bp[NB:NB + NT] + c_new * PNEW
    return (np.asarray(P["w_pos"])[None, None, :] * np.exp(np.clip(lin, -50, 5))).sum(-1)          # (peticiones, variantes)
truths = [("base + indicador libre de producto nuevo", B["base"][1][:-1], 0, B["base"][1][-1])] + [(f"modelo '{n}'", B[n][0], q, 0.0) for q, n in enumerate(MODES) if q]
nu = CTX[:, 0] > 0; share_new = PNEW.mean((0, 2))
print("   productos nuevos por lista de 10: " + " · ".join(f"{n} {10*s:.2f}" for n, s in zip(MODES, share_new)))
for tn, bp, q, c in truths:
    val = value(bp, q, c); d = val - val[:, [0]]; se = d.std(0) / np.sqrt(len(d))
    print(f"   verdad = {tn}")
    for v, n in enumerate(MODES):
        print(f"      lista de '{n:9s}' {100*val[:, v].mean():.3f} %  ({100*d[:, v].mean()/val[:, 0].mean():+.2f} % relativo ± {100*se[v]/val[:, 0].mean():.2f}) · usuarios nuevos {100*val[nu, v].mean():.3f} % · con historia {100*val[~nu, v].mean():.3f} %")
print(f"\n{time.time()-t_start:.0f} s")
