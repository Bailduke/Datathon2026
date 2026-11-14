"""H6/H7 · Hiperparámetros de los gustos latentes (nº de factores, encogimiento, alfa, caducidad de las cestas de hoy,
peso por repetición) y carrito añadido 2+ veces. Las cestas de cada petición se reconstruyen de los eventos y las
puntuaciones se recalculan fuera del replay; con los valores actuales deben reproducir las de producción."""
import zlib, numpy as np
from collections import defaultdict
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
h9 = pickle.load(open(os.path.join(C, "h9.pkl"), "rb")); z8 = np.load(os.path.join(C, "h8.npz"))
X, PN, AF, KI = h9["X"], h9["PN"], h9["AF"].astype(float), h9["KI"]
rid = KI[:, 0]; samp = np.array([bool(f["buy"]) or zlib.crc32(f["rid"].encode()) % 5 == 0 for f in fb]); m = samp[rid]
buy_list = np.array([bool(f["buy"]) for f in fb])[rid]
y = X[m, 1]; off = (np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]) + np.where(buy_list, 0.0, np.log(5.0)))[m]
t = np.array([fb[j]["t"] for j in rid])[m]; nu = X[m, 2] > 0; r_ = rid[m]; ii = KI[m, 1]
ctl = np.column_stack([z8["hrs"], np.log1p(z8["n_req"]), h9["EX"][:, 1], h9["EX"][:, 2]])[m]
G = rv.glm_design(X[m], PN[m], AF[m]); NB = rv.N_BASE
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); tags = z["tags"]; n = z["F"].shape[0]; K = 20
T = np.zeros((n, K), np.float32); T[np.arange(n), tags[:, 0]] = 1; T[np.arange(n), tags[:, 1]] = 1
e64 = np.load(os.path.join(HERE, "../scripts/.cache/eig.npz")); W64, V64 = e64["w"], e64["V"]
def factors(k):
    F = (V64[:, :k] * np.sqrt(np.abs(W64[:k]))).astype(np.float32); return F, F * np.sign(W64[:k]).astype(np.float32)
index = {p: i for i, p in enumerate(P["ids"])}; E = frozenset()
uev = defaultdict(list)
for (te, u, p, e) in events:
    i = index.get(p)
    if i is not None: uev[u].append((te, i, e[0]))
# cestas de cada lista de la submuestra: [(índices, nº de veces, edad en horas)] × 4
lists = np.unique(r_); LB = {}
for k in lists:
    f = fb[k]; u = f["user_id"]; tt = f["t"]
    hb = (P["bought"].get(u) or E) | (P["carted"].get(u) or E); hb = np.array(sorted(x for x in hb if x < n), np.int64)
    vs = [index[p] for p in rv.normalize_views(f["recent_views"]) if p in index and index[p] < n][-10:]
    c = {}; v = {}
    for (te, i, e) in uev.get(u, ()):
        if te >= tt: break
        d = c if e == "a" else v if e == "v" else None
        if d is not None:
            a = d.get(i); d[i] = (a[0] + 1 if a else 1, te)
    pk = lambda d: (np.array(list(d), np.int64), np.array([x[0] for x in d.values()], float), np.array([(tt - x[1]) / 3.6e6 for x in d.values()]))
    LB[k] = ((hb, np.ones(hb.size), np.zeros(hb.size)), (np.array(vs, np.int64), np.ones(len(vs)), np.arange(len(vs))[::-1].astype(float)), pk(c), pk(v))
bounds = np.flatnonzero(np.r_[True, r_[1:] != r_[:-1], True])
def feats(F, Fs, shr=(3, 1, 1, 1), al=6.0, wfn=None):
    selfdot = (Fs * F).sum(1); out = np.zeros((len(y), 8))
    for a, c_ in zip(bounds[:-1], bounds[1:]):
        it = ii[a:c_]; ok = (it >= 0) & (it < n)
        if not ok.any(): continue
        j = it[ok]
        for c, (b, cnt, age) in enumerate(LB[r_[a]]):
            if b.size == 0: continue
            w = np.ones(b.size) if wfn is None else wfn(c, cnt, age)
            pos = {int(x): q for q, x in enumerate(b)}; own = np.array([w[pos[int(x)]] if int(x) in pos else 0.0 for x in j]); Wt = w.sum()
            if Wt <= 0: continue
            out[a:c_, c][ok] = (Fs[j] @ (F[b] * w[:, None]).sum(0) - own * selfdot[j]) / (Wt - own + shr[c])
            cc = (T[b] * w[:, None]).sum(0); cl = cc[None, :] - own[:, None] * T[j]; tl = 2 * (Wt - own)
            out[a:c_, 4 + c][ok] = (T[j] * np.log((np.maximum(cl, 0) + al / K) / (np.maximum(tl, 0)[:, None] + al) * K)).sum(1)
    return out
F16, Fs16 = factors(16); f0 = feats(F16, Fs16)
print("comprobación: correlación de cada puntuación recalculada con la de producción:", np.round([np.corrcoef(f0[:, c], G[:, NB + c])[0, 1] for c in range(8)], 3))
des = lambda f: np.column_stack([G[:, :NB], f, ctl])
s_prod = score(np.column_stack([G, ctl]), y, off, t, r_, nu); s0 = score(des(f0), y, off, t, r_, nu)
print(f"submuestra {len(lists)} listas · {int(y.sum())} compras · base producción: Poisson {s_prod[0]:.1f} orden temp {s_prod[1]:.1f} CV {s_prod[2]:.1f} · recalculada: {s0[0]:.1f} {s0[1]:.1f} {s0[2]:.1f}")
print(f"{'variante':52s} {'Poisson':>8} {'ord.temp':>9} {'ord.CV':>8} {'nuevos':>8} {'historia':>9}"); res = {}
def run(label, f, extra=None):
    A = des(f) if extra is None else np.column_stack([des(f)] + extra); s = score(A, y, off, t, r_, nu) - s0; res[label] = s.tolist()
    print(f"{label:52s} {s[0]:+8.1f} {s[1]:+9.1f} {s[2]:+8.1f} {s[3]:+8.1f} {s[4]:+9.1f}", flush=True)
for k in (4, 8, 12, 24, 32, 48):
    F, Fs = factors(k); run(f"factores de co-compra: {k}", feats(F, Fs))
for sh in ((1, 1, 1, 1), (10, 1, 1, 1), (3, .3, .3, .3), (3, 3, 3, 3), (3, 1, 3, 3), (3, 3, 1, 1), (3, 1, 1, 3), (3, 1, .3, 1)):
    run(f"encogimiento {sh}", feats(F16, Fs16, shr=sh))
for al in (1.0, 3.0, 12.0, 24.0): run(f"alfa del perfil de etiquetas {al}", feats(F16, Fs16, al=al))
for tau in (1.0, 3.0, 8.0):
    run(f"cestas de hoy con caducidad exp(-edad/{tau:g} h)", feats(F16, Fs16, wfn=lambda c, cnt, age: np.exp(-age / tau) if c >= 2 else np.ones(cnt.size)))
run("cestas de hoy ponderadas por nº de veces (raíz)", feats(F16, Fs16, wfn=lambda c, cnt, age: np.sqrt(cnt) if c >= 2 else np.ones(cnt.size)))
run("cestas de hoy ponderadas por nº de veces", feats(F16, Fs16, wfn=lambda c, cnt, age: cnt if c >= 2 else np.ones(cnt.size)))
for rho in (0.5, 0.8): run(f"sesión ponderada por recencia ({rho}^k)", feats(F16, Fs16, wfn=lambda c, cnt, age: rho ** age if c == 1 else np.ones(cnt.size)))
# H7: carrito añadido 2+ veces y edad del carrito
lc2 = np.zeros(len(y)); lage = np.zeros(len(y))
for a, c_ in zip(bounds[:-1], bounds[1:]):
    b, cnt, age = LB[r_[a]][2]
    if b.size:
        d = {int(x): (cnt[q], age[q]) for q, x in enumerate(b)}
        for q, x in enumerate(ii[a:c_]):
            if int(x) in d: lc2[a + q] = d[int(x)][0] >= 2; lage[a + q] = d[int(x)][1]
LC = G[:, 4]
run("H7 carrito de hoy añadido 2+ veces", f0, [lc2 * LC]); run("H7 edad del carrito de hoy (horas)", f0, [lage * LC]); run("H7 carrito de hoy con más de 3 h", f0, [(lage > 3) * LC])
print(f"   impresiones de carrito de hoy {int(LC.sum())} · con 2+ añadidos {int((lc2*LC).sum())} · con más de 3 h {int(((lage>3)*LC).sum())}")
# parsimonia: una sola puntuación de co-compra y una de etiquetas (suma de las cuatro cestas)
A1 = np.column_stack([G[:, :NB], f0[:, :4].sum(1), f0[:, 4:].sum(1), ctl]); s = score(A1, y, off, t, r_, nu) - s0
print(f"{'parsimonia: 2 coeficientes en vez de 8':52s} {s[0]:+8.1f} {s[1]:+9.1f} {s[2]:+8.1f} {s[3]:+8.1f} {s[4]:+9.1f}")
import json; json.dump(res, open(os.path.join(C, "h6_res.json"), "w"))
pickle.dump({"LB": LB, "lists": lists}, open(os.path.join(C, "h6_baskets.pkl"), "wb"), protocol=4)
