"""H10 · Más ideas a nivel de producto: gusto por etiquetas dentro de la categoría, afinidad etiqueta-etiqueta (20×20),
tendencia de popularidad de hoy frente al histórico, e interacción usuario nuevo × co-compra con el carrito de hoy."""
import zlib, bisect, numpy as np
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
G = rv.glm_design(X[m], PN[m], AF[m]); NB = rv.N_BASE; A = np.column_stack([G, ctl])
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); tags = z["tags"]; n = z["F"].shape[0]; K = 20
T = np.zeros((n, K), np.float32); T[np.arange(n), tags[:, 0]] = 1; T[np.arange(n), tags[:, 1]] = 1
catg = np.asarray(P["cat"]); subg = np.asarray(P["sub"])
LB = pickle.load(open(os.path.join(C, "h6_baskets.pkl"), "rb"))["LB"]
Wt = np.load(os.path.join(HERE, "../scripts/.cache/W_tags.npy")); Wo = Wt - np.diag(np.diag(Wt))      # afinidad etiqueta-etiqueta sin la diagonal
bounds = np.flatnonzero(np.r_[True, r_[1:] != r_[:-1], True])
tagcat = np.zeros((len(y), 4)); tagpair = np.zeros((len(y), 4)); al = 6.0
for a, c_ in zip(bounds[:-1], bounds[1:]):
    it = ii[a:c_]; ok = (it >= 0) & (it < n)
    if not ok.any(): continue
    j = it[ok]
    for c, (b, cnt, age) in enumerate(LB[r_[a]]):
        if b.size == 0: continue
        Tb = T[b]; same = catg[b][None, :] == catg[j][:, None]; notself = b[None, :] != j[:, None]
        Mk = (same & notself).astype(np.float32); cc = Mk @ Tb; tot = cc.sum(1)                 # perfil con los productos de la cesta de SU categoría
        s = (T[j] * np.log((cc + al / K) / (tot[:, None] + al) * K)).sum(1); tagcat[a:c_, c][ok] = np.where(tot > 0, s, 0.0)
        Mn = notself.astype(np.float32); pr = ((T[j] @ Wo) * (Mn @ Tb)).sum(1) / (Mn.sum(1) + 1.0); tagpair[a:c_, c][ok] = pr
# tendencia de popularidad: compras orgánicas de hoy del producto hasta t, frente a lo esperado por su cuota histórica
index = {p: i for i, p in enumerate(P["ids"])}; pt = defaultdict(list); allp = []
for (te, u, p, e) in events:
    if e == "purchase":
        allp.append(te); i = index.get(p)
        if i is not None: pt[i].append(te)
allp = np.array(allp); share = (P["C_hist"] + 1.0) / (P["C_hist"].sum() + n)
trend = np.zeros(len(y)); trend2 = np.zeros(len(y))
for q in range(len(y)):
    i = ii[q]
    if 0 <= i < n:
        cl = bisect.bisect_left(pt.get(i, ()), t[q]); Tl = np.searchsorted(allp, t[q]); c2 = cl - bisect.bisect_left(pt.get(i, ()), t[q] - 7.2e6)
        trend[q] = np.log((cl + 5.0) / (share[i] * Tl + 5.0)); trend2[q] = np.log((c2 + 2.0) / (share[i] * (Tl - np.searchsorted(allp, t[q] - 7.2e6)) + 2.0))
print(f"submuestra {len(bounds)-1} listas · {int(y.sum())} compras")
ev = lambda cols, lab_: evaluate2(A, np.column_stack([A] + cols), y, off, t, r_, nu, lab_)
ev([tagcat[:, c] for c in range(4)], "etiquetas dentro de la categoría (4 cestas)")
ev([tagcat.sum(1)], "etiquetas dentro de la categoría (suma)")
ev([tagpair[:, c] for c in range(4)], "afinidad etiqueta-etiqueta 20×20 (4 cestas)")
ev([tagpair.sum(1)], "afinidad etiqueta-etiqueta (suma)")
ev([trend], "tendencia: compras de hoy / esperadas por histórico")
ev([trend2], "tendencia: últimas 2 h / esperadas")
n_ = nu.astype(float)
ev([n_ * G[:, NB + 2]], "usuario nuevo × co-compra con el carrito de hoy")
ev([n_ * G[:, NB + 2], n_ * G[:, NB + 5]], "usuario nuevo × (co-compra carrito hoy, etiquetas sesión)")
