import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache"); sys.path.insert(0, HERE)
from rt import q
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb")); tg = pickle.load(open(os.path.join(C, "tags.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids)
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"]); ns = sub.max() + 1
catdf = pd.DataFrame(q("SELECT product_id::String AS p, toString(brand) AS brand, price_eur::Float64 AS price, toString(created_at) AS created FROM shop.catalog", 120)["data"]).set_index("p").loc[ids]
brand = pd.factorize(catdf.brand)[0]
T = np.zeros((n, 20), np.float32)
for p, i in index.items():
    T[i, tg[p]] = 1
users = list(baskets); rng = np.random.default_rng(0); half = rng.random(len(users)) < 0.5
def build(mask):
    Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32); N = 0
    for k, u in enumerate(users):
        if not mask[k]: continue
        b = np.unique([index[p] for p in baskets[u][0] if p in index]).astype(int); N += 1
        if b.size:
            cnt[b] += 1
            if b.size > 1: Cc[np.ix_(b, b)] += 1
    np.fill_diagonal(Cc, 0)
    return np.log((Cc + 2.0) / (np.outer(cnt, cnt) / N + 2.0))
LA, LB = build(half), build(~half); L = build(np.ones(len(users), bool))
iu = np.triu_indices(n, 1)
def rel(a, b): return float(np.corrcoef(a[iu], b[iu])[0, 1])
print("fiabilidad (correlación entre mitades de usuarios) de la matriz de log-lift:", round(rel(LA, LB), 3))
shared = (T @ T.T)[iu]; sb = (brand[:, None] == brand[None, :])[iu]; ss = (sub[:, None] == sub[None, :])[iu]; sc = (cat[:, None] == cat[None, :])[iu]
l = L[iu]
print("log-lift medio por nº de etiquetas compartidas:", {int(k): round(float(l[shared == k].mean()), 3) for k in (0, 1, 2)}, "· pares:", {int(k): int((shared == k).sum()) for k in (0, 1, 2)})
print("misma marca: %.3f (n=%d) vs distinta %.3f" % (l[sb].mean(), sb.sum(), l[~sb].mean()))
print("misma subcat %.3f · misma cat otra subcat %.3f · otra cat %.3f" % (l[ss].mean(), l[sc & ~ss].mean(), l[~sc].mean()))
for nm, m in (("misma subcat", ss), ("misma cat otra subcat", sc & ~ss), ("otra cat", ~sc)):
    print(f"   {nm}: por etiquetas compartidas", {int(k): round(float(l[m & (shared == k)].mean()), 3) for k in (0, 1, 2)})
# medias por par de subcategorías
def block_means(M):
    S = np.zeros((ns, ns)); cntb = np.zeros((ns, ns))
    oh = np.zeros((n, ns), np.float32); oh[np.arange(n), sub] = 1
    S = oh.T @ M @ oh; cntb = np.outer(oh.sum(0), oh.sum(0)); return S / np.maximum(cntb, 1)
BM = block_means(L); Lb = BM[sub][:, sub]
print("varianza explicada por el par de subcategorías: %.3f" % (1 - ((L - Lb)[iu] ** 2).mean() / l.var()))
BMc = np.zeros((cat.max() + 1, cat.max() + 1)); ohc = np.zeros((n, cat.max() + 1), np.float32); ohc[np.arange(n), cat] = 1
Lc = (ohc.T @ L @ ohc / np.outer(ohc.sum(0), ohc.sum(0)))[cat][:, cat]
print("varianza explicada por el par de categorías: %.3f" % (1 - ((L - Lc)[iu] ** 2).mean() / l.var()))
# tras quitar par de subcategorías: efecto de etiquetas y fiabilidad del residuo
R = L - Lb; RA = LA - block_means(LA)[sub][:, sub]; RB = LB - block_means(LB)[sub][:, sub]
r = R[iu]
print("residuo (sin par de subcat): por etiquetas compartidas", {int(k): round(float(r[shared == k].mean()), 3) for k in (0, 1, 2)}, "· misma marca %.3f" % r[sb].mean())
print("fiabilidad del residuo entre mitades: %.3f" % rel(RA, RB))
# quitar además el efecto medio de etiquetas compartidas y de pares de etiquetas (20x20)
TT = np.linalg.lstsq(np.column_stack([shared == 1, shared == 2, np.ones(len(r))]).astype(float), r, rcond=None)[0]
print("ajuste lineal del residuo con etiquetas compartidas:", np.round(TT, 3))
def tagpair_fit(Rm):
    # modelo: R_ij ~ t_i' W t_j  (W 20x20 simétrica)  -> mínimos cuadrados por bloques
    G = T.T @ Rm @ T; Nn = (T.T @ np.ones((n, n), np.float32) @ T)
    return G / np.maximum(Nn, 1)
W = tagpair_fit(R); pred = T @ W @ T.T / 4.0
print("matriz etiqueta-etiqueta W: diagonal media %.3f · fuera de la diagonal media %.3f · desv. fuera %.3f" % (np.diag(W).mean(), W[~np.eye(20, dtype=bool)].mean(), W[~np.eye(20, dtype=bool)].std()))
WA, WB = tagpair_fit(RA), tagpair_fit(RB)
print("fiabilidad de W entre mitades: %.3f (fuera de la diagonal: %.3f)" % (np.corrcoef(WA.ravel(), WB.ravel())[0, 1], np.corrcoef(WA[~np.eye(20, dtype=bool)], WB[~np.eye(20, dtype=bool)])[0, 1]))
R2 = R - pred * (np.sum(R * pred) / np.sum(pred * pred))
RA2 = RA - (T @ WA @ T.T / 4.0); RB2 = RB - (T @ WB @ T.T / 4.0)
print("fiabilidad del residuo tras quitar etiquetas (par de subcat + etiquetas): %.3f" % rel(RA2, RB2))
print("  por misma marca en ese residuo: %.3f vs %.3f" % (R2[iu][sb].mean(), R2[iu][~sb].mean()))
# ¿producto concreto? popularidad conjunta: efecto 'producto i' (media de fila)
rowm = R2.mean(1); print("efecto fila del residuo: desv %.3f · corr con log popularidad %.2f" % (rowm.std(), np.corrcoef(rowm, np.log(P["C_hist"] + 1))[0, 1]))
np.save(os.path.join(C, "W_tags.npy"), W)
