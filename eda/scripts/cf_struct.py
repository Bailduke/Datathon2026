import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache"); sys.path.insert(0, HERE)
from rt import q
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids)
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"])
same_s = sub[:, None] == sub[None, :]; same_c = cat[:, None] == cat[None, :]
BK = {}
for u, (ps, es) in baskets.items():
    BK[u] = np.unique([index[p] for p in ps if p in index]).astype(int)
Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32)
for b in BK.values():
    if b.size:
        cnt[b] += 1
        if b.size > 1: Cc[np.ix_(b, b)] += 1
np.fill_diagonal(Cc, 0)
N = len(BK)
L = np.log((Cc + 2.0) / (np.outer(cnt, cnt) / N + 2.0))
for m in (same_s, same_c & ~same_s, ~same_c): L[m] -= L[m].mean()
np.fill_diagonal(L, 0)
row = q("SELECT * FROM shop.catalog LIMIT 1", 60)["data"][0]
print("campos del catálogo:", {k: row[k] for k in row})
cols = [k for k in row if k not in ("product_id",)]
catdf = pd.DataFrame(q("SELECT * FROM shop.catalog", 120)["data"]).set_index("product_id").loc[ids]
w, V = np.linalg.eigh(L.astype(np.float64))
o = np.argsort(-np.abs(w)); w, V = w[o], V[:, o]
print("autovalores top 10:", np.round(w[:10], 1))
lp = np.log(np.maximum(catdf["price_eur"].astype(float).values, 0.5))
# posición de precio dentro de la subcategoría
pos = pd.Series(lp).groupby(sub).rank(pct=True).values
lpop = np.log(cnt + 1)
for k in range(5):
    v = V[:, k]
    line = f"vector {k+1} (autovalor {w[k]:.1f}): corr log precio {np.corrcoef(v, lp)[0,1]:+.2f} · posición de precio en subcat {np.corrcoef(v, pos)[0,1]:+.2f} · log popularidad {np.corrcoef(v, lpop)[0,1]:+.2f}"
    for c in catdf.columns:
        if catdf[c].dtype == object and catdf[c].nunique() < 400:
            g = pd.Series(v).groupby(catdf[c].values)
            eta = ((g.mean() - v.mean()) ** 2 * g.size()).sum() / ((v - v.mean()) ** 2).sum()
            line += f" · eta2 {c} {eta:.2f}"
    print(line)
# ¿bloques? signo de los dos primeros vectores -> 4 grupos
grp = (V[:, 0] > 0).astype(int) * 2 + (V[:, 1] > 0).astype(int)
print("\nafinidad media entre grupos definidos por el signo de los 2 primeros vectores (tamaños", np.bincount(grp), "):")
print(np.round([[L[np.ix_(grp == a, grp == b)].mean() for b in range(4)] for a in range(4)], 3))
# precio: afinidad media por terciles de posición de precio
ter = np.minimum((pos * 3).astype(int), 2)
print("afinidad media por tercil de precio (barato, medio, caro):"); print(np.round([[L[np.ix_(ter == a, ter == b)].mean() for b in range(3)] for a in range(3)], 3))
qp = np.minimum((pd.Series(lp).rank(pct=True).values * 5).astype(int), 4)
print("afinidad media por quintil de precio absoluto:"); print(np.round([[L[np.ix_(qp == a, qp == b)].mean() for b in range(5)] for a in range(5)], 3))
