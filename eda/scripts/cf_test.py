"""¿Aporta la co-compra a nivel de producto (filtrado colaborativo item-item con la cesta histórica del usuario)
algo más que la probabilidad jerárquica? Se añade la puntuación al modelo de compra y se mira la logver."""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache"); sys.path.insert(0, HERE)
from rt import q
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids)
bp = os.path.join(C, "baskets.pkl")
if os.path.exists(bp):
    baskets = pickle.load(open(bp, "rb"))
else:
    baskets = {}
    for k in range(4):
        for x in q(f"""SELECT user_id::String AS u, groupArray(product_id::String) AS ps, groupArray(event_type::String) AS es FROM shop.history_events
                       WHERE event_type::String IN ('purchase','add_to_cart') AND cityHash64(user_id::String) % 4 = {k} GROUP BY u""", 600)["data"]:
            baskets[x["u"]] = (x["ps"], x["es"])
    pickle.dump(baskets, open(bp, "wb"), protocol=4)
users = list(baskets); urow = {u: k for k, u in enumerate(users)}
B = np.zeros((len(users), n), np.float32); K = np.zeros((len(users), n), np.float32)
for u, (ps, es) in baskets.items():
    r = urow[u]
    for p, e in zip(ps, es):
        i = index.get(p)
        if i is None: continue
        if e == "purchase": B[r, i] = 1
        else: K[r, i] = 1
print("usuarios con cesta:", len(users), "· compras distintas por usuario (media):", round(float(B.sum(1).mean()), 1), "· carritos:", round(float(K.sum(1).mean()), 1))
ni = B.sum(0) + 1.0
Cc = B.T @ B                                   # co-compras
np.fill_diagonal(Cc, 0)
lift = (Cc + 0.5) / (ni[:, None] * ni[None, :] / len(users) + 0.5)       # lift suavizado item-item
L = np.log(lift)
# misma subcategoría / categoría para separar lo que ya explica la jerarquía
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"])
same_s = sub[:, None] == sub[None, :]; same_c = cat[:, None] == cat[None, :]
print("log-lift medio: misma subcat %.2f · misma cat otra subcat %.2f · otra cat %.2f" % (L[same_s].mean(), L[same_c & ~same_s].mean(), L[~same_c].mean()))
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); uid = np.repeat([r["u"] for r in fb], 10)
cf_buy = np.zeros(len(df)); cf_res = np.zeros(len(df)); has = np.zeros(len(df))
Lres = L.copy()
for m in (same_s, same_c & ~same_s, ~same_c):   # residuo: lift relativo al típico de su nivel de parentesco
    Lres[m] -= L[m].mean()
cache = {}
for k in range(len(df)):
    u = uid[k]; i = index.get(pid[k]); r = urow.get(u)
    if i is None or r is None: continue
    if u not in cache:
        cache[u] = np.flatnonzero(B[r])
    b = cache[u]; b = b[b != i]
    if b.size == 0: continue
    has[k] = 1; cf_buy[k] = L[i, b].mean(); cf_res[k] = Lres[i, b].mean()
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb; df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); off = np.log(df.w.values)
def glm(cols):
    X = np.column_stack([np.ones(len(df))] + [np.asarray(c, float) for c in cols]); b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + off, -50, 20)); g = X.T @ (y - mu) - 1e-4 * b; H = (X * mu[:, None]).T @ X + 1e-4 * np.eye(len(b))
        s = np.linalg.solve(H, g); mm = np.abs(s).max(); b = b + (s / mm if mm > 1 else s)
        if mm < 1e-8: break
    mu = np.exp(np.clip(X @ b + off, -50, 20)); return b, np.sqrt(np.diag(np.linalg.inv(H))), float((y * np.log(mu) - mu).sum())
base = [df[c].values for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values]
b0, s0, ll0 = glm(base)
print(f"modelo base: logver {ll0:.1f} · impresiones con cesta: {100*has.mean():.0f} %")
for name, extra in [("tiene cesta", [has]), ("co-compra (log-lift medio con su cesta)", [has, cf_buy]),
                    ("co-compra residual (quitando cat/subcat)", [has, cf_res]), ("ambas", [has, cf_buy, cf_res])]:
    b, s, ll = glm(base + extra)
    print(f"  + {name:44s} dlogver {ll-ll0:+7.1f} · " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b[11:], s[11:])))
