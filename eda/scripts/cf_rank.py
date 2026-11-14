"""¿Basta una versión de rango bajo de la afinidad producto-producto (factores latentes)?"""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids)
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"])
same_s = sub[:, None] == sub[None, :]; same_c = cat[:, None] == cat[None, :]
users = list(baskets)
BK = {u: np.unique([index[p] for p in ps if p in index]).astype(int) for u, (ps, es) in baskets.items()}
Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32)
for b in BK.values():
    if b.size:
        cnt[b] += 1
        if b.size > 1: Cc[np.ix_(b, b)] += 1
np.fill_diagonal(Cc, 0)
L = np.log((Cc + 2.0) / (np.outer(cnt, cnt) / len(users) + 2.0)).astype(np.float64)
for m in (same_s, same_c & ~same_s, ~same_c): L[m] -= L[m].mean()
np.fill_diagonal(L, 0)
w, V = np.linalg.eigh(L); o = np.argsort(-np.abs(w)); w, V = w[o], V[:, o]
print("autovalores top 16:", np.round(w[:16], 0))
np.savez_compressed(os.path.join(C, "eig.npz"), w=w[:64], V=V[:, :64].astype(np.float32))
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); it = np.array([index.get(p, -1) for p in pid])
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
b0, s0, ll0 = glm(base); print(f"modelo base: logver {ll0:.1f} · referencia matriz completa: +284,7 (5 cestas)")
# cestas por petición (índices de producto)
E = np.zeros(0, int); bk = []
for r in fb:
    u, t = r["u"], int(r["t"]); e = ev.get(u); lc = lv = lb = E
    if e:
        lc = np.unique([index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "add_to_cart" and p in index]).astype(int)
        lv = np.unique([index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "view" and p in index]).astype(int)
        lb = np.unique([index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "purchase" and p in index]).astype(int)
    rv = np.unique([index[v["product_id"]] for v in (r["rv"] or []) if isinstance(v, dict) and v.get("product_id") in index]).astype(int)
    bk.append((BK.get(u, E), rv, lc, lv, lb))
pickle.dump(bk, open(os.path.join(C, "bk.pkl"), "wb"), protocol=4)
SHR = (3.0, 1.0, 1.0, 1.0, 1.0)
tagf = pickle.load(open(os.path.join(C, "tag_feats.pkl"), "rb"))["and"]
tags6 = [tagf[(nm, 6.0)] for nm in ["hist_compras", "hist_carrito", "sesion", "hoy_carrito", "hoy_vistas", "hoy_compras"]]
_, _, ll_t = glm(base + tags6); print(f"solo etiquetas: {ll_t - ll0:+.1f}")
for k in (2, 4, 6, 8, 12, 16, 24, 40, 64):
    F = V[:, :k] * np.sqrt(np.abs(w[:k])); sg = np.sign(w[:k])
    S = [np.zeros(len(df)) for _ in range(5)]; Sp = np.zeros(len(df))
    for a, bs in enumerate(bk):
        items = it[10 * a:10 * a + 10]; ok = items >= 0
        if not ok.any(): continue
        Fi = F[items[ok]]
        tot = np.zeros(k); ntot = 0
        for c, b in enumerate(bs):
            if b.size == 0: continue
            g = F[b].sum(0)                                  # suma de los vectores de la cesta
            # quitar el propio producto si está en la cesta
            inb = np.isin(items[ok], b)
            num = (Fi * sg) @ g - inb * ((Fi * sg) * Fi).sum(1)
            S[c][10 * a + np.flatnonzero(ok)] = num / (b.size - inb + SHR[c])
    b, s, ll = glm(base + S); b2, s2, ll2 = glm(base + S + tags6)
    print(f"rango {k:3d}: dlogver {ll-ll0:+7.1f} · con etiquetas {ll2-ll0:+7.1f} · coef " + " ".join(f"{bi:+.2f}" for bi in b[11:]))
