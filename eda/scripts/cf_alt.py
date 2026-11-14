"""Alternativas al PCA para resumir la co-compra en 16 números por producto. Misma prueba que cf_rank.py:
se calculan las puntuaciones de co-compra con cada representación (historia, sesión, carrito, vistas y compras
de hoy) y se mira cuánto mejora el modelo de compra (877 mil impresiones, 614 compras; 5 coeficientes en todas)."""
import os, pickle, numpy as np, pandas as pd, time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl")); pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb")); bk = pickle.load(open(os.path.join(C, "bk.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids); K = 16
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"]); same_s = sub[:, None] == sub[None, :]; same_c = cat[:, None] == cat[None, :]
users = list(baskets); BK = [np.unique([index[p] for p in baskets[u][0] if p in index]).astype(int) for u in users]
Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32)
for b in BK:
    if b.size:
        cnt[b] += 1
        if b.size > 1: Cc[np.ix_(b, b)] += 1
np.fill_diagonal(Cc, 0); N = len(users)
raw = np.log((Cc + 2.0) / (np.outer(cnt, cnt) / N + 2.0)).astype(np.float64)
L = raw.copy()
for m in (same_s, same_c & ~same_s, ~same_c): L[m] -= L[m].mean()
np.fill_diagonal(L, 0)
reps = {}
w, V = np.linalg.eigh(L); o = np.argsort(-np.abs(w))[:K]
F = V[:, o] * np.sqrt(np.abs(w[o])); reps["PCA de la tabla 'van juntos' (actual)"] = (F * np.sign(w[o]), F)
# solo autovalores positivos (PCA clásico, sin direcciones de repulsión)
o2 = np.argsort(-w)[:K]; F2 = V[:, o2] * np.sqrt(np.maximum(w[o2], 0)); reps["PCA solo con direcciones positivas"] = (F2, F2)
# NMF de la parte positiva (perfiles aditivos): actualizaciones multiplicativas
Pm = np.maximum(L, 0); rng = np.random.default_rng(0); W = rng.random((n, K)) * 0.1; H = rng.random((n, K)) * 0.1
for it in range(200):
    W *= (Pm @ H) / (W @ (H.T @ H) + 1e-9); H *= (Pm.T @ W) / (H @ (W.T @ W) + 1e-9)
reps["NMF (perfiles no negativos)"] = (W, H)
# factorización usuario-producto: SVD de la matriz usuario x producto normalizada (tf-idf + filas unitarias)
r_ = np.concatenate([np.full(b.size, k) for k, b in enumerate(BK)]); c_ = np.concatenate(BK)
idf = np.log(N / (cnt + 1.0)); val = idf[c_]; M = csr_matrix((val, (r_, c_)), shape=(N, n))
rn = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-9; M = csr_matrix((val / rn[r_], (r_, c_)), shape=(N, n))
u_, s_, vt = svds(M, k=K + 1); o3 = np.argsort(-s_)[1:]           # se descarta la primera dirección (popularidad)
Fu = vt[o3].T * s_[o3]; Fu = Fu / (np.linalg.norm(Fu, axis=1, keepdims=True) + 1e-9); reps["Factorización usuario-producto (SVD)"] = (Fu, Fu)
# k-means sobre el PCA: 16 perfiles discretos (afinidad = mismo perfil o no)
X = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9); cen = X[rng.choice(n, K, replace=False)]
for it in range(50):
    lab = (X @ cen.T).argmax(1); cen = np.array([X[lab == k].mean(0) if (lab == k).any() else cen[k] for k in range(K)]); cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9
Z = np.zeros((n, K)); Z[np.arange(n), lab] = 1; Z -= Z.mean(0); reps["16 grupos discretos (k-means)"] = (Z, Z)
print("tamaño de los grupos k-means:", np.bincount(lab, minlength=K).tolist())
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]]); pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); it_ = np.array([index.get(p, -1) for p in pid])
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb; df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb; df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); off = np.log(df.w.values); tt = df.t.values.astype(float); cut = np.quantile(tt, 0.7); tr, te = tt < cut, tt >= cut
def glm(X, yy, oo):
    b = np.zeros(X.shape[1]); b[0] = np.log(yy.sum() / np.exp(oo).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + oo, -50, 20)); g = X.T @ (yy - mu) - 1e-4 * b; H_ = (X * mu[:, None]).T @ X + 1e-4 * np.eye(len(b))
        s = np.linalg.solve(H_, g); m = np.abs(s).max(); b = b + (s / m if m > 1 else s)
        if m < 1e-8: break
    return b
def ll(X, b, mask): mu = np.exp(np.clip(X[mask] @ b + off[mask], -50, 20)); return float((y[mask] * np.log(mu) - mu).sum())
rid = df.rid.values
def order_ll(X, b):
    e = (X @ b + off)[te]; r = rid[te]; yy = y[te]; tot = 0.0; nn = 0
    for a in range(0, len(e), 10):
        if yy[a:a + 10].sum() > 0:
            ee = e[a:a + 10]; k = int(yy[a:a + 10].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); nn += 1
    return tot, nn
base = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values])
b0 = glm(base, y, off); ll0 = ll(base, b0, np.ones(len(df), bool)); bt = glm(base[tr], y[tr], off[tr]); o0, nn = order_ll(base, bt)
print(f"sin co-compra: logver {ll0:.1f} · orden fuera de muestra {o0:.1f} ({nn} compras)")
SHR = (3.0, 1.0, 1.0, 1.0, 1.0)
for name, (Fi, Fj) in reps.items():
    t0 = time.time(); S = np.zeros((len(df), 5)); self_ = (Fi * Fj).sum(1)
    for a, bs in enumerate(bk):
        items = it_[10 * a:10 * a + 10]; ok = items >= 0
        if not ok.any(): continue
        Fa = Fi[items[ok]]
        for c, b in enumerate(bs):
            if b.size == 0: continue
            inb = np.isin(items[ok], b); S[10 * a + np.flatnonzero(ok), c] = (Fa @ Fj[b].sum(0) - inb * self_[items[ok]]) / (b.size - inb + SHR[c])
    S /= (S[S != 0].std() + 1e-12)                        # misma escala para todas las representaciones
    Xf = np.column_stack([base, S]); bf = glm(Xf, y, off); bt = glm(Xf[tr], y[tr], off[tr]); o1, _ = order_ll(Xf, bt)
    print(f"  {name:44s} mejora dentro de muestra {ll(Xf, bf, np.ones(len(df), bool)) - ll0:+7.1f} · mejora del orden fuera de muestra {o1 - o0:+6.1f} · {time.time()-t0:.0f} s", flush=True)
