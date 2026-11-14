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
E = np.outer(cnt, cnt) / N
def resid(M):
    M = M.astype(np.float64).copy()
    for m in (same_s, same_c & ~same_s, ~same_c): M[m] -= M[m].mean()
    np.fill_diagonal(M, 0); return M
soft = lambda M, t: np.sign(M) * np.maximum(np.abs(M) - t, 0)
print("tabla l_ij: desviación típica %.3f · |l| mediana %.3f · p90 %.3f" % (L.std(), np.median(np.abs(L)), np.quantile(np.abs(L[::7, ::7]), 0.9)))
for e_ in (20.0, 40.0, 80.0, 160.0, 400.0): reps[f"Suavizado +{e_:.0f}"] = resid(np.log((Cc + e_) / (E + e_)))
w_, V_ = np.linalg.eigh(resid(np.log((Cc + 40.0) / (E + 40.0)))); o_ = np.argsort(-np.abs(w_))[:16]; F_ = V_[:, o_] * np.sqrt(np.abs(w_[o_]))
reps["Suavizado +40 y además PCA de 16 factores"] = (F_ * np.sign(w_[o_]), F_)
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
tagf = pickle.load(open(os.path.join(C, "tag_feats.pkl"), "rb"))["and"]
TG = np.column_stack([tagf[(nm, 6.0)] for nm in ["hist_compras", "hist_carrito", "sesion", "hoy_carrito", "hoy_vistas", "hoy_compras"]])
baseT = np.column_stack([base, TG]); bT = glm(baseT, y, off); llT = ll(baseT, bT, np.ones(len(df), bool)); oT, _ = order_ll(baseT, glm(baseT[tr], y[tr], off[tr]))
b0 = glm(base, y, off); ll0 = ll(base, b0, np.ones(len(df), bool)); bt = glm(base[tr], y[tr], off[tr]); o0, nn = order_ll(base, bt)
print(f"sin co-compra: logver {ll0:.1f} · orden fuera de muestra {o0:.1f} ({nn} compras) · solo etiquetas: dentro {llT - ll0:+.1f} · orden fuera {oT - o0:+.1f}")
SHR = (3.0, 1.0, 1.0, 1.0, 1.0)
for name, rep_ in reps.items():
    Fi, Fj = rep_ if isinstance(rep_, tuple) else (None, None); Lm = rep_
    t0 = time.time(); S = np.zeros((len(df), 5)); direct = Fi is None
    if not direct: self_ = (Fi * Fj).sum(1)
    for a, bs in enumerate(bk):
        items = it_[10 * a:10 * a + 10]; ok = items >= 0
        if not ok.any(): continue
        Fa = Fi[items[ok]] if not direct else None
        for c, b in enumerate(bs):
            if b.size == 0: continue
            inb = np.isin(items[ok], b)
            num = Lm[np.ix_(items[ok], b)].sum(1) if direct else Fa @ Fj[b].sum(0) - inb * self_[items[ok]]
            S[10 * a + np.flatnonzero(ok), c] = num / (b.size - inb + SHR[c])
    S /= (S[S != 0].std() + 1e-12)                        # misma escala para todas las representaciones
    Xf = np.column_stack([base, S]); bf = glm(Xf, y, off); bt = glm(Xf[tr], y[tr], off[tr]); o1, _ = order_ll(Xf, bt)
    XT = np.column_stack([baseT, S]); bfT = glm(XT, y, off); o1T, _ = order_ll(XT, glm(XT[tr], y[tr], off[tr]))
    print(f"  {name:52s} sin etiquetas: dentro {ll(Xf, bf, np.ones(len(df), bool)) - ll0:+7.1f} · orden fuera {o1 - o0:+6.1f}  |  con etiquetas: dentro {ll(XT, bfT, np.ones(len(df), bool)) - ll0:+7.1f} · orden fuera {o1T - o0:+6.1f}", flush=True)
