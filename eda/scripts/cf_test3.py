"""Estructura de la afinidad usuario-producto: ¿factores latentes (SVD) o vecinos (co-compra)? ¿Qué cesta?
¿Sirve la sesión / el carrito de hoy como cesta para los usuarios nuevos?"""
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
Bset, Kset = {}, {}
for u, (ps, es) in baskets.items():
    Bset[u] = np.unique([index[p] for p, e in zip(ps, es) if e == "purchase" and p in index]).astype(int)
    Kset[u] = np.unique([index[p] for p, e in zip(ps, es) if e != "purchase" and p in index]).astype(int)
BK = {u: np.union1d(Bset[u], Kset[u]) for u in users}
def cooc(sets):
    Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32)
    for b in sets.values():
        if b.size:
            cnt[b] += 1
            if b.size > 1: Cc[np.ix_(b, b)] += 1
    np.fill_diagonal(Cc, 0); return Cc, cnt
def lift_res(Cc, cnt, N, a, resid=True):
    L = np.log((Cc + a) / (np.outer(cnt, cnt) / N + a))
    if resid:
        for m in (same_s, same_c & ~same_s, ~same_c): L[m] -= L[m].mean()
    np.fill_diagonal(L, 0); return L.astype(np.float32)
N = len(users)
Ck, nk_ = cooc(BK)
Lk = lift_res(Ck, nk_, N, 2.0)                    # matriz con compras+carrito
print("matriz de afinidad (compras+carrito): desv. típica %.3f · p99 %.2f · p1 %.2f" % (Lk.std(), np.quantile(Lk[::7, ::7], 0.99), np.quantile(Lk[::7, ::7], 0.01)))
sv = np.linalg.svd(Lk[::2, ::2].astype(np.float64), compute_uv=False)
print("valores singulares (submuestra) top 12:", np.round(sv[:12] / sv[0], 3), "· energía top-5 %.2f top-10 %.2f top-20 %.2f top-50 %.2f" % tuple((sv[:k] ** 2).sum() / (sv ** 2).sum() for k in (5, 10, 20, 50)))
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); uid = np.repeat([r["u"] for r in fb], 10)
it = np.array([index.get(p, -1) for p in pid])
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb; df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); off = np.log(df.w.values)
def glm(cols, rows=None):
    X = np.column_stack([np.ones(len(df))] + [np.asarray(c, float) for c in cols]); yy, oo = y, off
    if rows is not None: X, yy, oo = X[rows], y[rows], off[rows]
    b = np.zeros(X.shape[1]); b[0] = np.log(max(yy.sum(), 1) / np.exp(oo).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + oo, -50, 20)); g = X.T @ (yy - mu) - 1e-4 * b; H = (X * mu[:, None]).T @ X + 1e-4 * np.eye(len(b))
        s = np.linalg.solve(H, g); mm = np.abs(s).max(); b = b + (s / mm if mm > 1 else s)
        if mm < 1e-8: break
    mu = np.exp(np.clip(X @ b + oo, -50, 20)); return b, np.sqrt(np.diag(np.linalg.inv(H))), float((yy * np.log(mu) - mu).sum())
base = [df[c].values for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values]
# cestas dependientes del tiempo: sesión (recent_views) y eventos en vivo anteriores a la petición
def agg(L, items, bs, k=3.0):
    ok = items >= 0
    out = np.zeros(len(items))
    if bs.size == 0 or not ok.any(): return out
    M = L[np.ix_(items[ok], bs)]
    M = np.where(items[ok][:, None] == bs[None, :], 0.0, M); nn = (items[ok][:, None] != bs[None, :]).sum(1)
    out[ok] = M.sum(1) / (nn + k); return out
s_hist = np.zeros(len(df)); s_ses = np.zeros(len(df)); s_lcart = np.zeros(len(df)); s_lview = np.zeros(len(df)); s_lbuy = np.zeros(len(df))
E = np.zeros(0, int)
for a, r in enumerate(fb):
    sl = slice(10 * a, 10 * a + 10); items = it[sl]; u, t = r["u"], int(r["t"])
    s_hist[sl] = agg(Lk, items, BK.get(u, E))
    rv = [index[v["product_id"]] for v in (r["rv"] or []) if isinstance(v, dict) and v.get("product_id") in index]
    s_ses[sl] = agg(Lk, items, np.unique(rv).astype(int), 1.0)
    e = ev.get(u)
    if e:
        lc = [index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "add_to_cart" and p in index]
        lv = [index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "view" and p in index]
        lb = [index[p] for p, k_, tt in zip(*e) if tt < t and k_ == "purchase" and p in index]
        s_lcart[sl] = agg(Lk, items, np.unique(lc).astype(int), 1.0); s_lview[sl] = agg(Lk, items, np.unique(lv).astype(int), 1.0)
        s_lbuy[sl] = agg(Lk, items, np.unique(lb).astype(int), 1.0)
np.savez_compressed(os.path.join(C, "cf_feats.npz"), s_hist=s_hist, s_ses=s_ses, s_lcart=s_lcart, s_lview=s_lview, s_lbuy=s_lbuy)
b0, s0, ll0 = glm(base); print(f"\nTODOS · base logver {ll0:.1f}")
def run(title, feats, rows=None, ref=None):
    b, s, ll = glm(base + feats, rows); print(f"  {title:52s} dlogver {ll-(ref if ref is not None else ll0):+7.1f} · " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b[11:], s[11:])))
    return ll
run("cesta histórica (compras+carrito)", [s_hist])
run("sesión (recent_views) como cesta", [s_ses])
run("carrito de hoy como cesta", [s_lcart])
run("vistas de hoy (shop.events) como cesta", [s_lview])
run("compras de hoy como cesta", [s_lbuy])
run("todas", [s_hist, s_ses, s_lcart, s_lview, s_lbuy])
for name, rows in (("USUARIOS NUEVOS", df.nu.values), ("USUARIOS CON HISTORIA", ~df.nu.values)):
    _, _, llr = glm(base, rows); print(f"{name} · {int(y[rows].sum())} compras · base logver {llr:.1f}")
    run("cesta histórica", [s_hist], rows, llr); run("sesión", [s_ses], rows, llr); run("carrito de hoy", [s_lcart], rows, llr)
    run("vistas de hoy", [s_lview], rows, llr); run("compras de hoy", [s_lbuy], rows, llr); run("todas", [s_hist, s_ses, s_lcart, s_lview, s_lbuy], rows, llr)
