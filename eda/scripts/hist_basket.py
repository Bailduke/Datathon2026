"""¿Qué define mejor el gusto histórico del usuario: lo que compró, lo que metió en el carrito, o el carrito abierto?"""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb"))
z = np.load(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/afinidad.npz"))
F, sg, tags = z["F"], z["sg"], z["tags"]; n = F.shape[0]; K = 20
Fs = F * sg; selfdot = (Fs * F).sum(1)
T = np.zeros((n, K), np.float32); T[np.arange(n), tags[:, 0]] = 1; T[np.arange(n), tags[:, 1]] = 1
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}
sets = {}
for u, (ps, es) in baskets.items():
    b = {index[p] for p, e in zip(ps, es) if e == "purchase" and p in index}; k = {index[p] for p, e in zip(ps, es) if e != "purchase" and p in index}
    sets[u] = (b, k)
pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); it = np.array([index.get(p, -1) for p in pid])
VAR = {"compras ∪ carrito (actual)": lambda b, k: b | k, "solo compras": lambda b, k: b, "solo carrito": lambda b, k: k,
       "carrito sin comprar": lambda b, k: k - b, "compras nunca en carrito... (control)": lambda b, k: b - k}
def scores(bset, items, shr=3.0, al=6.0):
    cf = np.zeros(len(items)); tg = np.zeros(len(items))
    ok = items >= 0
    if not bset or not ok.any(): return cf, tg
    b = np.fromiter(bset, np.int64, len(bset)); ii = items[ok]
    inb = np.isin(ii, b).astype(float)
    num = Fs[ii] @ F[b].sum(0) - inb * selfdot[ii]; cf[ok] = num / (b.size - inb + shr)
    cc = T[b].sum(0); tot = cc.sum()
    cl = cc[None, :] - inb[:, None] * T[ii]; tl = tot - 2 * inb
    tg[ok] = (T[ii] * np.log((cl + al / K) / (tl[:, None] + al) * K)).sum(1)
    return cf, tg
feats = {v: (np.zeros(len(df)), np.zeros(len(df))) for v in VAR}
for a, r in enumerate(fb):
    s = sets.get(r["u"])
    if s is None: continue
    sl = slice(10 * a, 10 * a + 10)
    for v, f in VAR.items():
        cf, tg = scores(f(*s), it[sl]); feats[v][0][sl] = cf; feats[v][1][sl] = tg
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
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
cfz = np.load(os.path.join(C, "cf_feats.npz")); tagf = pickle.load(open(os.path.join(C, "tag_feats.pkl"), "rb"))["and"]
ctrl = [cfz["s_ses"], cfz["s_lcart"], cfz["s_lview"]] + [tagf[(nm, 6.0)] for nm in ("sesion", "hoy_carrito", "hoy_vistas")]
base = [df[c].values for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values] + ctrl
b0, s0, ll0 = glm(base); print(f"base (con sesión y cestas de hoy): logver {ll0:.1f}"); k0 = len(base) + 1
for v in VAR:
    b, s, ll = glm(base + list(feats[v])); print(f"  cesta histórica = {v:38s} dlogver {ll-ll0:+7.1f} · co-compra {b[k0]:+.2f}±{s[k0]:.2f} · etiquetas {b[k0+1]:+.2f}±{s[k0+1]:.2f}")
b, s, ll = glm(base + list(feats["solo compras"]) + list(feats["carrito sin comprar"]))
print(f"  compras y carrito sin comprar por separado       dlogver {ll-ll0:+7.1f} · " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b[k0:], s[k0:])))
