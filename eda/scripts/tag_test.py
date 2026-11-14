"""¿Explican las etiquetas de la descripción la afinidad usuario-producto? Perfil de etiquetas del usuario a partir
de cada 'cesta' (historia, sesión, carrito/vistas/compras de hoy) y puntuación del producto mostrado."""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache"); sys.path.insert(0, HERE)
from rt import q
from tags import parse, TAGS
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb")); baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb"))
tg = pickle.load(open(os.path.join(C, "tags.pkl"), "rb"))
for x in q("SELECT product_id::String AS p, toString(description) AS d FROM shop.catalog_updates WHERE change::String = 'new_product'", 120)["data"]:
    tg.setdefault(x["p"], parse(x["d"]))
K = len(TAGS)
def tv(ps):
    c = np.zeros(K)
    for p in ps:
        for t in tg.get(p, ()): c[t] += 1
    return c
hist_b = {u: tv([p for p, e in zip(ps, es) if e == "purchase"]) for u, (ps, es) in baskets.items()}
hist_k = {u: tv([p for p, e in zip(ps, es) if e != "purchase"]) for u, (ps, es) in baskets.items()}
ALPHAS = (2.0, 6.0, 20.0)
names = ["hist_compras", "hist_carrito", "sesion", "hoy_carrito", "hoy_vistas", "hoy_compras"]
S_or = {(nm, a): np.zeros(len(df)) for nm in names for a in ALPHAS}; S_and = {(nm, a): np.zeros(len(df)) for nm in names for a in ALPHAS}
Z = np.zeros(K)
m = 0
for r in fb:
    u, t = r["u"], int(r["t"]); shown = (r["shown"] or [])[:10]
    e = ev.get(u)
    lc = lv = lb = ()
    if e:
        lc = [p for p, k_, tt in zip(*e) if tt < t and k_ == "add_to_cart"]; lv = [p for p, k_, tt in zip(*e) if tt < t and k_ == "view"]
        lb = [p for p, k_, tt in zip(*e) if tt < t and k_ == "purchase"]
    cs = {"hist_compras": hist_b.get(u, Z), "hist_carrito": hist_k.get(u, Z),
          "sesion": tv([v.get("product_id") for v in (r["rv"] or []) if isinstance(v, dict)]),
          "hoy_carrito": tv(set(lc)), "hoy_vistas": tv(set(lv)), "hoy_compras": tv(set(lb))}
    for j, p in enumerate(shown):
        ts = tg.get(p)
        if not ts: continue
        for nm, c in cs.items():
            tot = c.sum()
            if tot == 0: continue
            # el propio producto no cuenta en su perfil (si está en la cesta)
            for a in ALPHAS:
                th = (c + a / K) / (tot + a)
                S_or[(nm, a)][m + j] = np.log(th[ts].sum() / (len(ts) / K))
                S_and[(nm, a)][m + j] = np.log(th[ts] * K).sum()
    m += 10
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
base = [df[c].values for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values]
b0, s0, ll0 = glm(base); print(f"modelo base: logver {ll0:.1f}   (referencia co-compra: historia +89,7 · sesión +153 · carrito hoy +183 · vistas hoy +161 · compras hoy +117 · todas +284,7)")
def run(title, feats):
    b, s, ll = glm(base + feats); print(f"  {title:50s} dlogver {ll-ll0:+7.1f} · " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b[11:], s[11:]))); return ll
for nm in names:
    for a in ALPHAS:
        run(f"{nm} · alfa {a} · suma de cuotas", [S_or[(nm, a)]]); run(f"{nm} · alfa {a} · suma de logs", [S_and[(nm, a)]])
a = 6.0
run("todas (suma de cuotas, alfa 6)", [S_or[(nm, a)] for nm in names]); run("todas (suma de logs, alfa 6)", [S_and[(nm, a)] for nm in names])
cf = np.load(os.path.join(C, "cf_feats.npz"))
ll_cf = run("co-compra, las 5", [cf[k] for k in ("s_hist", "s_ses", "s_lcart", "s_lview", "s_lbuy")])
run("etiquetas (logs) + co-compra", [S_and[(nm, a)] for nm in names] + [cf[k] for k in ("s_hist", "s_ses", "s_lcart", "s_lview", "s_lbuy")])
pickle.dump({"or": S_or, "and": S_and}, open(os.path.join(C, "tag_feats.pkl"), "wb"), protocol=4)
