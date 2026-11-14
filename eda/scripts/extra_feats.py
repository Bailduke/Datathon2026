"""¿Queda señal sin usar? Sobre el modelo de compra (P jerárquica + clases de afinidad) se añaden, de una en una,
variables de producto y de contexto, y se mira cuánto mejora la log-verosimilitud."""
import os, sys, pickle, bisect, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
sys.path.insert(0, HERE)
from rt import q
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
df["pid"] = np.concatenate([(r["shown"] or [])[:10] for r in fb])
df["dev"] = np.repeat([r["dev"] for r in fb], 10)
price0 = {p: float(x) for p, x in zip(P["ids"], P["price"])}
ch = {}
for x in q("SELECT product_id::String AS p, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t, price_eur::Float64 AS price, change::String AS c FROM shop.catalog_updates WHERE change::String IN ('price_change','new_product') ORDER BY ts::DateTime64(3)", 120)["data"]:
    ch.setdefault(x["p"], []).append((int(x["t"]), float(x["price"])))
    if x["c"] == "new_product": price0.setdefault(x["p"], float(x["price"]))
def price_at(p, t):
    L = ch.get(p)
    if not L: return price0.get(p, np.nan)
    k = bisect.bisect_right(L, (t, 1e18)) - 1
    return L[k][1] if k >= 0 else price0.get(p, np.nan)
df["price"] = [price_at(p, t) for p, t in zip(df.pid.values, df.t.values)]
df["price0"] = df.pid.map(price0)
df["sale"] = (df.price < df.price0 - 1e-9).astype(float)
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb
df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); off = np.log(df.w.values)
def glm(cols):
    X = np.column_stack([np.ones(len(df))] + [np.asarray(c[1], float) for c in cols]); b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + off, -50, 20)); g = X.T @ (y - mu) - 1e-4 * b; H = (X * mu[:, None]).T @ X + 1e-4 * np.eye(len(b))
        s = np.linalg.solve(H, g); m = np.abs(s).max(); b = b + (s / m if m > 1 else s)
        if m < 1e-8: break
    mu = np.exp(np.clip(X @ b + off, -50, 20)); return b, np.sqrt(np.diag(np.linalg.inv(H))), float((y * np.log(mu) - mu).sum())
base = [(c, df[c].values) for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [("LCxlp", (df.LC * df.lp).values)]
b0, se0, ll0 = glm(base)
print(f"modelo base: logver {ll0:.1f}")
def test(title, extra):
    b, se, ll = glm(base + extra)
    k = len(base) + 1
    print(f"  + {title:44s} dlogver {ll - ll0:+7.1f} · " + " ".join(f"{n} {bi:+.2f}±{si:.2f}" for (n, _), bi, si in zip(extra, b[k:], se[k:])))
lpr = np.log(np.maximum(df.price.fillna(df.price.median()).values, 0.5))
test("log precio", [("logprecio", lpr - lpr.mean())])
test("rebajado ahora", [("rebaja", df.sale.values)])
test("rebajado x log precio", [("rebaja", df.sale.values), ("logprecio", lpr - lpr.mean())])
test("producto nuevo ya en 'NEW' -> popularidad", [("logpop", np.log((df["pop"].values + 20) / 300))])
test("misma categoría que el último visto", [("misma_cat", df.same_cat.values)])
test("misma subcategoría", [("misma_sub", df.same_sub.values)])
test("página category", [("pcat", (df.page == "category").values)])
test("device", [("ios", (df.dev == "ios").values), ("web", (df.dev == "web").values)])
test("nº de vistas en la sesión", [("n_rv", np.minimum(df.n_rv.values, 6))])
test("minutos desde la última vista (log)", [("ldt", np.log1p(np.minimum(df.dlast.values, 3600) / 60) * (df.n_rv.values > 0))])
test("actividad en vivo del usuario (log nº eventos)", [("lact", np.log1p(df.u_live_n.values))])
test("min desde su último evento en vivo (log)", [("lrec", np.log1p(np.minimum(df.u_dt.values, 36000) / 60))])
test("afinidad cat (log1p n_uc)", [("afc", np.log1p(df.n_uc.values))])
test("historia total (log1p n_u)", [("lnu", np.log1p(df.n_u.values))])
test("vista histórica sin carrito", [("HV", (((df.hv_n > 0) & (df.hc_n == 0) & (df.lc_n == 0)) * nb).values)])
test("vista en vivo sin carrito", [("LV", (((df.lv_n > 0) & (df.lc_n == 0)) * nb).values)])
test("nuevo x lp", [("nxlp", (df.nuevo * df.lp).values)])
test("nuevo x LC", [("nxLC", (df.nuevo * df.LC).values)])
test("LC añadido 2+ veces", [("LC2", ((df.lc_n >= 2) * nb).values)])
hrs = ((df.t.values // 3600000) % 24).astype(int)
test("hora del día (UTC)", [(f"h{h}", (hrs == h)) for h in sorted(set(hrs))[1:]])
test("posición libre (slots 2..10)", [(f"s{k}", (df.slot.values == k)) for k in range(1, 10)])
