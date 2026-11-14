import os, sys, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb"))
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
nopen = {u: len(c - P["bought"].get(u, frozenset())) for u, c in P["carted"].items()}
u_of = {r["rid"]: r["u"] for r in fb}
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
df["n_hc_u"] = np.repeat([nopen.get(u_of[r], 0) for r in df.rid.values[::10]], 10)
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb
df["HV"] = ((df.hv_n > 0) & (df.hc_n == 0) & (df.lc_n == 0)) * nb
df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); w = df.w.values
Z = {c: df[c].values.astype(float) for c in ["lp", "LC", "HC", "V0", "V1", "HV", "B", "NEW", "nuevo", "home"]}
lnh = np.log(np.maximum(df.n_hc_u.values, 1) / 10.0)

def rate_mult(p, exps=0):
    # p: const nuevo home theta | LC HC V0 V1 HV B NEW | [exps]
    th = p[3] + (p[11] * (Z["LC"] + Z["HC"]) if exps == 1 else p[11] * Z["LC"] + p[12] * Z["HC"] if exps == 2 else 0.0)
    extra = p[11] * Z["HC"] * lnh if exps == 3 else 0.0
    return np.exp(np.clip(p[0] + p[1] * Z["nuevo"] + p[2] * Z["home"] + th * Z["lp"] + p[4] * Z["LC"] + p[5] * Z["HC"] + p[6] * Z["V0"]
                          + p[7] * Z["V1"] + p[8] * Z["HV"] + p[9] * Z["B"] + p[10] * Z["NEW"] + extra, -40, 5))

def rate_add(p, dil=False):
    base = np.exp(np.clip(p[3] * Z["lp"] + p[8] * Z["HV"] + p[9] * Z["B"] + p[10] * Z["NEW"], -40, 20))
    hc = np.exp(p[5] + (p[11] * lnh if dil else 0.0)) * Z["HC"]
    bonus = (np.exp(p[4]) * Z["LC"] + hc + np.exp(p[6]) * Z["V0"] + np.exp(p[7]) * Z["V1"]) * (1 - Z["B"])
    return np.exp(np.clip(p[0] + p[1] * Z["nuevo"] + p[2] * Z["home"], -40, 5)) * (base + bonus)

def fit(fn, p0, title):
    def nll(p):
        mu = w * fn(p)
        return -float((y * np.log(mu + 1e-300) - mu).sum()) + 1e-4 * float(np.sum(np.asarray(p) ** 2))
    r = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 200000})
    r = minimize(nll, r.x, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 200000})
    print(f"{title:44s} logver {-r.fun:9.1f} · k={len(p0):2d} · AIC {2*r.fun + 2*len(p0):8.1f} · " + " ".join(f"{v:+.2f}" for v in r.x))
    return r.x

print("parámetros: const nuevo home theta | LC HC V0 V1 HV B NEW | extra")
p0 = [-8.8, 0.9, -1.0, 1.0, 2, 1.2, 1.9, 1.3, -0.6, -2.4, 1.0]
pm = fit(rate_mult, p0, "M  multiplicativo")
pm1 = fit(lambda p: rate_mult(p, 1), p0 + [0.0], "M1 mult. + exponente común del carrito")
pm2 = fit(lambda p: rate_mult(p, 2), p0 + [0.0, 0.0], "M2 mult. + exponente LC y HC")
pm3 = fit(lambda p: rate_mult(p, 3), p0 + [0.0], "M3 mult. + dilución por nº de carrito hist.")
pa = fit(rate_add, [-8.8, 0.9, -1.0, 1.0, 3.5, 2.5, 2.7, 1.0, -0.6, -2.4, 1.0], "A  aditivo")
pa2 = fit(lambda p: rate_add(p, True), [-8.8, 0.9, -1.0, 1.0, 3.5, 2.5, 2.7, 1.0, -0.6, -2.4, 1.0, 0.0], "A2 aditivo + dilución del carrito hist.")

print("\ncalibración por clase y tramo de P jerárquica: compras observadas / esperadas")
df["tr"] = pd.cut(df.pn.fillna(1e-9), [0, 0.001, 0.003, 0.01, 1], labels=["P<0,001", "0,001-0,003", "0,003-0,01", "P>0,01"])
preds = {"M": w * rate_mult(pm), "M1": w * rate_mult(pm1, 1), "A": w * rate_add(pa)}
for cls in ["LC", "HC", "V0", "V1"]:
    m = df[cls].values > 0
    g = pd.DataFrame({"tr": df.tr[m], "imp": 1, "obs": y[m], **{k: v[m] for k, v in preds.items()}}).groupby("tr", observed=True).sum()
    print(f"-- {cls}"); print(g.round(1).to_string())
m = (df.LC.values + df.HC.values + df.V0.values + df.V1.values + df.B.values + df.NEW.values) == 0
g = pd.DataFrame({"tr": df.tr[m], "imp": 1, "obs": y[m], **{k: v[m] for k, v in preds.items()}}).groupby("tr", observed=True).sum()
print("-- productos sin señal"); print(g.round(1).to_string())
print("\ncarrito histórico según cuántos productos tiene el usuario en el carrito (conv. por unidad de examen)")
h = df[df.HC > 0].copy(); h["n"] = pd.cut(h.n_hc_u, [0, 3, 8, 15, 30, 1000], right=False)
print(h.groupby("n", observed=True).agg(imp=("y", "size"), compras=("y", "sum"), exam=("w", "sum")).assign(conv=lambda t: (100 * t.compras / t.exam).round(3)).to_string())
