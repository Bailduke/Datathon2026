"""¿Cómo entran las señales usuario-producto (carrito de hoy, carrito histórico, visto en la sesión)
sobre la probabilidad jerárquica P_j? Compara estructuras con (A) Poisson sobre todas las impresiones y
(B) verosimilitud de la compra dentro de la lista, con validación cruzada."""
import os, sys, pickle, numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb"))
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float)
nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb
df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb
df["V1"] = (df.k_rv >= 1) * nb
df["HV"] = ((df.hv_n > 0) & (df.hc_n == 0) & (df.lc_n == 0)) * nb
df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3)   # centrado en P = 0,001
df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
df["logpop"] = np.log((df["pop"] + 20) / 300.0); df["afs"] = np.log1p(df.n_us); df["afc"] = np.log1p(df.n_uc)
y = df.y.values.astype(float); off = np.log(df.w.values)
print("P jerárquica de lo mostrado: mediana %.5f · p90 %.4f · p99 %.3f" % tuple(np.nanquantile(df.pn, [0.5, 0.9, 0.99])))

def glm(cols, l2=1e-4):
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) if isinstance(c, str) else c[1] for c in cols])
    names = ["const"] + [c if isinstance(c, str) else c[0] for c in cols]
    b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + off, -50, 20))
        g = X.T @ (y - mu) - l2 * b; H = (X * mu[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g); m = np.abs(step).max()
        b = b + (step / m if m > 1 else step)
        if m < 1e-8: break
    mu = np.exp(np.clip(X @ b + off, -50, 20)); se = np.sqrt(np.diag(np.linalg.inv(H)))
    return names, b, se, float((y * np.log(mu) - mu).sum())

def show(cols, title):
    names, b, se, ll = glm(cols)
    print(f"\n== {title} · logver {ll:.1f} · {len(b)} parámetros")
    print("   " + " | ".join(f"{n} {bi:+.2f}±{si:.2f}" for n, bi, si in zip(names, b, se)))
    return ll

lst = ["nuevo", "home"]
cls = ["LC", "HC", "V0", "V1", "HV", "B", "NEW"]
show(lst + ["lp"], "A0 · solo P jerárquica (theta = coef de lp)")
show(lst + ["lp"] + cls, "A1 · + clases (multiplicativo)")
inter = [("LCxlp", (df.LC * df.lp).values), ("HCxlp", (df.HC * df.lp).values), ("Vxlp", ((df.V0 + df.V1) * df.lp).values)]
show(lst + ["lp"] + cls + inter, "A2 · + exponente propio por clase")
show(lst + ["lp"] + cls + inter + ["same_cat", "logpop", "afs", "afc"], "A3 · + covariables planas (¿qué le falta a la jerárquica?)")
show(lst + cls + ["same_cat", "logpop", "afs", "afc"], "A4 · modelo plano sin P jerárquica")
inter2 = [("LCxnuevo", (df.LC * df.nuevo).values), ("HCxnuevo", (df.HC * df.nuevo).values), ("Vxnuevo", ((df.V0 + df.V1) * df.nuevo).values),
          ("lpxnuevo", (df.lp * df.nuevo).values), ("LC2", ((df.lc_n >= 2) * nb).values), ("LCxhome", (df.LC * df.home).values)]
show(lst + ["lp"] + cls + inter + inter2, "A5 · A2 + interacciones con usuario nuevo")

# ---------- (B) verosimilitud dentro de la lista, con validación cruzada
buy = df.groupby("rid").y.transform("sum").values > 0
d = df[buy].sort_values(["rid", "slot"])
n = len(d) // 10
G = lambda c: d[c].values.astype(float).reshape(n, 10)
A = {c: G(c) for c in ["lp", "LC", "HC", "V0", "V1", "HV", "B", "NEW", "w", "pn", "y"]}
yi = A["y"].argmax(1); rows = np.arange(n)
fold = np.random.default_rng(1).integers(0, 5, n)

def ll_of(logs, idx):
    s = np.log(A["w"][idx]) + logs[idx]
    s = s - s.max(1, keepdims=True)
    return float((s[np.arange(len(yi[idx])), yi[idx]] - np.log(np.exp(s).sum(1))).mean())

def mult(p, th_cls=False):
    lin = p[0] * A["LC"] + p[1] * A["HC"] + p[2] * A["V0"] + p[3] * A["V1"] + p[4] * A["HV"] + p[5] * A["B"] + p[6] * A["NEW"]
    th = p[7] * np.ones_like(A["lp"])
    if th_cls:
        th = th + p[8] * A["LC"] + p[9] * A["HC"] + p[10] * (A["V0"] + A["V1"])
    return th * A["lp"] + lin

def addi(p):
    base = np.exp(p[7] * A["lp"] + p[4] * A["HV"] + p[5] * A["B"] + p[6] * A["NEW"])
    return np.log(base + np.exp(p[0]) * A["LC"] + np.exp(p[1]) * A["HC"] + np.exp(p[2]) * A["V0"] + np.exp(p[3]) * A["V1"] + 1e-300)

def cv(fn, p0, title, fixed=None):
    def obj(p, idx):
        return -ll_of(fn(p), idx) + 1e-4 * float(np.sum(np.asarray(p) ** 2))
    test = []
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        r = minimize(obj, p0, args=(tr,), method="L-BFGS-B")
        test.append(ll_of(fn(r.x), te) * len(te))
    r = minimize(obj, p0, args=(rows,), method="L-BFGS-B")
    print(f"   {title:46s} LL dentro {ll_of(fn(r.x), rows):.4f} · LL validación cruzada {sum(test)/n:.4f} · " + " ".join(f"{v:+.2f}" for v in r.x))
    return r.x

print(f"\n== (B) compra dentro de la lista · {n} listas con compra")
print(f"   {'solo posición':46s} LL {ll_of(np.zeros_like(A['lp']), rows):.4f}")
print(f"   {'P jerárquica tal cual (theta = 1)':46s} LL {ll_of(A['lp'] - 20 * A['B'], rows):.4f}")
print("   parámetros: LC HC V0 V1 HV B NEW theta [theta_LC theta_HC theta_V]")
cv(lambda p: mult(np.r_[np.zeros(7), p]), [1.0], "solo theta")
cv(lambda p: mult(np.r_[np.zeros(5), p]), [-2.0, 0.0, 1.0], "theta + ya comprado + producto nuevo")
cv(mult, [2, 1, 1.5, 1, 0, -2, 0, 1.0], "multiplicativo por clase")
cv(lambda p: mult(p, True), [2, 1, 1.5, 1, 0, -2, 0, 1.0, 0, 0, 0], "multiplicativo + exponente propio por clase")
cv(addi, [-3, -5, -4, -4, 0, -2, 0, 1.0], "aditivo (bonus fijo por clase)")
