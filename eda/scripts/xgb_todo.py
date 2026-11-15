"""¿Y si XGBoost sustituye a la parte jerárquica? Mismas impresiones (877 mil, 614 compras), ajuste con el primer 70 % del
tiempo y evaluación en el resto. Se compara el orden dentro de la lista."""
import os, pickle, numpy as np, pandas as pd, xgboost as xgb, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl")); pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb"))
cf = np.load(os.path.join(C, "cf_feats.npz")); tg = pickle.load(open(os.path.join(C, "tag_feats.pkl"), "rb"))["and"]
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
B = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - B
lp = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), np.log(1e-3)) - np.log(1e-3)
cls = np.column_stack([(df.lc_n > 0) * nb, ((df.lc_n == 0) & (df.hc_n > 0)) * nb, (df.k_rv == 0) * nb, (df.k_rv >= 1) * nb, B, df.pn.isna().astype(float)])
taste = np.column_stack([cf[k] for k in ("s_hist", "s_ses", "s_lcart", "s_lview")] + [tg[(nm, 6.0)] for nm in ("hist_carrito", "sesion", "hoy_carrito", "hoy_vistas")])
ctx = np.column_stack([df.nu.astype(float), (df.page == "home").astype(float)])
raw = np.column_stack([np.log(df["pop"] + 20), df.same_cat, df.same_sub, df.ses_cat, df.ses_sub, np.log1p(df.n_us), np.log1p(df.n_uc), np.log1p(df.n_u), df.n_rv,
                       (df.page == "category").astype(float), df.lc_n.clip(0, 5), df.hc_n.clip(0, 5), np.log1p(df.hv_n), np.log1p(df.lv_n), df.k_rv, df["new"],
                       np.log1p(df.u_live_n), pd.factorize(df["cat"])[0]])
y = df.y.values.astype(float); off = np.log(df.w.values); t = df.t.values.astype(float); rid = df.rid.values
c1, c0 = np.quantile(t, 0.7), np.quantile(t, 0.56); tr, va, te, trva = t < c0, (t >= c0) & (t < c1), t >= c1, t < c1
print(f"ajuste: {int(y[trva].sum())} compras · evaluación: {int(y[te].sum())} compras")
def order(eta):
    e = eta[te]; yy = y[te]; tot = n = hit = 0
    for a in range(0, len(e), 10):
        if yy[a:a + 10].sum() > 0:
            ee = e[a:a + 10]; k = int(yy[a:a + 10].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); hit += int(ee.argmax() == k); n += 1
    return tot, 100 * hit / n
def glm(X):
    A = np.column_stack([np.ones(len(X)), X]); b = np.zeros(A.shape[1]); b[0] = np.log(y[trva].sum() / np.exp(off[trva]).sum())
    for _ in range(300):
        mu = np.exp(np.clip(A[trva] @ b + off[trva], -50, 20)); g = A[trva].T @ (y[trva] - mu) - 1e-3 * b; H = (A[trva] * mu[:, None]).T @ A[trva] + 1e-3 * np.eye(len(b))
        s = np.linalg.solve(H, g); m = np.abs(s).max(); b = b + (s / m if m > 1 else s)
        if m < 1e-8: break
    return A @ b + off
def boost(X, depth):
    p = dict(objective="count:poisson", eta=0.05, max_depth=depth, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, max_delta_step=0.7, tree_method="hist", nthread=8, seed=1, eval_metric="poisson-nloglik")
    c = np.log(y[tr].sum() / np.exp(off[tr]).sum())
    m = xgb.train(p, xgb.DMatrix(X[tr], y[tr], base_margin=off[tr] + c), 3000, [(xgb.DMatrix(X[va], y[va], base_margin=off[va] + c), "v")], early_stopping_rounds=100, verbose_eval=False)
    n_it = m.best_iteration + 1; c = np.log(y[trva].sum() / np.exp(off[trva]).sum())
    m2 = xgb.train(p, xgb.DMatrix(X[trva], y[trva], base_margin=off[trva] + c), max(int(n_it * 1.15), 1))
    return m2.predict(xgb.DMatrix(X, base_margin=off + c), output_margin=True), n_it
def show(name, eta, extra=""):
    o, h = order(eta); print(f"  {name:66s} orden {o:8.1f} · acierta {h:.0f} %{extra}")
show("solo posición", off.copy())
show("NUESTRO: lineal con probabilidad jerárquica + clases + gustos", glm(np.column_stack([ctx, lp, cls, taste])))
show("lineal SIN jerárquica: variables crudas + clases + gustos", glm(np.column_stack([ctx, raw[:, :-1], cls, taste])))
for d in (2, 3, 4):
    e, n = boost(np.column_stack([ctx, raw, cls, taste]), d); show(f"XGBoost SIN jerárquica (profundidad {d}): crudas + clases + gustos", e, f" · {n} árboles")
for d in (2, 3):
    e, n = boost(np.column_stack([ctx, raw, cls, taste, lp]), d); show(f"XGBoost de todo, CON la jerárquica como una variable más (prof. {d})", e, f" · {n} árboles")
