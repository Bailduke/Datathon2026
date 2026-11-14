"""¿Ordenaría mejor un modelo de árboles (XGBoost) que el log-lineal, con las MISMAS variables?
Ajuste con el primer 70 % del tiempo, evaluación en el 30 % final."""
import os, numpy as np, xgboost as xgb
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "../lab/.cache/base_arrays.npz"), allow_pickle=True)
A, y, off, rid, t = z["A"][:, :19].astype(np.float64), z["y"].astype(float), z["off"].astype(float), z["rid"], z["t"].astype(float)
names = ["const", "nuevo", "home", "logP", "carrito_hoy", "carrito_hist", "ult_visto", "visto_antes", "ya_comprado", "LCxlogP", "sinP",
         "cf_hist", "cf_ses", "cf_carr", "cf_vis", "tag_hist", "tag_ses", "tag_carr", "tag_vis"]
cut, cut_val = np.quantile(t, 0.7), np.quantile(t, 0.56)
tr, va, te = t < cut_val, (t >= cut_val) & (t < cut), t >= cut
trva = t < cut
print(f"impresiones: ajuste {int(trva.sum())} ({int(y[trva].sum())} compras) · evaluación {int(te.sum())} ({int(y[te].sum())} compras)")
def glm(X, yy, oo, l2=1e-4):
    b = np.zeros(X.shape[1]); b[0] = np.log(yy.sum() / np.exp(oo).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + oo, -50, 20)); g = X.T @ (yy - mu) - l2 * b; H = (X * mu[:, None]).T @ X + l2 * np.eye(len(b))
        s = np.linalg.solve(H, g); m = np.abs(s).max(); b = b + (s / m if m > 1 else s)
        if m < 1e-8: break
    return b
def report(name, eta_te):
    """eta_te: log de la tasa prevista (con el offset de posición) en evaluación."""
    mu = np.exp(eta_te); ll = float((y[te] * eta_te - mu).sum())
    r = rid[te]; o = np.argsort(r, kind="stable"); r, e, yy = r[o], eta_te[o], y[te][o]
    b = np.flatnonzero(np.r_[True, r[1:] != r[:-1], True]); tot = n = hit = 0
    for a_, b_ in zip(b[:-1], b[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); mx = ee.max()
            tot += ee[k] - (mx + np.log(np.exp(ee - mx).sum())); hit += int(ee.argmax() == k); n += 1
    print(f"  {name:44s} logver Poisson {ll:9.1f} · orden: logver {tot:8.1f} ({tot/n:.4f} por compra) · acierta el comprado {100*hit/n:.1f} %")
    return tot
print("referencia: solo posición"); report("solo posición", off[te] + np.log(y[trva].sum() / np.exp(off[trva]).sum()))
b = glm(A[trva], y[trva], off[trva]); base = report("log-lineal (el nuestro)", A[te] @ b + off[te])
F = A[:, 1:]
def fit_xgb(**kw):
    p = dict(objective="count:poisson", eta=0.05, max_depth=3, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=5.0, max_delta_step=0.7, tree_method="hist", nthread=8, seed=1, eval_metric="poisson-nloglik"); p.update(kw)
    dtr = xgb.DMatrix(F[tr], y[tr], base_margin=off[tr] + np.log(y[tr].sum() / np.exp(off[tr]).sum()))
    dva = xgb.DMatrix(F[va], y[va], base_margin=off[va] + np.log(y[tr].sum() / np.exp(off[tr]).sum()))
    m = xgb.train(p, dtr, 3000, [(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    n_it = m.best_iteration + 1
    c = np.log(y[trva].sum() / np.exp(off[trva]).sum())
    m2 = xgb.train(p, xgb.DMatrix(F[trva], y[trva], base_margin=off[trva] + c), int(n_it * 1.15))
    return m2.predict(xgb.DMatrix(F[te], base_margin=off[te] + c), output_margin=True), n_it, m2
for d in (1, 2, 3, 4, 6):
    eta_te, n_it, m = fit_xgb(max_depth=d)
    report(f"XGBoost profundidad {d} ({n_it} árboles)", eta_te)
# XGBoost sobre el residuo del log-lineal: ¿queda algo que aprender?
c = A @ b + off
p = dict(objective="count:poisson", eta=0.03, max_depth=3, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
         max_delta_step=0.7, tree_method="hist", nthread=8, seed=1, eval_metric="poisson-nloglik")
m = xgb.train(p, xgb.DMatrix(F[tr], y[tr], base_margin=c[tr]), 2000, [(xgb.DMatrix(F[va], y[va], base_margin=c[va]), "val")], early_stopping_rounds=100, verbose_eval=False)
n_it = m.best_iteration + 1
m2 = xgb.train(p, xgb.DMatrix(F[trva], y[trva], base_margin=c[trva]), max(n_it, 1))
report(f"log-lineal + XGBoost sobre su residuo ({n_it} árboles)", m2.predict(xgb.DMatrix(F[te], base_margin=c[te]), output_margin=True))
