"""Regresión de Poisson (enlace log, offset = peso de la posición) de la compra de cada producto mostrado.
exp(coef) = multiplicador de la probabilidad de compra con todo lo demás igual."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, ".cache", "imp_df.pkl"))
df["bought"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float)
nb = 1 - df.bought
F = pd.DataFrame(index=df.index)
F["const"] = 1.0
F["ya_comprado"] = df.bought
F["carrito_vivo"] = ((df.lc_n > 0) * nb)
F["carrito_hist"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
F["visto_ultimo"] = (df.k_rv == 0) * nb
F["visto_sesion_otro"] = (df.k_rv >= 1) * nb
F["vista_hist_sin_carrito"] = ((df.hv_n > 0) & (df.hc_n == 0) & (df.lc_n == 0)) * nb
F["vista_vivo_sin_carrito"] = ((df.lv_n > 0) & (df.lc_n == 0) & (df.k_rv < 0)) * nb
F["misma_cat_ultimo"] = df.same_cat.astype(float)
F["misma_subcat_ultimo"] = df.same_sub.astype(float)
F["pag_home"] = (df.page == "home").astype(float)
F["pag_category"] = (df.page == "category").astype(float)
F["usuario_nuevo"] = df.nu.astype(float)
F["log_pop"] = np.log((df["pop"] + 20) / 300.0)
F["producto_nuevo"] = df.new.astype(float)
F["afinidad_subcat"] = np.log1p(df.n_us)
F["afinidad_cat"] = np.log1p(df.n_uc) - np.log1p(df.n_us)
y = df.y.values.astype(float); off = np.log(df.w.values)

def fit(cols, l2=1e-4, iters=200):
    X = F[cols].values.astype(float); b = np.zeros(len(cols)); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(iters):
        mu = np.exp(X @ b + off)
        g = X.T @ (y - mu) - l2 * b; H = (X * mu[:, None]).T @ X + l2 * np.eye(len(cols))
        step = np.linalg.solve(H, g)
        m = np.abs(step).max()
        if m > 1.0: step = step / m                     # Newton amortiguado
        b = b + step
        if m < 1e-8: break
    mu = np.exp(X @ b + off); ll = float((y * np.log(mu) - mu).sum())
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se, ll

def show(cols, title):
    b, se, ll = fit(cols)
    print(f"\n== {title} · logver = {ll:.1f} · {len(cols)} parámetros")
    for c, bi, si in zip(cols, b, se):
        print(f"   {c:26s} coef {bi:+.3f} ± {si:.3f}   x{np.exp(bi):7.2f}   z={bi/si:+.1f}")
    return b, ll

base = list(F.columns)
b, ll_full = show(base, "modelo completo")

# interacciones del carrito
F["cv_x_misma_cat"] = F.carrito_vivo * F.misma_cat_ultimo
F["cv_x_logpop"] = F.carrito_vivo * F.log_pop
F["cv_x_nuevo"] = F.carrito_vivo * F.usuario_nuevo
F["cv_x_home"] = F.carrito_vivo * F.pag_home
F["cv_2mas"] = ((df.lc_n >= 2) * nb)
F["ch_x_misma_cat"] = F.carrito_hist * F.misma_cat_ultimo
F["ch_x_logpop"] = F.carrito_hist * F.log_pop
F["ch_2mas"] = ((df.lc_n == 0) & (df.hc_n >= 2)) * nb
F["ch_reciente14"] = ((df.lc_n == 0) & (df.hc_n > 0) & (df.hc_days < 14)) * nb
F["vs_x_logpop"] = (F.visto_ultimo + F.visto_sesion_otro) * F.log_pop
F["nuevo_x_logpop"] = F.usuario_nuevo * F.log_pop
F["nuevo_x_misma_cat"] = F.usuario_nuevo * F.misma_cat_ultimo
show(base + ["cv_x_misma_cat", "cv_x_logpop", "cv_x_nuevo", "cv_x_home", "cv_2mas", "ch_x_misma_cat", "ch_x_logpop", "ch_2mas",
             "ch_reciente14", "vs_x_logpop", "nuevo_x_logpop", "nuevo_x_misma_cat"], "con interacciones")
print("\ncobertura (impresiones / compras):")
for c in ["carrito_vivo", "carrito_hist", "visto_ultimo", "visto_sesion_otro", "misma_cat_ultimo", "usuario_nuevo", "producto_nuevo", "ya_comprado"]:
    m = F[c].values > 0
    print(f"   {c:22s} {int(m.sum()):7d} imp ({100*m.mean():5.2f} %)  {int(y[m].sum()):4d} compras ({100*y[m].sum()/y.sum():5.1f} %)")
