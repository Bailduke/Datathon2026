import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, ".cache", "imp_df.pkl"))
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
df["bought"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(int)
df["cart"] = (((df.lc_n > 0) | (df.hc_n > 0)) & (df.bought == 0)).astype(int)

def tab(d, by, title):
    g = d.groupby(by, observed=True).agg(imp=("y", "size"), exam=("w", "sum"), compras=("y", "sum"), clics=("clk", "sum"))
    g["conv_x_exam_%"] = (100 * g.compras / g.exam).round(3)
    g["clic_x_exam_%"] = (100 * g.clics / g.exam).round(2)
    d3 = d[d.slot < 3].groupby(by, observed=True).agg(imp_s13=("y", "size"), compras_s13=("y", "sum"))
    g = g.join(d3); g["conv_s13_%"] = (100 * g.compras_s13 / g.imp_s13).round(3)
    g["exam"] = g.exam.round(0)
    print("\n== " + title); print(g.to_string())

tab(df, "slot", "por posición (todas las listas)")
tab(df, "page", "por página")
tab(df, ["nu"], "usuario nuevo")
tab(df, ["bought", "cart"], "ya comprado / en carrito sin comprar")

def bucket(x, edges, labels):
    return pd.cut(x, edges, labels=labels, right=False)

nb = df[df.bought == 0].copy()
nb["carrito_vivo"] = bucket(nb.lc_dt, [0, 120, 600, 1800, 3600, 3 * 3600, 1e8, 2e9],
                            ["<2min", "2-10min", "10-30min", "30-60min", "1-3h", ">3h", "sin carrito vivo"])
tab(nb, "carrito_vivo", "NO comprados: tiempo desde el último 'añadir al carrito' EN VIVO")
nb["carrito_hist"] = np.where(nb.lc_n > 0, "tiene carrito vivo", np.where(nb.hc_n == 0, "nunca", np.where(nb.hc_days < 2, "hist <2d", np.where(nb.hc_days < 14, "hist 2-14d", "hist >14d"))))
tab(nb, "carrito_hist", "NO comprados: carrito SOLO histórico (días desde el último añadido)")
nc = nb[(nb.lc_n == 0)].copy()
nc["vista_vivo"] = bucket(nc.lv_dt, [0, 120, 600, 1800, 3600, 3 * 3600, 1e8, 2e9],
                          ["<2min", "2-10min", "10-30min", "30-60min", "1-3h", ">3h", "sin vista en vivo"])
tab(nc, "vista_vivo", "NO comprados y SIN carrito vivo: tiempo desde la última VISTA en vivo del producto")
tab(nc, "k_rv", "idem: posición en recent_views (0 = último visto; -1 = no está)")
nn = nc[(nc.lv_n == 0) & (nc.k_rv < 0)].copy()
tab(nn, ["page", "same_cat", "same_sub"], "sin carrito vivo ni vista en vivo: relación con el último producto visto")
nn["hist"] = np.where(nn.hc_n > 0, "carrito hist", np.where(nn.hv_n > 0, "vista hist", "nada"))
tab(nn, ["hist"], "sin señal en vivo: historia con el producto")
