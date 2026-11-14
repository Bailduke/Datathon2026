import numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99); R = pickle.load(open(os.path.join(C, "price.pkl"), "rb"))
X, PN, AF, KI, CT = R["X"], R["PN"], R["AF"], R["KI"], R["CT"].astype(float)
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
v2 = np.array([fb[j]["mv"] for j in rid]) == "tengo-aurilla-v2"; pre = t < V2_START; G = rv.glm_design(X, PN, AF); hrs = (t - events[0][0]) / 3.6e6
for nm, A in (("sin controles", G), ("3 controles (peticiones, vistos, carrito)", np.column_stack([G, CT])), ("+ hora (4)", np.column_stack([G, CT, hrs])), ("solo hora + peticiones", np.column_stack([G, hrs, CT[:, 0]]))):
    b, _, _ = fit(A[pre], y[pre], off[pre]); mu = np.exp(np.clip(A @ b + off, -50, 20))
    print(f"{nm:44s} obs/esp del v2: todas {y[v2].sum()/mu[v2].sum():.2f} · nuevos {y[v2&nu].sum()/mu[v2&nu].sum():.2f} · historia {y[v2&~nu].sum()/mu[v2&~nu].sum():.2f}"
          + " · por hora desde su arranque: " + " ".join(f"{y[v2&(t>=V2_START+h*3.6e6)&(t<V2_START+(h+1)*3.6e6)].sum()/max(mu[v2&(t>=V2_START+h*3.6e6)&(t<V2_START+(h+1)*3.6e6)].sum(),1e-9):.2f}" for h in range(3)))
A4 = np.column_stack([G, CT, hrs]); b, se, _ = fit(A4, y, off); print("coef de la hora con todo: %+.3f ± %.3f por hora" % (b[-1], se[-1]))
for h in range(3):
    m = v2 & (t >= V2_START + h * 3.6e6) & (t < V2_START + (h + 1) * 3.6e6)
    if m.sum(): print(f"   v2 hora {h}: {m.sum()//10} listas · conversión real {1000*y[m].sum()/m.sum():.2f} %")
