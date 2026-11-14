"""H2 · ¿Acierta el modelo de compra con las listas del propio v2? ¿Cambian los pesos al incluirlas?"""
import sys, numpy as np
from lab import *
P = load_P(); rt = rv.make_rt(); rv.load_affinity(rt, P, rv.CFG)
fb, events, cat = get_data(refresh="--refresh" in sys.argv)
R = base_replay(P, fb, events, cat)
X, PN, AF, KI = R["X"], R["PN"], R["AF"], R["KI"]
t = np.array([fb[k]["t"] for k in KI[:, 0]]); mv = np.array([fb[k]["mv"] for k in KI[:, 0]]); nu = X[:, 2] > 0
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); A = rv.glm_design(X, PN, AF)
v2 = mv == "tengo-aurilla-v2"; pre = t < V2_START
print(f"listas {len(fb)} · compras {int(y.sum())} · listas v2 {v2.sum()//10} con {int(y[v2].sum())} compras · última petición {time.strftime('%H:%M', time.localtime(t.max()/1000))}")
b_pre, se_pre, _ = fit(A[pre], y[pre], off[pre]); b_all, se_all, _ = fit(A, y, off)
mu = np.exp(np.clip(A @ b_pre + off, -50, 20))
def line(name, m):
    n = m.sum() / 10; o = y[m].sum(); e = mu[m].sum()
    print(f"  {name:34s} listas {n:7.0f} · real {100*o/n:.2f} % ± {100*np.sqrt(max(o,1))/n:.2f} · previsto {100*e/n:.2f} %  (obs/esp {o/e:.2f})")
print("Modelo ajustado SOLO con lo anterior a v2; previsión sobre las listas que v2 mostró de verdad:")
line("v2 · todas", v2); line("v2 · usuarios nuevos", v2 & nu); line("v2 · con historia", v2 & ~nu)
line("v0 (dentro de muestra) · todas", mv == "tengo-aurilla-v0"); line("resto anterior (dentro de muestra)", pre & (mv != "tengo-aurilla-v0"))
# calibración por tramo de tasa prevista dentro de v2
q = np.quantile(mu[v2] / np.exp(off[v2]), [0, .5, .8, .95, .99, 1.0]); r_ = mu / np.exp(off)
print("v2, por tramo de tasa prevista (por unidad de atención):")
for a, b_ in zip(q[:-1], q[1:]):
    m = v2 & (r_ >= a) & (r_ <= b_); print(f"   tasa {100*a:.3f}–{100*b_:.3f} %: imp {m.sum():6d} · obs {int(y[m].sum()):3d} · esp {mu[m].sum():6.1f}")
# por clase en v2
for nm, c in (("carrito hoy", 4), ("carrito antiguo", 5), ("último visto", 6), ("visto antes", 7)):
    m = v2 & (X[:, c] > 0); print(f"   v2 {nm:16s} imp {m.sum():6d} · obs {int(y[m].sum()):3d} · esp {mu[m].sum():6.1f}")
m = v2 & (X[:, 4:8].sum(1) == 0); print(f"   v2 sin clase         imp {m.sum():6d} · obs {int(y[m].sum()):3d} · esp {mu[m].sum():6.1f}")
print("por posición en v2: obs/esp", " ".join(f"{int(y[v2 & (X[:,0]==s)].sum())}/{mu[v2 & (X[:,0]==s)].sum():.1f}" for s in range(10)))
print("coeficientes: antes de v2 -> con todo")
for n_, a, sa, c, sc in zip(rv.GLM_COLS, b_pre, se_pre, b_all, se_all):
    print(f"   {n_:16s} {a:+.3f}±{sa:.3f} -> {c:+.3f}±{sc:.3f}")
