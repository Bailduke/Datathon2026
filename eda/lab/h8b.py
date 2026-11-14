"""H8b · ¿Cambian los pesos de producto al controlar la caída horaria y la fatiga del usuario? ¿Mejora el orden?"""
import numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99); R = base_replay(P, fb, events, cat)
X, PN, AF, KI = R["X"], R["PN"], R["AF"], R["KI"]; z = np.load(os.path.join(C, "h8.npz"))
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; off = np.log(w); A = rv.glm_design(X, PN, AF)
rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0; mv = np.array([fb[j]["mv"] for j in rid])
ctl = np.column_stack([z["hrs"], np.log1p(z["n_req"])])
r, b1, se1 = evaluate2(A, np.column_stack([A, ctl]), y, off, t, rid, nu, "controles de lista: horas + peticiones previas")
b0, se0, _ = fit(A, y, off)
print("coeficientes de producto sin / con controles:")
for n_, a, sa, c, sc in zip(rv.GLM_COLS, b0, se0, b1, se1): print(f"   {n_:16s} {a:+.3f}±{sa:.3f} -> {c:+.3f}±{sc:.3f}")
# ¿acierta ahora el nivel de v2 fuera de muestra?
pre = t < V2_START; A1 = np.column_stack([A, ctl]); bp, _, _ = fit(A1[pre], y[pre], off[pre]); mu = np.exp(np.clip(A1 @ bp + off, -50, 20))
for nm, m in (("v2 todas", mv == "tengo-aurilla-v2"), ("v2 nuevos", (mv == "tengo-aurilla-v2") & nu), ("v2 historia", (mv == "tengo-aurilla-v2") & ~nu)):
    print(f"   con controles, ajustado antes de v2 · {nm:12s} obs {int(y[m].sum())} · esp {mu[m].sum():.1f} · obs/esp {y[m].sum()/mu[m].sum():.2f}")
print("repetición del producto, con controles:")
A1 = np.column_stack([A, ctl]); s = z["seen_n"]
evaluate2(A1, np.column_stack([A1, np.log1p(s)]), y, off, t, rid, nu, "log(1+veces ya mostrado)")
evaluate2(A1, np.column_stack([A1, (s >= 2) & (s <= 3), s >= 4]), y, off, t, rid, nu, "mostrado 2-3 veces / 4+ veces")
evaluate2(A1, np.column_stack([A1, s >= 4]), y, off, t, rid, nu, "mostrado 4+ veces")
