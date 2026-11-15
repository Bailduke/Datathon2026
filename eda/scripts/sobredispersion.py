"""¿Las compras de cada producto siguen una Poisson con tasa q_i · T (T = compras de la tienda)? Índice de dispersión
(chi-cuadrado de Pearson / grados de libertad; 1 = Poisson) con el histórico por día y con el stream por cuarto de hora."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rt import q
def disp(M, label):
    """M: productos x periodos (recuentos). Esperado = total del producto x cuota del periodo."""
    M = M[M.sum(1) >= 30]; T = M.sum(0) / M.sum(); E = np.outer(M.sum(1), T)
    chi = ((M - E) ** 2 / np.maximum(E, 1e-9)).sum(1) / (M.shape[1] - 1)
    print(f"{label}: {M.shape[0]} productos x {M.shape[1]} periodos · dispersión mediana {np.median(chi):.2f} · media {chi.mean():.2f} · p90 {np.quantile(chi, .9):.2f} · p99 {np.quantile(chi, .99):.2f} · % de productos > 1,5: {100*(chi > 1.5).mean():.0f} %")
    return chi, M, E
r = q("""SELECT product_id::String AS p, toDate(ts::DateTime64(3)) AS d, count() AS n FROM shop.history_events WHERE event_type::String = 'purchase' GROUP BY p, d""", 300)
h = pd.DataFrame(r["data"]); H = h.pivot(index="p", columns="d", values="n").fillna(0).values.astype(float)
H = H[:, 1:-1]                                              # sin el primer y el último día, que están incompletos
chi_h, Mh, Eh = disp(H, "HISTÓRICO, por día")
tot = Mh.sum(1)
for lo, hi in ((30, 100), (100, 300), (300, 1000), (1000, 1e9)):
    m = (tot >= lo) & (tot < hi)
    print(f"   productos con {lo:.0f}–{hi:.0f} compras: {int(m.sum()):5d} · dispersión mediana {np.median(chi_h[m]):.2f}")
# si fuera binomial negativa: var = mu + mu^2 / k  ->  estimación global de k por momentos
mu = Eh.ravel(); v = (Mh - Eh).ravel() ** 2
k = (mu ** 2).sum() / max((v - mu).sum(), 1e-9)
print(f"   exceso de varianza global: {100*((v.sum()/mu.sum()) - 1):.1f} % sobre Poisson · parámetro k de una binomial negativa ≈ {k:.0f} (k grande = casi Poisson)")
r = q("""SELECT product_id::String AS p, toStartOfInterval(ts::DateTime64(3), INTERVAL 15 MINUTE) AS b, count() AS n FROM shop.events WHERE event_type::String = 'purchase' GROUP BY p, b""", 300)
s = pd.DataFrame(r["data"]); S = s.pivot(index="p", columns="b", values="n").fillna(0).values.astype(float)[:, 1:-1]
chi_s, Ms, Es = disp(S, "STREAM, por cuarto de hora")
top = np.argsort(-chi_s)[:8]; print("   los más dispersos del stream (dispersión, compras):", [(round(float(chi_s[i]), 1), int(Ms[i].sum())) for i in top])
mu = Es.ravel(); v = (Ms - Es).ravel() ** 2
print(f"   exceso de varianza global: {100*((v.sum()/mu.sum()) - 1):.1f} % · k ≈ {(mu**2).sum()/max((v-mu).sum(),1e-9):.1f}")
