"""¿Por qué los productos nuevos se compran más de lo previsto dentro de la lista? Dos hipótesis:
 (H1) les faltan los factores de co-compra (su puntuación de co-compra es 0 mientras los demás de la lista la tienen positiva);
 (H2) efecto novedad propio del carrusel, independiente de la co-compra.
Usa la reproducción guardada por nuevos_backtest.py (.cache/nuevos_bt.npz). No consulta RawTree."""
import os, sys, pickle, importlib.util, logging, math, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C_ = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv7", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v7.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv7"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes())
z = np.load(os.path.join(C_, "nuevos_bt.npz")); X, G, PN, AF, NEW = z["X"], z["G"], z["PN"][:, 0], z["AF"].astype(float), z["NEWr"]
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; A = rv.glm_design(X, PN, AF); cols = list(range(3, A.shape[1])); NB = rv.N_BASE
names = list(rv.GLM_COLS[3:])
def fit(extra, lab):
    M = np.column_stack([A[:, cols]] + extra); b, se, ll, ng = rv.cond_fit(M, y, np.log(w), G)
    print(f"   logver {ll:9.1f} · " + " · ".join(f"{l} {bb:+.2f} ± {s:.2f}" for l, bb, s in zip(lab, b[len(cols):], se[len(cols):]))); return b, ll
print(f"{int(NEW.sum())} impresiones de productos nuevos, {int(y[NEW > 0].sum())} compras")
print("pesos actuales de los gustos:", {n: round(float(v), 2) for n, v in zip(names[NB - 3:], rv.cond_fit(A[:, cols], y, np.log(w), G)[0][NB - 3:])})
print("\n(0) sin indicador / con indicador único"); b0, ll0 = fit([], []); b1, _ = fit([NEW], ["nuevo"])
# cuánta co-compra 'tienen' los demás productos de la misma lista: suma ponderada de las 4 columnas cf con sus pesos
cf = AF[:, :4] @ b0[NB - 3:NB - 3 + 4]; tg = AF[:, 4:8] @ b0[NB - 3 + 4:NB - 3 + 8]
old = NEW == 0; ng = int(G.max()) + 1
m_cf = np.bincount(G[old], weights=cf[old], minlength=ng) / np.maximum(np.bincount(G[old], minlength=ng), 1)       # media de co-compra de los antiguos de la lista
print(f"\nco-compra (ya ponderada) de los productos de la lista: antiguos media {cf[old].mean():.2f} · nuevos media {cf[~old].mean():.3f}  → a los nuevos les 'falta' ≈ la media de sus compañeros de lista")
print(f"etiquetas (ya ponderadas): antiguos {tg[old].mean():.2f} · nuevos {tg[~old].mean():.2f}")
gap = m_cf[G]                                                                                                   # lo que le falta a un nuevo si fuese 'como sus compañeros'
print("\n(1) H1: el indicador se sustituye por 'nuevo × co-compra media de los compañeros de lista' (pendiente 1 = le falta justo eso)")
fit([NEW * gap], ["nuevo×cf_lista"]); fit([NEW, NEW * gap], ["nuevo", "nuevo×cf_lista"])
print("\n(2) El efecto 'nuevo' según cuánta co-compra hay en la lista (terciles de la media de los compañeros)")
q = np.quantile(gap[NEW > 0], [1 / 3, 2 / 3]); t3 = np.searchsorted(q, gap)
for k in range(3):
    mk = (NEW > 0) & (t3 == k); print(f"   tercil {k+1}: co-compra media de la lista {gap[mk].mean():.2f} · {int(mk.sum())} impresiones · {int(y[mk].sum())} compras")
fit([NEW * (t3 == k) for k in range(3)], ["nuevo·t1", "nuevo·t2", "nuevo·t3"])
print("\n(3) Listas sin ninguna co-compra (usuario sin cestas): ¿sigue habiendo efecto 'nuevo'?")
none = (np.bincount(G, weights=np.abs(AF[:, :4]).sum(1), minlength=ng) == 0)[G]
print(f"   impresiones de nuevos en listas sin co-compra: {int((NEW * none).sum())} · compras {int(y[(NEW > 0) & none].sum())}")
fit([NEW * none, NEW * ~none], ["nuevo·sin cf", "nuevo·con cf"])
