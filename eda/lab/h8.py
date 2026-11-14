"""H8 (nueva) · ¿De dónde sale la caída de conversión con las horas? Fatiga del usuario, repetición de productos
ya mostrados (descuento por impresión) o reloj."""
import numpy as np
from collections import defaultdict
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
R = base_replay(P, fb, events, cat)
X, PN, AF, KI = R["X"], R["PN"], R["AF"], R["KI"]
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; off = np.log(w); A = rv.glm_design(X, PN, AF)
k = KI[:, 0]; t = np.array([fb[j]["t"] for j in k]); rid = k; nu = X[:, 2] > 0
order = np.argsort(np.array([f["t"] for f in fb]), kind="stable")
first_row = np.r_[0, np.cumsum(np.bincount(k, minlength=len(fb)))]   # filas de cada lista (en orden de fb? no: de replay)
rows_of = defaultdict(list)
for m_, kk in enumerate(k): rows_of[kk].append(m_)
n_req = np.zeros(len(y)); n_buy = np.zeros(len(y)); seen_n = np.zeros(len(y)); seen_w = np.zeros(len(y)); seen_last = np.zeros(len(y)); t_first = np.zeros(len(y))
ureq = defaultdict(int); ubuy = defaultdict(int); uexp = defaultdict(dict); ufirst = {}
wp = np.asarray(P["w_pos"])
for kk in order:
    f = fb[kk]; u = f["user_id"]; rws = rows_of.get(kk, ())
    ex = uexp[u]; ufirst.setdefault(u, f["t"])
    for m_ in rws:
        i = KI[m_, 1]; n_req[m_] = ureq[u]; n_buy[m_] = ubuy[u]; t_first[m_] = (f["t"] - ufirst[u]) / 3.6e6
        e = ex.get(i)
        if e: seen_n[m_], seen_w[m_] = e
    for m_ in rws:
        i = KI[m_, 1]; e = ex.get(i, (0, 0.0)); ex[i] = (e[0] + 1, e[1] + wp[int(X[m_, 0])])
    ureq[u] += 1; ubuy[u] += bool(f["buy"])
hrs = (t - t.min()) / 3.6e6
print(f"compras {int(y.sum())} · impresiones repetidas al mismo usuario: {100*(seen_n>0).mean():.1f} % · media de peticiones previas {n_req.mean():.2f}")
g = {"nuevos": nu, "historia": ~nu}
evaluate(A, np.column_stack([A, hrs]), y, off, t, g, rid, "horas desde el inicio")
evaluate(A, np.column_stack([A, np.log1p(n_req)]), y, off, t, g, rid, "log(1+peticiones previas del usuario)")
evaluate(A, np.column_stack([A, np.log1p(seen_n)]), y, off, t, g, rid, "log(1+veces ya mostrado ese producto)")
evaluate(A, np.column_stack([A, seen_n > 0]), y, off, t, g, rid, "ya mostrado antes (0/1)")
evaluate(A, np.column_stack([A, seen_w]), y, off, t, g, rid, "atención acumulada sobre ese producto")
evaluate(A, np.column_stack([A, hrs, np.log1p(n_req), np.log1p(seen_n)]), y, off, t, g, rid, "horas + peticiones + repetición")
A2 = np.column_stack([A, hrs])
evaluate(A2, np.column_stack([A2, np.log1p(seen_n)]), y, off, t, g, rid, "sobre 'horas': + repetición")
evaluate(A2, np.column_stack([A2, np.log1p(n_req)]), y, off, t, g, rid, "sobre 'horas': + peticiones previas")
evaluate(A2, np.column_stack([A2, np.log1p(n_buy)]), y, off, t, g, rid, "sobre 'horas': + compras previas en el carrusel")
evaluate(A2, np.column_stack([A2, t_first]), y, off, t, g, rid, "sobre 'horas': + horas desde su 1ª petición")
# tabla cruda: conversión por unidad de atención según veces mostrado
print("por unidad de atención según nº de veces ya mostrado (obs / esperado por el modelo base):")
b, _, _ = fit(A, y, off); mu = np.exp(np.clip(A @ b + off, -50, 20))
for a, c in ((0, 0), (1, 1), (2, 3), (4, 9), (10, 999)):
    m = (seen_n >= a) & (seen_n <= c); print(f"   {a}-{c}: imp {m.sum():7d} · obs {int(y[m].sum()):4d} · esp {mu[m].sum():7.1f} · obs/esp {y[m].sum()/max(mu[m].sum(),1e-9):.2f}")
np.savez_compressed(os.path.join(C, "h8.npz"), hrs=hrs, n_req=n_req, seen_n=seen_n, seen_w=seen_w, n_buy=n_buy)
