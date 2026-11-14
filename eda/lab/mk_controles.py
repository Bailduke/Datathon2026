"""Genera eda/lab/recommender-v4.controles.py = fichero en vivo ACTUAL + controles de lista en la calibración.
Solo toca replay_feedback y calibrate_live. Falla si el fichero en vivo ya no contiene los fragmentos esperados."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
s = (ROOT / "tengo-aurilla-v0/recommender-v4.py").read_text()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)
rep('''N_BASE = 11                                                # columnas anteriores a los gustos latentes''',
    '''N_BASE = 11                                                # columnas anteriores a los gustos latentes
# Controles de lista (no cambian el orden; evitan que la fatiga del usuario se confunda con la afinidad y acercan
# la previsión al nivel real): peticiones previas del usuario, productos vistos hoy y en el carrito hoy.
CTL_NAMES = ("ctl_peticiones_previas", "ctl_vistos_hoy", "ctl_carrito_hoy")''')
rep('''    pol_ctx = np.zeros((n_pol, 3))                         # nuevo, home, compró (con la lista real)''',
    '''    pol_ctx = np.zeros((n_pol, 3))                         # nuevo, home, compró (con la lista real)
    CT = np.zeros((M, len(CTL_NAMES)), np.float32)         # controles de lista de cada impresión
    pol_ctl = np.zeros((n_pol, len(CTL_NAMES)))
    n_req: dict = defaultdict(int)''')
rep('''            nuevo, home = float(r["nu"] == "true"), float(r["page"] == "home")''',
    '''            nuevo, home = float(r["nu"] == "true"), float(r["page"] == "home")
            ctl = (math.log1p(n_req[u]), math.log1p(len(st.viewed_live.get(u) or ())), math.log1p(len(lcs)))
            n_req[u] += 1''')
rep('''                pol_ctx[n_k] = (nuevo, home, float(bool(r["buy"])))''',
    '''                pol_ctx[n_k] = (nuevo, home, float(bool(r["buy"])))
                pol_ctl[n_k] = ctl''')
rep('''                if 0 <= i < Pn.size:
                    PN[m] = float(Pn[i])
                    AF[m] = ts_[:, i]
                m += 1
    return dict(X=X[:m], PN=PN[:m], AF=AF[:m], pol=pol, pol_ctx=pol_ctx)''',
    '''                if 0 <= i < Pn.size:
                    PN[m] = float(Pn[i])
                    AF[m] = ts_[:, i]
                CT[m] = ctl
                m += 1
    return dict(X=X[:m], PN=PN[:m], AF=AF[:m], CT=CT[:m], pol=pol, pol_ctx=pol_ctx, pol_ctl=pol_ctl)''')
rep('''    A = glm_design(X, PN, R["AF"])
    use_taste = cfg.taste and AFF is not None and y.sum() >= 300
    if not use_taste:
        A = A[:, :N_BASE]
    b, se, ll = poisson_fit(A, y, np.log(w))
    _, _, ll0 = poisson_fit(A[:, [0, 1, 2, 3, 8, 10]], y, np.log(w))
    _, _, ll1 = poisson_fit(A[:, :N_BASE], y, np.log(w))''',
    '''    use_taste = cfg.taste and AFF is not None and y.sum() >= 300
    n_t = 2 * N_BASKETS if use_taste else 0
    A = np.column_stack([glm_design(X, PN, R["AF"])[:, :N_BASE + n_t], R["CT"].astype(np.float64)])
    names = GLM_COLS[:N_BASE + n_t] + CTL_NAMES
    i_ctl = list(range(N_BASE + n_t, A.shape[1]))
    b, se, ll = poisson_fit(A, y, np.log(w))
    _, _, ll0 = poisson_fit(A[:, [0, 1, 2, 3, 8, 10] + i_ctl], y, np.log(w))
    _, _, ll1 = poisson_fit(A[:, list(range(N_BASE)) + i_ctl], y, np.log(w))''')
rep('''    for name, bi, si in zip(GLM_COLS, b, se):
        log.info("[calib]   %-14s %+.3f ± %.3f   (x%.2f)", name, bi, si, math.exp(min(bi, 20)))''',
    '''    for name, bi, si in zip(names, b, se):
        log.info("[calib]   %-22s %+.3f ± %.3f   (x%.2f)", name, bi, si, math.exp(min(bi, 20)))''')
rep('''    if use_taste:
        lin = lin + pol[..., 5:] @ b[N_BASE:N_BASE + 2 * N_BASKETS]''',
    '''    if use_taste:
        lin = lin + pol[..., 5:] @ b[N_BASE:N_BASE + 2 * N_BASKETS]
    lin = lin + (R2["pol_ctl"] @ b[i_ctl])[:, None, None]''')
rep('''            "coef": {n: [float(bi), float(si)] for n, bi, si in zip(GLM_COLS, b, se)}, "compras_por_clase": buys,''',
    '''            "coef": {n: [float(bi), float(si)] for n, bi, si in zip(names, b, se)}, "compras_por_clase": buys,''')
(Path(__file__).resolve().parent / "recommender-v4.controles.py").write_text(s)
print("generado a partir del fichero en vivo actual")
