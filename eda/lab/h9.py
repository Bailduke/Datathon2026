"""H9 (nueva) · ¿Saciedad más allá del producto? Si el usuario ya ha comprado HOY algo de la misma subcategoría,
categoría o etiquetas, ¿compra menos lo parecido? (en el modelo actual una compra SUMA afinidad a su subcategoría)."""
import numpy as np
from lab import *
P = load_P(); rt = rv.make_rt(); rv.load_affinity(rt, P, rv.CFG); fb, events, cat = get_data(max_age_h=99)
E = frozenset()
def extra(st, S, r, info, sh, t):
    u = r["user_id"]; out = np.zeros((len(sh), 8), np.float32)
    lb = (st.bought.get(u) or E) - (st.bought_hist.get(u) or E); hb = st.bought_hist.get(u) or E
    out[:, 0] = np.log1p(len(lb)); out[:, 1] = np.log1p(len(st.viewed_live.get(u) or E)); out[:, 2] = np.log1p(len(st.carted_live.get(u) or E))
    if lb or hb:
        lbs = np.fromiter(lb, np.int64, len(lb)); hbs = np.fromiter(hb, np.int64, len(hb))
        ls, lc = st.sub[lbs], st.cat[lbs]; hs = st.sub[hbs]; Tl = st.tagmat[lbs].sum(0) if len(lb) else None
        for j, i in enumerate(sh):
            if i < 0: continue
            out[j, 3] = (ls == st.sub[i]).sum(); out[j, 4] = (lc == st.cat[i]).sum(); out[j, 5] = (hs == st.sub[i]).sum()
            if Tl is not None: out[j, 6] = float(st.tagmat[i] @ Tl) / len(lb)
    out[:, 7] = len(hb)
    return out
R = base_replay(P, fb, events, cat, extra_fn=extra, n_extra=8)
pickle.dump({k: R[k] for k in ("X", "PN", "AF", "KI", "EX")}, open(os.path.join(C, "h9.pkl"), "wb"), protocol=4)
X, PN, AF, KI, EX = R["X"], R["PN"], R["AF"], R["KI"], R["EX"].astype(float); z = np.load(os.path.join(C, "h8.npz"))
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
A = np.column_stack([rv.glm_design(X, PN, AF), z["hrs"], np.log1p(z["n_req"])])
print(f"compras {int(y.sum())} · impresiones con compra de hoy en la misma subcat {100*(EX[:,3]>0).mean():.1f} % · misma cat {100*(EX[:,4]>0).mean():.1f} %")
ev = lambda cols, lab_: evaluate2(A, np.column_stack([A] + cols), y, off, t, rid, nu, lab_)
ev([EX[:, 0]], "lista: log(1+compras de hoy del usuario)")
ev([EX[:, 1]], "lista: log(1+productos vistos hoy)")
ev([EX[:, 2]], "lista: log(1+productos en carrito hoy)")
ev([EX[:, 0], EX[:, 1], EX[:, 2]], "lista: las tres")
ev([EX[:, 3] > 0], "producto: ya compró hoy en su subcategoría")
ev([EX[:, 4] > 0], "producto: ya compró hoy en su categoría")
ev([np.log1p(EX[:, 3]), np.log1p(EX[:, 4])], "producto: log compras hoy subcat / cat")
ev([np.log1p(EX[:, 5])], "producto: log compras históricas en su subcat")
ev([EX[:, 6]], "producto: etiquetas compartidas con lo comprado hoy")
