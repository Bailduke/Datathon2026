import json, numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
D = pickle.load(open(os.path.join(C, "h_rho2.pkl"), "rb")); R, VARS = D["R"], D["VARS"]
X, PN, AF, KI, EX = R["X"], R["PN"], R["AF"], R["KI"], R["EX"].astype(float)
z = np.load(os.path.join(C, "h8.npz")); h9 = pickle.load(open(os.path.join(C, "h9.pkl"), "rb"))
assert np.array_equal(h9["KI"], KI)
m = ~np.isnan(EX[:, 0]); rid = KI[:, 0]; buy_list = np.array([bool(f["buy"]) for f in fb])[rid]
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]) + np.where(buy_list, 0.0, np.log(5.0))
t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
ctl = np.column_stack([z["hrs"], np.log1p(z["n_req"]), h9["EX"][:, 1], h9["EX"][:, 2]])
print(f"submuestra: {m.sum()//10} listas (todas las que tienen compra + 1 de cada 5 del resto, ponderadas) · {int(y[m].sum())} compras · nuevos {int(y[m & nu].sum())} · historia {int(y[m & ~nu].sum())}")
des = lambda pn: np.column_stack([rv.glm_design(X[m], pn[m], AF[m]), ctl[m]])
s0 = score(des(PN), y[m], off[m], t[m], rid[m], nu[m])
print(f"base (rho 0,65 · lam_cat 1 · pesos 1/0,25/0,02 · b0 20000): Poisson {s0[0]:.1f} · orden temporal {s0[1]:.1f} · orden CV {s0[2]:.1f}")
print(f"{'variante':46s} {'Poisson':>8} {'ord.temp':>9} {'ord.CV':>8} {'nuevos':>8} {'historia':>9}")
res = {}
for v, var in enumerate(VARS):
    s = score(des(EX[:, v]), y[m], off[m], t[m], rid[m], nu[m]) - s0; res[json.dumps(var)] = s.tolist()
    print(f"{json.dumps(var):46s} {s[0]:+8.1f} {s[1]:+9.1f} {s[2]:+8.1f} {s[3]:+8.1f} {s[4]:+9.1f}", flush=True)
json.dump(res, open(os.path.join(C, "h_rho2_res.json"), "w"))
