"""H4 (posición libre) y H5 (pesos distintos para usuarios nuevos / con historia), con controles de lista."""
import numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
R = pickle.load(open(os.path.join(C, "h9.pkl"), "rb")); z = np.load(os.path.join(C, "h8.npz"))
X, PN, AF, KI, EX = R["X"], R["PN"], R["AF"], R["KI"], R["EX"].astype(float)
y = X[:, 1]; off = np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]); rid = KI[:, 0]; t = np.array([fb[j]["t"] for j in rid]); nu = X[:, 2] > 0
G = rv.glm_design(X, PN, AF); ctl = np.column_stack([z["hrs"], np.log1p(z["n_req"]), EX[:, 1], EX[:, 2]])
A = np.column_stack([G, ctl]); n = nu.astype(float)
np.savez_compressed(os.path.join(C, "base_arrays.npz"), A=A, y=y, off=off, rid=rid, t=t, nu=nu, slot=X[:, 0])
ev = lambda cols, lab_: evaluate2(A, np.column_stack([A] + cols), y, off, t, rid, nu, lab_)
r0, b0, s0 = evaluate2(G, A, y, off, t, rid, nu, "controles de lista (4) sobre el modelo actual")
ev([(X[:, 0] == s) for s in range(1, 10)], "H4 posición libre (9 indicadores)")
ev([np.log1p(X[:, 0])], "H4 una pendiente extra: log(1+slot)")
ev([n * G[:, 3]], "H5 nuevo × log P jerárquica")
ev([n * G[:, c] for c in (4, 5, 6, 7)], "H5 nuevo × clases (4)")
ev([n * G[:, 11 + c] for c in range(8)], "H5 nuevo × gustos (8)")
ev([n * G[:, 11 + c] for c in (1, 2, 3, 5, 6, 7)], "H5 nuevo × gustos de sesión y de hoy (6)")
ev([n * G[:, 3]] + [n * G[:, c] for c in (4, 5, 6, 7)] + [n * G[:, 11 + c] for c in range(8)], "H5 todo × nuevo (13)")
