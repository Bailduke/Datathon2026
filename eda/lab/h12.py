"""H12 · ¿Compran los usuarios nuevos productos distintos? Popularidad de hoy entre usuarios nuevos frente a la global."""
import zlib, bisect, numpy as np
from collections import defaultdict
from lab import *
P = load_P(); fb, events, cat = pickle.load(open(os.path.join(C, "dl_2040.pkl"), "rb"))
h9 = pickle.load(open(os.path.join(C, "h9.pkl"), "rb")); z8 = np.load(os.path.join(C, "h8.npz"))
X, PN, AF, KI = h9["X"], h9["PN"], h9["AF"].astype(float), h9["KI"]
rid = KI[:, 0]; samp = np.array([bool(f["buy"]) or zlib.crc32(f["rid"].encode()) % 5 == 0 for f in fb]); m = samp[rid]
buy_list = np.array([bool(f["buy"]) for f in fb])[rid]
y = X[m, 1]; off = (np.log(np.asarray(P["w_pos"])[X[:, 0].astype(int)]) + np.where(buy_list, 0.0, np.log(5.0)))[m]
t = np.array([fb[j]["t"] for j in rid])[m]; nu = X[m, 2] > 0; r_ = rid[m]; ii = KI[m, 1]; n = len(P["ids"])
ctl = np.column_stack([z8["hrs"], np.log1p(z8["n_req"]), h9["EX"][:, 1], h9["EX"][:, 2]])[m]
A = np.column_stack([rv.glm_design(X[m], PN[m], AF[m]), ctl])
index = {p: i for i, p in enumerate(P["ids"])}; old = set(P["users"])
pn_ = defaultdict(list); po_ = defaultdict(list); alln, allo = [], []
for (te, u, p, e) in events:
    if e != "purchase": continue
    i = index.get(p)
    if u in old:
        allo.append(te)
        if i is not None: po_[i].append(te)
    else:
        alln.append(te)
        if i is not None: pn_[i].append(te)
alln, allo = np.array(alln), np.array(allo); print(f"compras orgánicas de hoy: usuarios nuevos {len(alln)} · con historia {len(allo)}")
f_new = np.zeros(len(y)); f_seg = np.zeros(len(y)); a = 5.0
for q in range(len(y)):
    i = ii[q]
    if not (0 <= i < n): continue
    cn = bisect.bisect_left(pn_.get(i, ()), t[q]); co = bisect.bisect_left(po_.get(i, ()), t[q]); Tn = np.searchsorted(alln, t[q]); To = np.searchsorted(allo, t[q])
    lr = np.log((cn + a) / (Tn + 1.0)) - np.log((co + a * (To + 1.0) / (Tn + 1.0)) / (To + 1.0))     # log cuota entre nuevos − log cuota entre antiguos (suavizado)
    f_new[q] = lr * nu[q]; f_seg[q] = lr * (1.0 if nu[q] else -1.0)
print("desv. típica del log-cociente de cuotas (nuevos/antiguos) en lo mostrado: %.2f" % np.std(f_seg))
evaluate2(A, np.column_stack([A, f_new]), y, off, t, r_, nu, "popularidad entre nuevos / antiguos (solo usuarios nuevos)")
evaluate2(A, np.column_stack([A, f_seg]), y, off, t, r_, nu, "popularidad del propio segmento (con signo)")
