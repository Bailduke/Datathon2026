"""¿Se quitan compras entre sí los productos parecidos de una lista? Regresión a nivel de lista con los datos actuales:
log E[compra] = c + nuevo + home + gamma·log S + delta·concentración, con S = suma de las tasas previstas de sus 10 productos."""
import os, sys, pickle, importlib.util, logging, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); cfg = rv.CFG
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb")); z = np.load(os.path.join(C, "assaig.npz"))
X, PN, AF, gid = z["Xf"], z["PNf"], z["AFf"], z["gid"]
st = rv.State(P, cfg)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
sub = np.zeros(len(X), int); ca = np.zeros(len(X), int); v2 = np.zeros(len(X), bool); m = 0
for k in order:
    r = fb[k]; new = str(r.get("mv") or "").startswith(("tengo-aurilla-v2", "aurilla-tela", "transformada"))
    for pid in (r["shown"] or [])[:10]:
        i = st.index.get(pid, -1); sub[m] = st.sub[i] if i >= 0 else -1; ca[m] = st.cat[i] if i >= 0 else -1; v2[m] = new; m += 1
assert m == len(X)
w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; y = X[:, 1]; A = rv.glm_design(X, PN, AF)
b, se, _ = rv.poisson_fit(A, y, np.log(w))
item = w * np.exp(np.clip(A[:, 3:] @ b[3:], -50, 20))          # tasa del producto sin los términos de lista
d = pd.DataFrame({"g": gid, "y": y, "s": item, "sub": sub, "cat": ca, "nuevo": X[:, 2], "home": X[:, 3], "v2": v2})
L = d.groupby("g").agg(y=("y", "sum"), S=("s", "sum"), nuevo=("nuevo", "first"), home=("home", "first"), v2=("v2", "first"), nsub=("sub", "nunique"), ncat=("cat", "nunique"))
hs = d.groupby(["g", "sub"]).s.sum().groupby("g").apply(lambda v: float(((v / v.sum()) ** 2).sum())); hc = d.groupby(["g", "cat"]).s.sum().groupby("g").apply(lambda v: float(((v / v.sum()) ** 2).sum()))
L["hhi_sub"] = hs; L["hhi_cat"] = hc
print(f"{len(L)} listas, {int(L.y.sum())} compras · de v2 en adelante: {int(L.v2.sum())} listas, {int(L[L.v2].y.sum())} compras")
print("listas de v2 en adelante: subcategorías distintas por lista (mediana) %d · categorías distintas %d · listas con una sola categoría %.0f %%" % (L[L.v2].nsub.median(), L[L.v2].ncat.median(), 100 * (L[L.v2].ncat == 1).mean()))
def reg(D, cols, title):
    Z = np.column_stack([np.ones(len(D)), D.nuevo, D.home, np.log(D.S)] + [D[c].values for c in cols]); bb, ss, _ = rv.poisson_fit(Z, D.y.values.astype(float), np.zeros(len(D)))
    names = ["gamma (log S)"] + cols
    print(f"  {title:34s} " + " · ".join(f"{n} {v:+.2f} ± {s:.2f}" for n, v, s in zip(names, bb[3:], ss[3:])))
for nm, D in (("TODAS LAS LISTAS", L), ("SOLO v2 EN ADELANTE", L[L.v2]), ("v2+, páginas con sesión", L[L.v2 & (L.home == 0)])):
    print(nm); reg(D, [], "solo saturación"); reg(D, ["hhi_sub"], "+ concentración por subcategoría"); reg(D, ["hhi_cat"], "+ concentración por categoría"); reg(D, ["hhi_sub", "hhi_cat"], "+ ambas")
# lectura directa: compras observadas / esperadas según lo concentrada que está la lista (v2 en adelante)
D = L[L.v2].copy(); Z = np.column_stack([np.ones(len(D)), D.nuevo, D.home, np.log(D.S)]); bb, _, _ = rv.poisson_fit(Z, D.y.values.astype(float), np.zeros(len(D))); D["esp"] = np.exp(Z @ bb)
for col, lab in (("nsub", "subcategorías distintas en la lista"), ("ncat", "categorías distintas en la lista")):
    D["tr"] = pd.cut(D[col], [0, 2, 4, 6, 8, 11], labels=["1–2", "3–4", "5–6", "7–8", "9–10"])
    t = D.groupby("tr", observed=True).agg(listas=("y", "size"), compras=("y", "sum"), esperadas=("esp", "sum")); t["obs/esp"] = (t.compras / t.esperadas).round(2); t["±"] = (t["obs/esp"] / np.sqrt(t.compras.clip(lower=1))).round(2)
    print(f"\nv2 en adelante, por {lab}:"); print(t.round(1).to_string())
