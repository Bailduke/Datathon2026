"""Variantes de la puntuación de co-compra: suavizado, forma de agregar la cesta y qué entra en la cesta."""
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
df = pd.read_pickle(os.path.join(C, "imp_df.pkl"))
pn = pickle.load(open(os.path.join(C, "imp_pn.pkl"), "rb")); fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
baskets = pickle.load(open(os.path.join(C, "baskets.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}; n = len(ids)
sub, cat = np.asarray(P["sub"]), np.asarray(P["cat"])
same_s = sub[:, None] == sub[None, :]; same_c = cat[:, None] == cat[None, :]
users = list(baskets); urow = {u: k for k, u in enumerate(users)}
Bset, Kset = {}, {}
for u, (ps, es) in baskets.items():
    Bset[u] = np.unique([index[p] for p, e in zip(ps, es) if e == "purchase" and p in index]).astype(int)
    Kset[u] = np.unique([index[p] for p, e in zip(ps, es) if e != "purchase" and p in index]).astype(int)

def cooc(sets):
    Cc = np.zeros((n, n), np.float32); cnt = np.zeros(n, np.float32)
    for b in sets.values():
        if b.size:
            cnt[b] += 1
            if b.size > 1: Cc[np.ix_(b, b)] += 1
    np.fill_diagonal(Cc, 0); return Cc, cnt

def lift_res(Cc, cnt, N, a):
    L = np.log((Cc + a) / (np.outer(cnt, cnt) / N + a))
    for m in (same_s, same_c & ~same_s, ~same_c):
        L[m] -= L[m].mean()
    return L.astype(np.float32)

Cb, nb_ = cooc(Bset)
BK = {u: np.union1d(Bset[u], Kset[u]) for u in users}
Ck, nk_ = cooc(BK)
df["pn"] = np.concatenate([pn[r] for r in df.rid.values[::10]])
pid = np.concatenate([(r["shown"] or [])[:10] for r in fb]); uid = np.repeat([r["u"] for r in fb], 10)
it = np.array([index.get(p, -1) for p in pid])
df["B"] = ((df.lb_n > 0) | (df.hb_n > 0) | (df.fb_bought > 0)).astype(float); nb = 1 - df.B
df["LC"] = (df.lc_n > 0) * nb; df["HC"] = ((df.lc_n == 0) & (df.hc_n > 0)) * nb
df["V0"] = (df.k_rv == 0) * nb; df["V1"] = (df.k_rv >= 1) * nb; df["NEW"] = df.pn.isna().astype(float)
df["lp"] = np.where(df.pn > 0, np.log(np.maximum(df.pn, 1e-12)), 0.0) - np.log(1e-3); df.loc[df.NEW > 0, "lp"] = 0.0
df["nuevo"] = df.nu.astype(float); df["home"] = (df.page == "home").astype(float)
y = df.y.values.astype(float); off = np.log(df.w.values)
def glm(cols):
    X = np.column_stack([np.ones(len(df))] + [np.asarray(c, float) for c in cols]); b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(300):
        mu = np.exp(np.clip(X @ b + off, -50, 20)); g = X.T @ (y - mu) - 1e-4 * b; H = (X * mu[:, None]).T @ X + 1e-4 * np.eye(len(b))
        s = np.linalg.solve(H, g); mm = np.abs(s).max(); b = b + (s / mm if mm > 1 else s)
        if mm < 1e-8: break
    mu = np.exp(np.clip(X @ b + off, -50, 20)); return b, np.sqrt(np.diag(np.linalg.inv(H))), float((y * np.log(mu) - mu).sum())
base = [df[c].values for c in ["nuevo", "home", "lp", "LC", "HC", "V0", "V1", "B", "NEW"]] + [(df.LC * df.lp).values]
b0, s0, ll0 = glm(base); print(f"modelo base: logver {ll0:.1f}")
order = np.argsort(uid, kind="stable"); bounds = np.flatnonzero(np.r_[True, uid[order][1:] != uid[order][:-1], True])

def score(L, sets, agg, k=3.0):
    out = np.zeros(len(df)); nbask = np.zeros(len(df))
    for a, b_ in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b_]; u = uid[rows[0]]; bs = sets.get(u)
        if bs is None or bs.size == 0: continue
        items = it[rows]; ok = items >= 0
        if not ok.any(): continue
        M = L[np.ix_(items[ok], bs)]
        M = np.where(items[ok][:, None] == bs[None, :], 0.0, M)       # el propio producto no cuenta
        nn = np.maximum((items[ok][:, None] != bs[None, :]).sum(1), 1)
        s = M.sum(1)
        out[rows[ok]] = s / nn if agg == "media" else s / np.sqrt(nn) if agg == "raiz" else s / (nn + k)
        nbask[rows[ok]] = bs.size
    return out, nbask

def run(title, feats):
    b, s, ll = glm(base + feats)
    print(f"  {title:58s} dlogver {ll-ll0:+7.1f} · " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b[11:], s[11:])))

N = len(users)
for a in (0.5, 2.0, 5.0):
    L = lift_res(Cb, nb_, N, a)
    for agg in ("media", "raiz", "encogida"):
        sc, nbk = score(L, Bset, agg)
        run(f"cesta=compras · suavizado {a} · agregación {agg}", [sc])
L = lift_res(Ck, nk_, N, 2.0)
for agg in ("media", "encogida"):
    sc, nbk = score(L, BK, agg); run(f"cesta=compras+carrito (matriz idem) · suav 2 · {agg}", [sc])
L = lift_res(Cb, nb_, N, 2.0)
sc_b, _ = score(L, Bset, "encogida"); sc_k, _ = score(L, Kset, "encogida")
run("compras y carrito por separado (matriz de compras)", [sc_b, sc_k])
sd = sc_b[sc_b != 0].std(); print(f"desv. típica de la puntuación (encogida, suav 2): {sd:.3f}")
q_ = pd.qcut(sc_b[sc_b != 0], 5, duplicates="drop")
t = pd.DataFrame({"q": q_, "y": y[sc_b != 0], "w": df.w.values[sc_b != 0]}).groupby("q", observed=True).agg(imp=("y", "size"), compras=("y", "sum"), exam=("w", "sum"))
t["conv_x_exam_%"] = (100 * t.compras / t.exam).round(3); print(t.to_string())
np.save(os.path.join(C, "cf_sc.npy"), sc_b)
