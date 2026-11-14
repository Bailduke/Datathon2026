"""Construye la tabla de impresiones (una fila por producto mostrado en nuestras listas en vivo) con lo
que se sabía del par usuario-producto ANTES de la petición. Salida: .cache/imp_df.pkl"""
import os, sys, pickle, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rt import q
C = os.path.join(HERE, ".cache")
fb = pickle.load(open(os.path.join(C, "imp_fb.pkl"), "rb"))
ev = pickle.load(open(os.path.join(C, "imp_ev.pkl"), "rb"))
hist = pickle.load(open(os.path.join(C, "imp_hist.pkl"), "rb"))
P = pickle.load(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb"))
ids = list(P["ids"]); index = {p: i for i, p in enumerate(ids)}
sub = [P["subs"][s] for s in P["sub"]]; cat = [P["cats"][c] for c in P["cat"]]
psub = {p: sub[i] for p, i in index.items()}; pcat = {p: cat[i] for p, i in index.items()}
pnew = set()
for x in q("SELECT product_id::String AS p, toString(category) AS c, toString(subcategory) AS s FROM shop.catalog_updates WHERE change::String = 'new_product'", 120)["data"]:
    if x["p"] not in index:
        psub[x["p"]], pcat[x["p"]] = x["s"], x["c"]; pnew.add(x["p"])
pop = {p: float(P["C_hist"][i]) for p, i in index.items()}
urow = {u: k for k, u in enumerate(P["users"])}
N = P["N"]; subs_idx = {s: k for k, s in enumerate(P["subs"])}
sub_cat = P["sub_cat"]; cats_of_sub = {P["subs"][k]: P["cats"][sub_cat[k]] for k in range(len(P["subs"]))}
Ncat = np.zeros((N.shape[0], len(P["cats"])), np.float32)
for k in range(len(P["subs"])):
    Ncat[:, sub_cat[k]] += N[:, k]
cat_idx = {c: k for k, c in enumerate(P["cats"])}
Ntot = Ncat.sum(1)

evu, ut = {}, {}
for u, (ps, es, ts) in ev.items():
    d = {}
    for p, e, t in zip(ps, es, ts):
        d.setdefault(p, []).append((t, e[0]))          # v / a / p
    evu[u] = d; ut[u] = np.asarray(ts, np.int64)
hd = {(x["u"], x["p"]): x for x in hist}
fbuy = {}
for r in fb:
    if r["buy"]:
        fbuy.setdefault(r["u"], []).append((r["t"], r["buy"]))
T_HIST_END = max(x["tc"] for x in hist) if hist else 0

rows = []
BIG = 1e9
for r in fb:
    u, t = r["u"], int(r["t"])
    rv = r["rv"] or []
    rvp = [(v.get("product_id") if isinstance(v, dict) else v) for v in rv]
    rvt = []
    for v in rv:
        try:
            rvt.append(pd.Timestamp(v["ts"]).value // 10**6)
        except Exception:
            rvt.append(0)
    last = rvp[-1] if rvp else None
    ls_, lc_ = psub.get(last), pcat.get(last)
    ses_cats = {pcat.get(p) for p in rvp}
    ses_subs = {psub.get(p) for p in rvp}
    dlast = (t - rvt[-1]) / 1000 if rvt and rvt[-1] else BIG
    ue = ut.get(u)
    n_before = int(np.searchsorted(ue, t)) if ue is not None else 0
    u_dt = (t - int(ue[n_before - 1])) / 1000 if n_before else BIG
    row = urow.get(u)
    d = evu.get(u, {})
    fbu = fbuy.get(u, ())
    for j, p in enumerate((r["shown"] or [])[:10]):
        lv_n = lc_n = lb_n = 0; lv_dt = lc_dt = lb_dt = BIG
        for (te, e) in d.get(p, ()):
            if te >= t: break
            if e == "v": lv_n += 1; lv_dt = (t - te) / 1000
            elif e == "a": lc_n += 1; lc_dt = (t - te) / 1000
            else: lb_n += 1; lb_dt = (t - te) / 1000
        h = hd.get((u, p))
        s_, c_ = psub.get(p), pcat.get(p)
        k_rv = (len(rvp) - 1 - max(i for i, x in enumerate(rvp) if x == p)) if p in rvp else -1
        rows.append((r["rid"], j, int(p == r["buy"]), int(p == r["clk"]), r["page"], r["nu"] == "true", r["sb"], r["mv"], len(rvp),
                     k_rv, dlast, int(s_ == ls_ and s_ is not None), int(c_ == lc_ and c_ is not None), int(c_ in ses_cats), int(s_ in ses_subs),
                     lv_n, lv_dt, lc_n, lc_dt, lb_n, lb_dt,
                     h["nv"] if h else 0, h["nc"] if h else 0, h["nb"] if h else 0,
                     (t - h["tc"]) / 86400e3 if h and h["nc"] else BIG, (t - h["tv"]) / 86400e3 if h and h["nv"] else BIG,
                     int(any(tb < t and pb == p for tb, pb in fbu)),
                     n_before, u_dt, pop.get(p, 0.0), int(p in pnew),
                     float(N[row, subs_idx[s_]]) if row is not None and s_ in subs_idx else 0.0,
                     float(Ncat[row, cat_idx[c_]]) if row is not None and c_ in cat_idx else 0.0,
                     float(Ntot[row]) if row is not None else 0.0, c_ or "?", t))
cols = ["rid", "slot", "y", "clk", "page", "nu", "sb", "mv", "n_rv", "k_rv", "dlast", "same_sub", "same_cat", "ses_cat", "ses_sub",
        "lv_n", "lv_dt", "lc_n", "lc_dt", "lb_n", "lb_dt", "hv_n", "hc_n", "hb_n", "hc_days", "hv_days", "fb_bought",
        "u_live_n", "u_dt", "pop", "new", "n_us", "n_uc", "n_u", "cat", "t"]
df = pd.DataFrame(rows, columns=cols)
df["w"] = np.asarray(P["w_pos"])[df["slot"].values]
df.to_pickle(os.path.join(C, "imp_df.pkl"))
print("impresiones:", len(df), "| listas:", df.rid.nunique(), "| compras:", int(df.y.sum()), "| clics:", int(df.clk.sum()))
print(df.groupby("sb").agg(listas=("rid", "nunique"), compras=("y", "sum")).to_string())
