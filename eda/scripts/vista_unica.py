import os, sys, pickle, importlib.util, logging, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb"))
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); bought_hist = P["bought"]; idx = {p: i for i, p in enumerate(P["ids"])}
rows = []
for r in fb:
    vs = rv.normalize_views(r["recent_views"]); n = len(vs); sh = (r["shown"] or [])[:10]
    v2 = str(r.get("mv") or "").startswith(("tengo-aurilla-v2", "aurilla-tela", "transformada"))
    first = vs[0] if vs else None
    pos1 = sh.index(first) if first in sh else -1                     # posición del primer visto (o único)
    posl = sh.index(vs[-1]) if vs and vs[-1] in sh else -1
    hb = bought_hist.get(r["user_id"], ())
    rows.append((n, r["nu"] == "true", v2, pos1, posl, bool(r["buy"]), r["buy"] == first if first else False, r["buy"] == (vs[-1] if vs else None), r["page"], idx.get(first, -1) in hb if first else False))
d = pd.DataFrame(rows, columns=["n", "nuevo", "v2", "pos1", "posl", "buy", "buy_first", "buy_last", "page", "ya"])
print("peticiones por nº de vistas en la sesión:", d.n.clip(upper=6).value_counts(normalize=True).sort_index().round(3).to_dict())
print("páginas en peticiones con 1 vista:", d[d.n == 1].page.value_counts(normalize=True).round(2).to_dict())
for lab, D in (("TODAS LAS LISTAS", d), ("LISTAS DEL v2 EN ADELANTE", d[d.v2])):
    print("\n" + lab)
    for nm, sel in (("sesión de 1 vista", D.n == 1), ("sesión de 2+ vistas (primer visto)", D.n >= 2)):
        S = D[sel & ~D.ya]; tot = len(S)
        g = S.assign(donde=np.where(S.pos1 < 0, "no mostrado", np.where(S.pos1 == 0, "posición 1", np.where(S.pos1 <= 2, "posición 2–3", "posición 4–10"))))
        t = g.groupby("donde").agg(listas=("buy", "size"), compras_lista=("buy", "sum"), compras_del_visto=("buy_first", "sum"))
        t["% de listas"] = (100 * t.listas / tot).round(1); t["conv. lista %"] = (100 * t.compras_lista / t.listas).round(2); t["compra el visto %"] = (100 * t.compras_del_visto / t.listas).round(2)
        print(f" {nm}: {tot} listas"); print(t.reindex(["posición 1", "posición 2–3", "posición 4–10", "no mostrado"]).to_string())
