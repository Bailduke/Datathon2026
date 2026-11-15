"""¿Lo que mostramos cambia lo que el simulador hace después? Con el stream guardado (sin RawTree):
 (A) ¿un clic o una compra en el carrusel aparece luego como evento del usuario en shop.events?
 (B) ¿el producto clicado entra en las vistas recientes de su siguiente petición?
 (C) ¿la conversión del usuario depende de cuántas peticiones lleva y de si ya compró en el carrusel?"""
import os, pickle, bisect, numpy as np, collections
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb"))
clk = pickle.load(open(os.path.join(C, "clicks.pkl"), "rb"))
ev = collections.defaultdict(list)
for t, u, p, e in events: ev[(u, p, e)].append(t)
def has(u, p, e, lo, hi):
    l = ev.get((u, p, e))
    if not l: return False
    k = bisect.bisect_left(l, lo); return k < len(l) and l[k] < hi
tmax = events[-1][0]; W = 120e3
print(f"{len(fb)} listas · {sum(1 for r in fb if r['buy'])} con compra · {sum(1 for r in fb if r['rid'] in clk)} con clic")
print("\n(A) ¿Aparece en shop.events? (ventana de 2 min después; entre paréntesis, la misma ventana ANTES, como referencia)")
def rate(pairs, e):
    a = [has(u, p, e, t, t + W) for t, u, p in pairs]; b = [has(u, p, e, t - W, t) for t, u, p in pairs]; return np.mean(a), np.mean(b), len(pairs)
buys = [(r["t"], r["user_id"], r["buy"]) for r in fb if r["buy"] and r["t"] + W < tmax]
clicks = [(r["t"], r["user_id"], clk[r["rid"]][0]) for r in fb if r["rid"] in clk and clk[r["rid"]][0] and r["t"] + W < tmax]
ctrl = [(r["t"], r["user_id"], next(p for p in r["shown"] if p != clk[r["rid"]][0])) for r in fb if r["rid"] in clk and clk[r["rid"]][0] and r["t"] + W < tmax and len(r["shown"] or []) > 1]
for nm, pairs in (("producto COMPRADO en el carrusel", buys), ("producto CLICADO en el carrusel", clicks), ("mostrado en la misma lista, no clicado", ctrl)):
    for e in ("view", "add_to_cart", "purchase"):
        a, b, n = rate(pairs, e); print(f"   {nm:40s} → evento '{e:11s}' del mismo usuario y producto: {100*a:5.1f} %  ({100*b:4.1f} % antes) · n={n}")
print("\n(B) Siguiente petición del mismo usuario (en menos de 10 min): ¿el producto está en sus vistas recientes?")
by_u = collections.defaultdict(list)
for r in fb: by_u[r["user_id"]].append(r)
for l in by_u.values(): l.sort(key=lambda r: r["t"])
res = collections.defaultdict(lambda: [0, 0]); gaps = collections.defaultdict(list)
for u, l in by_u.items():
    for a, b in zip(l[:-1], l[1:]):
        dt = b["t"] - a["t"]; c = clk.get(a["rid"], ("", ""))[0]; gaps["con clic" if c else "sin clic"].append(dt / 1000)
        if dt > 600e3: continue
        rv = {x["product_id"] for x in (b["recent_views"] or [])}; rv0 = {x["product_id"] for x in (a["recent_views"] or [])}
        for j, p in enumerate((a["shown"] or [])[:10]):
            if p in rv0: continue                                # ya estaba visto antes: no cuenta
            k = "clicado" if p == c else ("comprado" if p == a["buy"] else ("mostrado en posición 1, sin clic" if j == 0 else "mostrado en 2–10, sin clic"))
            res[k][0] += p in rv; res[k][1] += 1
for k, (a, n) in res.items(): print(f"   {k:34s} {100*a/max(n,1):6.2f} %  (n={n})")
print("   tiempo hasta la siguiente petición (mediana): " + " · ".join(f"{k} {np.median(v):.0f} s" for k, v in gaps.items()))
print("\n(C) Compras por cada 100 listas según el número de orden de la petición del usuario en el día (páginas con sesión, sin home)")
T = collections.defaultdict(lambda: [0, 0]); T2 = collections.defaultdict(lambda: [0, 0])
for u, l in by_u.items():
    nb = 0
    for k, r in enumerate(l):
        if r["page"] != "home":
            o = "1" if k == 0 else "2" if k == 1 else "3–5" if k < 5 else "6–10" if k < 10 else "11–20" if k < 20 else "21+"
            T[o][0] += bool(r["buy"]); T[o][1] += 1
            if k >= 2: T2["ya compró en el carrusel hoy" if nb else "aún no ha comprado en el carrusel"][0] += bool(r["buy"]); T2["ya compró en el carrusel hoy" if nb else "aún no ha comprado en el carrusel"][1] += 1
        nb += bool(r["buy"])
for o in ("1", "2", "3–5", "6–10", "11–20", "21+"):
    a, n = T[o]; print(f"   petición nº {o:6s} {100*a/max(n,1):5.2f} % ± {100*np.sqrt(max(a,1))/max(n,1):4.2f}  (n={n})")
for o, (a, n) in T2.items(): print(f"   desde la 3.ª petición, {o:34s} {100*a/max(n,1):5.2f} % ± {100*np.sqrt(max(a,1))/max(n,1):4.2f}  (n={n})")
