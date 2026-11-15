"""Dos preguntas para la final (new products, new trends), solo con el stream ya descargado (.cache/dl_latest.pkl, sin consultar RawTree):
 (A) ¿Los productos nuevos venden como dice el prior (cuota mediana de su subcategoría)? ¿Hay efecto novedad? ¿Las vistas tempranas predicen sus ventas?
 (B) ¿La tendencia es solo de subcategoría? ¿Dentro de la subcategoría caliente se reparte como siempre? ¿Hay ráfagas por etiqueta o categoría?"""
import os, pickle, numpy as np, calendar, time
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb"))
P = pickle.loads(open(os.path.join(HERE, "../../tengo-aurilla-v0/.cache/priors_live.pkl"), "rb").read())
tags = pickle.load(open(os.path.join(C, "tags.pkl"), "rb"))
ids = list(P["ids"]); sub = list(P["sub"]); subs = P["subs"]; sidx = {s: k for k, s in enumerate(subs)}
ts_ms = lambda s: calendar.timegm(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) * 1000
launch = {}
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in launch and r["subcategory"] in sidx:
        launch[r["product_id"]] = ts_ms(r["ts"]); ids.append(r["product_id"]); sub.append(sidx[r["subcategory"]])
index = {p: i for i, p in enumerate(ids)}; n = len(ids); n0 = 5000; sub = np.array(sub); S = len(subs)
L = np.full(n, -1.0); L[n0:] = [launch[p] for p in ids[n0:]]
ev = [(t, index[p], e) for (t, u, p, e) in events if p in index]
T = np.array([x[0] for x in ev], float); I = np.array([x[1] for x in ev]); K = np.array([{"view": 0, "add_to_cart": 1, "purchase": 2}.get(x[2], 3) for x in ev])
t0, t1 = T.min(), T.max(); H = 3600e3
print(f"stream {(t1-t0)/H:.1f} h · {len(T)} eventos · compras {int((K==2).sum())} · vistas {int((K==0).sum())} · productos nuevos {n-n0}")
print("lanzamientos por hora desde el inicio:", np.bincount(((L[n0:] - t0) / H).astype(int)).tolist())

# ---------- (A) productos nuevos ----------
print("\n(A1) Productos nuevos frente a los antiguos de su misma subcategoría, misma franja horaria")
print("     razón = eventos por producto nuevo / eventos por producto antiguo (media) de la subcategoría")
edges = [0, 0.5, 1, 2, 4, 8, 16]
for kind, name in ((2, "compras"), (0, "vistas ")):
    m = K == kind; Tm, Im = T[m], I[m]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        obs = exp_mean = exp_med = 0.0
        # por hora de reloj para emparejar: eventos de antiguos por subcategoría en cada tramo [L+a, L+b)
        for Lg in np.unique(L[n0:]):
            lo, hi = Lg + a * H, min(Lg + b * H, t1)
            if hi <= lo: continue
            w = (Tm >= lo) & (Tm < hi); cnt = np.bincount(Im[w], minlength=n).astype(float)
            g = np.flatnonzero(L == Lg)
            frac = (hi - lo) / ((b - a) * H)
            for s in np.unique(sub[g]):
                old = cnt[:n0][sub[:n0] == s]; new = cnt[g[sub[g] == s]]
                obs += new.sum(); exp_mean += len(new) * old.mean(); exp_med += len(new) * P["m_sub"][s] / max(P["C_hist"][sub[:n0] == s].sum() / P["T_hist"], 1e-12) * old.sum()
        out.append((a, b, obs, exp_mean, exp_med))
    for a, b, o, em, emd in out:
        se = np.sqrt(max(o, 1)) / max(em, 1e-9)
        print(f"     {name} {a:4.1f}–{b:4.1f} h tras el lanzamiento: observado {o:7.0f} · si fuese como la media {em:8.1f} → razón {o/max(em,1e-9):5.2f} ± {se:4.2f} · frente al prior (mediana) {o/max(emd,1e-9):5.2f}")

# (A2) ¿se puede predecir pronto qué producto nuevo venderá? vistas y compras de la primera media hora frente a compras de 0,5–6 h
print("\n(A2) Productos nuevos: ¿qué predice las compras de 0,5–6 h tras el lanzamiento?")
g = np.flatnonzero((L > 0) & (L + 6 * H <= t1))
def cnt_win(kind, a, b):
    c = np.zeros(n)
    m = (K == kind) & (I >= n0); d = (T[m] - L[I[m]]) / H; w = (d >= a) & (d < b); np.add.at(c, I[m][w], 1); return c
v0, c0, p0, p1 = cnt_win(0, 0, .5), cnt_win(1, 0, .5), cnt_win(2, 0, .5), cnt_win(2, .5, 6)
# normalizar por subcategoría: actividad esperada de la subcategoría (compras totales de la sub en el stream / nº productos)
tot_s = np.bincount(sub[I[K == 2]], minlength=S) / np.bincount(sub, minlength=S)
rel = lambda x: x[g] / tot_s[sub[g]]
from scipy.stats import spearmanr
y = rel(p1)
print(f"     {len(g)} productos con 6 h de vida · compras 0,5–6 h por producto: media {p1[g].mean():.1f}, mediana {np.median(p1[g]):.0f}, p10 {np.percentile(p1[g],10):.0f}, p90 {np.percentile(p1[g],90):.0f}")
print(f"     primera media hora por producto: vistas {v0[g].mean():.1f} · carritos {c0[g].mean():.2f} · compras {p0[g].mean():.2f}")
for nm, x in (("vistas 0–30 min", rel(v0)), ("carritos 0–30 min", rel(c0)), ("compras 0–30 min", rel(p0)), ("vistas+10·carritos+40·compras", rel(v0 + 10 * c0 + 40 * p0))):
    r = spearmanr(x, y)[0]; print(f"     Spearman con compras posteriores (relativas a la subcategoría): {nm:30s} {r:5.2f}")
print(f"     dispersión entre productos nuevos de la cuota relativa posterior: cv = {y.std()/y.mean():.2f} (Poisson puro daría {1/np.sqrt(p1[g].mean()):.2f})")
# precio dentro de la subcategoría y etiquetas como predictores a priori
price = {r["product_id"]: float(r["price"]) for r in cat if r["change"] == "new_product"}
pr_all = np.concatenate([P["price"], [price[p] for p in ids[n0:]]])
zq = np.array([(pr_all[sub == sub[i]] > pr_all[i]).mean() for i in g])       # 1 = el más barato
print(f"     Spearman con el cuantil de precio (1 = barato): {spearmanr(zq, y)[0]:5.2f}")

# ---------- (B) niveles de la tendencia ----------
print("\n(B1) Episodios de tendencia por subcategoría (cubos de 15 min, cuota >= 2,5× la mediana y >= 30 compras)")
m = K == 2; b15 = ((T[m] - t0) // (15 * 60e3)).astype(int); nb = b15.max(); keep = b15 < nb; b15 = b15[keep]; Ip = I[m][keep]
PS = np.zeros((nb, S)); np.add.at(PS, (b15, sub[Ip]), 1); tot = PS.sum(1); share = PS / tot[:, None]; base = np.median(share, 0); lift = share / (base + 1e-9)
hot = (lift >= 2.5) & (PS >= 30); eps = [(b, s) for b, s in zip(*np.nonzero(hot))]
hs = sorted(set(s for _, s in eps))
print(f"     {len(eps)} cubos calientes en {len(hs)} subcategorías: " + ", ".join(f"{subs[s]}({sum(1 for _, x in eps if x == s)})" for s in hs))
PP = np.zeros((nb, n)); np.add.at(PP, (b15, Ip), 1)
print("     Dentro de la subcategoría caliente, ¿se reparte entre productos como fuera del episodio?")
G = df = 0.0; top = []
for s in hs:
    bs = np.array([b for b, x in eps if x == s]); cool = np.setdiff1d(np.arange(nb), np.concatenate([bs, bs - 1, bs + 1]))
    pr = np.flatnonzero(sub == s); hotc = PP[bs][:, pr].sum(0); coolc = PP[cool][:, pr].sum(0)
    ok = coolc >= 5                                             # productos con base suficiente (los nuevos recién lanzados quedan fuera)
    e = coolc[ok] / coolc[ok].sum() * hotc[ok].sum(); o = hotc[ok]
    nz = o > 0; G += 2 * (o[nz] * np.log(o[nz] / e[nz])).sum(); df += ok.sum() - 1
    # ruido del propio 'e' (estimado con coolc): var relativa adicional ≈ hot/cool
    infl = 1 + hotc[ok].sum() / coolc[ok].sum()
    top.append((subs[s], int(hotc.sum()), len(pr), float((o / e).max()), float(np.corrcoef(o, e)[0, 1]), infl))
for nm, tot_h, k, mx, cr, infl in top: print(f"       {nm:28s} compras en episodio {tot_h:5d} · {k:3d} productos · corr(observado, esperado) {cr:4.2f} · mayor razón producto {mx:4.1f} · inflación esperada de G {infl:4.2f}")
# ¿el episodio aplana o concentra el reparto?  log(observado) frente a log(esperado) dentro de la sub caliente (pendiente 1 = proporcional)
xs, ys = [], []
for s_ in hs:
    bs = np.array([b for b, x in eps if x == s_]); cool = np.setdiff1d(np.arange(nb), np.concatenate([bs, bs - 1, bs + 1]))
    pr = np.flatnonzero(sub == s_); hotc = PP[bs][:, pr].sum(0); coolc = PP[cool][:, pr].sum(0); ok = coolc >= 5
    e = coolc[ok] / coolc[ok].sum() * hotc[ok].sum(); xs.append(np.log(e)); ys.append(hotc[ok])
xs = np.concatenate(xs); ys = np.concatenate(ys); beta = 1.0
for _ in range(50):                                             # Poisson: y ~ exp(a + beta·log e), Newton en 2 parámetros
    X = np.c_[np.ones_like(xs), xs]; a_b = np.array([0.0, beta]) if _ == 0 else a_b
    mu = np.exp(X @ a_b); a_b = a_b + np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (ys - mu))
se = np.sqrt(np.linalg.inv(X.T @ (X * mu[:, None]))[1, 1]); beta = a_b[1]
print(f"     pendiente de log(observado) sobre log(esperado) en episodio: {beta:.2f} ± {se:.2f} (Poisson; con sobredispersión el error real es mayor)")
print(f"     G/gl = {G/df:.2f}  (1 ≈ reparto idéntico al habitual; hay que compararlo con la inflación esperada)")

print("\n(B2) Quitando el efecto de subcategoría, ¿quedan ráfagas por otra dimensión? (z por cubo de 15 min; var ≈ 1 = no)")
wi = PP.sum(0) + 0.5; ws = np.bincount(sub, weights=wi, minlength=S); wi = wi / ws[sub]          # cuota habitual del producto dentro de su subcategoría
EXP = PS[:, sub] * wi[None, :]                                                                 # esperado por producto y cubo dado el total de la subcategoría
def dim_report(name, lab, nl):
    O = np.zeros((nb, nl)); E = np.zeros((nb, nl))
    for l in range(nl): cols = np.flatnonzero(lab == l); O[:, l] = PP[:, cols].sum(1); E[:, l] = EXP[:, cols].sum(1)
    ok = E >= 10; z = (O[ok] - E[ok]) / np.sqrt(E[ok])
    print(f"     {name:34s} celdas {ok.sum():6d} · var(z) {z.var():5.2f} · max z {z.max():5.1f} · celdas con z>4: {(z>4).sum()}")
    return O, E
# etiquetas: dos por producto → dos pasadas (etiqueta 1 y etiqueta 2 como dimensiones separadas no tiene sentido; se usa pertenencia)
allt = sorted({t for v in tags.values() for t in v}); tid = {t: k for k, t in enumerate(allt)}
tags_of = lambda p: tags.get(p, [])          # los productos nuevos sin etiqueta en la caché quedan fuera de esta dimensión
TG = np.zeros((n, len(allt)), bool)
for i, p in enumerate(ids):
    for t in tags_of(p): TG[i, tid[t]] = True
print(f"     etiquetas por producto: media {TG.sum(1).mean():.2f} (nuevos: {TG[n0:].sum(1).mean():.2f})")
O = PP @ TG; E = EXP @ TG; ok = E >= 10; z = (O[ok] - E[ok]) / np.sqrt(E[ok])
print(f"     {'etiqueta (20)':34s} celdas {ok.sum():6d} · var(z) {z.var():5.2f} · max z {z.max():5.1f} · celdas con z>4: {(z>4).sum()}")
# etiqueta dentro de la subcategoría caliente
zz = []
for b, s in eps:
    cols = sub == s; o = PP[b, cols] @ TG[cols]; e = EXP[b, cols] @ TG[cols]; k = e >= 5; zz.extend(((o[k] - e[k]) / np.sqrt(e[k])).tolist())
zz = np.array(zz); print(f"     {'etiqueta dentro de la sub caliente':34s} celdas {len(zz):6d} · var(z) {zz.var():5.2f} · max z {zz.max():5.1f}")
# tercil de precio dentro de la subcategoría, y nuevo/antiguo
q3 = np.zeros(n, int)
for s in range(S):
    c = np.flatnonzero(sub == s); q3[c] = np.minimum((np.argsort(np.argsort(pr_all[c])) * 3 // len(c)), 2)
dim_report("tercil de precio en la subcategoría", q3, 3)
dim_report("nuevo frente a antiguo", (np.arange(n) >= n0).astype(int), 2)
# referencia: la propia subcategoría frente a su cuota habitual (aquí sí debe verse la tendencia)
E_s = tot[:, None] * (PS.sum(0) / PS.sum())[None, :]; ok = E_s >= 10; z = (PS[ok] - E_s[ok]) / np.sqrt(E_s[ok])
print(f"     {'(referencia) subcategoría':34s} celdas {ok.sum():6d} · var(z) {z.var():5.2f} · max z {z.max():5.1f} · celdas con z>4: {(z>4).sum()}")
catl = np.asarray(P["sub_cat"])[sub]; PC = np.zeros((nb, 10)); np.add.at(PC, (b15, catl[Ip]), 1)
# categoría una vez descontada la subcategoría caliente: quitar las subcategorías calientes y mirar el resto de la categoría
msk = ~np.isin(sub, hs); PCr = np.zeros((nb, 10)); np.add.at(PCr, (b15[msk[Ip]], catl[Ip][msk[Ip]]), 1)
E_c = PCr.sum(1)[:, None] * (PCr.sum(0) / PCr.sum())[None, :]; z = (PCr - E_c) / np.sqrt(E_c)
print(f"     {'categoría (sin las subs calientes)':34s} celdas {z.size:6d} · var(z) {z.var():5.2f} · max z {z.max():5.1f} · celdas con z>4: {(z>4).sum()}")
# ¿la categoría de la sub caliente también sube durante el episodio?
r = []
for b, s in eps:
    c = P["sub_cat"][s]; r.append(PCr[b, c] / E_c[b, c])
print(f"     resto de la categoría de la sub caliente durante el episodio: razón observada/esperada {np.mean(r):.2f} ± {np.std(r)/np.sqrt(len(r)):.2f}")
