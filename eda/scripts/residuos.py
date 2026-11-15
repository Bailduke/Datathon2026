"""Residuos del modelo (v6, reproducción guardada en .cache/assaig.npz): compras observadas / esperadas por tramos de muchas variables.
Razón = 1 → el modelo acierta el nivel en ese tramo; lejos de 1 → patrón que el modelo no recoge.
Solo se ven los productos que nosotros mostramos (selección): un tramo con razón alta está infravalorado al menos entre los mostrados.
No consulta RawTree."""
import os, sys, pickle, importlib.util, logging, calendar, time, bisect, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes())
fb, events, cat = pickle.load(open(os.path.join(C, "dl_latest.pkl"), "rb"))
z = np.load(os.path.join(C, "assaig.npz")); X, PN, AF, gid = z["Xf"], z["PNf"], z["AFf"], z["gid"]
y = X[:, 1]; pos = X[:, 0].astype(int); w = np.asarray(P["w_pos"])[pos]
A = rv.glm_design(X, PN, AF); b, se, ll = rv.poisson_fit(A, y, np.log(w)); eta = A @ b; mu = w * np.exp(eta)
print(f"{len(y)} impresiones · {int(y.sum())} compras · esperadas {mu.sum():.0f} · logver {ll:.1f}")

# ---------- variables por fila ----------
ms = lambda s: calendar.timegm(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) * 1000
ids = list(P["ids"]); sub = list(P["sub"]); subs = list(P["subs"]); sidx = {s: k for k, s in enumerate(subs)}; price = list(P["price"]); launch = [None] * 5000
pch = {}
for r in cat:
    p = r["product_id"]
    if r["change"] == "new_product" and p not in set(ids[5000:]) and r["subcategory"] in sidx:
        ids.append(p); sub.append(sidx[r["subcategory"]]); price.append(float(r["price"])); launch.append(ms(r["ts"]))
    elif r["change"] == "price_change": pch.setdefault(p, []).append((ms(r["ts"]), float(r["price"])))
index = {p: i for i, p in enumerate(ids)}; n = len(ids); sub = np.array(sub); price = np.array(price)
base = price.copy()
for p, l in pch.items():
    if p in index: base[index[p]] = max(base[index[p]], max(v for _, v in l)); l.sort()
zq = np.zeros(n); rk = np.zeros(n)
for s in range(len(subs)):
    c = np.flatnonzero(sub == s); zq[c] = 1 - np.argsort(np.argsort(base[c])) / max(len(c) - 1, 1)          # 1 = el más barato
    co = c[c < 5000]; rk[co] = np.argsort(np.argsort(-P["C_hist"][co])) / max(len(co) - 1, 1)                # 0 = el más vendido
tags = pickle.load(open(os.path.join(C, "tags.pkl"), "rb"))
uev = {}
for t, u, p, e in events: uev.setdefault(u, []).append((t, e))
ut = {u: [x[0] for x in l] for u, l in uev.items()}; ub = {u: [x[0] for x in l if x[1] == "purchase"] for u, l in uev.items()}
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"]); t0 = events[0][0]
M = len(y); V = {k: np.full(M, -1, int) for k in ("page", "dev", "reg", "nu", "hora", "nv", "edad", "rk", "zq", "pabs", "sale", "nev", "rec", "ncomp", "cat", "mismo", "tag")}
LAB = dict(page=["home", "category", "product"], dev=list(P["devs"]), reg=list(P["regs"]), nu=["con historial", "usuario nuevo"],
           hora=[f"{2*h:02d}–{2*h+2:02d} h" for h in range(9)], nv=["0 vistas", "1", "2", "3", "4+"],
           edad=["antiguo", "nuevo <1 h", "1–2 h", "2–4 h", "4–8 h", "8+ h"], rk=["top 10 %", "10–25 %", "25–50 %", "50–75 %", "75–100 %"],
           zq=["más caro", "2", "3", "4", "más barato"], pabs=["<20 €", "20–50", "50–100", "100–250", "250+"],
           sale=["sin rebaja", "rebaja <15 %", "15–30 %", "30 %+"], nev=["0 eventos", "1–2", "3–5", "6–10", "11–20", "21–50", "51+"],
           rec=["sin eventos", "<1 min", "1–5 min", "5–30 min", "30 min+"], ncomp=["0 compras hoy", "1", "2", "3+"], cat=list(P["cats"]),
           mismo=["sin vistas", "es un visto", "misma subcategoría que el último visto", "misma categoría", "otra categoría"],
           tag=["0 etiquetas comunes con el último visto", "1", "2"])
pg = {v: k for k, v in enumerate(LAB["page"])}; dv = {v: k for k, v in enumerate(LAB["dev"])}; rg = {v: k for k, v in enumerate(LAB["reg"])}
subcat = np.asarray(P["sub_cat"]); m = 0
for g, k in enumerate(order):
    r = fb[k]; t = r["t"]; u = r["user_id"]; sh = (r["shown"] or [])[:10]; L = len(sh); sl = slice(m, m + L)
    vs = [index[x] for x in rv.normalize_views(r["recent_views"]) if x in index]
    V["page"][sl] = pg.get(r["page"], -1); V["dev"][sl] = dv.get(r["device"], -1); V["reg"][sl] = rg.get(r["region"], -1); V["nu"][sl] = int(r["nu"] == "true")
    V["hora"][sl] = min(int((t - t0) / 7200e3), 8); V["nv"][sl] = min(len(vs), 4)
    tl = ut.get(u, ()); ne = bisect.bisect_left(tl, t); V["nev"][sl] = np.searchsorted([1, 3, 6, 11, 21, 51], ne, side="right")
    V["rec"][sl] = 0 if ne == 0 else 1 + int(np.searchsorted([60e3, 300e3, 1800e3], t - tl[ne - 1], side="right"))
    V["ncomp"][sl] = min(bisect.bisect_left(ub.get(u, ()), t), 3)
    lv = vs[-1] if vs else -1; tl_ = set(tags.get(ids[lv], ())) if lv >= 0 else None
    for j, pid in enumerate(sh):
        i = index.get(pid, -1)
        if i < 0: continue
        V["edad"][m + j] = 0 if i < 5000 else 1 + int(np.searchsorted([1, 2, 4, 8], (t - launch[i]) / 3600e3, side="right"))
        if i < 5000: V["rk"][m + j] = int(np.searchsorted([.10, .25, .50, .75], rk[i], side="right"))
        V["zq"][m + j] = min(int(zq[i] * 5), 4); V["cat"][m + j] = subcat[sub[i]]
        pr = base[i]; l = pch.get(pid)
        if l:
            q = bisect.bisect_right(l, (t, 1e18)) - 1
            if q >= 0 and t - l[q][0] < 7200e3: pr = l[q][1]
        d = 1 - pr / base[i]; V["sale"][m + j] = 0 if d < 0.005 else 1 + int(np.searchsorted([.15, .30], d, side="right"))
        V["pabs"][m + j] = int(np.searchsorted([20, 50, 100, 250], pr, side="right"))
        if lv < 0: V["mismo"][m + j] = 0
        else:
            V["mismo"][m + j] = 1 if i in vs else 2 if sub[i] == sub[lv] else 3 if subcat[sub[i]] == subcat[sub[lv]] else 4
            if i not in vs and pid in tags: V["tag"][m + j] = len(tl_ & set(tags[pid]))
    m += L
assert m == M
def qbin(x, k, fmt="{:.2f}"):
    ok = np.isfinite(x); e = np.unique(np.quantile(x[ok], np.linspace(0, 1, k + 1))); v = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)
    return v, [f"{fmt.format(e[q])}…{fmt.format(e[q+1])}" for q in range(len(e) - 1)]
V["pos"] = pos; LAB["pos"] = [f"posición {k+1}" for k in range(10)]
V["lp"], LAB["lp"] = qbin(eta, 10); V["logh"], LAB["logh"] = qbin(np.log(np.maximum(PN, 1e-12)), 10)
tr = AF[:, 8].astype(float); V["trend"] = np.searchsorted([-0.3, -0.1, 0.1, 0.4, 0.8, 1.5], tr, side="right"); LAB["trend"] = ["<−0,3", "−0,3…−0,1", "≈0", "0,1…0,4", "0,4…0,8", "0,8…1,5", ">1,5"]
for c, nm in ((0, "cf_hist"), (1, "cf_ses"), (3, "cf_vhoy"), (4, "tag_hist"), (5, "tag_ses")):
    x = AF[:, c].astype(float); nzm = x != 0; v = np.zeros(M, int); q, lab = qbin(x[nzm], 5); v[nzm] = 1 + q; V[nm] = v; LAB[nm] = ["= 0"] + lab
TIT = dict(pos="Posición en la lista", page="Página", dev="Dispositivo", reg="Región", nu="Tipo de usuario", hora="Hora del stream (desde las 10:00 CET)", nv="Vistas recientes en la petición",
           edad="Edad del producto", rk="Ventas históricas dentro de la subcategoría", zq="Precio dentro de la subcategoría", pabs="Precio absoluto", sale="Rebaja activa",
           nev="Eventos conocidos del usuario hoy", rec="Tiempo desde el último evento del usuario", ncomp="Compras del usuario hoy", cat="Categoría del producto",
           mismo="Relación con el último visto", tag="Etiquetas comunes con el último visto (no vistos)", lp="Decil de la predicción del modelo", logh="Decil de log h (jerárquica)",
           trend="Tendencia de la subcategoría (log τ)", cf_hist="Co-compra con el historial", cf_ses="Co-compra con la sesión", cf_vhoy="Co-compra con las vistas de hoy",
           tag_hist="Etiquetas con el historial", tag_ses="Etiquetas con la sesión")

# ---------- dos tipos de residuo ----------
# NIVEL: Poisson sobre todas las impresiones (¿acierta cuánta gente compra en esa situación?). No afecta al orden si la variable es del usuario.
# ORDEN: dentro de cada lista con compra, P(comprado = j | lista) = softmax(log w_j + x_j·b). Es lo que decide el ranking.
has = np.bincount(gid, weights=y, minlength=int(gid.max()) + 1)[gid] > 0
Mx = A[has][:, 3:]; yc = y[has]; off = np.log(w[has]); _, inv = np.unique(gid[has], return_inverse=True); ng = inv.max() + 1; bc = np.zeros(Mx.shape[1])
for _ in range(200):
    e = Mx @ bc + off; mx = np.full(ng, -np.inf); np.maximum.at(mx, inv, e); ex = np.exp(e - mx[inv]); p = ex / np.bincount(inv, weights=ex, minlength=ng)[inv]
    xb = np.zeros((ng, Mx.shape[1])); np.add.at(xb, inv, Mx * p[:, None])
    st = np.linalg.solve((Mx * p[:, None]).T @ Mx - xb.T @ xb + 1e-3 * np.eye(Mx.shape[1]), Mx.T @ (yc - p) - 1e-3 * bc); k = np.abs(st).max(); bc += st / k if k > 1 else st
    if k < 1e-8: break
V["lpc"], LAB["lpc"] = qbin(Mx @ bc, 10); TIT["lpc"] = "Decil de la puntuación (dentro de la lista)"
tmp = np.zeros(M, int); tmp[has] = V.pop("lpc"); V["lpc"] = tmp
ITEM = ["edad", "cat", "rk", "logh", "zq", "pabs", "sale", "mismo", "tag", "trend", "cf_hist", "cf_ses", "cf_vhoy", "tag_hist", "tag_ses", "pos", "lpc"]
USER = ["nev", "rec", "ncomp", "nv", "hora", "nu", "page", "dev", "reg"]
def table(keys, obs, exp, var, mask, emin):
    R = {}
    for k in keys:
        v = V[k][mask]; res = []
        for q, lab in enumerate(LAB[k]):
            mk = v == q; e = exp[mk].sum(); o = obs[mk].sum()
            if e < emin: continue
            sd = np.sqrt(var[mk].sum()); res.append((lab, int(mk.sum()), o, e, o / e, sd / e, (o - e) / sd))
        R[k] = res
    return R
R_ord = table(ITEM, yc, p, p * (1 - p), has, 2.0)
R_niv = table(USER, y, mu, mu, np.ones(M, bool), 3.0)
OUT = os.path.join(HERE, "..", "residuos"); os.makedirs(OUT, exist_ok=True)
def dump(R, name, title):
    rank = sorted(R, key=lambda k: -sum(r[6] ** 2 for r in R[k]) / max(len(R[k]), 1))
    L = [title, "", "Variables ordenadas por desajuste (media de z² por tramo; ≈1 = nada que ver)"]
    L += [f"  {TIT[k]:52s} {sum(r[6]**2 for r in R[k])/max(len(R[k]),1):6.2f}" for k in rank]
    for k in rank:
        L += ["", TIT[k]] + [f"   {lab:42s} n={nn:8d} obs={o:6.0f} esp={e:7.1f} razón={ra:5.2f} ±{s:4.2f} z={zz:+5.1f}{'  <<<' if abs(zz) >= 3 else ''}" for lab, nn, o, e, ra, s, zz in R[k]]
    open(os.path.join(OUT, name), "w").write("\n".join(L) + "\n"); print("\n".join(L[:3 + len(rank)])); return rank
k_ord = dump(R_ord, "tabla_orden.txt", f"ORDEN dentro de la lista · {int(ng)} listas con compra · compras observadas / esperadas por el modelo (v6) entre los 10 productos mostrados")
k_niv = dump(R_niv, "tabla_nivel.txt", f"NIVEL · {M} impresiones, {int(y.sum())} compras · compras observadas / esperadas (Poisson, v6). Variables del usuario: no cambian el orden de la lista")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, textwrap
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6b7280", "#2f6fb3", "#c2570c"
def fig(R, keys, name, title, sub):
    nc = 3; nr = -(-len(keys) // nc); f, axs = plt.subplots(nr, nc, figsize=(16, 3.9 * nr), squeeze=False)
    for ax, k in zip(axs.ravel(), keys):
        res = R[k]; x = np.arange(len(res)); ra = np.array([r[4] for r in res]); s = np.array([r[5] for r in res]); zz = np.array([r[6] for r in res])
        lo = np.maximum(ra - 1.96 * s, 0.05); hi = ra + 1.96 * s; flag = np.abs(zz) >= 3
        ax.axhline(1, color=MUTED, lw=1, zorder=1)
        for m_, c_ in ((~flag, BLUE), (flag, ORANGE)):
            if m_.any(): ax.errorbar(x[m_], ra[m_], yerr=[ra[m_] - lo[m_], hi[m_] - ra[m_]], fmt="o", color=c_, ms=7, lw=2, capsize=0, zorder=3)
        for xi, r_, f_ in zip(x, ra, flag):
            if f_: ax.annotate(f"{r_:.2f}".replace(".", ","), (xi, r_), xytext=(7, 4), textcoords="offset points", fontsize=9, color=INK)
        ax.set_yscale("log"); top = max(2.0, hi.max() * 1.15); bot = min(0.5, max(lo.min() * 0.87, 0.05)); ax.set_ylim(bot, top)
        tk = [t for t in (0.1, 0.2, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 5, 8) if bot <= t <= top]; ax.set_yticks(tk); ax.set_yticklabels([str(t).replace(".", ",") for t in tk]); ax.minorticks_off()
        ax.set_xticks(x); ax.set_xticklabels([textwrap.fill(r[0], 13) for r in res], fontsize=8, rotation=0 if len(res) <= 7 else 40, ha="center" if len(res) <= 7 else "right")
        ax.set_title(TIT[k], fontsize=11, color=INK, loc="left"); ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#d1d5db")
        ax.tick_params(colors=MUTED, length=0); ax.set_xlim(-0.6, len(res) - 0.4)
    for ax in axs.ravel()[len(keys):]: ax.axis("off")
    from matplotlib.lines import Line2D
    f.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=2, label="dentro del ruido"), Line2D([], [], marker="o", color=ORANGE, lw=2, label="desajuste claro (|z| ≥ 3)")],
             loc="upper right", frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(0.99, 0.995))
    f.suptitle(title, x=0.01, ha="left", fontsize=15, color=INK, y=0.998); f.text(0.01, 1 - 0.36 / (3.9 * nr), sub, fontsize=10, color=MUTED, ha="left", va="top")
    f.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (3.9 * nr))); f.savefig(os.path.join(OUT, name), dpi=110, facecolor="white"); plt.close(f)
fig(R_ord, k_ord, "residuos_orden.png", "Residuos de ORDEN: compras observadas / esperadas dentro de la lista",
    "1 = el modelo acierta. Por encima de 1 = esos productos se compran más de lo que el modelo cree (están demasiado abajo). Barras: intervalo del 95 %. Escala logarítmica.")
fig(R_niv, k_niv, "residuos_nivel.png", "Residuos de NIVEL: compras observadas / esperadas por situación del usuario",
    "Variables del usuario o de la petición: desplazan la conversión de toda la lista, no el orden entre productos. Barras: intervalo del 95 %. Escala logarítmica.")
print("\nescrito en", os.path.normpath(OUT))
