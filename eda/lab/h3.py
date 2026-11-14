"""H3 · ¿Satura la lista? Conversión real de la lista frente a la suma prevista S de sus productos, y si la
concentración (misma subcategoría / categoría / etiquetas) resta."""
import numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
R = base_replay(P, fb, events, cat)
X, PN, AF, KI = R["X"], R["PN"], R["AF"], R["KI"]
y = X[:, 1]; w = np.asarray(P["w_pos"])[X[:, 0].astype(int)]; off = np.log(w); A = rv.glm_design(X, PN, AF)
b, se, _ = fit(A, y, off)
bi = b.copy(); bi[1] = bi[2] = 0.0                       # tasa del producto sin los efectos de lista
r = np.exp(np.clip(A @ bi, -50, 20)) * w                  # contribución prevista de cada impresión (salvo constante de lista)
k = KI[:, 0]; nl = len(fb)
S = np.bincount(k, weights=r, minlength=nl); Y = np.bincount(k, weights=y, minlength=nl)
nu = np.array([f["nu"] == "true" for f in fb], float); home = np.array([f["page"] == "home" for f in fb], float)
mv = np.array([f["mv"] for f in fb]); t = np.array([f["t"] for f in fb])
sub = np.concatenate([np.asarray(P["sub"]), np.full(4000, -1)]); catg = np.concatenate([np.asarray(P["cat"]), np.full(4000, -1)])
i = KI[:, 1]; si = np.where((i >= 0) & (i < 5000), sub[np.clip(i, 0, 8999)], -1 - (i % 97)); ci = np.where((i >= 0) & (i < 5000), catg[np.clip(i, 0, 8999)], -1 - (i % 11))
def hhi(g):
    key = k.astype(np.int64) * 1000 + (g + 200)
    u, inv = np.unique(key, return_inverse=True); m = np.bincount(inv, weights=r)
    return np.bincount((u // 1000).astype(int), weights=m ** 2, minlength=nl) / np.maximum(S, 1e-300) ** 2
Hs, Hc = hhi(si), hhi(ci)
top = np.bincount(k, weights=r * (X[:, 0] == 0), minlength=nl) / np.maximum(S, 1e-300)       # peso del slot 1 en S
ok = S > 0
print(f"listas {ok.sum()} · compras {int(Y[ok].sum())}")
print("conversión real frente a S (suma prevista), por deciles de S en cada grupo de listas:")
for name, m in (("baseline/otras", ok & (mv == "")), ("v0", ok & (mv == "tengo-aurilla-v0")), ("v2", ok & (mv == "tengo-aurilla-v2"))):
    qs = np.quantile(S[m], [0, .25, .5, .75, .9, 1.0]); out = []
    for a, c in zip(qs[:-1], qs[1:]):
        mm = m & (S >= a) & (S <= c); out.append(f"S~{S[mm].mean()*1e5:6.1f}: {int(Y[mm].sum()):3d}/{mm.sum():5d}={100*Y[mm].mean():.2f}%")
    print(f"  {name:15s} " + " | ".join(out))
def lfit(cols, names, m=ok):
    Z = np.column_stack([np.ones(m.sum())] + [c[m] for c in cols]); bb, ss, l = fit(Z, Y[m], np.zeros(m.sum()))
    print(f"   logver {l:9.1f} · " + " ".join(f"{n} {x:+.3f}±{s:.3f}" for n, x, s in zip(["const"] + names, bb, ss))); return l
lS = np.log(np.maximum(S, 1e-12))
print("Poisson a nivel de lista: log E[compra] = c + nuevo + home + gamma·log S (+ concentración)")
l0 = lfit([nu, home, lS], ["nuevo", "home", "gamma"])
lfit([nu, home, lS, Hs], ["nuevo", "home", "gamma", "HHI_subcat"])
lfit([nu, home, lS, Hc], ["nuevo", "home", "gamma", "HHI_cat"])
lfit([nu, home, lS, top], ["nuevo", "home", "gamma", "peso_slot1"])
lfit([nu, home, lS, (t - t.min()) / 3.6e6], ["nuevo", "home", "gamma", "horas"])
lfit([nu, home, lS, (mv == "tengo-aurilla-v0").astype(float), (mv == "tengo-aurilla-v2").astype(float)], ["nuevo", "home", "gamma", "es_v0", "es_v2"])
lfit([nu, home, lS, (t - t.min()) / 3.6e6, (mv == "tengo-aurilla-v0").astype(float), (mv == "tengo-aurilla-v2").astype(float)], ["nuevo", "home", "gamma", "horas", "es_v0", "es_v2"])
print("solo listas anteriores a las 18:10 UTC (sin v0/v2):"); m0 = ok & (t < 1794679800000)
lfit([nu, home, lS], ["nuevo", "home", "gamma"], m0); lfit([nu, home, lS, (t - t.min()) / 3.6e6], ["nuevo", "home", "gamma", "horas"], m0)
print("media de HHI_subcat: otras %.2f · v0 %.2f · v2 %.2f | HHI_cat: %.2f · %.2f · %.2f" % (Hs[ok & (mv == "")].mean(), Hs[mv == "tengo-aurilla-v0"].mean(), Hs[mv == "tengo-aurilla-v2"].mean(), Hc[ok & (mv == "")].mean(), Hc[mv == "tengo-aurilla-v0"].mean(), Hc[mv == "tengo-aurilla-v2"].mean()))
