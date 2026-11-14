"""H1 · Vistas del histórico como cesta de gustos (etiquetas y co-compra) e indicador 'visto y nunca en el carrito'."""
import numpy as np
from lab import *
P = load_P(); fb, events, cat = get_data(max_age_h=99)
B = np.load(os.path.join(C, "base_arrays.npz")); A, y, off, rid, t, nu = B["A"], B["y"], B["off"], B["rid"], B["t"], B["nu"]
KI = pickle.load(open(os.path.join(C, "h9.pkl"), "rb"))["KI"]; assert np.array_equal(KI[:, 0], rid)
hv = pickle.load(open(os.path.join(C, "hist_views.pkl"), "rb"))
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); F, sg, tags = z["F"], z["sg"], z["tags"]; n = F.shape[0]; K = 20
Fs = F * sg; selfdot = (Fs * F).sum(1); T = np.zeros((n, K), np.float32); T[np.arange(n), tags[:, 0]] = 1; T[np.arange(n), tags[:, 1]] = 1
E = frozenset(); print("usuarios con vistas históricas:", len(hv), "· productos distintos vistos (media): %.0f" % np.mean([len(v[0]) for v in hv.values()]))
def sc(b, wts, ii, shr=3.0, al=6.0):
    """Puntuaciones de co-compra y etiquetas de los productos ii con la cesta b (pesos wts), sin contarse a sí mismos."""
    cf = np.zeros(len(ii)); tg = np.zeros(len(ii)); ok = (ii >= 0) & (ii < n)
    if b.size == 0 or not ok.any(): return cf, tg
    j = ii[ok]; pos = {int(x): k for k, x in enumerate(b)}; own = np.array([wts[pos[int(x)]] if int(x) in pos else 0.0 for x in j])
    W = wts.sum(); num = Fs[j] @ (F[b] * wts[:, None]).sum(0) - own * selfdot[j]; cf[ok] = num / (W - own + shr)
    cc = (T[b] * wts[:, None]).sum(0); cl = cc[None, :] - own[:, None] * T[j]; tl = 2 * (W - own)
    tg[ok] = (T[j] * np.log((cl + al / K) / (tl[:, None] + al) * K)).sum(1); return cf, tg
names = ["vistas (únicas)", "vistas (ponderadas por nº, raíz)", "vistas nunca en carrito"]
feat = {nm: [np.zeros(len(y)), np.zeros(len(y))] for nm in names}; hvonly = np.zeros(len(y)); nhv = np.zeros(len(y))
bounds = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1], True])
for a, c in zip(bounds[:-1], bounds[1:]):
    u = fb[rid[a]]["user_id"]; v = hv.get(u)
    if v is None: continue
    b = v[0].astype(np.int64); cnt = v[1].astype(float); ii = KI[a:c, 1]
    cart = (P["carted"].get(u) or E) | (P["bought"].get(u) or E); notc = np.array([x not in cart for x in b])
    for nm, (bb, ww) in zip(names, ((b, np.ones(b.size)), (b, np.sqrt(cnt)), (b[notc], np.ones(int(notc.sum()))))):
        cf, tg = sc(bb, ww, ii); feat[nm][0][a:c] = cf; feat[nm][1][a:c] = tg
    s = set(b[notc].tolist()); hvonly[a:c] = [int(x) in s for x in ii]; nhv[a:c] = b.size
print(f"compras {int(y.sum())} (con historia {int(y[~nu].sum())}) · impresiones 'visto y nunca en carrito': {100*hvonly.mean():.1f} %")
ev = lambda cols, lab_: evaluate2(A, np.column_stack([A] + cols), y, off, t, rid, nu, lab_)
for nm in names:
    ev(feat[nm], f"cesta = {nm}: co-compra + etiquetas"); ev([feat[nm][1]], f"cesta = {nm}: solo etiquetas")
ev([hvonly], "indicador: visto en el histórico y nunca en carrito")
ev(feat[names[0]] + [hvonly], "vistas únicas (2) + indicador")
ev([feat[names[0]][1], hvonly], "etiquetas de vistas únicas + indicador")
np.savez_compressed(os.path.join(C, "h1_feats.npz"), cf=feat[names[0]][0], tg=feat[names[0]][1], hvonly=hvonly, cf_nc=feat[names[2]][0], tg_nc=feat[names[2]][1])
