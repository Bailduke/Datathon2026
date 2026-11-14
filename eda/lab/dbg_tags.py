import numpy as np
from lab import *
P = load_P(); fb, events, cat = pickle.load(open(os.path.join(C, "dl_2040.pkl"), "rb"))
h9 = pickle.load(open(os.path.join(C, "h9.pkl"), "rb")); AF, KI = h9["AF"].astype(float), h9["KI"]
z = np.load(os.path.join(rv.CACHE, "afinidad.npz")); tags = z["tags"]; n = 5000; K = 20
T = np.zeros((n, K)); T[np.arange(n), tags[:, 0]] = 1; T[np.arange(n), tags[:, 1]] = 1
print("productos con 2 etiquetas distintas:", int((T.sum(1) == 2).sum()), "· con etiquetas repetidas/1:", int((T.sum(1) == 1).sum()), "· tags -1:", int((tags < 0).sum()))
E = frozenset(); al = 6.0; shown = 0; diffs = []
for a in range(0, len(KI), 10):
    k = KI[a, 0]; u = fb[k]["user_id"]; hb = (P["bought"].get(u) or E) | (P["carted"].get(u) or E)
    if len(hb) < 3: continue
    b = np.array(sorted(hb)); cc = T[b].sum(0); tot = cc.sum()
    for q in range(10):
        i = KI[a + q, 1]
        if not (0 <= i < n): continue
        own = float(i in hb); mine = (T[i] * np.log((cc - own * T[i] + al / K) / (tot - 2 * own + al) * K)).sum()
        diffs.append((mine, AF[a + q, 4], own))
        if shown < 6 and abs(mine - AF[a + q, 4]) > 0.05:
            shown += 1; print(f"  usuario {u} producto {i} en cesta={own:.0f} |cesta|={len(hb)} · mío {mine:+.3f} · producción {AF[a+q,4]:+.3f} · etiquetas {tags[i]} cc={cc[tags[i]]}")
    if len(diffs) > 60000: break
d = np.array(diffs); print("corr %.4f · |dif| media %.4f · fracción con |dif|>0,05: %.3f · entre los que están en la cesta: %.3f · fuera: %.3f" % (np.corrcoef(d[:, 0], d[:, 1])[0, 1], np.abs(d[:, 0] - d[:, 1]).mean(), (np.abs(d[:, 0] - d[:, 1]) > .05).mean(), (np.abs(d[:, 0] - d[:, 1]) > .05)[d[:, 2] > 0].mean(), (np.abs(d[:, 0] - d[:, 1]) > .05)[d[:, 2] == 0].mean()))
