"""Assaig de la final: usuaris sense història i amb pocs esdeveniments.
 (1) Pesos propis per a usuaris nous?  (2) Afinitat directa amb l'últim vist i màxim amb el carretó d'avui?
Es reprodueix el stream dues vegades: amb tota la informació de l'usuari i 'en fred' (només sessió, sense esdeveniments d'avui)."""
import os, sys, pickle, importlib.util, logging, time, numpy as np
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv6", os.path.join(HERE, "../../tengo-aurilla-v0/recommender-v6.py")); rv = importlib.util.module_from_spec(spec); sys.modules["rv6"] = rv; spec.loader.exec_module(rv)
P = pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes()); rv.apply_params_v6(P); cfg = rv.CFG; cfg.thompson = False
f = os.path.join(C, "dl_latest.pkl")
if os.path.exists(f) and time.time() - os.path.getmtime(f) < 3 * 3600: fb, events, cat = pickle.load(open(f, "rb"))
else:
    fb, events, cat = rv.download_feedback(rv.make_rt()); pickle.dump((fb, events, cat), open(f, "wb"), protocol=4)
rv.load_affinity(None, P, cfg)
NB, NT = rv.N_BASE, rv.N_TASTE; w_pos = np.asarray(P["w_pos"])
R_full = rv.replay_feedback(P, cfg, fb, events, cat)
def cold(self, user, i, kind, organic=True, ts=None):          # l'usuari no acumula res d'avui: només compres (sacietat), popularitat i tendència
    if not user or i is None or i < 0 or i >= self.n: return
    if ts is not None and organic: self.trend.add(int(self.sub[i]), ts)
    if kind == "purchase":
        self.add_bought(user, i)
        if organic: self.C_live[i] += 1; self.T_live += 1
orig = rv.State.user_event; rv.State.user_event = cold
R_cold = rv.replay_feedback(P, cfg, fb, events, cat); rv.State.user_event = orig
# mateix ordre de files que el replay, per afegir-hi columnes pròpies
z = np.load(rv.CACHE / "afinidad.npz"); F, sg = z["F"], z["sg"]; Fs = F * sg; nF = F.shape[0]
st = rv.State(P, cfg)
for r in cat:
    if r["change"] == "new_product" and r["product_id"] not in st.index: st.add_product(r, 0.0)
index = st.index
carts = {}
for t, u, p, e in events:
    if e == "add_to_cart" and p in index: carts.setdefault(u, []).append((t, index[p]))
order = sorted(range(len(fb)), key=lambda k: fb[k]["t"])
M = len(R_full["X"]); gid = np.zeros(M, int); last = np.zeros(M); cmax = np.zeros(M); tt = np.zeros(M); m = 0
for g, k in enumerate(order):
    r = fb[k]; vs = [index[x] for x in rv.normalize_views(r["recent_views"]) if x in index]
    vl = vs[-1] if vs else -1
    cu = [i for (t, i) in carts.get(r["user_id"], ()) if t < r["t"] and i < nF]
    for pid in (r["shown"] or [])[:10]:
        i = index.get(pid, -1); gid[m] = g; tt[m] = r["t"]
        if 0 <= i < nF:
            if 0 <= vl < nF and vl != i: last[m] = float(Fs[i] @ F[vl])
            cj = [j for j in cu if j != i]
            if cj: cmax[m] = float((F[cj] @ Fs[i]).max())
        m += 1
assert m == M
def fit(A, y, off, l2=1e-3):
    b = np.zeros(A.shape[1]); b[0] = np.log(max(y.sum(), 1) / np.exp(off).sum())
    for _ in range(300):
        mu = np.exp(np.clip(A @ b + off, -50, 20)); g_ = A.T @ (y - mu) - l2 * b; H = (A * mu[:, None]).T @ A + l2 * np.eye(len(b))
        s = np.linalg.solve(H, g_); mx = np.abs(s).max(); b = b + (s / mx if mx > 1 else s)
        if mx < 1e-8: break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))
def order_ll(A, b, off, y, gid_, mask):
    e = (A @ b + off)[mask]; yy = y[mask]; gg = gid_[mask]; tot = 0.0; n = 0; hit = 0
    o = np.argsort(gg, kind="stable"); e, yy, gg = e[o], yy[o], gg[o]; bnd = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1], True])
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        if yy[a_:b_].sum() > 0:
            ee = e[a_:b_]; k = int(yy[a_:b_].argmax()); tot += ee[k] - (ee.max() + np.log(np.exp(ee - ee.max()).sum())); hit += int(ee.argmax() == k); n += 1
    return tot, n, hit
np.savez_compressed(os.path.join(C, "assaig.npz"), Xf=R_full["X"], PNf=R_full["PN"], AFf=R_full["AF"], Xc=R_cold["X"], PNc=R_cold["PN"], AFc=R_cold["AF"], gid=gid, last=last, cmax=cmax, tt=tt)
cut = np.quantile(tt, 0.7); tr, te = tt < cut, tt >= cut
names = list(rv.GLM_COLS)
for label, R in (("AMB TOTA LA INFORMACIÓ", R_full), ("EN FRED (només sessió, sense esdeveniments d'avui)", R_cold)):
    X, PN, AF = R["X"], R["PN"], R["AF"]; y = X[:, 1]; off = np.log(w_pos[X[:, 0].astype(int)]); nou = X[:, 2] > 0
    A = rv.glm_design(X, PN, AF)
    keep = [k for k in range(A.shape[1]) if np.abs(A[nou][:, k]).sum() > 0 or k == 0]          # columnes amb variació en usuaris nous
    print(f"\n=== {label} · {int(nou.sum())} impressions d'usuaris nous, {int(y[nou].sum())} compres ({int(y[nou & te].sum())} a la part d'avaluació)")
    b_all, _ = fit(A[tr], y[tr], off[tr]); o_all = order_ll(A, b_all, off, y, gid, te & nou)
    Ak = A[:, keep]; b_new, se_new = fit(Ak[tr & nou], y[tr & nou], off[tr & nou]); o_new = order_ll(Ak, b_new, off, y, gid, te & nou)
    o_pos = order_ll(A[:, :1], np.zeros(1), off, y, gid, te & nou)
    print(f"  ordre fora de mostra en usuaris nous ({o_all[1]} compres): només posició {o_pos[0]:.1f} · pesos generals {o_all[0]:.1f} (encerta {100*o_all[2]/o_all[1]:.0f} %) · pesos propis {o_new[0]:.1f} (encerta {100*o_new[2]/o_new[1]:.0f} %) · diferència {o_new[0]-o_all[0]:+.1f}")
    print("  pesos   general -> propis d'usuaris nous:")
    for kk, k in enumerate(keep):
        if names[k] in ("const", "usuario_nuevo", "sin_P", "ya_comprado"): continue
        print(f"     {names[k]:16s} {b_all[k]:+.2f} -> {b_new[kk]:+.2f} ± {se_new[kk]:.2f}")
    # idea 2: afinitat directa amb l'últim vist i màxim amb el carretó d'avui
    for nm, extra in (("últim vist", [last]), ("màxim amb el carretó d'avui", [cmax]), ("totes dues", [last, cmax])):
        if label.startswith("EN FRED") and nm != "últim vist": continue
        A2 = np.column_stack([A] + extra); b2, se2 = fit(A2[tr], y[tr], off[tr]); bf, sef = fit(A2, y, off)
        o2 = order_ll(A2, b2, off, y, gid, te); o1 = order_ll(A, b_all, off, y, gid, te)
        o2n = order_ll(A2, b2, off, y, gid, te & nou)
        print(f"  + {nm:28s} pes {' / '.join(f'{v:+.2f}±{s:.2f}' for v, s in zip(bf[-len(extra):], sef[-len(extra):]))} · millora de l'ordre fora de mostra: tots {o2[0]-o1[0]:+.1f} ({o1[1]} compres) · nous {o2n[0]-o_all[0]:+.1f}")

# pregunta clau: a la final, els pesos estimats amb tota la informació s'aplicaran a usuaris "en fred". Quant es perd?
Xf, Xc = R_full["X"], R_cold["X"]; y = Xf[:, 1]; off = np.log(w_pos[Xf[:, 0].astype(int)]); nou = Xf[:, 2] > 0
Af = rv.glm_design(Xf, R_full["PN"], R_full["AF"]); Ac = rv.glm_design(Xc, R_cold["PN"], R_cold["AF"])
b_full, _ = fit(Af[tr], y[tr], off[tr]); b_cold, _ = fit(Ac[tr], y[tr], off[tr])
print("\n=== USUARIS NOUS EN FRED, part d'avaluació: quins pesos ordenen millor?")
for nm, bb in (("pesos estimats amb tota la informació (el que faria ara el model)", b_full), ("pesos estimats en fred", b_cold)):
    o = order_ll(Ac, bb, off, y, gid, te & nou); print(f"  {nm:66s} ordre {o[0]:.1f} · encerta {100*o[2]/o[1]:.0f} %")
o = order_ll(Ac[:, :1], np.zeros(1), off, y, gid, te & nou); print(f"  {'només posició':66s} ordre {o[0]:.1f} · encerta {100*o[2]/o[1]:.0f} %")
for nm, msk in (("pàgina de producte o categoria", Xf[:, 3] == 0), ("portada", Xf[:, 3] > 0)):
    m_ = te & nou & msk; o1 = order_ll(Ac, b_full, off, y, gid, m_); o2 = order_ll(Ac, b_cold, off, y, gid, m_); o0 = order_ll(Ac[:, :1], np.zeros(1), off, y, gid, m_)
    print(f"  {nm}: {o1[1]} compres · posició {o0[0]:.1f} · pesos actuals {o1[0]:.1f} · pesos en fred {o2[0]:.1f}")
