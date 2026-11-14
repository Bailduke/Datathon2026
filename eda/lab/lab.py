"""Laboratorio offline: utilidades comunes de backtest. No escribe nada fuera de eda/lab/."""
import os, sys, pickle, time, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, ".cache")
spec = importlib.util.spec_from_file_location("rv", os.path.join(HERE, "recommender-lab.py"))
rv = importlib.util.module_from_spec(spec); sys.modules["rv"] = rv; spec.loader.exec_module(rv)
assert str(rv.CACHE) == C, rv.CACHE
V2_START = 1794684200000          # 2026-11-14 19:23:20 UTC en ms: arranque de tengo-aurilla-v2

def load_P():
    return pickle.loads((rv.CACHE / "priors_live.pkl").read_bytes())

def get_data(refresh=False, max_age_h=1.0):
    f = os.path.join(C, "dl.pkl")
    if os.path.exists(f) and not refresh and time.time() - os.path.getmtime(f) < max_age_h * 3600:
        return pickle.load(open(f, "rb"))
    assert time.strftime("%d%H%M") < "150930", "sin consultas a RawTree después de las 09:30 del domingo"
    if os.path.exists(f): os.replace(f, f.replace("dl.pkl", "dl_prev.pkl"))     # la copia de laboratorio de download_feedback lee dl.pkl si existe
    d = rv.download_feedback(rv.make_rt()); pickle.dump(d, open(f, "wb"), protocol=4); return d

def base_replay(P, fb, events, cat, tag="base", **kw):
    f = os.path.join(C, f"replay_{tag}_{len(fb)}.pkl")
    if os.path.exists(f) and not kw:
        return pickle.load(open(f, "rb"))
    rv.CFG.thompson = False
    R = rv.replay_feedback(P, rv.CFG, fb, events, cat, **kw); R.pop("state", None)
    if not kw: pickle.dump(R, open(f, "wb"), protocol=4)
    return R

def fit(A, y, off, l2=1e-4):
    return rv.poisson_fit(A, y, off, l2)

def ll(A, y, off, b):
    mu = np.exp(np.clip(A @ b + off, -50, 20)); return float((y * np.log(mu) - mu).sum())

def evaluate(A0, A1, y, off, t, groups=None, rid=None, label=""):
    """Compara el diseño A1 (con la idea) con A0 (base): logver dentro de muestra, fuera de muestra temporal
    (ajuste con el primer 70 % del tiempo, evaluación con el 30 % final) y validación cruzada por lista (5 partes)."""
    out = {}
    b0, _, l0 = fit(A0, y, off); b1, se1, l1 = fit(A1, y, off); out["dentro"] = l1 - l0
    cut = np.quantile(t, 0.7); tr, te = t <= cut, t > cut
    c0, _, _ = fit(A0[tr], y[tr], off[tr]); c1, _, _ = fit(A1[tr], y[tr], off[tr])
    out["temporal"] = ll(A1[te], y[te], off[te], c1) - ll(A0[te], y[te], off[te], c0); out["n_te"] = int(y[te].sum())
    rng = np.random.default_rng(0); u = np.unique(rid); fo = dict(zip(u, rng.integers(0, 5, u.size))); fold = np.array([fo[r] for r in rid])
    cv = 0.0; cvg = {g: 0.0 for g in (groups or {})}
    for k in range(5):
        tr, te = fold != k, fold == k
        c0, _, _ = fit(A0[tr], y[tr], off[tr]); c1, _, _ = fit(A1[tr], y[tr], off[tr])
        cv += ll(A1[te], y[te], off[te], c1) - ll(A0[te], y[te], off[te], c0)
        for g, m in (groups or {}).items():
            mm = te & m; cvg[g] += ll(A1[mm], y[mm], off[mm], c1) - ll(A0[mm], y[mm], off[mm], c0)
    out["cv"] = cv; out.update({f"cv_{g}": v for g, v in cvg.items()})
    k0 = A0.shape[1]
    print(f"{label:46s} dentro {out['dentro']:+7.1f} · temporal {out['temporal']:+7.1f} ({out['n_te']} compras) · CV {cv:+7.1f}"
          + "".join(f" · {g} {v:+.1f}" for g, v in cvg.items())
          + " · coef " + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b1[k0:], se1[k0:])), flush=True)
    return out, b1, se1

def slate(A, y, off, b, rid, mask=None):
    """Log-verosimilitud de QUÉ producto se compró dentro de cada lista con compra (los efectos de lista se cancelan).
    Es la métrica que mide la calidad del orden. Devuelve (suma, nº de listas)."""
    eta = A @ b + off
    sel = np.flatnonzero(y > 0) if mask is None else np.flatnonzero((y > 0) & mask)
    lists = np.unique(rid[sel]); inl = np.isin(rid, lists)
    e = eta[inl]; r = rid[inl]; yy = y[inl]
    u, inv = np.unique(r, return_inverse=True); mx = np.full(u.size, -1e9); np.maximum.at(mx, inv, e)
    den = np.log(np.bincount(inv, weights=np.exp(e - mx[inv]))) + mx
    if mask is not None: keep = (yy > 0) & mask[inl]
    else: keep = yy > 0
    return float((e[keep] - den[inv][keep]).sum()), int(keep.sum())

def evaluate2(A0, A1, y, off, t, rid, nu, label="", verbose=True):
    """Calidad del ORDEN: logver dentro de la lista, fuera de muestra temporal (ajuste con el primer 70 % del tiempo)
    y por validación cruzada de 5 partes por lista; separada en usuarios nuevos y con historia."""
    b0, _, l0 = fit(A0, y, off); b1, se1, l1 = fit(A1, y, off)
    cut = np.quantile(t, 0.7); tr, te = t <= cut, t > cut
    c0, _, _ = fit(A0[tr], y[tr], off[tr]); c1, _, _ = fit(A1[tr], y[tr], off[tr])
    tmp = slate(A1[te], y[te], off[te], c1, rid[te])[0] - slate(A0[te], y[te], off[te], c0, rid[te])[0]
    rng = np.random.default_rng(0); u = np.unique(rid); fo = np.zeros(rid.max() + 1, int); fo[u] = rng.integers(0, 5, u.size); fold = fo[rid]
    cv = cvn = cvo = 0.0
    for k in range(5):
        tr2, te2 = fold != k, fold == k
        d0, _, _ = fit(A0[tr2], y[tr2], off[tr2]); d1, _, _ = fit(A1[tr2], y[tr2], off[tr2])
        a = A1[te2], y[te2], off[te2], d1, rid[te2]; z = A0[te2], y[te2], off[te2], d0, rid[te2]
        cv += slate(*a)[0] - slate(*z)[0]; cvn += slate(*a, nu[te2])[0] - slate(*z, nu[te2])[0]; cvo += slate(*a, ~nu[te2])[0] - slate(*z, ~nu[te2])[0]
    k0 = A0.shape[1]
    if verbose:
        print(f"{label:44s} Poisson dentro {l1-l0:+6.1f} | orden: temporal {tmp:+6.1f} · CV {cv:+6.1f} (nuevos {cvn:+.1f} · historia {cvo:+.1f}) | coef "
              + " ".join(f"{bi:+.2f}±{si:.2f}" for bi, si in zip(b1[k0:], se1[k0:])), flush=True)
    return dict(dentro=l1 - l0, temporal=tmp, cv=cv, cv_nuevos=cvn, cv_historia=cvo), b1, se1

def score(A, y, off, t, rid, nu):
    """Valores absolutos para comparar diseños no anidados: Poisson dentro de muestra y logver del orden fuera de muestra."""
    _, _, l = fit(A, y, off)
    cut = np.quantile(t, 0.7); tr, te = t <= cut, t > cut
    c, _, _ = fit(A[tr], y[tr], off[tr]); tmp = slate(A[te], y[te], off[te], c, rid[te])[0]
    rng = np.random.default_rng(0); u = np.unique(rid); fo = np.zeros(rid.max() + 1, int); fo[u] = rng.integers(0, 5, u.size); fold = fo[rid]
    cv = cvn = cvo = 0.0
    for k in range(5):
        tr2, te2 = fold != k, fold == k
        d, _, _ = fit(A[tr2], y[tr2], off[tr2]); a = A[te2], y[te2], off[te2], d, rid[te2]
        cv += slate(*a)[0]; cvn += slate(*a, nu[te2])[0]; cvo += slate(*a, ~nu[te2])[0]
    return np.array([l, tmp, cv, cvn, cvo])
