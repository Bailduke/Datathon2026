"""Descarga única: productos vistos por cada usuario en el histórico (con nº de vistas). eda/lab/.cache/hist_views.pkl"""
import numpy as np
from lab import *
assert time.strftime("%d%H%M") < "150930"
P = load_P(); index = {p: i for i, p in enumerate(P["ids"])}; rt = rv.make_rt()
out = {}; NP = 24; tot = 0; t0 = time.time()
for k in range(NP):
    rows = rt.query(f"""SELECT u, groupArray(p) AS ps, groupArray(c) AS cs FROM (
        SELECT user_id::String AS u, product_id::String AS p, count() AS c FROM shop.history_events
        WHERE event_type::String = 'view' AND cityHash64(user_id::String) % {NP} = {k} GROUP BY u, p) GROUP BY u""", 600)
    for r in rows:
        ii = np.array([index.get(p, -1) for p in r["ps"]], np.int32); cc = np.array(r["cs"], np.int32); ok = ii >= 0
        out[r["u"]] = (ii[ok].astype(np.int16), np.minimum(cc[ok], 30000).astype(np.int16)); tot += int(ok.sum())
    print(f"parte {k}: {len(out)} usuarios · {tot} pares · {time.time()-t0:.0f} s", flush=True); time.sleep(4)
pickle.dump(out, open(os.path.join(C, "hist_views.pkl"), "wb"), protocol=4)
