"""Descarga a disco todo lo necesario para analizar nuestras impresiones del carrusel en vivo:
listas mostradas + contexto de la petición, eventos orgánicos en vivo de cada usuario y el resumen
histórico de cada par (usuario, producto mostrado). Salida: .cache/imp_*.pkl"""
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt import q, TEAM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(OUT, exist_ok=True)


def run(sql, timeout=600, tries=3):
    for k in range(tries):
        r = q(sql, timeout)
        if isinstance(r, dict) and r.get("data") is not None:
            return r["data"]
        print("   reintento:", str(r)[:300]); time.sleep(2 + 3 * k)
    raise RuntimeError("consulta fallida")


t0 = time.time()
fb = []
for k in range(4):
    fb += run(f"""SELECT f.request_id::String AS rid, r.user_id::String AS u, toUnixTimestamp64Milli(r.ts::DateTime64(3)) AS t,
        toString(r.context.page) AS page, toString(r.context.device) AS dev, toString(r.is_new_user) AS nu,
        r.recent_views AS rv, f.shown::Array(String) AS shown, ifNull(toString(f.purchased), '') AS buy,
        ifNull(toString(f.clicked), '') AS clk, toString(f.served_by) AS sb, toString(f.model_version) AS mv,
        ifNull(toString(f.fallback_reason), '') AS fr
        FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
        WHERE cityHash64(f.request_id::String) % 4 = {k} AND f.request_id::String != '__init__'""")
    print(f"feedback parte {k}: {len(fb)} filas acumuladas ({time.time()-t0:.0f} s)", flush=True)
pickle.dump(fb, open(os.path.join(OUT, "imp_fb.pkl"), "wb"), protocol=4)

lim = run("SELECT toString(min(ts::DateTime64(3))) AS a, toString(max(ts::DateTime64(3))) AS b, count() AS n FROM shop.events")[0]
print("shop.events:", lim, flush=True)
NP = 12
ev = {}
for k in range(NP):
    rows = run(f"""SELECT user_id::String AS u, groupArray(product_id::String) AS p, groupArray(event_type::String) AS e,
        groupArray(toUnixTimestamp64Milli(ts::DateTime64(3))) AS t
        FROM (SELECT user_id, product_id, event_type, ts FROM shop.events WHERE cityHash64(user_id::String) % {NP} = {k}
              ORDER BY ts::DateTime64(3)) GROUP BY u""")
    for x in rows:
        ev[x["u"]] = (x["p"], x["e"], x["t"])
    print(f"eventos parte {k}: {len(ev)} usuarios ({time.time()-t0:.0f} s)", flush=True)
pickle.dump(ev, open(os.path.join(OUT, "imp_ev.pkl"), "wb"), protocol=4)

hist = []
for k in range(4):
    hist += run(f"""SELECT user_id::String AS u, product_id::String AS p,
        countIf(event_type::String = 'view') AS nv, countIf(event_type::String = 'add_to_cart') AS nc,
        countIf(event_type::String = 'purchase') AS nb,
        toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'view')) AS tv,
        toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'add_to_cart')) AS tc,
        toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'purchase')) AS tb
        FROM shop.history_events
        WHERE cityHash64(user_id::String) % 4 = {k} AND (user_id::String, product_id::String) IN (
            SELECT DISTINCT r.user_id::String, arrayJoin(f.shown::Array(String))
            FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
            WHERE f.request_id::String != '__init__' AND cityHash64(r.user_id::String) % 4 = {k})
        GROUP BY u, p""", 900)
    print(f"histórico parte {k}: {len(hist)} pares ({time.time()-t0:.0f} s)", flush=True)
pickle.dump(hist, open(os.path.join(OUT, "imp_hist.pkl"), "wb"), protocol=4)
print("hecho en", round(time.time() - t0), "s")
