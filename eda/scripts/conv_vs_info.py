"""Conversión real del carrusel según cuánta información teníamos del usuario en el instante de la petición.
Información = eventos del usuario conocidos entonces: los del histórico (60 días) + los del stream anteriores a la petición."""
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rt import q, TEAM
out = os.path.join(HERE, ".cache", "conv_info.pkl")
t0 = time.time()
rows = q(f"""WITH rq AS (
    SELECT f.request_id::String AS rid, r.user_id::String AS u, r.ts::DateTime64(3) AS t, toString(f.model_version) AS mv,
           toString(f.served_by) AS sb, ifNull(toString(f.purchased), '') != '' AS buy, toString(r.is_new_user) AS nu,
           toString(r.context.page) AS page, length(r.recent_views::Array(JSON)) AS nrv
    FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
    WHERE f.request_id::String != '__init__'),
  hist AS (SELECT user_id::String AS u, count() AS nh, countIf(event_type::String = 'purchase') AS nhb,
                  countIf(event_type::String = 'add_to_cart') AS nhc FROM shop.history_events GROUP BY u),
  live AS (SELECT rq.rid AS rid, countIf(e.t < rq.t) AS nl, countIf(e.t < rq.t AND e.k = 'add_to_cart') AS nlc
           FROM rq INNER JOIN (SELECT user_id::String AS u, ts::DateTime64(3) AS t, event_type::String AS k FROM shop.events) AS e ON e.u = rq.u GROUP BY rid),
  prev AS (SELECT a.request_id::String AS rid, countIf(b.t < a.ts::DateTime64(3)) AS np
           FROM shop.requests AS a INNER JOIN (SELECT user_id::String AS u, ts::DateTime64(3) AS t FROM shop.requests) AS b ON b.u = a.user_id::String GROUP BY rid)
  SELECT rq.mv AS mv, rq.sb AS sb, rq.buy AS buy, rq.nu AS nu, rq.page AS page, rq.nrv AS nrv, toUnixTimestamp64Milli(rq.t) AS t,
         ifNull(hist.nh, 0) AS nh, ifNull(hist.nhb, 0) AS nhb, ifNull(hist.nhc, 0) AS nhc, ifNull(live.nl, 0) AS nl, ifNull(live.nlc, 0) AS nlc, ifNull(prev.np, 0) AS np
  FROM rq LEFT JOIN hist ON hist.u = rq.u LEFT JOIN live ON live.rid = rq.rid LEFT JOIN prev ON prev.rid = rq.rid""", 600)
if rows.get("data") is None:
    print(str(rows)[:600]); sys.exit(1)
pickle.dump(rows["data"], open(out, "wb"), protocol=4)
print(len(rows["data"]), "listas ·", round(time.time() - t0), "s")
