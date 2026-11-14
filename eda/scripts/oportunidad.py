import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q, TEAM
Z = "toDateTime64('1971-01-01 00:00:00',3)"
r = q(f"""WITH rq AS (SELECT request_id::String AS rid, user_id::String AS u, ts::DateTime64(3) AS t, toString(context.page) AS page,
                    toString(is_new_user) AS nu FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL 20 MINUTE),
 up AS (SELECT user_id::String AS u, product_id::String AS p,
          maxIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta_last, minIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta,
          minIf(ts::DateTime64(3), event_type::String='purchase') AS tp
        FROM (SELECT user_id, product_id, ts, event_type FROM shop.history_events WHERE event_type::String IN ('add_to_cart','purchase')
              UNION ALL SELECT user_id, product_id, ts, event_type FROM shop.events WHERE event_type::String IN ('add_to_cart','purchase'))
        WHERE user_id::String IN (SELECT u FROM rq) GROUP BY u, p),
 per AS (SELECT rq.rid AS rid, any(rq.page) AS page, any(rq.nu) AS nu,
          countIf(up.ta > {Z} AND up.ta < rq.t AND NOT (up.tp > {Z} AND up.tp < rq.t)) AS n_open,
          countIf(up.ta > {Z} AND up.ta < rq.t AND NOT (up.tp > {Z} AND up.tp < rq.t) AND up.ta_last > rq.t - INTERVAL 24 HOUR AND up.ta_last < rq.t) AS n_recent,
          countIf(up.ta > {Z} AND up.ta < rq.t AND NOT (up.tp > {Z} AND up.tp < rq.t) AND up.ta_last > rq.t - INTERVAL 30 MINUTE AND up.ta_last < rq.t) AS n_30m
        FROM rq LEFT JOIN up ON up.u = rq.u GROUP BY rid)
 SELECT nu AS usuario_nuevo, count() AS peticiones,
        round(100*countIf(n_open = 0)/count(),1) AS pct_sin_carrito, round(100*countIf(n_open BETWEEN 1 AND 9)/count(),1) AS pct_1a9, round(100*countIf(n_open >= 10)/count(),1) AS pct_10mas,
        round(avg(n_open),1) AS media_abiertos, round(100*countIf(n_recent > 0)/count(),1) AS pct_con_reciente_24h, round(avg(n_recent),2) AS media_recientes,
        round(100*countIf(n_30m > 0)/count(),1) AS pct_con_carrito_30min
 FROM per GROUP BY nu ORDER BY nu""", 900)
rows = r.get("data")
print("PETICIONES DE LOS ÚLTIMOS 20 MIN · productos en carrito sin comprar disponibles por petición")
if rows is None: print(r)
else:
    for x in rows: print("  ", x)
# qué está mostrando el v4
r = q(f"""WITH sb AS (SELECT s.request_id::String AS rid, s.items::Array(String) AS items, rq.user_id::String AS u, rq.ts::DateTime64(3) AS t
                 FROM {TEAM}.submissions AS s INNER JOIN shop.requests AS rq ON rq.request_id::String = s.request_id::String
                 WHERE s.model_version::String = 'tengo-aurilla-v0'),
 up AS (SELECT user_id::String AS u, product_id::String AS p,
          maxIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta_last, minIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta,
          minIf(ts::DateTime64(3), event_type::String='purchase') AS tp
        FROM (SELECT user_id, product_id, ts, event_type FROM shop.history_events WHERE event_type::String IN ('add_to_cart','purchase')
              UNION ALL SELECT user_id, product_id, ts, event_type FROM shop.events WHERE event_type::String IN ('add_to_cart','purchase'))
        WHERE user_id::String IN (SELECT u FROM sb) GROUP BY u, p),
 it AS (SELECT sb.rid AS rid, arrayJoin(arrayEnumerate(sb.items)) AS slot, sb.items[slot] AS p, sb.u AS u, sb.t AS t FROM sb)
 SELECT count(DISTINCT it.rid) AS listas, round(countIf(up.ta > {Z} AND up.ta < it.t AND NOT (up.tp > {Z} AND up.tp < it.t)) / count(DISTINCT it.rid), 2) AS carrito_por_lista,
        round(countIf(slot <= 3 AND up.ta > {Z} AND up.ta < it.t AND NOT (up.tp > {Z} AND up.tp < it.t)) / count(DISTINCT it.rid), 2) AS carrito_en_slots_1a3,
        round(countIf(up.ta_last > it.t - INTERVAL 24 HOUR AND up.ta_last < it.t AND NOT (up.tp > {Z} AND up.tp < it.t)) / count(DISTINCT it.rid), 2) AS recientes_por_lista,
        countIf(up.tp > {Z} AND up.tp < it.t) AS ya_comprados_mostrados
 FROM it LEFT JOIN up ON up.u = it.u AND up.p = it.p""", 900)
print("\nLISTAS ENVIADAS POR tengo-aurilla-v0:", r.get("data") or r)
