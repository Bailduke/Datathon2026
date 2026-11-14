import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q, TEAM
CAT = "(SELECT product_id::String AS pid, category::String AS cat FROM shop.catalog)"
UP = """(SELECT user_id::String AS u, product_id::String AS p,
          minIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta, minIf(ts::DateTime64(3), event_type::String='purchase') AS tp
        FROM (SELECT user_id, product_id, ts, event_type FROM shop.history_events WHERE event_type::String IN ('add_to_cart','purchase')
              UNION ALL SELECT user_id, product_id, ts, event_type FROM shop.events WHERE event_type::String IN ('add_to_cart','purchase'))
        GROUP BY u, p)"""
Z = "toDateTime64('1971-01-01 00:00:00',3)"
r = q(f"""WITH imp AS (
   SELECT rq.user_id::String AS u, f.ts::DateTime64(3) AS t, arrayJoin(arrayEnumerate(f.shown::Array(String))) AS slot,
     (f.shown::Array(String))[slot] AS p, toUInt8(ifNull(toString(f.purchased),'') = (f.shown::Array(String))[slot]) AS b,
     if(length(rq.recent_views) > 0, arrayElement(arrayMap(x -> x.product_id::String, rq.recent_views::Array(JSON)), -1), '') AS lastv
   FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS rq ON rq.request_id::String = f.request_id::String
   WHERE f.request_id::String != '__init__')
 SELECT if(slot <= 3, '1-3', '4-10') AS pos,
   multiIf(up.u != '' AND up.tp > {Z} AND up.tp < imp.t, '1 ya comprado', up.u != '' AND up.ta > {Z} AND up.ta < imp.t, '2 en carrito sin comprar', '3 resto') AS estado,
   multiIf(imp.lastv = '', 'home (sin sesión)', c1.cat = c2.cat, 'categoría de la sesión', 'otra categoría') AS rel,
   count() AS impresiones, sum(b) AS compras, round(100*sum(b)/count(),3) AS conv_pct
 FROM imp LEFT JOIN {UP} AS up ON up.u = imp.u AND up.p = imp.p
 LEFT JOIN {CAT} AS c1 ON c1.pid = imp.p LEFT JOIN {CAT} AS c2 ON c2.pid = imp.lastv
 GROUP BY pos, estado, rel ORDER BY pos, estado, rel""", 900)
rows = r.get("data")
if rows is None: print(r); raise SystemExit
print(f"{'slots':<6}{'estado':<26}{'relación con la sesión':<26}{'impr.':>9}{'compras':>9}{'conv %':>8}")
for x in rows: print(f"{x['pos']:<6}{x['estado']:<26}{x['rel']:<26}{x['impresiones']:>9}{x['compras']:>9}{x['conv_pct']:>8}")
