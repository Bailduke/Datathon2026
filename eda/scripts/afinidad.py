import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q, TEAM
UP = """(SELECT user_id::String AS u, product_id::String AS p,
          minIf(ts::DateTime64(3), event_type::String='view') AS tv, minIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta,
          minIf(ts::DateTime64(3), event_type::String='purchase') AS tp,
          countIf(event_type::String='view') AS nv
        FROM ({src}) WHERE cityHash64(user_id::String) % {mod} = 0 GROUP BY u, p)"""
EST = """multiIf(up.u = '', '4 sin interacción previa',
                up.tp > toDateTime64('1971-01-01 00:00:00',3) AND up.tp < imp.t, '1 ya comprado',
                up.ta > toDateTime64('1971-01-01 00:00:00',3) AND up.ta < imp.t, '2 en carrito, sin comprar',
                up.tv > toDateTime64('1971-01-01 00:00:00',3) AND up.tv < imp.t, '3 visto, sin carrito ni compra',
                '4 sin interacción previa')"""
def show(title, rows):
    print("\n" + title)
    print(f"{'slots':<8}{'estado':<34}{'impresiones':>12}{'compras':>9}{'conv %':>9}")
    for r in rows: print(f"{r['pos']:<8}{r['estado']:<34}{r['impresiones']:>12}{r['compras']:>9}{r['conv_pct']:>9}")
# --- histórico (muestra 1/10 de usuarios)
r = q(f"""WITH imp AS (SELECT user_id::String AS u, ts::DateTime64(3) AS t, arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
     (shown::Array(String))[slot] AS p, toUInt8(ifNull(toString(purchased),'') = (shown::Array(String))[slot]) AS b
     FROM shop.history_sessions WHERE cityHash64(user_id::String) % 10 = 0)
  SELECT if(slot <= 3, '1-3', '4-10') AS pos, {EST} AS estado, count() AS impresiones, sum(b) AS compras, round(100*sum(b)/count(),3) AS conv_pct
  FROM imp LEFT JOIN {UP.format(src="SELECT user_id, product_id, ts, event_type FROM shop.history_events", mod=10)} AS up ON up.u = imp.u AND up.p = imp.p
  GROUP BY pos, estado ORDER BY pos, estado""", 600)
show("CARRUSEL ANTIGUO (history_sessions, 1/10 de usuarios)", r.get("data") or [r])
# --- en vivo (todos los usuarios con feedback)
src = "SELECT user_id, product_id, ts, event_type FROM shop.history_events UNION ALL SELECT user_id, product_id, ts, event_type FROM shop.events"
r = q(f"""WITH imp AS (SELECT rq.user_id::String AS u, f.ts::DateTime64(3) AS t, arrayJoin(arrayEnumerate(f.shown::Array(String))) AS slot,
     (f.shown::Array(String))[slot] AS p, toUInt8(ifNull(toString(f.purchased),'') = (f.shown::Array(String))[slot]) AS b
     FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS rq ON rq.request_id::String = f.request_id::String
     WHERE f.request_id::String != '__init__' AND cityHash64(rq.user_id::String) % 2 = 0)
  SELECT if(slot <= 3, '1-3', '4-10') AS pos, {EST} AS estado, count() AS impresiones, sum(b) AS compras, round(100*sum(b)/count(),3) AS conv_pct
  FROM imp LEFT JOIN {UP.format(src=src, mod=2)} AS up ON up.u = imp.u AND up.p = imp.p
  GROUP BY pos, estado ORDER BY pos, estado""", 900)
show("CARRUSEL EN VIVO (nuestro feedback, 1/2 de usuarios)", r.get("data") or [r])
