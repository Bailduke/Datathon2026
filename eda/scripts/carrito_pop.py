import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q, TEAM
UP = """(SELECT user_id::String AS u, product_id::String AS p,
          minIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta, minIf(ts::DateTime64(3), event_type::String='purchase') AS tp,
          countIf(event_type::String='add_to_cart') AS n_cart, maxIf(ts::DateTime64(3), event_type::String='add_to_cart') AS ta_last
        FROM (SELECT user_id, product_id, ts, event_type FROM shop.history_events WHERE event_type::String IN ('add_to_cart','purchase')
              UNION ALL SELECT user_id, product_id, ts, event_type FROM shop.events WHERE event_type::String IN ('add_to_cart','purchase'))
        GROUP BY u, p)"""
POP = "(SELECT product_id::String AS pid, count() AS c FROM shop.history_events WHERE event_type::String='purchase' GROUP BY pid)"
Z = "toDateTime64('1971-01-01 00:00:00',3)"
base = f"""WITH imp AS (
   SELECT rq.user_id::String AS u, f.ts::DateTime64(3) AS t, arrayJoin(arrayEnumerate(f.shown::Array(String))) AS slot,
     (f.shown::Array(String))[slot] AS p, toUInt8(ifNull(toString(f.purchased),'') = (f.shown::Array(String))[slot]) AS b
   FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS rq ON rq.request_id::String = f.request_id::String
   WHERE f.request_id::String != '__init__' AND length(rq.recent_views) > 0)"""
def run(title, group, where="1"):
    r = q(base + f""" SELECT multiIf(up.u != '' AND up.tp > {Z} AND up.tp < imp.t, '1 ya comprado', up.u != '' AND up.ta > {Z} AND up.ta < imp.t, '2 carrito', '3 resto') AS estado,
        {group} AS g, count() AS impresiones, sum(b) AS compras, round(100*sum(b)/count(),3) AS conv_pct
      FROM imp LEFT JOIN {UP} AS up ON up.u = imp.u AND up.p = imp.p LEFT JOIN {POP} AS pp ON pp.pid = imp.p
      WHERE {where} GROUP BY estado, g HAVING estado != '1 ya comprado' ORDER BY estado, g""", 900)
    rows = r.get("data")
    print("\n" + title)
    if rows is None: print(r); return
    for x in rows: print(f"  {x['estado']:<12}{str(x['g']):<28}{x['impresiones']:>9}{x['compras']:>8}{x['conv_pct']:>8}")
run("Por popularidad del producto (compras en el histórico), todas las posiciones, peticiones con sesión:",
    "multiIf(pp.c < 30, 'a <30 compras', pp.c < 100, 'b 30-99', pp.c < 300, 'c 100-299', 'd 300+')")
run("Productos del carrito: por nº de veces añadido", "multiIf(up.n_cart <= 1, '1 vez', up.n_cart = 2, '2 veces', '3+ veces')", "up.u != ''")
run("Productos del carrito: por antigüedad del último añadido", "multiIf(dateDiff('hour', up.ta_last, imp.t) < 24, 'a <24 h', dateDiff('day', up.ta_last, imp.t) < 14, 'b 1-14 días', 'c >14 días')", "up.u != ''")
