# Recomendador en tiempo real · team_03 · Datathon FME 2026

Solución del equipo **team_03** al reto de RawTree: responder en menos de 3 segundos, con una lista de
10 productos, a cada petición de un carrusel de recomendaciones, sobre un stream en vivo de una tienda
simulada (usuarios, productos y tendencias nuevos mientras corre).

El modelo es **estadístico, pequeño y explicable**: una probabilidad jerárquica de compra multiplicada por
una afinidad usuario-producto. Tiene unas pocas decenas de parámetros, todos con interpretación, y cada
decisión de diseño se comprobó con datos antes de entrar. No hay redes ni árboles: se probaron y no
mejoraban (ver [Lo que se probó y se descartó](#lo-que-se-probó-y-se-descartó)).

- **Código de la final:** [tengo-aurilla-v0/recommender-v8.py](tengo-aurilla-v0/recommender-v8.py) (un solo fichero, 2.457 líneas, solo `numpy` y `requests`).
- **Documento del modelo:** [tengo-aurilla-v0/modelo-neta-v8.pdf](tengo-aurilla-v0/modelo-neta-v8.pdf) (19 páginas: definición, estimación, validación y limitaciones).
- **Recorrido de una petición, en una página:** [docs/pipeline-de-una-peticion-transformada-neta-v8.html](docs/pipeline-de-una-peticion-transformada-neta-v8.html).

## Qué corrió en la final

| | |
|---|---|
| Ventana | Domingo 15/11/2026, 10:00–12:00 CET |
| Programa | `tengo-aurilla-v0/recommender-v8.py`, un solo proceso |
| Versión declarada | `transformada-neta-v8` |
| Pesos | Calibrados a las 09:11 con todo el feedback acumulado (3.814 compras en 236.000 listas); copia en [tengo-aurilla-v0/params_final_v8.json](tengo-aurilla-v0/params_final_v8.json) |
| Tiempo de cálculo por petición | p50 1,0 ms · p99 2,2 ms |
| Antigüedad de la petición al enviar la respuesta | p50 0,75 s · p95 1,1 s (límite: 3 s) |
| **Resultado** | **2.º puesto** en el reto principal (clasificación final, con la valoración del jurado) · **1.º en el reto Tukey** con el [cuadro de mando](quadre-comandament/) |
| Leaderboard automático | puntuación 56,21, la tercera más alta (las dos primeras: 58,28 y 57,73) |
| Conversión | 2,189 % (baseline 0,750 % · referencia de la organización 2,141 %) |
| Conversión con usuarios «cold» | 2,553 % |
| Respuestas a tiempo | 99,93 % (16 de 21.600 perdidas, 15 de ellas por una caída de 5 s de la API a las 11:56) |
| Ingresos por petición | 1,146 € |

## El modelo en una página

Para el usuario `u` y el producto `i`, en una petición:

```
score(i) = h(i)^θ · A(u, i)          se envían los 10 productos elegibles de mayor score
```

**`h(i)`: probabilidad jerárquica de compra.** Categoría → subcategoría → producto.

- *Categoría y subcategoría:* Dirichlet-multinomial por usuario. Parte de la cuota poblacional y se
  actualiza con los eventos del usuario (una compra cuenta 1, un carrito 0,25, una vista 0,02) y con lo que
  ha visto en la sesión.
- *Producto dentro de la subcategoría:* logit condicional con la popularidad (Gamma-Poisson: histórico más
  compras en vivo), el precio relativo dentro de la subcategoría según el perfil de precio del usuario, y
  si está rebajado.

**`A(u, i)`: afinidad usuario-producto.** Multiplica, no redistribuye.

- *Clases:* el producto está en el carrito de hoy, en un carrito del histórico, es el único visto en la
  sesión, el primero de varios vistos, u otro de los vistos.
- *Gustos:* cuatro «cestas» del usuario (historia, sesión, carrito de hoy, vistas de hoy) comparadas con
  el producto por dos vías: las **etiquetas** de la descripción (cada producto lleva 2 de 20) y la
  **co-compra** (16 factores latentes del log-lift producto-producto, una vez descontado lo que ya explica
  la jerarquía).
- *Tendencia:* por subcategoría, eventos recientes entre eventos esperados
  (media móvil de 5 minutos frente a la cuota habitual de 4 horas).

**Elegibles:** productos en stock que el usuario no ha comprado (ni en el histórico, ni en la tienda, ni
en nuestro carrusel).

**Producto nuevo (v8):** no tiene historia de co-compra. En vez de ceros, hereda la media de los factores
de los productos de su subcategoría que comparten alguna etiqueta con él.

## Qué aprende del stream

| Pieza | Cuándo se actualiza | De dónde |
|---|---|---|
| Sesión del usuario | En cada petición | `recent_views` de `shop.requests` |
| Gustos del usuario (categorías, carrito y vistas de hoy, perfil de precio) | Con cada evento suyo | `shop.events`, cada 10 s |
| Productos ya comprados | Con cada compra, orgánica o de nuestro carrusel | `shop.events` y `team_03.feedback` |
| Popularidad de cada producto | Con cada compra orgánica | `shop.events` |
| Tendencia por subcategoría | Con cada evento; las medias avanzan cada minuto | `shop.events` |
| Precio, rebaja, stock y productos nuevos | En ≈ 1 s; el stock se revalida al enviar | `shop.catalog_updates` |
| Pesos de la afinidad, de la tendencia y `θ` | Con `--calibrate`: reproduce el stream y reajusta con todo nuestro feedback (≈ 5 min) | `team_03.feedback` |

Un usuario nuevo empieza solo con la sesión y la jerarquía poblacional, y gana precisión con cada evento.
Un producto nuevo entra con la popularidad típica de su subcategoría, sus etiquetas y los factores
heredados, y su popularidad se corrige con las compras desde su lanzamiento.

## Cómo se validó

Tres herramientas, de menos a más fiable:

1. **Reproducción del stream.** Se vuelve a pasar todo el stream con el código de producción y se mide la
   verosimilitud de *qué producto de la lista se compró* (logit condicional dentro de cada lista), dentro
   de muestra, con corte temporal 70/30 y con validación cruzada.
2. **Residuos.** Compras observadas entre esperadas, por tramos de cada variable
   ([eda/residuos/](eda/residuos/)).
3. **A/B en vivo.** Dos versiones a la vez, cada una con la mitad de las peticiones (`--shard K/N`, por
   hash del `request_id`, sin duplicados), comparadas en el leaderboard por versión y en compras reales.

| Cambio | Evidencia |
|---|---|
| Tendencia por subcategoría (v6) | A/B: 2,08 % → 2,49 % de compras reales (± 0,19) |
| Tres clases de sesión y pesos estimados dentro de la lista (v7) | A/B contra v6: leaderboard 2,09 → 2,16; compras reales 1,98 % → 2,29 % (± 0,21) |
| Factores heredados para productos nuevos (v8) | Residuos: los productos nuevos se compraban 2,4–2,7 veces lo esperado y era el único fallo de orden. Reproducción: log-verosimilitud −3424,3 → −3398,9 sin parámetros nuevos. A/B contra v7: leaderboard 2,16 → 2,26; compras reales 2,33 % → 2,13 % (± 0,18) |

El A/B del v8 no demuestra la mejora: las dos medidas se contradicen y ambas están dentro del ruido. La
ganancia esperada es pequeña (en torno al +0,5 % relativo) porque los productos nuevos son pocas compras.
Se eligió para la final porque arregla un fallo identificado, no dio errores y no iba peor.

### Lo que se probó y se descartó

Todo con datos, y casi todo por la misma razón: no mejoraba el orden fuera de muestra.

- **Modelos más flexibles:** XGBoost (para la afinidad y para todo el modelo), NMF, factorización
  usuario-producto, binomial negativa.
- **Reglas de lista:** diversidad forzada por categoría o subcategoría, fatiga de exposición, pesos de
  posición estimados en vivo.
- **Señales:** clics como objetivo de ajuste, antigüedad de las vistas, vistas repetidas, vistas del
  histórico como cesta, tiempo desde la reposición, edad del producto nuevo, tendencias por etiqueta o por
  precio, precio y rebaja dentro del carrusel.
- **Segmentos:** pesos propios para usuarios nuevos, priors por dispositivo.
- **Productos nuevos:** cambiar su prior de popularidad, un multiplicador fijo de «novedad» (el exceso no
  era novedad, era la co-compra que les faltaba).

El diario de los experimentos está en [eda/lab/HALLAZGOS.md](eda/lab/HALLAZGOS.md).

## Cómo ejecutarlo

Requiere Python 3.10 o posterior, `numpy` y `requests`, y las credenciales de RawTree.

```bash
python -m venv .venv && .venv/bin/pip install numpy requests
cp .env.example .env        # rellenar TEAM_ID y RAWTREE_API_KEY (no se sube nunca al repo)
```

Desde la raíz del repo:

```bash
.venv/bin/python tengo-aurilla-v0/recommender-v8.py --build       # estima la jerarquía con el histórico y la guarda en .cache/
.venv/bin/python tengo-aurilla-v0/recommender-v8.py --calibrate   # reajusta afinidad, tendencia y θ con el feedback en vivo
.venv/bin/python tengo-aurilla-v0/recommender-v8.py --smoke       # responde 30 peticiones reales sin enviarlas
.venv/bin/python tengo-aurilla-v0/recommender-v8.py --dry-run     # sistema completo sin escribir en RawTree
.venv/bin/python tengo-aurilla-v0/recommender-v8.py               # en vivo
```

Opciones útiles:

| Opción | Para qué |
|---|---|
| `--model-version NOMBRE` | Nombre con el que aparece la versión en el leaderboard |
| `--shard K/N` | Responde solo las peticiones con `hash(request_id) % N == K` (A/B sin duplicados) |
| `--set clave=valor` | Cambia un parámetro; por ejemplo `--set impute_new=false` reproduce el v7 y `--set trend=false` quita la tendencia |
| `--eval N` | Evaluación offline con N impresiones del histórico |

En la final corrió dentro de un bucle que lo relanza si cae:

```bash
while true; do .venv/bin/python tengo-aurilla-v0/recommender-v8.py --model-version transformada-neta-v8; sleep 1; done
```

**Solo un proceso puede escribir a la vez** en `team_03.submissions`, salvo que se reparta el tráfico con
`--shard`.

## Arquitectura del proceso en vivo

```
shop.requests ──► request_loop ×2 (cada 50 ms) ──► answer() ──► push_loop ──► team_03.submissions
                                                      ▲
                         estado en memoria (foto inmutable por petición)
                                                      ▲
     events_loop (10 s)      catalog_loop (0,75 s)      feedback_loop (2 s)      health_loop
       shop.events          shop.catalog_updates        team_03.feedback        métricas y log
```

- `answer()` no hace ninguna consulta: todo lo que necesita está en memoria.
- Al arrancar reconstruye el estado desde RawTree (catálogo, eventos, tendencias y compras del carrusel)
  en unos 20 segundos y no repite las peticiones ya respondidas.
- Una petición que ya no llegaría a tiempo se descarta en vez de enviarse tarde.

## Mapa del repositorio

| Ruta | Contenido |
|---|---|
| [tengo-aurilla-v0/](tengo-aurilla-v0/) | El modelo final y sus versiones anteriores (`recommender-v4.py` … `recommender-v8.py`), cada una con su documento (`modelo*.pdf` y su `.tex`) |
| [eda/scripts/](eda/scripts/) | Un script por pregunta: cada decisión del modelo tiene aquí su comprobación |
| [eda/residuos/](eda/residuos/) | Análisis de residuos y de productos nuevos que llevó al v8 |
| [eda/lab/](eda/lab/) | Laboratorio offline y diario de experimentos (`HALLAZGOS.md`) |
| [eda/](eda/) | Cuadernos de exploración inicial, figuras y la serie del leaderboard (`leaderboard_log.jsonl`, una foto cada 5 minutos) |
| [quadre-comandament/](quadre-comandament/) | Cuadro de mando en Streamlit (facturación, stock, rebajas y rendimiento por versión); solo lee de RawTree |
| [starter/](starter/), [recommender-v2.py](recommender-v2.py), [offline_build.py](offline_build.py), [model.txt](model.txt) | Primer sistema («Bayes-TS», del sábado), sustituido por el modelo jerárquico |
| [control_tower_agent.py](control_tower_agent.py) | Agente del reto secundario «RawTree Control Tower» |
| [docs/](docs/) | Estructura de los datos y fuentes |

### Historia de versiones

| Versión | Fichero | Qué añade |
|---|---|---|
| Bayes-TS | `starter/`, `recommender-v2.py` | Primer sistema: priors Beta jerárquicas, Thompson Sampling contextual y MMR |
| v4 | `tengo-aurilla-v0/recommender-v4.py` | Modelo jerárquico más afinidad con gustos latentes (etiquetas y co-compra) |
| v5 | `recommender-v5.py` | Más suavizado en la co-compra, `θ` sin recorte |
| v6 | `recommender-v6.py` | Tendencia por subcategoría; A/B con `--shard` |
| v7 | `recommender-v7.py` | Tres clases de sesión; pesos estimados dentro de cada lista |
| v8 | `recommender-v8.py` | Factores de co-compra heredados para productos nuevos (**final**) |

## Limitaciones conocidas

- Con un usuario que empieza de cero, el modelo ordena solo con la sesión y la jerarquía poblacional;
  mejora según acumula eventos.
- Tras la imputación, los productos nuevos se siguen comprando más de lo previsto: la media de los vecinos
  recupera solo parte de su co-compra real.
- Las mejoras del v7 y del v8 son del orden del ruido de un A/B de tres horas; se apoyan en la
  reproducción del stream y en los residuos, no en el A/B.
