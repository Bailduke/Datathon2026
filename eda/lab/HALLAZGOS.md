# Laboratorio offline · tengo-aurilla-v2

Diario de backtests. Nada de lo que hay aquí está desplegado: el proceso en vivo usa
`tengo-aurilla-v0/recommender-v4.py`; el laboratorio trabaja con la copia `eda/lab/recommender-lab.py`
y su propia caché `eda/lab/.cache/`.

**Datos:** primera tanda, 101.956 listas mostradas a nuestro equipo hasta el sábado 20:40 CET, 851 compras
(67 del v2). Repaso con datos frescos hasta las 21:44: 113.323 listas, 1.122 compras, de ellas 338 en
14.466 listas del v2 (conversión real del v2: 2,34 % ± 0,13; nuevos 3,35 %; con historia 1,90 %).

**Cómo se mide.** Se reproduce el stream con el código de producción y se ajusta la regresión de Poisson
de la compra. Para cada idea se dan tres cifras, siempre como diferencia respecto al modelo actual:

- *Poisson dentro*: log-verosimilitud dentro de muestra (optimista; solo orienta).
- *Orden temporal*: log-verosimilitud de **qué producto de la lista se compró**, ajustando con el primer
  70 % del tiempo y evaluando en el 30 % final. Es la cifra que mide si la lista quedaría mejor ordenada.
- *Orden CV*: lo mismo con validación cruzada de 5 partes, separada en usuarios nuevos y con historia.
- En el repaso, además, *solo v2*: se ajusta sin las listas del v2 y se evalúa el orden solo en ellas.

Con 850–1.100 compras, diferencias de menos de ±5 puntos en las cifras de orden son ruido.

## Conclusión

**No he encontrado ninguna mejora grande.** Tras 16 hipótesis, lo que queda son retoques de alrededor del
1–2 % cada uno, ninguno confirmado con claridad en las listas del propio v2. El v2 está cerca de lo que
estos datos permiten con estas señales. La hipótesis principal (vistas del histórico para cerrar la
distancia en usuarios con historia) no se sostiene.

Lo único que recomiendo adoptar es de higiene, no de conversión: los **controles de lista en la
calibración**, que dejan la previsión cerca del nivel real y no tocan el camino en vivo.

## Resumen, de más a menos evidencia

| # | Hipótesis | Poisson dentro | Orden temporal | Orden CV | Veredicto |
|---|---|---|---|---|---|
| H8 | Controles de lista en la calibración (peticiones previas, vistos hoy, carrito hoy) | +317 | +4,8 | +1,5 | **Adoptar** (higiene): la previsión del v2 pasa de sobreestimar un 60 % a un 20 %; el orden no cambia |
| PRECIO | Quitar del nivel 3 toda la pendiente de precio (también r_u) | −9,2 | +7,7 | +12,2 | **Candidata menor**: +7,8 también solo en v2. La propuesta de dejar solo β_u·r_u no pierde nada (+0,8 / +0,9 / +1,9) |
| H5b | Usuario nuevo × co-compra con el carrito de hoy (un coeficiente) | +48,2 | +3,2 | +12,5 | **Candidata menor**: positiva en las tres pruebas con datos frescos (+3,7 solo en v2); vale ≲1 % |
| W | Pesos de recuento compra 0 · carrito 1 · vista 0,1 (hoy 1 / 0,25 / 0,02), kappa_c 5, lam_cat 0,5 | +28,5 | +6,4 | +18,2 | **Dudosa**: no se confirma en las listas del v2 (−2,3); vale ±2 % según el juez |
| H2 | Recalibrar con el feedback del v2 | — | — | — | Los pesos se mueven poco; la previsión sube un 1 % (2,51 → 2,53 %) |
| H5 | Todos los pesos de gustos distintos para usuarios nuevos | +57,7 | 0,0 | +12,8 | Se reduce a H5b |
| RHO | Decaimiento de la sesión rho ∈ {0,2 … 1,0} con lam_cat ∈ {0,5; 1; 2} | −10 a +1,5 | −5 a +2,7 | −10 a +4,2 | **Ruido**: perfil plano entre 0,2 y 0,65; se queda 0,65. lam_cat 0,5 un pelo mejor (+4), 2 peor (−6) |
| B0 | Fuerza del prior de popularidad b0 | +1,6 / −8,0 | +0,8 / −3,5 | +1,3 / −5,3 | Se queda 20.000 (2.000 da igual; 200.000 empeora) |
| H6 | Hiperparámetros de gustos (factores, encogimiento, alfa, caducidad, pesos) | −35 a +10 | −4 a +4 | −24 a +7 | **Se quedan** 16 factores, encogimiento (3,1,1,1), alfa 6, sin caducidad; 8 coeficientes mejor que 2 |
| H8c | Descontar productos ya mostrados al mismo usuario | +2,8 | −25,7 | +0,1 | **Descartar**: con datos frescos empeora fuera de muestra |
| H1 | Vistas del histórico como cesta de gustos | +21,3 | −2,9 | −3,5 | **Descartar**: gana dentro de muestra, no ordena mejor |
| H1b | Indicador «visto en el histórico y nunca en el carrito» | +3,1 | +1,9 | −1,9 | **Descartar** (coef −0,46 ± 0,19, sin efecto en el orden) |
| H3 | Los productos parecidos se canibalizan | — | — | — | **Descartar**: la lista no satura (exponente 0,97–1,1) y concentrar no resta |
| H9 | Saciedad por subcategoría / categoría / etiquetas | +1 a +9 | ≈0 | −4 a 0 | **Descartar** |
| H4 | Pesos de posición estimados en vivo | +8,5 | −4,5 | −8,6 | **Descartar**: se quedan los del histórico |
| H7 | Carrito añadido 2+ veces; antigüedad del carrito de hoy | +0,6 / +2,9 | −2,7 / +1,3 | +0,1 / +0,5 | **Descartar** (solo 104 impresiones con 2+ añadidos) |
| H10 | Etiquetas dentro de la categoría; afinidad etiqueta-etiqueta 20×20; tendencia de popularidad | +1 a +14 | −5 a 0 | −5 a +2 | **Descartar** las tres |
| H11 | Exposición de la popularidad solo con el producto en stock (Ian) | — | — | — | **Despreciable**: cambia log q en 0,016 de media en los 100 más vendidos; no implementado |
| H12 | Popularidad propia de los usuarios nuevos | 0,0 | −0,1 | −0,5 | **Descartar**: nuevos y antiguos compran los mismos productos |
| H13 | Compras que se nos escapan | — | — | — | Sin patrón identificable |

**Sin probar:** qué pasa con las cestas «de hoy» cuando tengan 24 h (el domingo el carrito del sábado; hoy
solo hay 10 h de datos y hasta ahí no caducan); popularidad por región o dispositivo (los eventos orgánicos
no los llevan); una factorización conjunta usuario-producto en lugar de co-compra producto-producto.

## H2 · ¿Acierta el modelo de compra con las listas que el v2 mostró de verdad?

Modelo ajustado solo con lo anterior al v2 (20:23 CET) y aplicado a las 3.099 listas del v2:

| Listas del v2 | Real | Previsto | Obs/esp |
|---|---|---|---|
| Todas | 2,16 % ± 0,26 | 3,69 % | 0,59 |
| Usuarios nuevos | 3,61 % ± 0,63 | 5,29 % | 0,68 |
| Con historia | 1,56 % ± 0,27 | 3,02 % | 0,52 |

La sobreestimación es pareja en todos los tramos de tasa prevista y en todas las posiciones: no es un
problema de los productos mejor puntuados sino un factor común a toda la lista. Lo mismo le pasaba al v0
dentro de muestra (obs/esp 0,67), mientras que las listas anteriores quedaban infraestimadas (1,18).
Los pesos apenas cambian al añadir las listas del v2 (todos dentro de un error estándar).

## H3 · ¿Se canibalizan los productos parecidos?

Regresión a nivel de lista: log E[compra] = c + nuevo + home + γ·log S, con S la suma prevista de sus
productos.

- γ = 0,97 ± 0,03 (1,08–1,10 al controlar la hora): la conversión de la lista crece en proporción a S, sin
  saturar.
- Concentración por subcategoría (índice de Herfindahl de S): coeficiente **+0,74 ± 0,17**. Las listas
  concentradas convierten más, no menos. Por categoría: +0,19 ± 0,16.
- Lo que sí aparece es una caída con las horas: −0,10 a −0,13 por hora a igualdad de lista, también en las
  listas anteriores a v0/v2. Eso explica el desajuste de H2 (ver H8).

Veredicto: no diversificar; `max_per_sub` se queda desactivado.

## H8 · ¿De dónde sale la caída con las horas?

Es del usuario, no del reloj ni de los productos:

| Variable de lista | Coeficiente | Poisson dentro |
|---|---|---|
| Horas desde el inicio | −0,14 ± 0,01 | +58 |
| log(1 + peticiones previas del usuario) | −0,62 ± 0,06 | +52 |
| log(1 + productos vistos hoy) | −0,55 ± 0,04 | +82 |
| Las tres de hoy juntas: compras / vistos / carrito | −0,04 / −1,07 / +1,39 | +235 |

Un usuario que hoy ha visto mucho y ha metido poco en el carrito compra menos de cualquier lista; uno con
mucho carrito, más. Con vistos y carrito de hoy dentro, la hora casi desaparece (−0,03 ± 0,01).

Nivel del v2 previsto por un modelo ajustado solo con lo anterior al v2 (datos hasta las 21:44, 338 compras
del v2), según los controles de lista que lleve:

| Controles | Obs/esp todas | Nuevos | Con historia |
|---|---|---|---|
| Ninguno (calibración actual) | 0,62 | 0,62 | 0,63 |
| Peticiones previas, vistos hoy, carrito hoy (lo implementado) | 0,81 | 0,79 | 0,83 |
| Los tres + hora | 0,87 | 0,85 | 0,88 |
| Solo hora + peticiones previas | 1,05 | 1,03 | 1,06 |

La hora no se ha metido en la versión implementada porque es la única variable sin mecanismo detrás y con
los otros tres controles su coeficiente es −0,025 ± 0,012 por hora. Con los tres controles:

- La previsión sobreestima un ~20 % en vez de un ~60 %.
- Los pesos de producto se mueven algo (carrito de hoy 0,79 → 1,26; co-compra con el carrito de hoy
  0,14 → 0,49; co-compra con las vistas de hoy 1,07 → 0,83), pero el orden no mejora de forma medible
  (temporal +4,8; CV +1,5).

Veredicto: añadirlos a `--calibrate` para que la previsión de nivel se acerque a la real. No cambian qué se enseña.

### H8c · Descontar lo ya mostrado al mismo usuario

El 31 % de las impresiones son de un producto que ese usuario ya había visto en una lista nuestra.
Observado/esperado según las veces ya mostrado: 0 → 1,06; 1 → 0,97; 2–3 → 0,70; 4–9 → 0,14 (2 compras
frente a 14,8 esperadas). Pero al controlar la fatiga del usuario el efecto del producto se queda en
−1,44 ± 0,72 solo para 4+ veces, y fuera de muestra no ayuda (temporal −10,2; CV +1,3).

Repaso con datos frescos (el v2 repite: el 23 % de sus impresiones ya se habían mostrado a ese usuario, el
1,9 % cuatro veces o más): coeficiente −1,01 ± 0,51, orden temporal **−25,7**, CV +0,1; en las listas del v2
lo mostrado 4+ veces dio 3 compras frente a 4,8 esperadas. Descartado.

## H9 · Saciedad más allá del producto

Si el usuario ya ha comprado hoy algo de la misma subcategoría (5 % de las impresiones) o categoría
(16 %): coeficientes +0,17 ± 0,10 y +0,13 ± 0,09, y el orden no mejora (CV −2,4 y −2,0). Etiquetas
compartidas con lo comprado hoy: +0,03 ± 0,09. No hay saciedad por parecido; descartado.

## H4 · Posición

Nueve indicadores de posición libres sobre los pesos del histórico: Poisson +8,5 dentro, pero el orden
empeora fuera de muestra (temporal −4,5; CV −8,6). Una pendiente extra en log(1+slot): −0,14 ± 0,05 dentro,
−3,1 / −1,3 fuera. Se mantienen los pesos del histórico.

## H5 · Pesos distintos para usuarios nuevos

- nuevo × log P jerárquica: −0,10 ± 0,08; nuevo × clases: nada.
- nuevo × gustos (8): Poisson +57,7; orden temporal 0,0; CV +12,8 (casi todo en usuarios nuevos). Lo lleva
  una sola interacción: co-compra con el carrito de hoy, **+1,25 ± 0,23** más en usuarios nuevos.

Dudoso: la mejora no aparece en el corte temporal. Pendiente probar esa interacción sola.

## RHO · Decaimiento de la sesión (pedido por Ian) y lam_cat

rho pondera las vistas de la sesión en S_c y S_s (la última pesa 1, la anterior rho, …). Se recalculó la
probabilidad jerárquica de cada producto mostrado con cada pareja (rho, lam_cat) y se reajustó todo el
modelo de compra (clases + gustos + controles de lista). Submuestra ponderada: las 851 listas con compra
(447 de usuarios nuevos, 404 con historia) y 1 de cada 5 del resto (21.052 listas).
Diferencias respecto a rho 0,65 · lam_cat 1 (lo que está en vivo):

| rho | lam_cat 0,5: Poisson / orden temp / orden CV | lam_cat 1 | lam_cat 2 |
|---|---|---|---|
| 0,20 | +0,4 / +2,7 / +4,0 | −0,0 / +1,7 / +0,7 | −5,5 / −0,6 / −5,5 |
| 0,35 | +0,9 / +2,5 / +4,1 | +0,2 / +1,2 / +0,7 | −5,4 / −1,4 / −5,7 |
| 0,50 | +1,3 / +2,2 / +4,2 | +0,3 / +0,6 / +0,5 | −5,7 / −2,2 / −6,2 |
| 0,65 | +1,5 / +1,8 / +4,1 | **0 (actual)** | −6,4 / −3,1 / −7,0 |
| 0,80 | +1,4 / +1,4 / +3,8 | −0,8 / −0,7 / −0,8 | −7,6 / −4,0 / −8,2 |
| 1,00 | +0,5 / +0,7 / +3,0 | −2,6 / −1,7 / −2,2 | −10,0 / −5,2 / −10,0 |

Por segmento (orden CV): con lam_cat 0,5 la mejora se reparte, nuevos +2,4 y con historia +1,7, para
cualquier rho entre 0,2 y 0,65.

Lectura:

- **rho no importa** entre 0,2 y 0,65: toda la fila cabe en ±1 punto. Solo rho = 1 (todas las vistas pesan
  igual) sale algo peor, y poco. El indicio previo («misma categoría que el último visto» +0,35) ya no
  aparece con los gustos latentes dentro: la co-compra y las etiquetas de la sesión recogen esa señal.
- **lam_cat sí tiene dirección** (0,5 mejor que 1, 2 claramente peor), pero +4 puntos con 851 compras es
  ruido. No se mueve con rho: el óptimo de lam_cat es el mismo en todas las filas.
- Recomendación: dejar rho = 0,65 y lam_cat = 1. Si se quiere, lam_cat = 0,5 no hace daño.

## B0 · Fuerza del prior de popularidad

b0 = 2.000: +1,6 / +0,8 / +1,3 (indiferente). b0 = 200.000: −8,0 / −3,5 / −5,3 (peor, sobre todo en
usuarios con historia, −4,7). Se queda 20.000.

## H1 · Vistas del histórico como cesta de gustos

59.841 usuarios tienen vistas en el histórico, 147 productos distintos de media (frente a ~15 de compras y
carrito). Puntuaciones de co-compra y de etiquetas con esa cesta, añadidas al modelo actual:

| Cesta | Poisson dentro | Orden temporal | Orden CV (historia) | Coeficientes |
|---|---|---|---|---|
| Vistas únicas: co-compra + etiquetas | +21,3 | −2,9 | −3,5 (−3,1) | +1,73 ± 0,30 / +0,36 ± 0,14 |
| Vistas únicas: solo etiquetas | +4,6 | −7,2 | −4,9 (−4,1) | +0,44 ± 0,15 |
| Ponderadas por nº de vistas | +18,5 | −3,6 | −3,7 (−3,3) | +1,65 ± 0,31 / +0,37 ± 0,14 |
| Solo vistas nunca metidas en el carrito | +6,8 | +1,4 | −0,8 (−1,3) | +0,52 ± 0,29 / −0,33 ± 0,09 |
| Indicador «visto y nunca en el carrito» (9 % de las impresiones) | +3,1 | +1,9 | −1,9 | −0,46 ± 0,19 |

La co-compra con las vistas tiene un coeficiente grande dentro de muestra, pero no mejora la elección del
producto dentro de la lista en ninguna de las dos pruebas fuera de muestra: lo que añade es nivel de lista
(usuarios más o menos compradores), no orden. La hipótesis de que las vistas históricas cerrarían la
distancia en usuarios con historia **no se sostiene**. Descartado; además habría costado ~40 MB de estado y
una descarga pesada en cada arranque.

## W · Pesos de recuento (compra / carrito / vista), con kappa_c y lam_cat

Los recuentos por tipo se reconstruyeron del histórico (compras y carritos por usuario; vistas de la
descarga de H1) y coinciden exactamente con el `N` de producción con los pesos actuales. Misma submuestra
ponderada que en RHO. Diferencias respecto a lo que está en vivo (1 / 0,25 / 0,02 · kappa_c 2 · lam_cat 1):

| Pesos compra / carrito / vista | kappa_c | lam_cat | Poisson | Orden temp | Orden CV | nuevos | historia |
|---|---|---|---|---|---|---|---|
| 1 / 0,25 / 0 | 2 | 1 | −8,1 | −2,3 | −3,7 | −0,6 | −3,1 |
| 1 / 0,25 / 0,1 | 2 | 1 | +12,1 | +4,0 | +4,4 | +1,1 | +3,3 |
| 1 / 1 / 0,02 | 2 | 1 | +6,5 | +5,9 | +5,7 | +3,1 | +2,6 |
| 2 / 0,25 / 0,02 | 2 | 1 | −12,8 | −3,1 | −5,8 | −3,0 | −2,9 |
| 0,5 / 0,25 / 0,02 | 2 | 1 | +7,7 | +2,0 | +3,6 | +2,0 | +1,6 |
| 1 / 1 / 0,1 | 2 | 1 | +13,8 | +7,3 | +7,6 | +3,0 | +4,6 |
| 0,5 / 1 / 0,1 | 2 | 1 | +19,3 | +8,6 | +10,1 | +4,1 | +6,0 |
| 0 / 1 / 0,1 | 2 | 1 | +24,9 | +9,8 | +12,7 | +5,3 | +7,4 |
| 0 / 1 / 0,1 | 1 | 1 | +18,7 | +8,3 | +6,7 | +0,8 | +5,9 |
| **0 / 1 / 0,1** | **5** | 1 | +28,7 | +10,5 | +18,2 | +10,5 | +7,7 |
| 0 / 1 / 0,1 | 10 | 1 | +25,3 | +9,8 | +19,3 | +12,8 | +6,6 |
| 0 / 1 / 0,1 | 20 | 1 | +15,5 | +8,3 | +17,3 | +13,3 | +4,0 |
| **0 / 1 / 0,1** | **5** | **0,5** | +28,3 | +11,2 | +20,3 | +11,5 | +8,9 |
| 0 / 1 / 0,1 | 5 | 2 | +24,0 | +8,6 | +12,9 | +7,3 | +5,5 |
| 0 / 1 / 0,05 | 5 | 1 | +25,6 | +10,0 | +17,7 | +10,7 | +7,1 |
| 0 / 1 / 0,2 | 5 | 1 | +29,8 | +10,6 | +17,6 | +9,9 | +7,7 |
| 0 / 1 / 0 | 5 | 1 | +17,2 | +8,9 | +15,1 | +10,5 | +4,6 |
| 0 / 2 / 0,1 | 5 | 1 | +22,1 | +11,1 | +16,9 | +9,6 | +7,3 |
| 0 / 0,5 / 0,1 | 5 | 1 | +28,5 | +9,0 | +16,9 | +10,3 | +6,7 |
| 1 / 0,25 / 0,02 (actual) | 5 | 1 | +2,7 | +1,5 | +5,1 | +5,0 | 0,0 |

Lectura:

- La dirección es monótona y la misma en las dos pruebas fuera de muestra: **la compra no añade afinidad a
  su categoría más allá del carrito que la precedió** (toda compra pasó antes por el carrito), el carrito
  pesa más de lo que tenía y las vistas también. La forma simple es «recuento = carritos + 0,1 · vistas».
- La zona buena es ancha (carrito 0,5–2; vista 0,05–0,2; compra 0–0,25; kappa_c 5–10): no hay que afinar.
- Tamaño: +10 a +11 en el corte temporal y +18 a +20 en validación cruzada con 851 compras. Es unas 2–4
  veces el umbral de ruido; pequeña pero consistente, y mejora en los dos segmentos.
- kappa_s (20 o 200) no cambia nada.

### Comparación cruzada de políticas (¿cuánta conversión vale?)

La log-verosimilitud dice que el modelo mejora, no cuánto mejora la lista. Para medirlo, cada variante se
calibra entera, elige sus listas para las últimas 30.000 peticiones, y todas las listas se valoran con el
modelo de compra de cada variante (el «juez»). Con controles de lista, la previsión ya está en el nivel
real (el v2 midió 2,16 % ± 0,26 en esas horas; leaderboard 2,4).

| Lista elegida por | Juez: modelo actual | Juez: modelo W | Juez: lam_cat 0,5 |
|---|---|---|---|
| Modelo actual | **2,788 %** | 2,608 % | 2,758 % |
| W (0 / 1 / 0,1 · kappa_c 5 · lam_cat 0,5) | 2,734 % | **2,660 %** | 2,727 % |
| Solo lam_cat 0,5 | 2,781 % | 2,623 % | **2,765 %** |

Cada juez prefiere su propia lista, como era de esperar. Lo que importa es el tamaño: entre el modelo
actual y W hay **±2 % relativo** según a qué juez se crea (por segmento: nuevos +2,8 % y con historia
+1,4 % con el juez W; −2,7 % y −1,4 % con el juez actual). El juez W ajusta mejor los datos (+30 de
log-verosimilitud), así que lo más probable es una ganancia de alrededor del 2 %: de 2,40 a 2,45.
lam_cat 0,5 solo: ±0,3 %, nada.

**Veredicto sobre W:** mejora real pero pequeña. Cambiarla obliga a sustituir los recuentos `N` del modelo
(un `priors_live.pkl` nuevo) y a recalibrar; no lo recomiendo para la final salvo que se quiera apurar ese
2 %. Queda preparada en `eda/lab/.cache_counts/` (ver «Qué hay listo para desplegar»).

## H11 · Exposición de la popularidad solo con el producto en stock (planteada por Ian)

Tamaño del efecto, con las roturas de `catalog_updates` y las 116.469 compras orgánicas del stream (10,7 h):

| Grupo | Con alguna rotura | Exposición del stream sin stock (media / p90 / máx) | Cambio en log q_i (medio / máx) |
|---|---|---|---|
| Catálogo (5.000) | 6,5 % | 0,7 % / 0 % / 37,5 % | 0,001 / 0,066 |
| 100 más vendidos | 67 % | 9,2 % / 18,7 % / 37,5 % | 0,016 / 0,066 |
| 500 más vendidos | 34 % | 4,0 % / 9,5 % / 37,5 % | — |
| Productos nuevos (300) | 0 % | 0 % | 0 |

- Los superventas pasan sin stock el 9 % del stream, pero el stream es solo el 17 % de su exposición total
  (T_hist = 548.160 compras frente a 116.469 de hoy), así que log q_i cambia 0,016 de media. Para comparar,
  la desviación típica de log q_i entre productos es 1,67.
- Ningún producto nuevo se ha quedado sin stock: en ellos la corrección es exactamente cero.
- Proyectado al domingo a las 10:00 (stream 2,6 veces más largo): 0,033 de media, 0,14 como máximo.

Veredicto: el razonamiento es correcto pero el efecto es despreciable con estos datos; no se implementa.
Limitación: no sabemos si en el histórico también hubo roturas (allí no hay `catalog_updates`).

## H6 · Hiperparámetros de los gustos latentes

Las cestas de cada petición se reconstruyeron de los eventos y las puntuaciones se recalcularon fuera del
replay (correlación con las de producción: co-compra 1,00–0,997; etiquetas 0,96–0,98, así que las cifras
valen como comparación entre variantes, no como valor absoluto).

| Variante | Poisson | Orden temp | Orden CV | nuevos | historia |
|---|---|---|---|---|---|
| Factores de co-compra: 4 / 8 / 12 | −30,9 / −12,7 / −17,8 | +0,1 / −2,1 / −1,9 | −23,9 / −13,0 / −10,7 | −17,3 / −8,7 / −6,7 | −6,6 / −4,3 / −4,0 |
| Factores: 24 / 32 / 48 | +3,2 / +4,9 / −9,6 | +0,6 / +1,1 / +3,9 | +5,6 / +6,9 / −2,4 | +5,5 / +6,9 / −0,5 | +0,1 / 0,0 / −1,9 |
| Encogimiento (1,1,1,1) / (10,1,1,1) | +0,2 / −1,1 | −0,3 / +1,2 | +0,3 / +0,2 | | |
| Encogimiento (3; 0,3; 0,3; 0,3) | +10,1 | +1,9 | +2,9 | +1,6 | +1,3 |
| Encogimiento (3,3,3,3) | −20,2 | −2,3 | −7,9 | −6,0 | −1,9 |
| Alfa de etiquetas 1 / 3 / 12 / 24 | −4,0 / −0,8 / −0,8 / −4,6 | +2,8 / +1,4 / −1,9 / −4,1 | +1,2 / +0,8 / −1,5 / −4,0 | | |
| Cestas de hoy con caducidad de 1 h / 3 h / 8 h | −35,5 / −2,6 / +2,7 | −0,2 / +2,2 / +1,7 | −22,8 / −3,1 / +0,9 | −26,6 / −4,3 / +0,6 | +3,8 / +1,1 / +0,3 |
| Cestas de hoy ponderadas por nº de veces | −1,5 | +1,0 | +0,1 | | |
| Sesión ponderada por recencia 0,5^k / 0,8^k | −7,8 / −1,2 | −1,8 / −0,4 | −4,5 / −1,2 | | |
| Dos coeficientes en vez de ocho | −28,1 | −5,7 | −7,7 | +4,4 | −12,1 |

- 16 factores están bien: menos empeora claramente; 24–32 dan +6 en CV y +1 en el corte temporal (ruido).
- **Las cestas de hoy no deben caducar**: con vida media de 1 h los usuarios nuevos pierden 27 puntos. Hasta
  donde hay datos (10 h), lo que el usuario vio o metió en el carrito por la mañana sigue valiendo por la
  tarde. No se puede comprobar todavía qué pasa a las 24 h.
- Encogimiento, alfa y ponderaciones: indiferentes dentro de un rango amplio.
- Juntar las cuatro cestas en una sola puntuación empeora, sobre todo con historia (−12).

## H7 · Carrito repetido y antigüedad

Carrito de hoy añadido 2+ veces: +0,6 / −2,7 / +0,1 (solo 104 impresiones). Antigüedad del carrito de hoy
en horas: +2,9 / +1,3 / +0,5. Más de 3 h: +2,5 / +2,3 / +1,0. Nada que usar.

## H10 · Otras ideas a nivel de producto

| Idea | Poisson | Orden temp | Orden CV | Coeficiente |
|---|---|---|---|---|
| Perfil de etiquetas solo con los productos de la cesta de su misma categoría | +2,3 | −0,5 | +2,3 | −0,05 ± 0,02 |
| Afinidad etiqueta-etiqueta 20×20 (sin la diagonal) | +10,4 | −4,1 | −4,6 | +1,38 ± 0,31 |
| Tendencia: compras de hoy frente a lo esperado por el histórico | +1,4 | 0,0 | −0,4 | −0,25 ± 0,15 |
| Tendencia de las últimas 2 h | +1,5 | −0,5 | −1,6 | −0,15 ± 0,09 |
| Usuario nuevo × co-compra con el carrito de hoy | +52,0 | −2,2 | +10,8 | +1,45 ± 0,14 |

Las cuatro primeras, descartadas. La última es la única señal de H5: coeficiente muy significativo dentro
de muestra, pero el corte temporal no la respalda. Revisar con el feedback de la noche.

## H12 · ¿Compran los usuarios nuevos productos distintos?

Cuota de cada producto en las compras orgánicas de hoy entre usuarios nuevos (35.124 compras) frente a
usuarios con historia (81.345). El log-cociente tiene una desviación típica de 0,16 en lo mostrado y no
predice nada (coeficiente −0,03 ± 0,34; orden −0,1 / −0,5). Una sola popularidad para todos está bien.

## H13 · ¿Qué compras se nos escapan?

Para cada una de las 851 compras del carrusel, posición que ocupaba el producto comprado en el orden del v2
entre todos los elegibles en ese instante. En las 594 compras hechas sobre listas de otras políticas
(baseline y modelos anteriores) el v2 lo tenía en su top-10 el 69 % (con historia) y el 71 % (nuevos) de las
veces; en el top-30, el 85–87 %; mediana, 3ª posición. Las que quedan fuera son casi todas productos «sin
clase» (162 de 179), repartidas por páginas sin ningún patrón, con una puntuación a la mitad de la del 10º
(mediana 0,46) y en posición mediana 27. No hay un grupo identificable de compras perdidas.
(Medida sesgada a favor de lo que enseñaba el baseline; sirve para buscar huecos, no como cifra de acierto.)

## Repaso con datos frescos (hasta las 21:44; 1.122 compras, 338 del v2)

- **H2 · recalibrar con el v2.** Con la calibración de controles, los pesos pasan de
  θ 0,78 → 0,73; carrito de hoy 0,79 → 1,12; carrito histórico 0,40 → 0,69; último visto 1,78 → 1,60;
  co-compra historia / sesión / carrito hoy / vistas hoy 0,32 / 0,94 / 0,19 / 1,01 → 0,36 / 0,84 / −0,03 / 0,61;
  etiquetas casi iguales. La lista prevista mejora un 1 % (2,51 → 2,53 %; con historia 2,08 → 2,10 %).
  Recalibrar el domingo sigue siendo lo correcto, pero no hay que esperar un salto.
- **W · pesos de recuento.** Poisson +28,5; orden temporal +6,4; CV +18,2 (nuevos +8,7; historia +9,5). Pero
  ajustando sin las listas del v2 y evaluando solo en ellas: −640,1 el modelo actual, −642,4 con W. En las
  listas que más se parecen a las que mostraremos, W no gana. Baja a «dudosa».
- **H5b · usuario nuevo × co-compra con el carrito de hoy.** +1,19 ± 0,12; orden temporal +3,2; CV +12,5
  (nuevos +11,6); solo v2 +3,7. Con ella dentro, el coeficiente base de la co-compra con el carrito de hoy
  queda en −0,82 ± 0,13 (usuarios con historia) y +0,37 en nuevos. Implementada en la copia de laboratorio
  (`taste_new`); la previsión con ella sube de 2,61 a 2,64 %.

## PRECIO · ¿Hace falta la pendiente de precio en el nivel 3? (propuesta del coordinador)

Hoy la utilidad lleva pos_i · (β_0 + β_u·r_u + β_dispositivo + β_región). Replay con la región y el
dispositivo reales de cada petición (10 regiones, 3 dispositivos), 1.122 compras:

| Variante | Poisson | Orden temp | Orden CV | nuevos | historia | Solo v2 |
|---|---|---|---|---|---|---|
| Solo β_u·r_u (la propuesta) | −6,4 | +0,8 | +0,9 | +0,1 | +0,8 | +1,9 |
| β_0 + β_u·r_u (sin dispositivo ni región) | +2,6 | 0,0 | +5,3 | +4,8 | +0,5 | +1,4 |
| Sin región | +0,7 | −1,1 | +1,2 | +1,3 | −0,1 | +0,1 |
| Sin dispositivo | +4,0 | +1,8 | +5,1 | +3,8 | +1,2 | +1,8 |
| Sin β_0 (resto igual) | −18,5 | −2,6 | −9,6 | −6,8 | −2,8 | −1,6 |
| Sin r_u (resto igual) | +3,6 | +7,2 | +11,3 | +8,4 | +2,9 | +6,1 |
| **Sin precio** (ni β_0, ni r_u, ni dispositivo, ni región) | −9,2 | +7,7 | +12,2 | +8,8 | +3,4 | +7,8 |

- La propuesta (un solo coeficiente) **se confirma fuera de muestra**: no pierde orden en ninguna prueba.
  Los −6 de Poisson son de nivel de lista, no de orden.
- Se puede ir más lejos: sin ninguna pendiente de precio el orden mejora entre +8 y +12 en las tres pruebas,
  incluida la de las listas del v2. Es coherente con que precio y rebaja no mueven la compra en el
  carrusel: β_u = 9,2 se estimó con compras orgánicas y en el carrusel estorba un poco.
- Tamaño: en el borde del ruido; no vale más de un 1 %. Desplegarlo no requiere código, solo poner
  `b_p0 = 0`, `b_u = 0`, `b_dev = {}`, `b_reg = {}` en los parámetros guardados y recalibrar. Es una
  simplificación del modelo (quita el perfil de precio del usuario) más que una mejora.

## Qué hay listo para desplegar (nada está desplegado)

1. **Controles de lista en la calibración** — `eda/lab/recommender-v4.controles.py`.
   Es el fichero en vivo más 30 líneas, todas dentro de `replay_feedback` y `calibrate_live`: tres columnas
   de lista en la regresión (peticiones previas del usuario, productos vistos hoy, productos en el carrito
   hoy) y su uso en la previsión. **No toca `answer()` ni nada del camino en vivo.** Probado con
   `py_compile` y una calibración en seco con los datos en caché (`eda/lab/test_controles.py`).
   Efecto: la previsión pasa a estar en el nivel real (v2: 2,8 % previsto frente a 2,2–2,4 % medido, en vez
   de 3,6 %) y los pesos se estiman sin la confusión con la fatiga del usuario. La lista que sale es
   prácticamente la misma (2,78 % frente a 2,79 % con el mismo juez).
   Para adoptarlo: sustituir `tengo-aurilla-v0/recommender-v4.py` por ese fichero **antes** de la
   calibración del domingo; el proceso en vivo no necesita nada más que el reinicio de siempre.
2. **Pesos de recuento W** (ganancia estimada ±2 %) — `eda/lab/.cache_counts/priors_live.pkl` tiene el `N`
   recalculado (compra 0 · carrito 1 · vista 0,1), kappa_c = 5 y lam_cat = 0,5; `eda/lab/mk_counts.py` lo
   genera y lo calibra en seco. Para desplegarlo harían falta tres pasos: cambiar en `Config`
   `w_purchase = 0.0`, `w_cart = 1.0`, `w_view = 0.1`; copiar ese `priors_live.pkl` a
   `tengo-aurilla-v0/.cache/`; y lanzar `--calibrate`. Ojo: un `--build` posterior volvería a generar `N`
   con los pesos de `Config`, así que el primer paso no es opcional. **No lo recomiendo para la final.**
3. **Usuario nuevo × co-compra con el carrito de hoy (H5b)** — implementado solo en
   `eda/lab/recommender-lab.py`: un parámetro `taste_new` que `--calibrate` estima y `answer()` suma a la
   puntuación de co-compra del carrito de hoy cuando `is_new_user` es cierto. Probado con `py_compile` y
   calibración en seco. Es un cambio en el camino en vivo (dos líneas en `answer()`); ganancia ≲1 %.
4. `eda/lab/recommender-lab.py` es la copia de trabajo (controles + H5b + ganchos de laboratorio: variantes
   de rho y de pesos de recuento, reutilización de la descarga en caché). No es para desplegar tal cual, y
   parte de la versión del fichero en vivo anterior al cambio del cuantil de precio (equivalente).

Nota: `recommender-v4.controles.py` se generó a partir del fichero en vivo de las 21:10. Si el fichero en
vivo ha cambiado después (cuantil de precio sin el −½), hay que volver a aplicar el parche sobre la versión
actual en lugar de copiar ese fichero: son 30 líneas, todas en `replay_feedback` y `calibrate_live`.
