# Quadre de comandament

Dashboard en Streamlit per a direcció: facturació, gestió d'estoc, rebaixes i estoc mort, amb les dades en viu de RawTree. També compara el rendiment de cada versió del recommender.

Només **llegeix** de RawTree. No escriu res ni interfereix amb el recommender que estigui en marxa.

## Com s'executa

1. Instal·la els paquets (només el primer cop):

   ```
   pip install streamlit plotly pandas numpy requests
   ```

2. Arrenca'l **des de dins d'aquesta carpeta** (així agafa el tema fosc de `.streamlit/config.toml`):

   ```
   cd quadre-comandament
   streamlit run dashboard.py
   ```

3. S'obre al navegador a <http://localhost:8501>.

## Què necessita

- **El `.env` de l'equip**, amb `TEAM_ID` i `RAWTREE_API_KEY`. El busca dins d'aquesta carpeta, a `Datathon2026/` o a `Datathon2026/datathon-fme-2026/`. No el pugis mai al repositori.
- **La carpeta `tengo-aurilla-v0/`** al mateix nivell que aquesta. Del recommender més nou (`recommender-vN.py` amb el número més alt) n'importa només el client de RawTree per connectar-se, sense arrencar res. També llegeix el `model_version` de cada fitxer per saber quina etiqueta correspon a cada versió. Si no la troba, es connecta directament a l'API de RawTree.

## Què llegeix de RawTree

| Taula | Per a què |
|---|---|
| `shop.catalog`, `shop.catalog_updates` | Catàleg vigent, ruptures d'estoc, rebaixes i productes nous |
| `shop.events` | Vistes, carrets i compres d'avui |
| `shop.history_events` | Els 60 dies anteriors (comparació i estoc mort) |
| `shop.requests` | Filtrar el feedback per regió i dispositiu |
| `team_03.feedback` | Resultats del carrusel i rendiment per `model_version` |

Les consultes es guarden 30 segons; el botó ⟳ de dalt a la dreta les refresca a l'instant.

## Pantalles

- **Portada:** quatre blocs. Fes clic a «Veure detall →» per entrar a cadascun.
- **Retorn econòmic:** facturació, tiquet mitjà, conversió, el nostre motor contra la reserva, versions del recommender i mapa categoria × regió.
- **Gestió d'estoc:** prioritat de reposició, euros perduts per ruptures, matriu de decisió i llista de reposició.
- **Rebaixes:** efecte mitjà de les rebaixes, detall per categoria i cost màxim per unitat perquè siguin rendibles (sense suposar cap marge).
- **Estoc mort:** productes actius sense vendes, separats entre els que es miren però no es compren i els que gairebé ningú mira.

Està pensat per a una pantalla de 1920×1080.
