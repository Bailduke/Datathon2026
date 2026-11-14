# Bayes-TS · Real-Time Recommender (Datathon FME 2026)

Implementació modular de l'especificació **v3 “Bayes-TS”** de `model.txt`, compatible amb
el kit original (`pull → predict → push` i el notebook `starter/offline_eval.ipynb`).

## Arrencada ràpida

```bash
cd Datathon2026
cp .env.example .env              # posa TEAM_ID i RAWTREE_API_KEY
pip install -r starter/requirements.txt

python offline_build.py           # priors + team_NN.scratch + state/latest.npz
pytest -q starter/tests/test_engine.py
python starter/recommender.py     # servei desatès (SIGTERM desa checkpoint)
```

`offline_build.py --quick --skip-cooc --no-scratch` fa un build de fum en ~12 s.
`python starter/recommender.py --once` fa un cicle real de pull/answer/push i surt.

## Arquitectura

```
shop.requests ──► request_loop (0.25 s) ──► answer() ──► OUTBOX ──► push_loop (0.35 s) ──► team_NN.submissions
                       │                       ▲
                       └── Snapshot immutable ─┘
feedback_loop (1.5 s)   events_loop (10 s)   catalog_loop (0.75 s)   persist_loop (60 s)   health_loop (10 s)
```

| Mòdul | Responsabilitat |
|---|---|
| `starter/config.py` | `Config` (Secció 6.4) + overrides `BAYES_*` i càrrega de `.env` |
| `starter/rawtree.py` | client HTTP thread-safe, `SETTINGS max_execution_time=25`, backoff 429/413/408/5xx, taules buides, rellotge servidor |
| `starter/text.py` | tokenització determinista + TF-IDF hashed (`crc32`, `D=512`, L2) |
| `starter/state.py` | `Snapshot` immutable, `StateHolder` atòmic, `LRU` amb TTL, CSR, `float16+base64` |
| `starter/priors.py` | IPS/ESS, jerarquia Beta-Binomial, orgànics, transicions, Laplace/IRLS, calibratge, persistència scratch |
| `starter/features.py` | parsing defensiu de 5 formats de `recent_views`, centroide, τ_view/τ_buy, matriu contextual `Z` |
| `starter/scoring.py` | candidats, Thompson Sampling contextual, utilitat preu, MMR/quotes, `answer()` |
| `starter/workers.py` | tots els bucles asíncrons, CUSUM/TV, circuit breaker, `regime_shift()` |
| `starter/recommender.py` | entrypoint + wrappers `pull/predict/push/recommend/best_sellers` |
| `offline_build.py` | pipeline SQL offline i commit en dues fases (`meta` al final) |
| `control_tower_agent.py` | procés independent del Side Challenge “RawTree Control Tower” |

## Model en una pàgina

1. **Offline.** IPS amb correcció de posició `w_k`, clipping `ω_max=50`, ESS (mai comptes crus),
   decaïment `H_hist=21` dies. Prior jeràrquic per moments (Global → Categoria → Subcategoria →
   Producte) amb shrinkage per precisió (`κ1=κ2=0.5`, `N_max=300`). Senyals orgànics amb `K_org=0.01`.
   Dues matrius CSR top-200 (`T_view`, `T_buy`) amb correcció de popularitat `n_i^{-0.5}`, prior
   estructural i diagonal exclosa. Regressió logística Bernoulli ponderada per `w_k`, Laplace per
   Newton/IRLS amb damping, i calibratge `(a,T)` sobre holdout temporal de 7 dies.
2. **Online.** Filtre dur (`active & ~unavailable & price>0 & ~bought`) → 100 candidats amb quotes
   (35 transicions / 20 contingut / 10 tendències / 15 priors / 10 exploració / 10 seguretat) →
   Thompson Sampling contextual (una `β̃` per petició) → `U = p̃(1+ρq)` → slot 1 exploit,
   slots 2-8 MMR+quotes, slots 9-10 exploració ε-uniforme → reparació `best_safe`.
   Sempre 10 IDs únics i elegibles; p99 mesurat ≈ 3 ms amb N=5.000.
3. **Streaming.** Feedback amb biaix de posició i doble escala (7 dies / 20 min), ADF
   Sherman-Morrison amb reinjecció de variància, refit cada 30 min; ring buffer de 240 min per
   tendències; catàleg incremental amb tombstone immediat; persistència incremental i snapshot
   complet horari; circuit breaker `NORMAL → DEGRADED → SAFE → RECOVER`.

## Les 8 correccions aplicades

1. `order_slate(..., rng)` rep el generador explícit.
2. `Config.lambda_T_view` / `lambda_T_buy` i pesos per pàgina `(0.8,0.3)` / `(0.4,0.9)`.
3. `cold_reserve` compara `snap.subcategory[snap.index[v]]` amb `snap.new_subcats`.
4. Tots els sets d'usuari són `frozenset` amb *copy-on-write* (`LRU.add_to_set`).
5. Co-compra amb `ts::DateTime64(3) AS ts` i `AND v.ts <= pur.ts`.
6. Un `product_id` nou desconegut només activa `wake_catalog`; `answer()` no fa HTTP.
7. `L_beta` es precalcula al worker i es publica dins del `Snapshot`; el sampling fa `mu + L @ z`.
8. El rànquing i l'actualització fan servir exclusivament Bernoulli + `w_k` (Seccions 2-6);
   l'Apèndix C (softmax antic) s'ignora.

## Side Challenge · Control Tower

`control_tower_agent.py` corre en un procés separat. Consulta `shop.events`,
`shop.catalog_updates` i `team_NN.feedback`, manté el bessó digital a
`team_NN.factory_inventory` / `team_NN.production_orders`, escriu les investigacions SQL i les
decisions a `team_NN.agent_traces`, i publica `kind='control_tower_policy'` a `team_NN.scratch`.
`health_loop` llegeix aquesta fila i aplica només paràmetres **clampats**
(`restock_boost ∈ [0.8,1.25]`, `trend_scale ∈ [0.5,1.5]`, `temperature_mult ∈ [0.9,1.15]`,
`exploration_delta ∈ [-0.1,0.1]`). Si la clau del equip és de rol de base de dades i no pot
crear taules, l'agent desa el mateix bessó digital com a files `kind='factory_inventory'`,
`kind='production_orders'` i `kind='agent_trace'` a `team_NN.scratch` (fallback transparent).

```bash
python control_tower_agent.py --once
python control_tower_agent.py --interval 15
```

## Validació

* `pytest -q starter/tests/test_engine.py`: parsing (5 formats), filtre, 10.000 peticions
  sintètiques (100% 10 IDs únics), p99 < 50 ms, Beta amb biaix de posició, new product,
  stock-out/back-in-stock, `regime_shift`, Cholesky, `frozenset`, SQL causal.
* `offline_build.py`: escriu `state/latest.npz` i `team_NN.scratch` amb `meta` verificada.
* `python starter/recommender.py --once`: prova integrada contra el stream de pràctica.

## Operació

`caffeinate -dimsu python starter/recommender.py` (macOS) o un servei `systemd` amb
`Restart=always`. Els logs indiquen `clock_offset_s`; si supera 2 s cal revisar NTP. El diumenge
a les 09:00 UTC `health_loop` activa automàticament `regime_shift()` i després fa annealing.
