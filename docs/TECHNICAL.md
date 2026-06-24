# Technical Documentation

Architecture, modules, data flows, security notes, and interview framing for the
**Market Regime Stress-Test Lab for RL Trading Agents**.

For setup and usage, see the [root README](../README.md).

---

## 1. System overview

The project compares three reinforcement-learning trading agents (**DQN, PPO,
A2C**) against a passive **buy-and-hold** baseline across five historically
labeled market regimes. Everything runs locally: market data is cached as CSV,
trained models are saved as `.zip` files, and a Streamlit dashboard replays the
held-out evaluation slice on demand.

```
 yfinance ──► data cache ──► features (+ z-score) ──► train/test split
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  ▼                                                 ▼
                        train.py  (RL training)                        evaluate.py (battle replay)
                                  │                                                 │
                          models/*.zip                                     leaderboard + histories
                                                                                    │
                                                                              app.py (Streamlit UI)
                                                                                    │
                                                                          explain.py (Groq / rules)
```

There is **no database**. State is the filesystem: `data/cache/`, `scalers/`,
and `models/`.

---

## 2. Module map

| Module | Responsibility |
|--------|----------------|
| `config.py` | Central settings: paths, portfolio params, training modes, tickers, feature columns, scenario labels, thresholds. |
| `scenarios.py` | The five regime windows (`recession`, `bull_run`, `rate_hike`, `covid_crash`, `recovery`) as date ranges. |
| `data.py` | Downloads OHLCV + macro (VIX, 10Y yield, SPY trend) from Yahoo Finance, caches CSVs, and merges them per ticker/scenario. |
| `features.py` | Technical indicators (RSI, MACD, Bollinger, volatility, 1-day return), time-ordered train/test split, and `FeatureScaler` (z-score). |
| `trading_env.py` | Gymnasium `TradingEnv` (single dataframe) and `RandomTickerEnv` (multi-ticker training wrapper). |
| `baselines.py` | `run_buy_and_hold()` — passive benchmark aligned to the RL timeline. |
| `train.py` | Trains one model per agent per scenario, fits/saves scalers, writes a config sidecar JSON. |
| `evaluate.py` | `run_battle()` — loads models, replays the eval split, computes metrics + bootstrap Sharpe CI, ranks competitors. |
| `explain.py` | Interview-style narrative via Groq API, with a rule-based fallback. |
| `app.py` | Streamlit dashboard (UI only) — sidebar controls, winner cards, equity curves, leaderboard, analysis. |
| `tests/test_evaluate.py` | Unit tests for the evaluation/metrics logic. |

---

## 3. Data flow

### 3.1 Download and cache (`data.py`)

1. `download_ohlcv(ticker, start, end)` — pulls auto-adjusted OHLCV, flattens any
   multi-index columns, normalizes the `Date` index, and caches to
   `data/cache/{ticker}_{start}_{end}.csv`.
2. `download_macro(start, end)` — pulls `^VIX`, `^TNX` (10Y yield), and `SPY`,
   computes the 200-day SMA and `spy_above_sma200` flag, forward-fills, and
   caches to `data/cache/macro_{start}_{end}.csv`.
3. `get_merged_data(ticker, scenario)` — left-joins macro onto OHLCV by date,
   forward-fills macro gaps, and drops remaining NaNs.

Cached files are reused on subsequent runs, so only the first run needs network
access.

### 3.2 Feature engineering and splitting (`features.py`)

1. `add_technical_features()` adds `rsi`, `macd`, `bb_upper`, `bb_lower`,
   `volatility`, and `return_1d`. Macro columns (`vix`, `yield_10y`,
   `spy_above_sma200`) come from the merge step.
2. `train_test_split_df()` performs a **time-ordered** split (no shuffling) using
   `config.TRAIN_RATIO` (0.70 → first 70% train, last 30% eval).
3. `FeatureScaler` fits z-score statistics on the **train split only** and is
   persisted to `scalers/{scenario}_{ticker}.json`. The same scaler transforms
   the eval split to avoid look-ahead leakage.

The model-facing observation is the scaled `FEATURE_COLS` vector. The raw
`Close` price is intentionally left unscaled because the environment needs it for
trade accounting.

### 3.3 Environment (`trading_env.py`)

- **Action space:** `Discrete(3)` → `0=hold`, `1=buy`, `2=sell`.
- **Observation:** the scaled feature row for the **previous** day (`step_idx-1`),
  i.e. a one-day observation lag so the agent never sees same-day information it
  would trade on.
- **Trade mechanics:** buys/sells one share at the day's `Close`, charging
  `TRANSACTION_COST_BPS` (5 bps default) per trade.
- **Portfolio value:** `cash + shares × close` (consistent everywhere).
- **Reward:** a rolling (≈21-day) Sharpe-style ratio of recent portfolio returns.
- `RandomTickerEnv` picks a random ticker from `TICKERS` each episode, so one
  model per scenario learns across the whole basket (AAPL, MSFT, SPY, TLT, GLD).

### 3.4 Training (`train.py`)

For each scenario it fits all scalers, then for each agent it builds a
`RandomTickerEnv`, trains an SB3 `MlpPolicy` for the resolved timestep count, and
saves:

- `models/{AGENT}_{scenario}.zip` — the trained policy.
- `models/{AGENT}_{scenario}_config.json` — reproducibility sidecar (tickers,
  initial balance, train ratio, timesteps, mode, transaction cost, seed,
  timestamp).

Timestep resolution precedence: `--timesteps` > `--mode` > `config.TRAINING_MODE`
(`quick`=5k, `dev`=30k, `final`=100k). Default run trains 5 scenarios × 3 agents
= **15 models**; `--scenario` / `--agent` filter to a subset.

### 3.5 Evaluation (`evaluate.py`)

`run_battle(ticker, scenario)`:

1. `get_eval_data()` rebuilds features, splits, and applies the saved scaler to
   the eval slice.
2. Runs `run_buy_and_hold()` and each available RL model deterministically over
   the eval window, collecting daily portfolio histories.
3. `compute_metrics()` derives Sharpe (annualized × √252 only when the eval
   window ≥ `ANNUALIZE_SHARPE_MIN_DAYS`), total return, max drawdown, and final
   value.
4. `bootstrap_sharpe_ci()` resamples daily returns (500 draws, seeded) for a 95%
   Sharpe confidence interval.
5. `enrich_metrics()` flags `low_activity` agents (|return| < 1%) so the UI can
   warn about "sat in cash" Sharpe winners.
6. Results are sorted by Sharpe; both the Sharpe winner and the return leader are
   reported (they can differ), along with `missing_models`, `dates`, histories,
   and macro summary.

### 3.6 Presentation (`app.py`, `explain.py`)

- `app.py` is UI only. It calls `run_battle()` (cached on a `models_cache_key()`
  derived from model file mtimes), renders winner cards, equity curves
  (dollar/return views), a leaderboard, and the analysis text.
- `explain.py` produces the narrative: if `GROQ_API_KEY` is set it calls Groq
  (`config.GROQ_MODEL`); otherwise it falls back to a deterministic rule-based
  template. Groq failures silently fall back too.

---

## 4. Configuration knobs (`config.py`)

| Setting | Effect |
|---------|--------|
| `INITIAL_BALANCE` | Starting cash per episode/battle ($10,000). |
| `TRANSACTION_COST_BPS` | Per-trade fee (5 bps = 0.05%). |
| `TRAINING_MODE` / `TIMESTEPS_BY_MODE` | quick / dev / final training lengths. |
| `TRAIN_RATIO` | Train/eval split fraction (0.70). |
| `TICKERS` | Training basket and UI dropdown. |
| `MACRO_TICKERS` | Symbols for VIX, 10Y yield, SPY. |
| `FEATURE_COLS` | Observation vector fed to agents. |
| `SHORT_WINDOW_DAYS` / `ANNUALIZE_SHARPE_MIN_DAYS` | Noise warning + Sharpe annualization gate. |
| `LOW_ACTIVITY_RETURN_THRESHOLD` | Threshold for the low-activity flag. |
| `GROQ_MODEL` | LLM used for analysis. |
| `SCENARIO_LABELS` | Human-readable scenario names. |

---

## 5. Storage layout

| Path | Contents | Tracked in git? |
|------|----------|-----------------|
| `data/cache/` | OHLCV + macro CSVs | No (gitignored) |
| `scalers/` | Per scenario+ticker z-score JSON | No (gitignored) |
| `models/` | `*.zip` policies + `*_config.json` sidecars | `.zip` gitignored |
| `.env` | `GROQ_API_KEY` (optional) | No (gitignored) |

All three data folders are regenerated by running `train.py` (and the first
dashboard run for cache/scalers), so a fresh clone only needs dependencies plus a
training pass.

---

## 6. Security notes

- **Secrets:** the only secret is `GROQ_API_KEY`, loaded from `.env` via
  `python-dotenv`. `.env` is gitignored; commit only `.env.example`. For
  Streamlit Community Cloud, set the key in app secrets rather than in the repo.
- **No PII / no auth:** the app stores no user data and exposes no write API. It
  reads public market data and local model files only.
- **External calls:** outbound network is limited to Yahoo Finance (data) and
  Groq (optional analysis). If Groq is unreachable or the key is missing, the
  rule-based fallback keeps the app fully functional offline (after data cache
  exists).
- **Untrusted input:** scenario/ticker selections are constrained to the
  configured allow-lists (`SCENARIOS`, `TICKERS`) in the UI, avoiding arbitrary
  symbol fetches.
- **Model files:** `*.zip` policies are deserialized by Stable Baselines3. Only
  load models you trained yourself or trust, since pickle-based loading can
  execute code.

---

## 7. Testing

```bash
.venv/bin/pytest tests/ -v
```

`tests/test_evaluate.py` covers the evaluation/metrics path (Sharpe, drawdown,
return, history alignment) so changes to `evaluate.py` are guarded.

---

## 8. Interview framing

Use precise, honest language:

- **Pitch:** "A labeled-regime backtest lab that stress-tests DQN/PPO/A2C trading
  policies against buy-and-hold on out-of-sample eval slices, with macro-aware
  features and transaction costs."
- **Methodology rigor:** time-ordered 70/30 split, scaler fit on train only
  (no leakage), one-day observation lag, bootstrap Sharpe CIs, and annualization
  gated on window length.
- **Reading results:** Sharpe winner ≠ return leader is expected — a low-activity
  agent can win Sharpe by mostly sitting in cash, which the `low_activity` flag
  surfaces.
- **Honest limitations:** regime labels are historical narratives (not a learned
  classifier); models fit past windows and are **not** forecasts; trading is
  simplified to one share/day; training is multi-ticker while eval is
  single-ticker; short windows (e.g. COVID crash) make Sharpe noisy.
- **One-liner:** "Historical replay only — not a forecast."
