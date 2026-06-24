# Market Regime Stress-Test Lab for RL Trading Agents

Compare **DQN, PPO, and A2C** reinforcement-learning trading agents against **buy-and-hold** across five historically labeled market windows (recession, bull run, rate hikes, COVID crash, recovery).

Built with Python, Stable Baselines3, yfinance, and Streamlit. Optional Groq API generates interview-style battle summaries.

---

## Features

- Five macro regime scenarios with date windows + VIX / 10Y yield / SPY trend
- Multi-ticker training basket (AAPL, MSFT, SPY, TLT, GLD)
- 70% train / 30% eval split (dashboard uses the **held-out eval slice**)
- Risk metrics: Sharpe, total return, max drawdown, bootstrap Sharpe CI
- Buy & hold baseline with aligned timeline vs RL agents
- Low-activity warnings when Sharpe winners barely trade
- Interactive Streamlit dashboard with equity curves and leaderboard

---

## Prerequisites

- **Python 3.11+** (tested on 3.13)
- **macOS / Linux / Windows**
- Internet access (yfinance downloads)
- Optional: [Groq API key](https://console.groq.com) for AI analysis

**No database required** — data is cached as CSV files; models saved as `.zip` files.

---

## Installation

```bash
cd "/path/to/Market Regime Stress-Test Lab for RL Trading Agents"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Environment variables

Copy the example file and add your key:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | No | AI battle analysis in the dashboard |

Without `GROQ_API_KEY`, the app uses a built-in rule-based summary.

---

## Data & storage (no database)

| Path | Purpose |
|------|---------|
| `data/cache/` | Downloaded OHLCV + macro CSVs |
| `scalers/` | Z-score scaler JSON per scenario+ticker |
| `models/` | Trained RL models (`*.zip`) + config sidecars |

First run downloads market data automatically. No migrations or SQL setup.

---

## Train models

Edit `config.py`:

```python
TRAINING_MODE = "quick"   # ~5–15 min  (5k steps)
TRAINING_MODE = "dev"     # ~30–75 min (30k steps)
TRAINING_MODE = "final"   # ~2–4 hr   (100k steps)
```

Then:

```bash
.venv/bin/python train.py
.venv/bin/python train.py --mode dev
.venv/bin/python train.py --scenario bull_run --agent PPO
.venv/bin/python train.py --help
```

Trains **15 models** (3 agents × 5 scenarios) unless you filter with `--scenario` / `--agent`.

---

## Run the dashboard locally

```bash
.venv/bin/streamlit run app.py
```

Open **http://localhost:8501** → pick scenario + ticker → **Run Battle**.

---

## Run tests

```bash
.venv/bin/pytest tests/ -v
```

---

## Configuration (`config.py`)

| Setting | What it does |
|---------|----------------|
| `INITIAL_BALANCE` | Starting cash ($10,000 default) |
| `TRANSACTION_COST_BPS` | Fee per trade (5 bps default) |
| `TRAINING_MODE` | quick / dev / final timesteps |
| `TRAIN_RATIO` | 70% train, 30% eval |
| `TICKERS` | Stocks in training + UI dropdown |

---

## Deployment (Streamlit Community Cloud)

1. Push repo to GitHub (exclude `.env`, `data/cache/`, `models/*.zip` or commit demo models)
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Main file: `app.py`
4. Add `GROQ_API_KEY` in app secrets
5. Run `python train.py` in CI or upload models before deploy

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python: command not found` | Use `python3` or `.venv/bin/python` |
| Missing RL agents in UI | Run `.venv/bin/python train.py` |
| Agents show ~0% return | Retrain with `TRAINING_MODE = "dev"` or `"final"` |
| Sharpe winner ≠ highest return | Expected — see sidebar “Who is competing?” |
| Groq analysis fails | Check `.env`; rule-based fallback runs automatically |
| Stale results after retrain | Refresh browser; cache keys on model file timestamps |

---

## Honest limitations

1. Regime labels are historical narratives, not a formal regime classifier.
2. Models train on past windows — **not** future predictions.
3. RL buys one share per day max — simplified vs real portfolio sizing.
4. Multi-ticker training, single-ticker eval — see analysis notes in the UI.
5. Short scenarios (e.g. COVID crash) produce noisy Sharpe estimates.

---

## Documentation

- **[docs/TECHNICAL.md](docs/TECHNICAL.md)** — architecture, modules, flows, security, interview framing

---

## Resume bullet (example)

> Built Market Regime Stress-Test Lab for RL Trading Agents, a Streamlit app comparing DQN, PPO, and A2C against buy-and-hold across five labeled macro stress windows, with macro-aware features, transaction costs, out-of-sample eval splits, and Groq-generated interview narratives.
