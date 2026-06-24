"""
Central settings file — change most project knobs here.

PORTFOLIO (starting cash, fees):
  → INITIAL_BALANCE, TRANSACTION_COST_BPS
  → Used by trading_env.py, baselines.py, evaluate.py

TRAINING (how long / how hard to train RL models):
  → TRAINING_MODE = "quick" | "dev" | "final"
  → Or run: .venv/bin/python train.py --help

PORTFOLIO VALUE formula (same everywhere):
  portfolio_value = cash_in_account + (shares_owned × current_stock_price)
"""

from pathlib import Path

# ── Folder paths (usually no need to change) ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_CACHE_DIR = ROOT_DIR / "data" / "cache"   # Downloaded stock CSV files
SCALERS_DIR = ROOT_DIR / "scalers"             # Z-score scaler JSON files
MODELS_DIR = ROOT_DIR / "models"               # Saved RL model .zip files


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SETTINGS
# Change these to adjust starting money and trading fees.
# ══════════════════════════════════════════════════════════════════════════════

# Starting cash ($) at the beginning of every training episode and every battle.
# Example: 10_000 means each agent starts with $10,000.
INITIAL_BALANCE = 10_000

# Trading fee per buy or sell, in basis points (bps).
# 5 bps = 0.05% of the trade value. Higher = more realistic but harder to profit.
TRANSACTION_COST_BPS = 5


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# Pick a TRAINING_MODE, then run:  .venv/bin/python train.py
#
#   "quick"  → ~5–15 min  (smoke test, agents may barely trade)
#   "dev"    → ~30–75 min (normal development)
#   "final"  → ~2–4 hours (best quality for demos / resume)
#
# You can also override from the command line:
#   .venv/bin/python train.py --mode final
#   .venv/bin/python train.py --timesteps 20000
#   .venv/bin/python train.py --scenario bull_run --agent PPO
# ══════════════════════════════════════════════════════════════════════════════

TRAINING_MODE = "final"  # "quick" | "dev" | "final"

TIMESTEPS_BY_MODE = {
    "quick": 5_000,
    "dev": 30_000,
    "final": 100_000,
}

# Resolved step count used by train.py (derived from TRAINING_MODE).
TIMESTEPS = TIMESTEPS_BY_MODE[TRAINING_MODE]

# Which RL algorithms to train (one model per scenario per agent).
AGENTS = ["DQN", "PPO", "A2C"]

# Reproducibility — same seed gives similar training results each run.
RANDOM_SEED = 42

# Fraction of each scenario window used for training (rest is eval / test).
# 0.70 = first 70% train, last 30% eval (shown in the dashboard).
TRAIN_RATIO = 0.70


# ── Stocks the project downloads, trains on, and shows in the UI ─────────────
TICKERS = ["AAPL", "MSFT", "SPY", "TLT", "GLD"]


# ── Macro data symbols (VIX, 10-year yield, SPY trend) ───────────────────────
MACRO_TICKERS = {
    "vix": "^VIX",
    "yield_10y": "^TNX",
    "spy": "SPY",
}


# ── Metric / UI thresholds ───────────────────────────────────────────────────
SHORT_WINDOW_DAYS = 60           # Below this, dashboard warns Sharpe is noisy
ANNUALIZE_SHARPE_MIN_DAYS = 60   # Annualize Sharpe only if eval window is long enough

# Groq model for the Analysis section in the dashboard.
GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Feature columns fed to RL agents (not raw stock price — that stays for trading) ──
FEATURE_COLS = [
    "rsi",
    "macd",
    "bb_upper",
    "bb_lower",
    "volatility",
    "vix",
    "yield_10y",
    "spy_above_sma200",
    "return_1d",
]

# Flag agents with very small absolute return (often "sit in cash" policies).
LOW_ACTIVITY_RETURN_THRESHOLD = 0.01  # 1%
SCENARIO_LABELS = {
    "recession": "2008 Recession",
    "bull_run": "2017 Bull Run",
    "rate_hike": "2022 Rate Hike Cycle",
    "covid_crash": "COVID Crash",
    "recovery": "2020–2021 Recovery",
}
