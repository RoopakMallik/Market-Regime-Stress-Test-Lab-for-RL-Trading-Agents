"""
Evaluate competitors and build the battle leaderboard.

Main entry: run_battle(ticker, scenario)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from baselines import run_buy_and_hold
from config import (
    AGENTS,
    ANNUALIZE_SHARPE_MIN_DAYS,
    FEATURE_COLS,
    LOW_ACTIVITY_RETURN_THRESHOLD,
    MODELS_DIR,
    SCALERS_DIR,
    SHORT_WINDOW_DAYS,
)
from data import macro_summary
from features import FeatureScaler, add_technical_features, get_merged_data, train_test_split_df
from trading_env import TradingEnv

AGENT_CLASSES = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


def get_eval_data(ticker: str, scenario_name: str) -> tuple:
    """Load test split once (raw + scaled) — avoids duplicate downloads."""
    df = get_merged_data(ticker, scenario_name)
    df = add_technical_features(df)
    train_df, test_df = train_test_split_df(df)

    scaler_path = SCALERS_DIR / f"{scenario_name}_{ticker}.json"
    if scaler_path.exists():
        scaler = FeatureScaler.load(scaler_path)
    else:
        scaler = FeatureScaler().fit(train_df, FEATURE_COLS)
        scaler.save(scaler_path)

    scaled_test = scaler.transform(test_df, FEATURE_COLS)
    return test_df, scaled_test


def compute_metrics(portfolio_history: list[float], eval_days: int) -> dict:
    """Sharpe, return, drawdown, final value from a daily portfolio series."""
    ph = np.array(portfolio_history, dtype=float)
    if len(ph) < 2:
        return {
            "sharpe": 0.0,
            "sharpe_annualized": False,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "final_value": round(float(ph[-1]), 2),
        }

    returns = np.diff(ph) / (ph[:-1] + 1e-8)
    std = returns.std()
    sharpe_raw = float(returns.mean() / (std + 1e-8))

    annualized = eval_days >= ANNUALIZE_SHARPE_MIN_DAYS
    sharpe = sharpe_raw * np.sqrt(252) if annualized else sharpe_raw

    peak = np.maximum.accumulate(ph)
    drawdown = ((peak - ph) / (peak + 1e-8)).max()
    total_return = (ph[-1] - ph[0]) / (ph[0] + 1e-8)

    return {
        "sharpe": round(float(sharpe), 3),
        "sharpe_annualized": annualized,
        "max_drawdown": round(float(drawdown), 3),
        "total_return": round(float(total_return), 3),
        "final_value": round(float(ph[-1]), 2),
    }


def enrich_metrics(metrics: dict, trade_count: int | None = None) -> dict:
    """Add quality flags so the UI can warn about misleading Sharpe winners."""
    abs_return = abs(metrics.get("total_return", 0.0))
    metrics["low_activity"] = abs_return < LOW_ACTIVITY_RETURN_THRESHOLD
    if trade_count is not None:
        metrics["trade_count"] = trade_count
    return metrics


def bootstrap_sharpe_ci(
    portfolio_history: list[float],
    eval_days: int,
    n: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    ph = np.array(portfolio_history, dtype=float)
    if len(ph) < 20:
        return None

    returns = np.diff(ph) / (ph[:-1] + 1e-8)
    annualized = eval_days >= ANNUALIZE_SHARPE_MIN_DAYS
    samples = []

    rng = np.random.default_rng(42)
    for _ in range(n):
        draw = rng.choice(returns, size=len(returns), replace=True)
        std = draw.std()
        s = float(draw.mean() / (std + 1e-8))
        if annualized:
            s *= np.sqrt(252)
        samples.append(s)

    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return round(low, 3), round(high, 3)


def align_history(history: list[float], target_len: int) -> list[float]:
    """Pad or trim so every competitor has the same number of chart points."""
    if not history:
        return [0.0] * target_len
    if len(history) == target_len:
        return history
    if len(history) > target_len:
        return history[:target_len]
    return history + [history[-1]] * (target_len - len(history))


def evaluate_rl_agent(model, df_eval, feature_cols: list[str] | None = None) -> tuple[dict, list[float], int]:
    feature_cols = feature_cols or FEATURE_COLS
    env = TradingEnv(df_eval, feature_cols)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    metrics = compute_metrics(env.history, eval_days=len(df_eval))
    enrich_metrics(metrics, trade_count=env.trade_count)
    return metrics, env.history, env.trade_count


def _model_path(agent_name: str, scenario_name: str) -> Path:
    return MODELS_DIR / f"{agent_name}_{scenario_name}.zip"


def models_cache_key() -> str:
    """Change when any model file is updated (for Streamlit cache busting)."""
    zips = list(MODELS_DIR.glob("*.zip"))
    if not zips:
        return "no-models"
    return str(int(max(p.stat().st_mtime for p in zips)))


def model_exists(agent_name: str, scenario_name: str) -> bool:
    return _model_path(agent_name, scenario_name).exists()


def run_battle(ticker: str, scenario_name: str) -> dict:
    test_df, scaled_eval = get_eval_data(ticker, scenario_name)

    eval_days = len(test_df)
    short_window_warning = eval_days < SHORT_WINDOW_DAYS
    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in test_df.index]
    target_len = len(dates)

    results: list[dict] = []
    histories: dict[str, list[float]] = {}

    bh_history = align_history(run_buy_and_hold(test_df), target_len)
    bh_metrics = enrich_metrics(compute_metrics(bh_history, eval_days), trade_count=1)
    bh_metrics["agent"] = "Buy & Hold"
    ci = bootstrap_sharpe_ci(bh_history, eval_days)
    if ci:
        bh_metrics["sharpe_ci_low"], bh_metrics["sharpe_ci_high"] = ci
    results.append(bh_metrics)
    histories["Buy & Hold"] = bh_history

    missing_models: list[str] = []
    for agent_name in AGENTS:
        path = _model_path(agent_name, scenario_name)
        if not path.exists():
            missing_models.append(agent_name)
            continue

        model = AGENT_CLASSES[agent_name].load(path)
        metrics, history, trades = evaluate_rl_agent(model, scaled_eval)
        metrics["agent"] = agent_name
        ci = bootstrap_sharpe_ci(history, eval_days)
        if ci:
            metrics["sharpe_ci_low"], metrics["sharpe_ci_high"] = ci
        results.append(metrics)
        histories[agent_name] = align_history(history, target_len)

    results.sort(key=lambda row: row["sharpe"], reverse=True)
    sharpe_winner = results[0] if results else {}
    return_winner = max(results, key=lambda row: row["total_return"]) if results else {}

    return {
        "results": results,
        "histories": histories,
        "dates": dates,
        "eval_days": eval_days,
        "short_window_warning": short_window_warning,
        "macro_summary": macro_summary(scenario_name),
        "missing_models": missing_models,
        "ticker": ticker,
        "scenario": scenario_name,
        "sharpe_winner": sharpe_winner,
        "return_winner": return_winner,
        "same_winner": sharpe_winner.get("agent") == return_winner.get("agent"),
    }
