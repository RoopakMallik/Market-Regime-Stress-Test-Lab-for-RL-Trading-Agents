import numpy as np
import pandas as pd

from baselines import run_buy_and_hold
from config import INITIAL_BALANCE
from evaluate import align_history, compute_metrics, enrich_metrics
from trading_env import TradingEnv


def _sample_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Close": close,
            "rsi": rng.normal(0, 1, n),
            "macd": rng.normal(0, 1, n),
            "bb_upper": close + 1,
            "bb_lower": close - 1,
            "volatility": np.abs(rng.normal(0, 0.01, n)),
            "vix": rng.normal(20, 2, n),
            "yield_10y": rng.normal(2, 0.1, n),
            "spy_above_sma200": rng.integers(0, 2, n),
            "return_1d": rng.normal(0, 0.01, n),
        }
    )


def test_compute_metrics_positive_return():
    history = [10_000, 10_100, 10_200, 10_150]
    m = compute_metrics(history, eval_days=64)
    assert m["total_return"] > 0
    assert m["final_value"] == 10_150


def test_align_history_pads():
    assert align_history([1, 2], 4) == [1, 2, 2, 2]


def test_enrich_metrics_low_activity():
    m = enrich_metrics({"total_return": 0.002}, trade_count=0)
    assert m["low_activity"] is True


def test_bh_and_rl_same_history_length():
    df = _sample_df(12)
    bh = run_buy_and_hold(df)
    env = TradingEnv(df)
    env.reset()
    while True:
        _, _, done, trunc, _ = env.step(0)
        if done or trunc:
            break
    assert len(bh) == len(env.history) == len(df)


def test_trading_env_starts_cash():
    df = _sample_df(8)
    env = TradingEnv(df)
    env.reset()
    assert env.history[0] == INITIAL_BALANCE
    assert env.shares == 0
