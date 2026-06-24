"""
Buy-and-hold baseline — passive strategy for comparison.

Timeline matches TradingEnv:
  Day 0 = all cash
  Day 1 = buy as many shares as possible, then hold
"""

from __future__ import annotations

import pandas as pd

from config import INITIAL_BALANCE, TRANSACTION_COST_BPS


def run_buy_and_hold(df_eval: pd.DataFrame, initial_balance: float = INITIAL_BALANCE) -> list[float]:
    """Return daily portfolio values aligned with the RL eval timeline."""
    if df_eval.empty:
        return [float(initial_balance)]

    cost_rate = TRANSACTION_COST_BPS / 10_000.0
    prices = df_eval["Close"].astype(float).to_numpy()
    n = len(prices)

    # Day 0: start with cash only (same as RL reset).
    history = [float(initial_balance)]
    if n == 1:
        return history

    cash = float(initial_balance)
    shares = 0

    for day in range(1, n):
        price = float(prices[day])
        if day == 1 and shares == 0:
            all_in = price * (1 + cost_rate)
            shares = int(cash // all_in)
            cash -= shares * all_in
        history.append(cash + shares * price)

    return history
