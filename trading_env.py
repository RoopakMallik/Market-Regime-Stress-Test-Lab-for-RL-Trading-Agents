"""
Gymnasium trading environment — simulates portfolio value day by day.

PORTFOLIO = cash + (shares × close price)
Settings: config.INITIAL_BALANCE, config.TRANSACTION_COST_BPS
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import FEATURE_COLS, INITIAL_BALANCE, TICKERS, TRANSACTION_COST_BPS
from features import load_scaled_split


class TradingEnv(gym.Env):
    """Walk through one price dataframe; observe yesterday, trade at today's close."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        initial_balance: float = INITIAL_BALANCE,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or FEATURE_COLS
        self.initial_balance = initial_balance
        self.transaction_cost_rate = TRANSACTION_COST_BPS / 10_000.0
        self.trade_count = 0

        n_features = len(self.feature_cols)
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )

        self.step_idx = 1
        self.balance = self.initial_balance
        self.shares = 0
        self.history: list[float] = [self.initial_balance]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_idx = 1
        self.balance = self.initial_balance
        self.shares = 0
        self.trade_count = 0
        # Day 0: all cash (aligned with buy-and-hold baseline).
        self.history = [float(self.initial_balance)]
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        price = float(self.df.loc[self.step_idx, "Close"])

        if action == 1:
            cost = price * self.transaction_cost_rate
            total = price + cost
            if self.balance >= total:
                self.shares += 1
                self.balance -= total
                self.trade_count += 1
        elif action == 2 and self.shares > 0:
            proceeds = self.shares * price
            cost = proceeds * self.transaction_cost_rate
            self.balance += proceeds - cost
            self.shares = 0
            self.trade_count += 1

        portfolio = self.balance + self.shares * price
        self.history.append(float(portfolio))

        reward = self._sharpe_reward()
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df)
        truncated = False

        obs = (
            self._obs()
            if not terminated
            else np.zeros(len(self.feature_cols), dtype=np.float32)
        )
        return obs, reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        row = self.df.loc[self.step_idx - 1, self.feature_cols]
        return row.astype(np.float32).to_numpy()

    def _sharpe_reward(self) -> float:
        if len(self.history) < 10:
            return 0.0
        recent = np.array(self.history[-21:])
        rets = np.diff(recent) / (recent[:-1] + 1e-8)
        if len(rets) < 2 or rets.std() < 1e-8:
            return 0.0
        return float(rets.mean() / rets.std())

    def portfolio_value(self) -> float:
        idx = min(self.step_idx, len(self.df) - 1)
        price = float(self.df.loc[idx, "Close"])
        return self.balance + self.shares * price


class RandomTickerEnv(gym.Env):
    """Training wrapper: random ticker each episode (multi-ticker training)."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_name: str, split: str = "train", seed: int | None = None) -> None:
        super().__init__()
        self.scenario_name = scenario_name
        self.split = split
        self._rng = random.Random(seed)
        self._inner: TradingEnv | None = None

        self.feature_cols = list(FEATURE_COLS)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )

    def _new_inner(self) -> TradingEnv:
        ticker = self._rng.choice(TICKERS)
        df, _ = load_scaled_split(ticker, self.scenario_name, split=self.split)
        if len(df) < 3:
            raise ValueError(f"Not enough rows to train on {ticker} / {self.scenario_name}")
        return TradingEnv(df, self.feature_cols)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._inner = self._new_inner()
        return self._inner.reset(seed=seed)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._inner is None:
            raise RuntimeError("Environment not reset")
        return self._inner.step(action)
