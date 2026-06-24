"""
Feature engineering and train/test splitting.

Pipeline per ticker + scenario:
  1. get_merged_data()        → raw prices + macro
  2. add_technical_features() → RSI, MACD, Bollinger Bands, etc.
  3. train_test_split_df()    → first 70% train, last 30% eval (config.TRAIN_RATIO)
  4. FeatureScaler            → z-score normalize using train stats only

The eval split (test) is what the dashboard replays when you click Run Battle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import FEATURE_COLS, SCALERS_DIR, TRAIN_RATIO
from data import get_merged_data


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index — momentum indicator (0–100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> pd.Series:
    """MACD line — difference between fast and slow moving averages."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


def _bollinger(close: pd.Series, length: int = 5, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Bollinger Band upper and lower lines around a rolling mean."""
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    return mid + std_mult * std, mid - std_mult * std


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns used by RL agents (see config.FEATURE_COLS)."""
    out = df.copy()
    close = out["Close"]

    out["rsi"] = _rsi(close, length=14)
    out["macd"] = _macd(close)
    out["bb_upper"], out["bb_lower"] = _bollinger(close, length=20)
    out["volatility"] = close.pct_change().rolling(20).std()
    out["return_1d"] = close.pct_change()
    out = out.dropna()
    return out


def train_test_split_df(df: pd.DataFrame, ratio: float = TRAIN_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split: no shuffling (important for time series).
    First `ratio` fraction → train; remainder → eval/test.
    """
    if len(df) < 10:
        raise ValueError("Not enough rows for train/test split")
    split_idx = max(1, int(len(df) * ratio))
    if split_idx >= len(df):
        split_idx = len(df) - 1
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


class FeatureScaler:
    """Z-score normalization: (value - mean) / std, fit on train data only."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        self.stats = {}
        for col in feature_cols:
            series = df[col].astype(float)
            mean = float(series.mean())
            std = float(series.std())
            if std < 1e-8:
                std = 1.0
            self.stats[col] = {"mean": mean, "std": std}
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in feature_cols:
            if col not in self.stats:
                raise KeyError(f"Scaler not fitted for column: {col}")
            mean = self.stats[col]["mean"]
            std = self.stats[col]["std"]
            out[col] = (out[col].astype(float) - mean) / std
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        scaler = cls()
        scaler.stats = json.loads(path.read_text())
        return scaler


def scaler_path(scenario_name: str, ticker: str) -> Path:
    return SCALERS_DIR / f"{scenario_name}_{ticker}.json"


def prepare_raw_splits(ticker: str, scenario_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline to train and test dataframes (not yet scaled)."""
    df = get_merged_data(ticker, scenario_name)
    df = add_technical_features(df)
    return train_test_split_df(df)


def load_scaled_split(
    ticker: str,
    scenario_name: str,
    split: str = "train",
) -> tuple[pd.DataFrame, FeatureScaler]:
    """Return scaled train or test data; Close price kept unscaled for trading."""
    train_df, test_df = prepare_raw_splits(ticker, scenario_name)
    path = scaler_path(scenario_name, ticker)

    if path.exists():
        scaler = FeatureScaler.load(path)
    else:
        scaler = FeatureScaler().fit(train_df, FEATURE_COLS)
        scaler.save(path)

    source = train_df if split == "train" else test_df
    scaled = scaler.transform(source, FEATURE_COLS)
    return scaled, scaler


def fit_all_scalers(scenario_name: str, tickers: list[str]) -> None:
    """Called at the start of training for each scenario — saves scalers/*.json."""
    for ticker in tickers:
        train_df, _ = prepare_raw_splits(ticker, scenario_name)
        scaler = FeatureScaler().fit(train_df, FEATURE_COLS)
        scaler.save(scaler_path(scenario_name, ticker))
