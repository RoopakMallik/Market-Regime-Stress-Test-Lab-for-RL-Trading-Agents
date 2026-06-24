"""
Download stock prices and macro data from Yahoo Finance.

Flow:
  download_ohlcv()  → single stock OHLCV
  download_macro()  → VIX, 10Y yield, SPY trend
  get_merged_data() → stock + macro joined for one ticker + scenario

Files are cached under data/cache/ so re-runs are fast.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from config import DATA_CACHE_DIR, MACRO_TICKERS
from scenarios import get_scenario_dates


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns multi-level column names — flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download Open/High/Low/Close/Volume for one ticker; cache to CSV."""
    cache_path = DATA_CACHE_DIR / f"{ticker}_{start}_{end}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return df

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No OHLCV data for {ticker} between {start} and {end}")

    df = _flatten_columns(raw)
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    df = df.dropna()

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def download_macro(start: str, end: str) -> pd.DataFrame:
    """Download VIX, 10-year Treasury yield, and SPY 200-day MA context."""
    cache_path = DATA_CACHE_DIR / f"macro_{start}_{end}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return df

    vix = yf.download(MACRO_TICKERS["vix"], start=start, end=end, progress=False)
    vix = _flatten_columns(vix)[["Close"]].rename(columns={"Close": "vix"})

    tnx = yf.download(MACRO_TICKERS["yield_10y"], start=start, end=end, progress=False)
    tnx = _flatten_columns(tnx)[["Close"]].rename(columns={"Close": "yield_10y"})

    spy = yf.download(
        MACRO_TICKERS["spy"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    spy = _flatten_columns(spy)[["Close"]].rename(columns={"Close": "spy_close"})
    spy["spy_sma200"] = spy["spy_close"].rolling(200, min_periods=20).mean()
    spy["spy_above_sma200"] = (spy["spy_close"] > spy["spy_sma200"]).astype(float)

    macro = vix.join(tnx, how="outer").join(spy, how="outer")
    macro = macro.sort_index().ffill().dropna()

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_csv(cache_path)
    return macro


def get_merged_data(ticker: str, scenario_name: str) -> pd.DataFrame:
    """Stock OHLCV + macro columns aligned by date for one ticker and scenario."""
    start, end = get_scenario_dates(scenario_name)
    ohlcv = download_ohlcv(ticker, start, end)
    macro = download_macro(start, end)

    merged = ohlcv.join(macro, how="left")
    merged[macro.columns] = merged[macro.columns].ffill()
    merged = merged.dropna()

    if merged.empty:
        raise ValueError(f"Merged dataset empty for {ticker} / {scenario_name}")

    return merged


def macro_summary(scenario_name: str) -> dict:
    """Summary stats for the dashboard sidebar (avg VIX, yield change, etc.)."""
    start, end = get_scenario_dates(scenario_name)
    macro = download_macro(start, end)
    if macro.empty:
        return {}

    vix_avg = float(macro["vix"].mean())
    yield_start = float(macro["yield_10y"].iloc[0])
    yield_end = float(macro["yield_10y"].iloc[-1])
    yield_change = yield_end - yield_start
    spy_above_pct = float(macro["spy_above_sma200"].mean())

    return {
        "start": start,
        "end": end,
        "avg_vix": round(vix_avg, 2),
        "yield_10y_start": round(yield_start, 2),
        "yield_10y_end": round(yield_end, 2),
        "yield_10y_change": round(yield_change, 2),
        "spy_above_sma200_pct": round(spy_above_pct * 100, 1),
    }
