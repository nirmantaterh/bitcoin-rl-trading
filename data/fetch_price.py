"""
Fetch daily BTC/USDT OHLCV from Binance public API.
Falls back to yfinance if Binance is unreachable.

Output: data/cache/btc_price.csv
Columns: date, open, high, low, close, volume
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "btc_price.csv"
BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Pull daily klines from Binance public REST. No API key needed."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    all_rows = []
    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        start_ms = rows[-1][0] + 86_400_000  # advance by one day
        time.sleep(0.1)  # polite rate limiting

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """Fallback: yfinance BTC-USD daily."""
    import yfinance as yf  # lazy import — optional dependency

    raw = yf.download("BTC-USD", start=start, end=end, interval="1d", progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame")

    # yfinance returns MultiIndex columns in newer versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"adj close": "adj_close"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    return raw[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def fetch_price(symbol: str = "BTCUSDT", start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch BTC price data with automatic fallback.
    Returns cached data if cache exists and covers the requested range.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use cache if it covers the full requested range
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        cached_max = cached["date"].max().date()
        req_end = pd.Timestamp(end).date()
        if cached["date"].min() <= pd.Timestamp(start) and cached_max >= req_end:
            mask = (cached["date"] >= start) & (cached["date"] <= end)
            print(f"[price] loaded {mask.sum()} rows from cache")
            return cached[mask].reset_index(drop=True)

    print(f"[price] fetching {symbol} {start} → {end}")
    try:
        df = fetch_binance(symbol, start, end)
        print(f"[price] Binance: {len(df)} rows")
    except Exception as e:
        print(f"[price] Binance failed ({e}), falling back to yfinance")
        df = fetch_yfinance(start, end)
        print(f"[price] yfinance: {len(df)} rows")

    df.to_csv(CACHE_FILE, index=False)
    print(f"[price] cached to {CACHE_FILE}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    df = fetch_price(args.symbol, args.start, args.end)
    print(df.head())
    print(f"Shape: {df.shape}, Date range: {df['date'].min().date()} → {df['date'].max().date()}")
