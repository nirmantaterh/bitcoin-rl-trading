"""
Technical indicators matching the paper's Table II feature set.

BMSB (Bull Market Support Band) parameters KS/KP are not disclosed in the paper.
We use KS=KP=0.02 as documented assumptions (see configs/default.yaml).

All indicators use only past data — no lookahead leakage.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bmsb(
    close: pd.Series,
    sma_window: int = 147,  # 21 weeks × 7
    ema_window: int = 140,  # 20 weeks × 7
    ks: float = 0.02,       # ponytail: assumed — paper does not disclose KS/KP
    kp: float = 0.02,
) -> pd.DataFrame:
    """
    Bull Market Support Band from the paper.

    Returns DataFrame with columns: bmsb_sma, bmsb_ema, bmsb_mid, bmsb_lower, bmsb_upper, bmsb_signal.
    bmsb_signal: 1 if price above band, -1 if below, 0 if inside.
    """
    s = sma(close, sma_window)
    e = ema(close, ema_window)

    mid = (s + e) / 2
    lower = mid * (1 - ks)
    upper = mid * (1 + kp)

    signal = pd.Series(0, index=close.index, dtype=float)
    signal[close > upper] = 1.0
    signal[close < lower] = -1.0

    return pd.DataFrame({
        "bmsb_sma": s,
        "bmsb_ema": e,
        "bmsb_mid": mid,
        "bmsb_lower": lower,
        "bmsb_upper": upper,
        "bmsb_signal": signal,
    })


def add_technical_features(
    df: pd.DataFrame,
    sma_window: int = 147,
    ema_window: int = 140,
    rsi_window: int = 14,
    volatility_window: int = 21,
    momentum_window: int = 10,
    bmsb_ks: float = 0.02,
    bmsb_kp: float = 0.02,
) -> pd.DataFrame:
    """
    Add all technical indicators in-place and return the DataFrame.
    NaN rows from rolling warm-up are retained — caller decides whether to drop them.
    """
    df = df.copy()
    close = df["close"]

    df["sma"] = sma(close, sma_window)
    df["ema"] = ema(close, ema_window)
    df["rsi"] = rsi(close, rsi_window)

    bmsb_df = bmsb(close, sma_window, ema_window, bmsb_ks, bmsb_kp)
    df = pd.concat([df, bmsb_df], axis=1)

    log_ret = np.log(close / close.shift(1))
    df["daily_return"] = log_ret
    df["volatility"] = log_ret.rolling(volatility_window, min_periods=volatility_window).std()
    df["momentum"] = close.pct_change(momentum_window)

    # Normalised position within BMSB band (0 = lower, 1 = upper, can exceed [0,1])
    band_width = df["bmsb_upper"] - df["bmsb_lower"]
    df["bmsb_position"] = (close - df["bmsb_lower"]) / band_width.replace(0, np.nan)

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_price import fetch_price

    price = fetch_price(start="2019-01-01", end="2022-12-31")
    out = add_technical_features(price)
    print(out.tail(5).to_string())
    print(f"\nShape: {out.shape}, NaN rows: {out.isna().any(axis=1).sum()}")
