"""
Full feature pipeline: price → technical → sentiment → scale → train/test split.

Design decisions:
- Scaler fit on train only, applied to both train and test (no leakage)
- NaN rows from rolling warm-up dropped after merge
- Returns raw DataFrames + fitted scaler so evaluation can inverse-transform prices
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from features.technical import add_technical_features
from features.sentiment import add_sentiment_features

FEATURE_COLS = [
    "sma", "ema", "rsi",
    "bmsb_mid", "bmsb_signal", "bmsb_position",
    "volatility", "momentum", "daily_return",
    "sentiment_score", "sentiment_ma5", "sentiment_lag1",
    "article_count",
    # raw price/position kept for env state — scaled separately
    "close",
]


def build_feature_matrix(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    train_end: str,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Build train and test feature matrices.

    Returns:
        train_df  — rows where date <= train_end, scaled
        test_df   — rows where date > train_end, scaled with train scaler
        scaler    — fitted MinMaxScaler (use to inverse-transform close prices)
    """
    if cfg is None:
        cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    feat_cfg = cfg.get("features", {})
    df = add_technical_features(
        price_df,
        sma_window=feat_cfg.get("sma_window", 147),
        ema_window=feat_cfg.get("ema_window", 140),
        rsi_window=feat_cfg.get("rsi_window", 14),
        volatility_window=feat_cfg.get("volatility_window", 21),
        momentum_window=feat_cfg.get("momentum_window", 10),
        bmsb_ks=feat_cfg.get("bmsb_ks", 0.02),
        bmsb_kp=feat_cfg.get("bmsb_kp", 0.02),
    )

    sent_cfg = cfg.get("sentiment", {})
    df = add_sentiment_features(df, sentiment_df, ma_window=feat_cfg.get("sentiment_ma_window", 5))

    # Drop NaN rows from rolling warm-up
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Split
    split_dt = pd.Timestamp(train_end)
    train_mask = df["date"] <= split_dt
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[~train_mask].copy().reset_index(drop=True)

    # Scale: fit on train only
    scaler = MinMaxScaler()
    cols_to_scale = [c for c in FEATURE_COLS if c in df.columns]
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    return train_df, test_df, scaler


def build_test_period(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    start: str,
    end: str,
    scaler: MinMaxScaler,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Build a single test period using a pre-fitted scaler.
    Used for out-of-sample evaluation on bear_2022 and recovery_2023 periods.
    """
    if cfg is None:
        cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    feat_cfg = cfg.get("features", {})
    df = add_technical_features(price_df, **{
        k: feat_cfg.get(k, v) for k, v in {
            "sma_window": 147, "ema_window": 140, "rsi_window": 14,
            "volatility_window": 21, "momentum_window": 10,
            "bmsb_ks": 0.02, "bmsb_kp": 0.02,
        }.items()
    })
    df = add_sentiment_features(df, sentiment_df, ma_window=feat_cfg.get("sentiment_ma_window", 5))
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    mask = (df["date"] >= start) & (df["date"] <= end)
    df = df[mask].copy().reset_index(drop=True)

    cols_to_scale = [c for c in FEATURE_COLS if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_price import fetch_price
    from data.fetch_sentiment import fetch_sentiment

    price = fetch_price(start="2019-01-01", end="2022-12-31")
    sent = fetch_sentiment(start="2019-01-01", end="2022-12-31", mode="finbert")
    train, test, scaler = build_feature_matrix(price, sent, train_end="2021-12-31")
    print(f"Train: {train.shape} | {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"Test:  {test.shape} | {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"Features: {[c for c in FEATURE_COLS if c in train.columns]}")
