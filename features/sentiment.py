"""
Merge sentiment scores into the price DataFrame and add derived sentiment features.
"""

import numpy as np
import pandas as pd


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ma_window: int = 5,
) -> pd.DataFrame:
    """
    Left-merge sentiment onto price DataFrame using backward merge_asof (no lookahead).

    sentiment_df must have at minimum: date, sentiment_score.
    Additional columns (sentiment_pos, sentiment_neg, sentiment_neu, article_count) are
    carried through if present.

    Derived features added:
      sentiment_ma5     — rolling 5-day mean of sentiment_score
      sentiment_lag1    — 1-day lagged sentiment_score
      sentiment_momentum — sentiment_score - sentiment_lag1
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    sent = sentiment_df.copy().sort_values("date").reset_index(drop=True)

    # merge_asof: for each price date, take the most recent sentiment date <= price date
    merged = pd.merge_asof(df, sent, on="date", direction="backward")

    # Forward-fill any gaps (weekends / missing news days)
    sentiment_cols = [c for c in sent.columns if c != "date"]
    merged[sentiment_cols] = merged[sentiment_cols].ffill()

    merged["sentiment_ma5"] = (
        merged["sentiment_score"].rolling(ma_window, min_periods=1).mean()
    )
    merged["sentiment_lag1"] = merged["sentiment_score"].shift(1)
    merged["sentiment_momentum"] = merged["sentiment_score"] - merged["sentiment_lag1"]

    return merged


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_price import fetch_price
    from data.fetch_sentiment import fetch_sentiment

    price = fetch_price(start="2022-01-01", end="2022-12-31")
    sent = fetch_sentiment(start="2022-01-01", end="2022-12-31", mode="finbert")
    out = add_sentiment_features(price, sent)
    print(out[["date", "close", "sentiment_score", "sentiment_ma5", "sentiment_lag1"]].tail(10).to_string())
    print(f"\nShape: {out.shape}")
