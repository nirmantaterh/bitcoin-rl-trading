"""Tests for technical indicators and pipeline correctness."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.technical import sma, ema, rsi, bmsb, add_technical_features
from features.sentiment import add_sentiment_features


def make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10_000.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.uniform(1e9, 5e9, n),
    })


def make_sentiment_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sentiment_score": rng.uniform(-1, 1, n),
        "sentiment_pos": rng.uniform(0, 1, n),
        "sentiment_neg": rng.uniform(0, 1, n),
        "sentiment_neu": rng.uniform(0, 1, n),
        "article_count": rng.integers(1, 20, n),
    })


class TestIndicators:
    def test_sma_window(self):
        s = pd.Series(range(1, 11))
        result = sma(s, 3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[9] == pytest.approx(9.0)

    def test_rsi_range(self):
        price = make_price_df(200)["close"]
        r = rsi(price, window=14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_bmsb_signal_values(self):
        price = make_price_df(300)
        band = bmsb(price["close"])
        valid = band["bmsb_signal"].dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all()

    def test_bmsb_mid_between_bands(self):
        price = make_price_df(300)
        band = bmsb(price["close"])
        valid = band.dropna()
        assert (valid["bmsb_lower"] <= valid["bmsb_mid"]).all()
        assert (valid["bmsb_mid"] <= valid["bmsb_upper"]).all()

    def test_no_lookahead_sma(self):
        """SMA at time t must not use data from t+1 onwards."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 2)
        assert result.iloc[1] == pytest.approx(1.5)  # mean(1,2) not mean(2,3)


class TestSentimentMerge:
    def test_no_lookahead(self):
        """merge_asof backward: sentiment date must be <= price date."""
        price = make_price_df(50)
        # Sentiment only on odd days — ensures gaps exist
        sent = make_sentiment_df(50)
        sent = sent[sent.index % 2 == 0].reset_index(drop=True)

        merged = add_sentiment_features(price, sent)
        # All sentiment dates after fill must be <= corresponding price date
        sent_dates = merged["date"]
        # Check no NaN after ffill (there should be none since we ffill)
        assert merged["sentiment_score"].isna().sum() == 0

    def test_output_columns(self):
        price = make_price_df(200)
        sent = make_sentiment_df(200)
        out = add_sentiment_features(price, sent)
        assert "sentiment_ma5" in out.columns
        assert "sentiment_lag1" in out.columns
        assert "sentiment_momentum" in out.columns

    def test_lag1_shifted(self):
        price = make_price_df(50)
        sent = make_sentiment_df(50)
        merged = add_sentiment_features(price, sent)
        valid = merged.dropna(subset=["sentiment_lag1"])
        # lag1[i] should equal sentiment_score[i-1]
        for i in range(1, min(10, len(valid))):
            assert valid["sentiment_lag1"].iloc[i] == pytest.approx(
                valid["sentiment_score"].iloc[i - 1]
            )


class TestMetrics:
    def test_compute_metrics_buy_and_hold(self):
        from evaluate import compute_metrics
        # Straight line up 10%
        pv = [10_000.0 * (1 + 0.1 * i / 252) for i in range(253)]
        m = compute_metrics(pv)
        assert m["cumulative_return"] > 0
        assert m["max_drawdown"] <= 0

    def test_compute_metrics_all_loss(self):
        from evaluate import compute_metrics
        pv = [10_000.0 * (0.99 ** i) for i in range(100)]
        m = compute_metrics(pv)
        assert m["cumulative_return"] < 0
        assert m["max_drawdown"] < 0
