"""Tests for BitcoinTradingEnv contract."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.technical import add_technical_features
from features.sentiment import add_sentiment_features
from features.pipeline import FEATURE_COLS
from env.bitcoin_env import BitcoinTradingEnv


def make_env(n: int = 200) -> BitcoinTradingEnv:
    rng = np.random.default_rng(42)
    close = 10_000.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    price_df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.uniform(1e9, 5e9, n),
    })
    sent_df = pd.DataFrame({
        "date": dates,
        "sentiment_score": rng.uniform(-1, 1, n),
        "sentiment_pos": rng.uniform(0, 1, n),
        "sentiment_neg": rng.uniform(0, 1, n),
        "sentiment_neu": rng.uniform(0, 1, n),
        "article_count": rng.integers(1, 20, n),
    })
    df = add_technical_features(price_df)
    df = add_sentiment_features(df, sent_df)
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns]).reset_index(drop=True)
    return BitcoinTradingEnv(df)


class TestEnvContract:
    def test_reset_returns_correct_shape(self):
        env = make_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "portfolio_value" in info

    def test_hold_action_no_trade(self):
        """Action near zero should not change portfolio composition."""
        env = make_env()
        env.reset(seed=0)
        initial_balance = env._balance
        initial_btc = env._btc_held
        env.step(np.array([0.05]))  # below hold_threshold=0.1
        assert env._balance == pytest.approx(initial_balance)
        assert env._btc_held == pytest.approx(initial_btc)

    def test_buy_reduces_balance(self):
        env = make_env()
        env.reset(seed=0)
        before = env._balance
        env.step(np.array([1.0]))  # full buy
        assert env._balance < before
        assert env._btc_held > 0

    def test_sell_increases_balance(self):
        env = make_env()
        env.reset(seed=0)
        env.step(np.array([1.0]))  # buy first
        btc_after_buy = env._btc_held
        balance_after_buy = env._balance
        env.step(np.array([-1.0]))  # full sell
        assert env._btc_held < btc_after_buy
        assert env._balance > balance_after_buy

    def test_episode_ends_at_last_step(self):
        env = make_env(n=200)
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            obs, _, terminated, truncated, _ = env.step(np.array([0.0]))
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_portfolio_history_length(self):
        env = make_env(n=200)
        env.reset(seed=0)
        for _ in range(10):
            env.step(np.array([0.0]))
        assert len(env.portfolio_history) == 11  # reset + 10 steps

    def test_action_space_contains_actions(self):
        env = make_env()
        for _ in range(20):
            a = env.action_space.sample()
            assert env.action_space.contains(a)
