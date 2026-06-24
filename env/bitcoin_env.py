"""
BitcoinTradingEnv — Gymnasium environment for BTC RL trading.

Observation space: 15-dim continuous vector (normalized by MinMaxScaler in pipeline)
Action space: Box([-1], [1]) — positive=buy fraction, negative=sell fraction, ~0=hold

Reward: log(portfolio_value_t / portfolio_value_{t-1})
  Paper says "maximize logarithmic cumulative return" but gives no exact formula.
  Log-return reward is the standard interpretation and is documented as an assumption.

Transaction costs: 0.1% buy + 0.1% sell (Binance standard).
Episode ends: last timestep OR portfolio < ruin_threshold * initial_amount.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from features.pipeline import FEATURE_COLS


class BitcoinTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 10_000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        hold_threshold: float = 0.1,
        reward_scaling: float = 1.0,
        ruin_threshold: float = 0.1,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.hold_threshold = hold_threshold
        self.reward_scaling = reward_scaling
        self.ruin_floor = initial_amount * ruin_threshold

        # State cols: all feature cols that exist in df + balance + btc_held
        self._feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        self._state_dim = len(self._feat_cols) + 2  # +2: balance_norm, btc_held_norm

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]),
            dtype=np.float32,
        )

        self._step_idx: int = 0
        self._balance: float = initial_amount
        self._btc_held: float = 0.0
        self._prev_portfolio: float = initial_amount
        self._portfolio_history: list[float] = []

    # ─── Gymnasium interface ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._balance = self.initial_amount
        self._btc_held = 0.0
        self._prev_portfolio = self.initial_amount
        self._portfolio_history = [self.initial_amount]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))
        price = self._current_price()

        if abs(a) < self.hold_threshold:
            pass  # hold
        elif a > 0:
            self._execute_buy(a, price)
        else:
            self._execute_sell(-a, price)

        portfolio_value = self._balance + self._btc_held * price
        self._portfolio_history.append(portfolio_value)

        reward = np.log(portfolio_value / self._prev_portfolio + 1e-9) * self.reward_scaling
        self._prev_portfolio = portfolio_value

        self._step_idx += 1
        terminated = (
            self._step_idx >= len(self.df) - 1
            or portfolio_value < self.ruin_floor
        )
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {
            "portfolio_value": portfolio_value,
            "balance": self._balance,
            "btc_held": self._btc_held,
        }

    def render(self):
        price = self._current_price()
        pv = self._balance + self._btc_held * price
        print(
            f"step={self._step_idx:4d}  price={price:10.2f}  "
            f"balance={self._balance:10.2f}  btc={self._btc_held:.6f}  "
            f"portfolio={pv:10.2f}"
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _current_price(self) -> float:
        idx = min(self._step_idx, len(self.df) - 1)
        # close column was scaled to [0,1] — use raw if available, else use scaled
        # pipeline stores normalised close; env works with it consistently
        return float(self.df.iloc[idx]["close"])

    def _execute_buy(self, fraction: float, price: float) -> None:
        spend = self._balance * fraction
        cost = spend * self.buy_cost_pct
        net_spend = spend - cost
        if price > 0 and net_spend > 0:
            self._btc_held += net_spend / price
            self._balance -= spend

    def _execute_sell(self, fraction: float, price: float) -> None:
        sell_btc = self._btc_held * fraction
        proceeds = sell_btc * price
        cost = proceeds * self.sell_cost_pct
        self._btc_held -= sell_btc
        self._balance += proceeds - cost

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_idx, len(self.df) - 1)
        row = self.df.iloc[idx]
        feats = row[self._feat_cols].values.astype(np.float32)

        # Normalise wallet state relative to initial amount
        price = float(row["close"])
        balance_norm = self._balance / self.initial_amount
        btc_norm = (self._btc_held * price) / self.initial_amount

        return np.concatenate([feats, [balance_norm, btc_norm]]).astype(np.float32)

    @property
    def portfolio_history(self) -> list[float]:
        return self._portfolio_history


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.fetch_price import fetch_price
    from data.fetch_sentiment import fetch_sentiment
    from features.pipeline import build_feature_matrix
    from gymnasium.utils.env_checker import check_env

    price = fetch_price(start="2019-01-01", end="2021-12-31")
    sent = fetch_sentiment(start="2019-01-01", end="2021-12-31", mode="finbert")
    train_df, _, _ = build_feature_matrix(price, sent, train_end="2021-12-31")

    env = BitcoinTradingEnv(train_df)
    print("Running gymnasium env_checker...")
    check_env(env)
    print("check_env passed.")

    obs, _ = env.reset(seed=42)
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
    for _ in range(5):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        env.render()
        if term:
            break
