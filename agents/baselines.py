"""
Non-RL baselines for comparison against RL agents.

BuyAndHoldAgent  — buy everything at step 0, hold until end.
RandomAgent      — random continuous action every step.

Both return portfolio_history lists for apples-to-apples metric comparison.
"""

from __future__ import annotations

import numpy as np

from env.bitcoin_env import BitcoinTradingEnv


class BuyAndHoldAgent:
    """Invest 100% of initial balance in BTC at the first step, hold forever."""

    def evaluate(self, env: BitcoinTradingEnv) -> list[float]:
        obs, _ = env.reset()
        # Buy everything at step 0
        obs, _, terminated, truncated, _ = env.step(np.array([1.0]))
        done = terminated or truncated
        while not done:
            obs, _, terminated, truncated, _ = env.step(np.array([0.0]))  # hold
            done = terminated or truncated
        return env.portfolio_history


class RandomAgent:
    """Sample random actions each step. Average over multiple seeds for stability."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate(self, env: BitcoinTradingEnv) -> list[float]:
        rng = np.random.default_rng(self.seed)
        obs, _ = env.reset(seed=self.seed)
        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        return env.portfolio_history

    @staticmethod
    def evaluate_multi_seed(env: BitcoinTradingEnv, seeds: list[int]) -> list[float]:
        """Return element-wise mean portfolio across multiple random seeds."""
        histories = []
        for s in seeds:
            agent = RandomAgent(seed=s)
            histories.append(agent.evaluate(env))

        # Pad shorter histories to longest length (ruin episodes end early)
        max_len = max(len(h) for h in histories)
        padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]
        return list(np.mean(padded, axis=0))
