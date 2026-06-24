"""
Thin wrapper around Stable Baselines3 A2C / SAC / TD3 for the BitcoinTradingEnv.

Using SB3 directly (not FinRL's DRLAgent) for tighter control over hyperparameters
and compatibility with the custom Gymnasium env.

A2C params match the paper: lr=0.007, n_steps=5, 50k timesteps.
SAC/TD3 params are extensions — documented in configs/default.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from env.bitcoin_env import BitcoinTradingEnv

_SB3_ALGORITHMS = {
    "a2c": "stable_baselines3.A2C",
    "sac": "stable_baselines3.SAC",
    "td3": "stable_baselines3.TD3",
}


def _import_algo(name: str):
    import importlib
    module_path, cls_name = _SB3_ALGORITHMS[name].rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


class RLAgent:
    """
    Wraps SB3 A2C / SAC / TD3 with a consistent train / predict interface.

    Usage:
        agent = RLAgent("sac", env, cfg["agents"]["sac"])
        agent.train(seed=42)
        portfolio = agent.evaluate(test_env)
    """

    def __init__(self, algo: str, env: BitcoinTradingEnv, algo_cfg: dict):
        self.algo = algo.lower()
        self.env = env
        self.cfg = algo_cfg
        self.model = None

        if self.algo not in _SB3_ALGORITHMS:
            raise ValueError(f"algo must be one of {list(_SB3_ALGORITHMS)}, got '{algo}'")

    def train(self, seed: int = 42, verbose: int = 0) -> "RLAgent":
        AlgoCls = _import_algo(self.algo)
        timesteps = self.cfg.get("timesteps", 50_000)
        policy = self.cfg.get("policy", "MlpPolicy")

        # Build kwargs from cfg, drop non-SB3 keys
        kwargs: dict[str, Any] = {"policy": policy, "env": self.env, "seed": seed, "verbose": verbose}
        for key in ["learning_rate", "n_steps", "buffer_size", "batch_size", "ent_coef"]:
            if key in self.cfg:
                kwargs[key] = self.cfg[key]

        self.model = AlgoCls(**kwargs)
        self.model.learn(total_timesteps=timesteps)
        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call train() before predict()")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def evaluate(self, eval_env: BitcoinTradingEnv) -> list[float]:
        """Run one episode on eval_env deterministically. Returns portfolio_history."""
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action = self.predict(obs)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        return eval_env.portfolio_history

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save — model not trained")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: str | Path) -> "RLAgent":
        AlgoCls = _import_algo(self.algo)
        self.model = AlgoCls.load(str(path), env=self.env)
        return self
