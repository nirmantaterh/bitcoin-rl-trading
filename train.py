"""
Train one RL agent on a specified period and evaluate on all test periods.

Usage:
    python train.py --agent sac --sentiment finbert --timesteps 100000
    python train.py --agent a2c --sentiment finbert --timesteps 50000
    python train.py --agent td3 --sentiment cryptobert --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import pandas as pd

from data.fetch_price import fetch_price
from data.fetch_sentiment import fetch_sentiment
from features.pipeline import build_feature_matrix, build_test_period
from env.bitcoin_env import BitcoinTradingEnv
from agents.rl_agent import RLAgent
from agents.baselines import BuyAndHoldAgent, RandomAgent
from evaluate import compute_metrics, print_metrics

RESULTS_DIR = Path("results")


def run(
    algo: str,
    sentiment_mode: str,
    timesteps: int | None,
    seeds: list[int],
    cfg_path: str = "configs/default.yaml",
    save_model: bool = True,
) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    algo_cfg = cfg["agents"][algo].copy()
    if timesteps is not None:
        algo_cfg["timesteps"] = timesteps

    # Fetch data
    full_start = data_cfg["train_start"]
    all_ends = [p["end"] for p in data_cfg["test_periods"]]
    full_end = max(all_ends)

    print(f"[train] fetching price {full_start} → {full_end}")
    price_df = fetch_price(start=full_start, end=full_end)

    print(f"[train] fetching sentiment ({sentiment_mode})")
    sentiment_df = fetch_sentiment(start=full_start, end=full_end, mode=sentiment_mode)

    # Build train / first test split
    train_end = data_cfg["train_end"]
    train_df, _, scaler = build_feature_matrix(price_df, sentiment_df, train_end=train_end, cfg=cfg)

    results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[train] {algo.upper()} | sentiment={sentiment_mode} | seed={seed}")
        print(f"{'='*60}")

        train_env = BitcoinTradingEnv(train_df, **cfg["env"])
        agent = RLAgent(algo, train_env, algo_cfg)
        agent.train(seed=seed, verbose=1)

        if save_model:
            model_path = RESULTS_DIR / "models" / f"{algo}_{sentiment_mode}_seed{seed}"
            agent.save(model_path)
            print(f"[train] saved model → {model_path}")

        seed_results = {}

        # Evaluate on each test period
        for period in data_cfg["test_periods"]:
            p_name = period["name"]
            test_df = build_test_period(
                price_df, sentiment_df,
                start=period["start"], end=period["end"],
                scaler=scaler, cfg=cfg,
            )
            if test_df.empty:
                print(f"[eval] {p_name}: no data, skipping")
                continue

            # RL agent
            test_env = BitcoinTradingEnv(test_df, **cfg["env"])
            rl_history = agent.evaluate(test_env)
            rl_metrics = compute_metrics(rl_history)
            print_metrics(f"{algo.upper()} + {sentiment_mode} | {p_name} | seed={seed}", rl_metrics)
            seed_results[p_name] = {"rl": rl_metrics}

            # Baselines (only need to run once per period — independent of seed)
            if seed == seeds[0]:
                bh_env = BitcoinTradingEnv(test_df, **cfg["env"])
                bh_history = BuyAndHoldAgent().evaluate(bh_env)
                bh_metrics = compute_metrics(bh_history)
                print_metrics(f"BuyAndHold | {p_name}", bh_metrics)
                seed_results[p_name]["buy_and_hold"] = bh_metrics

                rand_history = RandomAgent.evaluate_multi_seed(
                    BitcoinTradingEnv(test_df, **cfg["env"]),
                    seeds=algo_cfg.get("seeds", seeds),
                )
                rand_metrics = compute_metrics(rand_history)
                print_metrics(f"Random | {p_name}", rand_metrics)
                seed_results[p_name]["random"] = rand_metrics

        results[f"seed_{seed}"] = seed_results

    # Save results JSON
    out_file = RESULTS_DIR / f"{algo}_{sentiment_mode}_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[train] results saved → {out_file}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="sac", choices=["a2c", "sac", "td3"])
    p.add_argument("--sentiment", default="finbert", choices=["finbert", "cryptobert"])
    p.add_argument("--timesteps", type=int, default=None, help="Override config timesteps")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["agents"][args.agent]["seeds"]
    run(args.agent, args.sentiment, args.timesteps, seeds, args.config, not args.no_save)


if __name__ == "__main__":
    main()
