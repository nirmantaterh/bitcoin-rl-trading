"""
Single-command paper replication: A2C + FinBERT, paper hyperparameters.

The original paper used VADER; this repo replaces it with FinBERT (better domain fit).
A2C hyperparameters (lr=0.007, n_steps=5, 50k timesteps) match the paper exactly.
Other approximations are documented in README.md.

Usage:
    python run_paper.py
    python run_paper.py --seeds 42          # single seed, faster
    python run_paper.py --sentiment finbert  # explicit (default)
"""

import sys
from train import run

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Paper replication: A2C + FinBERT")
    p.add_argument("--sentiment", default="finbert", choices=["finbert", "cryptobert"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    p.add_argument("--timesteps", type=int, default=50_000)
    args = p.parse_args()

    print("=" * 60)
    print("  PAPER REPLICATION: A2C + FinBERT")
    print("  Algorithm: A2C (lr=0.007, n_steps=5, 50k steps)")
    print(f"  Sentiment: {args.sentiment}")
    print(f"  Seeds: {args.seeds}")
    print("  Approximations: reward=log-return, BMSB KS=KP=0.02")
    print("  See README.md for full replication notes")
    print("=" * 60)

    run(
        algo="a2c",
        sentiment_mode=args.sentiment,
        timesteps=args.timesteps,
        seeds=args.seeds,
    )
