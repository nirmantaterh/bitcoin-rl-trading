"""
Refresh all data caches to today.
Run manually or via GitHub Actions cron.

Usage: python data/refresh.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_price import fetch_price
from data.fetch_sentiment import fetch_sentiment


def refresh(end: str | None = None) -> None:
    today = end or date.today().isoformat()
    start = "2019-01-01"

    print(f"=== refreshing data: {start} → {today} ===\n")

    cache_dir = Path(__file__).parent / "cache"

    print("--- price ---")
    price_cache = cache_dir / "btc_price.csv"
    if price_cache.exists():
        price_cache.unlink()
    price_df = fetch_price(start=start, end=today)
    print(f"price rows: {len(price_df)}\n")

    print("--- sentiment (finbert) ---")
    sent_cache = cache_dir / "btc_sentiment_finbert.csv"
    if sent_cache.exists():
        sent_cache.unlink()
    sent_df = fetch_sentiment(start=start, end=today, mode="finbert", refresh=True)
    print(f"sentiment rows: {len(sent_df)}\n")

    print("=== refresh complete ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--end", default=None, help="End date (default: today)")
    args = p.parse_args()
    refresh(args.end)
