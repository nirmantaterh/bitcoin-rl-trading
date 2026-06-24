"""
Fetch BTC news headlines and score them with FinBERT or CryptoBERT.

Primary source: CryptoPanic API (free tier, set CRYPTOPANIC_API_KEY in .env)
Fallback: bundled sample headlines in data/cache/sample_headlines.csv

Modes:
  finbert    — ProsusAI/finbert, trained on financial news (best match for CryptoPanic headlines)
  cryptobert — ElKulako/cryptobert, trained on Reddit/Twitter crypto text
               (domain mismatch vs news headlines — use for comparison experiments only)

Output: data/cache/btc_sentiment_<mode>.csv
Columns: date, headline, source, sentiment_score, sentiment_pos, sentiment_neg, sentiment_neu
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
SAMPLE_CACHE = CACHE_DIR / "sample_headlines.csv"


# ─── Headline Fetching ────────────────────────────────────────────────────────

def fetch_cryptopanic(start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch BTC headlines from CryptoPanic API."""
    import requests

    headlines = []
    url = "https://cryptopanic.com/api/v1/posts/"
    page = 1

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    print("[sentiment] fetching from CryptoPanic...")
    while True:
        params = {
            "auth_token": api_key,
            "currencies": "BTC",
            "kind": "news",
            "public": "true",
            "page": page,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for post in results:
            pub_dt = pd.Timestamp(post["published_at"]).tz_convert(None)
            if pub_dt < start_dt:
                return pd.DataFrame(headlines)  # past our range, done
            if pub_dt <= end_dt:
                headlines.append({
                    "date": pub_dt.normalize(),
                    "headline": post["title"],
                    "source": post.get("source", {}).get("title", ""),
                })

        if not data.get("next"):
            break
        page += 1
        time.sleep(0.5)

    return pd.DataFrame(headlines)


def load_sample_headlines() -> pd.DataFrame:
    """Load bundled sample headlines for offline / no-API-key usage."""
    if SAMPLE_CACHE.exists():
        df = pd.read_csv(SAMPLE_CACHE, parse_dates=["date"])
        print(f"[sentiment] loaded {len(df)} sample headlines from cache")
        return df
    raise FileNotFoundError(
        f"No CryptoPanic API key and no sample cache found at {SAMPLE_CACHE}. "
        "Run data/fetch_sentiment.py --refresh with CRYPTOPANIC_API_KEY set."
    )


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_finbert(headlines: list[str], model_name: str = "ProsusAI/finbert", batch_size: int = 32) -> list[dict]:
    """Score headlines with FinBERT. Returns list of {pos, neg, neu, score} dicts."""
    import scipy.special
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"[sentiment] loading FinBERT ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    label_map = model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}
    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            logits = model(**inputs).logits.numpy()

        for row in logits:
            probs = scipy.special.softmax(row)
            scores = {label_map[j]: float(probs[j]) for j in range(3)}
            scores["sentiment_score"] = scores["positive"] - scores["negative"]
            results.append(scores)

        if (i // batch_size) % 10 == 0:
            print(f"[sentiment] scored {min(i + batch_size, len(headlines))}/{len(headlines)}")

    return results


def score_cryptobert(headlines: list[str], model_name: str = "ElKulako/cryptobert", batch_size: int = 32) -> list[dict]:
    """Score headlines with CryptoBERT.

    Note: CryptoBERT was trained on Reddit/Twitter crypto text — domain mismatch
    when scoring news headlines. Use finbert for production; cryptobert for ablation.
    """
    import torch
    from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

    print(f"[sentiment] loading CryptoBERT ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=64,
        truncation=True,
        padding="max_length",
        top_k=None,  # return all class scores, not just the top label
    )

    results = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        preds = pipe(batch)
        for pred_list in preds:
            scores = {p["label"].lower(): p["score"] for p in pred_list}
            pos = scores.get("bullish", 0.0)
            neg = scores.get("bearish", 0.0)
            neu = scores.get("neutral", 0.0)
            results.append({
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "sentiment_score": pos - neg,
            })

        if (i // batch_size) % 10 == 0:
            print(f"[sentiment] scored {min(i + batch_size, len(headlines))}/{len(headlines)}")

    return results


# ─── Daily Aggregation ────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-headline scores to daily level."""
    agg = df.groupby("date").agg(
        sentiment_score=("sentiment_score", "mean"),
        sentiment_pos=("positive", "mean"),
        sentiment_neg=("negative", "mean"),
        sentiment_neu=("neutral", "mean"),
        article_count=("sentiment_score", "count"),
        net_sentiment=("sentiment_score", "sum"),
    ).reset_index()
    return agg.sort_values("date").reset_index(drop=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def fetch_sentiment(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    mode: str = "finbert",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: fetch headlines → score → aggregate to daily.
    mode: "finbert" (default, best for news) or "cryptobert" (ablation only)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CACHE_DIR / f"btc_sentiment_{mode}.csv"

    if out_file.exists() and not refresh:
        df = pd.read_csv(out_file, parse_dates=["date"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        if mask.sum() > 0:
            print(f"[sentiment] loaded {mask.sum()} days from cache ({mode})")
            return df[mask].reset_index(drop=True)

    # 1. Get headlines
    api_key = os.getenv("CRYPTOPANIC_API_KEY", "")
    if api_key:
        headlines_df = fetch_cryptopanic(start, end, api_key)
    else:
        print("[sentiment] no CRYPTOPANIC_API_KEY — using bundled sample cache")
        headlines_df = load_sample_headlines()
        mask = (headlines_df["date"] >= start) & (headlines_df["date"] <= end)
        headlines_df = headlines_df[mask].copy()

    if headlines_df.empty:
        raise RuntimeError("No headlines fetched. Check API key or date range.")

    # 2. Score
    texts = headlines_df["headline"].tolist()
    if mode == "finbert":
        scores = score_finbert(texts)
    elif mode == "cryptobert":
        scores = score_cryptobert(texts)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'finbert' or 'cryptobert'.")

    scored_df = pd.concat([
        headlines_df.reset_index(drop=True),
        pd.DataFrame(scores),
    ], axis=1)

    # 3. Aggregate to daily
    daily = aggregate_daily(scored_df)

    daily.to_csv(out_file, index=False)
    print(f"[sentiment] cached {len(daily)} daily rows to {out_file}")
    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--mode", default="finbert", choices=["finbert", "cryptobert"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    df = fetch_sentiment(args.start, args.end, args.mode, args.refresh)
    print(df.head(10))
    print(f"Shape: {df.shape}")
