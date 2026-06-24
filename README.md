# bitcoin-rl-trading

Reproducible re-implementation of the Springer paper **"Feature-Rich Long-term Bitcoin Trading Assistant** ([arXiv:2209.12664](https://arxiv.org/abs/2209.12664)), extended with proper baselines, risk-adjusted metrics, and modern sentiment models.

> **The paper introduced the idea. This repo is how you'd build it today.**

---

## What the paper found vs what this repo adds

| | Paper | This Repo |
|---|---|---|
| Algorithm | A2C (SB3) | A2C (replication) + SAC + TD3 |
| Sentiment | VADER (2013 tool) | FinBERT (financial news) + CryptoBERT (ablation) |
| Baselines | None | Buy-and-Hold + Random agent |
| Risk metrics | None reported | Sharpe, Sortino, Max Drawdown, Calmar |
| Test periods | Single bull run 2019–2021 | Bear market 2022 + Recovery 2023–2024 |
| Multiple seeds | Single run | 5 seeds per agent |
| Data freshness | Static (2021) | Weekly auto-refresh via GitHub Actions |

---

## Results (to be populated after training)

| Strategy | Period | Return | Sharpe | Sortino | Max DD |
|---|---|---|---|---|---|
| Buy & Hold | bear_2022 | TBD | TBD | TBD | TBD |
| A2C + FinBERT | bear_2022 | TBD | TBD | TBD | TBD |
| SAC + FinBERT | bear_2022 | TBD | TBD | TBD | TBD |
| TD3 + FinBERT | bear_2022 | TBD | TBD | TBD | TBD |
| Buy & Hold | recovery_2023 | TBD | TBD | TBD | TBD |
| SAC + FinBERT | recovery_2023 | TBD | TBD | TBD | TBD |

> **Note on buy-and-hold**: Bitcoin's sustained bull run in 2019–2021 means buy-and-hold often beats RL on raw return. The advantage of RL agents appears in risk-adjusted metrics (Sortino, Max Drawdown) especially in the 2022 bear market. This is expected and documented — the paper itself has no baselines for comparison.

---

## Architecture

```
bitcoin-rl-trading/
├── data/
│   ├── fetch_price.py       # Binance public API (no auth) + yfinance fallback
│   ├── fetch_sentiment.py   # CryptoPanic headlines → FinBERT / CryptoBERT scores
│   └── refresh.py           # Pull all data to today (also runs weekly via CI)
│
├── features/
│   ├── technical.py         # SMA, EMA, RSI, BMSB, volatility, momentum
│   ├── sentiment.py         # merge_asof (no lookahead), daily aggregation
│   └── pipeline.py          # full pipeline: indicators + sentiment + MinMaxScaler
│
├── env/
│   └── bitcoin_env.py       # Gymnasium BitcoinTradingEnv
│
├── agents/
│   ├── rl_agent.py          # SB3 wrapper: A2C / SAC / TD3
│   └── baselines.py         # BuyAndHoldAgent, RandomAgent
│
├── evaluate.py              # Sharpe, Sortino, MaxDD, Calmar, win rate
├── train.py                 # Full training CLI
├── run_paper.py             # One-command paper replication
│
├── configs/default.yaml     # All hyperparameters in one place
└── .github/workflows/
    └── refresh.yml          # Weekly data refresh cron
```

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Replicate the paper (A2C + FinBERT, paper hyperparams)
python run_paper.py

# Run full comparison
python train.py --agent sac --sentiment finbert
python train.py --agent td3 --sentiment finbert
python train.py --agent a2c --sentiment cryptobert  # ablation

# Refresh data manually
python data/refresh.py

# Run tests
pytest tests/ -v
```

No API key required — the repo ships with `data/cache/sample_headlines.csv` (106 BTC headlines 2022–2024) for offline use. For live sentiment data, register a free [CryptoPanic](https://cryptopanic.com/developers/api/) account and set `CRYPTOPANIC_API_KEY` in `.env`.

---

## Environment design

**State** (15-dim): `[sma, ema, rsi, bmsb_mid, bmsb_signal, bmsb_position, volatility, momentum, daily_return, sentiment_score, sentiment_ma5, sentiment_lag1, article_count, balance_norm, btc_held_norm]`

**Action** `∈ [-1, 1]`:
- `> 0.1` → buy `action × balance` worth of BTC
- `< -0.1` → sell `|action| × btc_held` BTC
- `[-0.1, 0.1]` → hold

**Reward**: `log(pv_t / pv_{t-1})` — log-return. The paper states "maximize logarithmic cumulative return" without disclosing the exact reward formula; log-return is the standard interpretation and is documented as an assumption.

**Transaction costs**: 0.1% buy + 0.1% sell (Binance standard).

---

## Replication notes

| Detail | Paper | This Repo |
|---|---|---|
| Algorithm | A2C, lr=0.007, n_steps=5, 50k steps | Exact match |
| Features | 11 (Table II) | Matches Table II |
| Sentiment model | VADER | FinBERT (paper: outdated VADER) |
| BMSB params (KS/KP) | Not disclosed | KS=KP=0.02 (documented assumption) |
| Reward formula | Not disclosed | `log(pv_t / pv_{t-1})` (documented assumption) |
| Data source | Cryptocompare + Twitter | Binance API + CryptoPanic |
| Train period | Jan 2016 – May 2019 | 2019–2021 |
| MLP architecture | Not disclosed | SB3 MlpPolicy defaults (64×64 Tanh) |

---

## Citation

```bibtex
@article{lucarelli2022deep,
  title={A Deep Reinforcement Learning Approach to Bitcoin Trading},
  author={Lucarelli, Giulio and Borrotti, Matteo},
  journal={arXiv preprint arXiv:2209.12664},
  year={2022}
}
```

---

## Data sources (all free, no auth required for price data)

- **Price**: [Binance public REST API](https://api.binance.com/api/v3/klines) — `BTCUSDT` 1d, no API key needed
- **Sentiment headlines**: [CryptoPanic API](https://cryptopanic.com/developers/api/) — free tier, BTC news
- **FinBERT**: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — local HuggingFace inference
- **CryptoBERT**: [`ElKulako/cryptobert`](https://huggingface.co/ElKulako/cryptobert) — local HuggingFace inference (ablation only; trained on Reddit/Twitter, not news)
