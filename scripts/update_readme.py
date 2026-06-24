"""
Read results JSON files and patch the README results table in-place.

Usage:
    python scripts/update_readme.py
"""

import json
import re
from pathlib import Path

RESULTS_DIR = Path("results")
README = Path("README.md")

# Which strategies to show, in order
ROWS = [
    ("buy_and_hold",   "Buy & Hold",        "bear_2022"),
    ("a2c_finbert",    "A2C + FinBERT",     "bear_2022"),
    ("sac_finbert",    "SAC + FinBERT",     "bear_2022"),
    ("td3_finbert",    "TD3 + FinBERT",     "bear_2022"),
    ("a2c_cryptobert", "A2C + CryptoBERT",  "bear_2022"),
    ("buy_and_hold",   "Buy & Hold",        "recovery_2023"),
    ("a2c_finbert",    "A2C + FinBERT",     "recovery_2023"),
    ("sac_finbert",    "SAC + FinBERT",     "recovery_2023"),
    ("td3_finbert",    "TD3 + FinBERT",     "recovery_2023"),
]


def load_results() -> dict:
    """Load all *_results.json files into a flat dict keyed by (strategy, period)."""
    data = {}
    for f in RESULTS_DIR.glob("*_results.json"):
        with open(f) as fh:
            r = json.load(fh)
        # e.g. sac_finbert_results.json → strategy = "sac_finbert"
        strategy = f.stem.replace("_results", "")
        # Average metrics across seeds
        for seed_key, periods in r.items():
            for period, agents in periods.items():
                for agent_key, metrics in agents.items():
                    if agent_key in ("buy_and_hold", "random"):
                        key = (agent_key, period)
                    else:
                        key = (strategy, period)
                    if key not in data:
                        data[key] = {k: [] for k in metrics}
                    for k, v in metrics.items():
                        import math
                        if v is not None and not (isinstance(v, float) and math.isnan(v)):
                            data[key][k].append(v)

    # Average across seeds
    averaged = {}
    for key, metric_lists in data.items():
        import numpy as np
        averaged[key] = {k: round(float(np.mean(v)), 4) if v else float("nan")
                         for k, v in metric_lists.items()}
    return averaged


def fmt(val, pct: bool = True) -> str:
    if val != val:  # nan
        return "—"
    suffix = "%" if pct else ""
    return f"{val:+.2f}{suffix}" if pct else f"{val:.3f}"


def build_table(results: dict) -> str:
    header = (
        "| Strategy | Period | Return | Sharpe | Sortino | Max DD |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for strategy_key, label, period in ROWS:
        m = results.get((strategy_key, period))
        if m is None:
            ret = shr = srt = mdd = "TBD"
        else:
            ret = fmt(m.get("cumulative_return", float("nan")), pct=True)
            shr = fmt(m.get("sharpe_ratio", float("nan")), pct=False)
            srt = fmt(m.get("sortino_ratio", float("nan")), pct=False)
            mdd = fmt(m.get("max_drawdown", float("nan")), pct=True)
        rows.append(f"| {label} | {period} | {ret} | {shr} | {srt} | {mdd} |")

    return header + "\n".join(rows)


def patch_readme(table: str) -> None:
    text = README.read_text(encoding="utf-8")
    # Replace everything between the table header and the next --- or blank section
    pattern = (
        r"(\| Strategy \| Period \| Return \| Sharpe \| Sortino \| Max DD \|"
        r".*?)"
        r"(\n\n>|\n\n---)"
    )
    replacement = table + r"\2"
    new_text, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if n == 0:
        print("[update_readme] WARNING: could not find results table in README — printing table only")
        print(table)
        return
    README.write_text(new_text, encoding="utf-8")
    print(f"[update_readme] README updated ({n} replacement)")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("[update_readme] No results found in results/. Run train.py first.")
    else:
        print(f"[update_readme] Found results for {len(results)} strategy/period pairs")
        table = build_table(results)
        print("\n" + table)
        patch_readme(table)
