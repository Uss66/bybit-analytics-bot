"""
Refreshes only the data sources that feed strategy.py's currently ACTIVE
rules (RULES + AVOID_RULES), then re-runs their classifiers - the minimum
needed to keep testnet_trader.py's live signal current. Intentionally does
NOT touch Telegram/whale-alert scraping (fragile, rate-limit-prone, and no
active rule currently depends on it - see PARKED_HIGH_DUTY_CYCLE_RULES /
REJECTED_RULES in strategy.py) or GDELT (stopped per user request).

Active-rule data dependencies:
  fgi_greed              -> fear_greed_index      -> classify_fear_greed.py
  dxy_up/down_extreme    -> traditional_markets    -> classify_traditional_markets.py
  dvol_drop/spike_extreme -> deribit_dvol           -> classify_deribit_dvol.py
  technical_hype         -> events (RSS-derived)   -> classify_events.py

Usage (intended to run hourly, right before testnet_trader.py):
    python scripts/refresh_live_data.py
"""
import subprocess
import sys

STEPS = [
    ["fetch_bybit_ohlcv.py", "--top", "30", "--years", "0.05"],
    ["fetch_fear_greed.py"],
    ["classify_fear_greed.py"],
    ["fetch_traditional_markets.py", "--years", "1"],
    ["classify_traditional_markets.py"],
    ["fetch_deribit_dvol.py"],
    ["classify_deribit_dvol.py"],
    ["fetch_news_rss.py"],
    ["classify_events.py"],
]


def main():
    for step in STEPS:
        print(f"--- {' '.join(step)} ---")
        result = subprocess.run([sys.executable, *step], cwd=".", capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: {step[0]} exited with code {result.returncode} - continuing anyway "
                  f"(a stale source shouldn't block the whole tick, but check it)", file=sys.stderr)
    print("\nData refresh done.")


if __name__ == "__main__":
    main()
