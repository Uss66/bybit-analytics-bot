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

CRITICAL vs best-effort (2026-09-22): a stale signal source degrades the
score a little; a stale PRICE breaks the bot outright - the -8% stop-loss
and the trend filter are both comparisons against the latest close. That
distinction was not made here originally, and it cost six days: the OHLCV
step failed on every GitHub Actions run from the Supabase migration
(2026-09-15) onward, this script printed "WARNING ... continuing anyway",
and the tick kept evaluating stop-losses against a price from days ago
while every log looked healthy. Now a failed CRITICAL step makes this
script exit non-zero so the run goes red; testnet_trader.py separately
refuses to trade on a stale feed and alerts (MAX_PRICE_AGE_MINUTES).

Usage (runs every 15 minutes in CI, right before testnet_trader.py):
    python scripts/refresh_live_data.py
"""
import subprocess
import sys

# The symbols the live bot actually trades. Deliberately NOT "--top 30"
# any more (2026-09-22): the top-N lookup costs an extra /v5/market/tickers
# call returning every spot pair, and 23 of those 30 symbols are dead
# weight the live tick never reads. Keep in sync with testnet_trader.py's
# --symbols default.
LIVE_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT"

# (script + args, critical?) - critical steps fail the whole run.
STEPS = [
    (["fetch_bybit_ohlcv.py", "--symbols", LIVE_SYMBOLS, "--years", "0.05"], True),
    (["fetch_fear_greed.py"], False),
    (["classify_fear_greed.py"], False),
    (["fetch_traditional_markets.py", "--years", "1"], False),
    (["classify_traditional_markets.py"], False),
    (["fetch_deribit_dvol.py"], False),
    (["classify_deribit_dvol.py"], False),
    (["fetch_news_rss.py"], False),
    (["classify_events.py"], False),
]


def main():
    failed_critical = []
    for step, critical in STEPS:
        print(f"--- {' '.join(step)} ---")
        result = subprocess.run([sys.executable, *step], cwd=".", capture_output=False)
        if result.returncode == 0:
            continue
        if critical:
            failed_critical.append(step[0])
            print(f"ERROR: {step[0]} exited with code {result.returncode} - this is a CRITICAL "
                  f"source, the tick cannot trade safely without it", file=sys.stderr)
        else:
            print(f"WARNING: {step[0]} exited with code {result.returncode} - continuing anyway "
                  f"(a stale signal source shouldn't block the whole tick, but check it)", file=sys.stderr)

    if failed_critical:
        print(f"\nData refresh FAILED on critical source(s): {', '.join(failed_critical)}", file=sys.stderr)
        sys.exit(1)
    print("\nData refresh done.")


if __name__ == "__main__":
    main()
