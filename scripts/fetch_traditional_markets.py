"""
Fetch daily closes for traditional markets via Yahoo Finance's free public
chart API (no key) - for correlating crypto moves against broader macro
risk-on/risk-off regimes (equities, gold, dollar strength, bond yields,
volatility).

Usage:
    python scripts/fetch_traditional_markets.py --years 2
"""
import argparse
import time
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

TICKERS = {
    "SPX": "%5EGSPC",      # S&P 500
    "NDX": "%5EIXIC",      # Nasdaq Composite
    "GOLD": "GC=F",        # Gold futures
    "DXY": "DX-Y.NYB",     # US Dollar Index
    "VIX": "%5EVIX",       # CBOE Volatility Index
    "UST10Y": "%5ETNX",    # 10-Year Treasury yield (x10, e.g. 45.0 = 4.50%)
    "OIL": "CL=F",         # WTI Crude Oil futures
    "SILVER": "SI=F",      # Silver futures
    "PLATINUM": "PL=F",    # Platinum futures
    "COPPER": "HG=F",      # Copper futures (industrial/global-growth proxy)
    "NVDA": "NVDA",        # NVIDIA - AI/tech-trade bellwether, shows up constantly alongside crypto in news flow
}
BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(ticker_query: str, range_: str, retries: int = 3):
    """Retries on connection/DNS errors - discovered 2026-09-13 that
    query1.finance.yahoo.com intermittently fails DNS resolution when
    called right after the Bybit OHLCV/FGI/DVOL fetches earlier in the same
    hourly tick (those succeed every time; only this host is flaky), even
    though the same host resolves fine seconds later when tested manually.
    A single retry after a short pause has consistently worked in
    practice. Without this, DXY silently went stale for ~3 days (the
    exception was caught and logged, not fatal, but the dxy_up_extreme/
    dxy_down_extreme rules ran on 3-day-old data with nobody noticing until
    a manual review)."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL.format(ticker=ticker_query), params={"range": range_, "interval": "1d"},
                                 headers=HEADERS, timeout=20)
            resp.raise_for_status()
            result = resp.json()["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            return [(ts, c) for ts, c in zip(timestamps, closes) if c is not None]
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise last_exc


def upsert(conn, ticker: str, rows):
    if not rows:
        return 0
    values = [(ticker, datetime.fromtimestamp(ts, tz=timezone.utc), float(close)) for ts, close in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO traditional_markets (ticker, ts, close) VALUES %s
            ON CONFLICT (ticker, ts) DO UPDATE SET close = EXCLUDED.close
            """,
            values,
        )
    conn.commit()
    return len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=2)
    args = parser.parse_args()
    range_ = f"{args.years}y"

    conn = get_connection()
    for ticker, query in TICKERS.items():
        print(f"[{ticker}] fetching...", end=" ", flush=True)
        rows = fetch(query, range_)
        n = upsert(conn, ticker, rows)
        print(f"{n} rows")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
