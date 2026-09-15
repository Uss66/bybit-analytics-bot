"""
Fetch historical funding rate and open interest for Bybit perpetual futures
(category=linear) - these are positioning/leverage signals from the
derivatives market, distinct from the spot price data in `ohlcv`. Even
though the trading strategy is spot-only, funding rate spikes and open
interest buildups are a standard leading indicator of spot moves (crowded
positioning tends to unwind violently).

Usage:
    python scripts/fetch_bybit_derivatives.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --years 2
"""
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
from psycopg2.extras import execute_values
from tenacity import retry, stop_after_attempt, wait_exponential

from db import get_connection

BASE_URL = "https://api.bybit.com"
SESSION = requests.Session()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _get(path, params):
    resp = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data}")
    return data["result"]


def paginate(path, base_params, ts_field, start_ms, end_ms, window_days, limit=200):
    """Time-window pagination that doesn't assume a fixed record interval -
    if a window returns a full page (likely more data in that window), retry
    from the last timestamp instead of jumping a full window ahead."""
    items = []
    cur_start = start_ms
    while cur_start < end_ms:
        window_end = min(cur_start + window_days * 86_400_000, end_ms)
        result = _get(path, {**base_params, "startTime": cur_start, "endTime": window_end, "limit": limit})
        batch = result.get("list", [])
        if batch:
            batch_sorted = sorted(batch, key=lambda x: int(x[ts_field]))
            items.extend(batch_sorted)
            last_ts = int(batch_sorted[-1][ts_field])
            cur_start = last_ts + 1 if len(batch) >= limit else window_end
        else:
            cur_start = window_end
        time.sleep(0.05)
    return items


def upsert_funding(conn, symbol, rows):
    if not rows:
        return 0
    values = [(symbol, datetime.fromtimestamp(int(r["fundingRateTimestamp"]) / 1000, tz=timezone.utc), float(r["fundingRate"])) for r in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO funding_rate (symbol, ts, funding_rate) VALUES %s
            ON CONFLICT (symbol, ts) DO UPDATE SET funding_rate = EXCLUDED.funding_rate
            """,
            values,
        )
    conn.commit()
    return len(values)


def upsert_oi(conn, symbol, rows):
    if not rows:
        return 0
    values = [(symbol, datetime.fromtimestamp(int(r["timestamp"]) / 1000, tz=timezone.utc), float(r["openInterest"])) for r in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO open_interest (symbol, ts, open_interest) VALUES %s
            ON CONFLICT (symbol, ts) DO UPDATE SET open_interest = EXCLUDED.open_interest
            """,
            values,
        )
    conn.commit()
    return len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--years", type=float, default=2.0)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(args.years * 365))
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    conn = get_connection()
    for symbol in symbols:
        print(f"[{symbol}] funding rate...", end=" ", flush=True)
        funding = paginate("/v5/market/funding/history", {"category": "linear", "symbol": symbol},
                            "fundingRateTimestamp", start_ms, end_ms, window_days=60)
        n_funding = upsert_funding(conn, symbol, funding)
        print(f"{n_funding} rows")

        print(f"[{symbol}] open interest...", end=" ", flush=True)
        oi = paginate("/v5/market/open-interest", {"category": "linear", "symbol": symbol, "intervalTime": "1h"},
                       "timestamp", start_ms, end_ms, window_days=8)
        n_oi = upsert_oi(conn, symbol, oi)
        print(f"{n_oi} rows")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
