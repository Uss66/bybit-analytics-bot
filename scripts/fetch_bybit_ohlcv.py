"""
Fetch historical spot OHLCV candles from Bybit's public v5 API for the
top-N pairs by 24h turnover, and upsert them into the `ohlcv` hypertable.

Usage:
    python scripts/fetch_bybit_ohlcv.py --top 30 --years 2 --interval 60
"""
import argparse
import time
from datetime import datetime, timedelta, timezone

from psycopg2.extras import execute_values

import bybit_public
from db import get_connection

KLINE_LIMIT = 1000
# Pegged/stablecoin-style assets: not useful for an event-dependency price study.
STABLE_EXCLUDE = {"USDCUSDT", "XAUTUSDT", "RLUSDUSDT", "USD1USDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT", "PYUSDUSDT"}

# 2026-09-22: no longer a direct requests.get - bybit_public.get() falls back
# to the pg_net-inside-Postgres transport when Bybit's CloudFront geo-blocks
# the caller, which is exactly what had been silently killing this script on
# every GitHub Actions run since the Supabase migration. See bybit_public.py.
_get = bybit_public.get


def get_top_symbols(top_n: int) -> list[str]:
    result = _get("/v5/market/tickers", {"category": "spot"})
    tickers = [
        t for t in result["list"]
        if t["symbol"].endswith("USDT") and t["symbol"] not in STABLE_EXCLUDE
    ]
    tickers.sort(key=lambda t: float(t.get("turnover24h") or 0), reverse=True)
    return [t["symbol"] for t in tickers[:top_n]]


def fetch_klines(symbol: str, interval_min: int, start_ms: int, end_ms: int):
    rows = []
    cur_start = start_ms
    step_ms = interval_min * 60_000 * KLINE_LIMIT
    while cur_start < end_ms:
        window_end = min(cur_start + step_ms, end_ms)
        result = _get(
            "/v5/market/kline",
            {
                "category": "spot",
                "symbol": symbol,
                "interval": str(interval_min),
                "start": cur_start,
                "end": window_end,
                "limit": KLINE_LIMIT,
            },
        )
        batch = result.get("list", [])
        if batch:
            batch_sorted = sorted(batch, key=lambda x: int(x[0]))
            rows.extend(batch_sorted)
            last_ts = int(batch_sorted[-1][0])
            cur_start = last_ts + interval_min * 60_000
        else:
            cur_start = window_end
        print(".", end="", flush=True)
        time.sleep(0.05)
    return rows


def upsert_rows(conn, symbol: str, rows: list[list[str]]):
    if not rows:
        return 0
    values = [
        (
            symbol,
            datetime.fromtimestamp(int(r[0]) / 1000, tz=timezone.utc),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            float(r[4]),
            float(r[5]),
            float(r[6]),
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO ohlcv (symbol, ts, open, high, low, close, volume, turnover)
            VALUES %s
            ON CONFLICT (symbol, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume, turnover = EXCLUDED.turnover
            """,
            values,
        )
    conn.commit()
    return len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30, help="Number of top USDT spot pairs by turnover")
    parser.add_argument("--years", type=float, default=2.0, help="How many years of history to fetch")
    parser.add_argument("--interval", type=int, default=60, help="Candle interval in minutes (Bybit accepts 1,3,5,15,30,60,120,240,360,720,D,W,M)")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbol override, skips top-N lookup")
    args = parser.parse_args()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(args.years * 365))
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        print(f"Looking up top {args.top} USDT spot pairs by 24h turnover...")
        symbols = get_top_symbols(args.top)
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")

    conn = get_connection()
    total = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {symbol} ...", end=" ", flush=True)
        rows = fetch_klines(symbol, args.interval, start_ms, end_ms)
        n = upsert_rows(conn, symbol, rows)
        total += n
        print(f"{n} candles")
    conn.close()
    print(f"Done. Total candles upserted: {total}")


if __name__ == "__main__":
    main()
