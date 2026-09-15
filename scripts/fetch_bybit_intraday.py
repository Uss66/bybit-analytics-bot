"""
Fetch sub-hourly (e.g. 15-minute) OHLCV candles into `ohlcv_intraday` - a
SEPARATE table from `ohlcv` (see db/init/01_schema.sql comment) so this
doesn't collide with the hourly data the live testnet_trader.py bot
depends on. Built for the Elliott Wave high-frequency re-test (2026-09-14),
see project_elliott_wave_analysis memory.

Reuses fetch_bybit_ohlcv.py's own fetch_klines()/get_top_symbols() (same
verified pagination/retry logic, not reimplemented) - only the upsert
target differs.

Usage:
    python scripts/fetch_bybit_intraday.py --symbols BTCUSDT --interval 15 --years 2
"""
import argparse

from psycopg2.extras import execute_values

from db import get_connection
from fetch_bybit_ohlcv import get_top_symbols, fetch_klines
from datetime import datetime, timedelta, timezone


def upsert_rows(conn, symbol: str, interval_min: int, rows: list[list[str]]):
    if not rows:
        return 0
    values = [
        (symbol, interval_min, datetime.fromtimestamp(int(r[0]) / 1000, tz=timezone.utc),
         float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]))
        for r in rows
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO ohlcv_intraday (symbol, interval_min, ts, open, high, low, close, volume, turnover)
            VALUES %s
            ON CONFLICT (symbol, interval_min, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume, turnover = EXCLUDED.turnover
            """,
            values,
        )
    conn.commit()
    return len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=1, help="Number of top USDT spot pairs by turnover (ignored if --symbols given)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT", help="Comma-separated symbol override")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--interval", type=int, default=15, help="Candle interval in minutes")
    args = parser.parse_args()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(args.years * 365))
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else get_top_symbols(args.top)
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}, interval={args.interval}min, years={args.years}")

    conn = get_connection()
    total = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {symbol} @ {args.interval}min ...", end=" ", flush=True)
        rows = fetch_klines(symbol, args.interval, start_ms, end_ms)
        n = upsert_rows(conn, symbol, args.interval, rows)
        total += n
        print(f"{n} candles")
    conn.close()
    print(f"Done. Total candles upserted: {total}")


if __name__ == "__main__":
    main()
