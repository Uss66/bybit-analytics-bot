"""
Fetch Deribit's DVOL (options-implied volatility index, crypto's equivalent
of the VIX) via CryptoDataDownload's free daily CSV mirror - free, no key,
history back to 2021-03-24, updated daily.

Usage:
    python scripts/fetch_deribit_dvol.py
"""
import csv
import io
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

URL = "https://www.cryptodatadownload.com/cdd/DeriBit_volatility_OHLC_{symbol}.csv"
SYMBOLS = ["BTC", "ETH"]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(symbol: str) -> list[tuple]:
    resp = requests.get(URL.format(symbol=symbol), timeout=30, headers=HEADERS)
    resp.raise_for_status()
    lines = resp.text.splitlines()[1:]  # first line is a stray URL banner, not CSV header
    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        ts = datetime.strptime(row["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows.append((symbol, ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])))
    return rows


def upsert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO deribit_dvol (symbol, ts, open, high, low, close) VALUES %s
            ON CONFLICT (symbol, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    conn = get_connection()
    for symbol in SYMBOLS:
        rows = fetch(symbol)
        n = upsert(conn, rows)
        print(f"[{symbol}] {n} rows ({rows[-1][1].date()} -> {rows[0][1].date()})")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
