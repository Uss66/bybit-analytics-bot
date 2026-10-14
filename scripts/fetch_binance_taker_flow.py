"""
Hourly spot klines WITH the taker-buy split, from Binance's public data portal
(data.binance.vision) - free, no key, monthly zip files, history back to 2017.

Why Binance and not Bybit: Bybit's kline carries only volume and turnover; the
split of that volume into aggressive buys and aggressive sells (who crossed the
spread) is in Binance's kline as `taker_buy_quote_volume`. Order-flow imbalance
is a different thing from the volume z-score the project already rejected.

Stored in the LOCAL research database only (table taker_flow), never in the
live Supabase: it feeds studies, not the bot.

Binance stamps a candle with its OPEN time. Spot files switched from
milliseconds to MICROseconds at the start of 2025; both are handled. The value
of a candle is therefore known only at open_time + 1 hour.

Downloaded zips are untrusted data: each goes into a temp directory of its own,
is read with zipfile/csv as text, every field is parsed as a number, and
nothing from them is ever executed.

Usage:
    python scripts/fetch_binance_taker_flow.py
    python scripts/fetch_binance_taker_flow.py --symbols BTCUSDT --start 2024-01
"""
import argparse
import csv
import io
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from psycopg2.extras import execute_values

from db import get_connection

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def to_ts(raw: str) -> datetime:
    v = int(raw)
    if v > 10**14:                       # microseconds (spot files from 2025)
        v //= 1000
    return datetime.fromtimestamp(v / 1000, tz=timezone.utc)


def parse_zip(blob: bytes, symbol: str):
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            text = zf.read(name).decode("utf-8", errors="strict")
            for rec in csv.reader(io.StringIO(text)):
                if len(rec) < 11 or not rec[0].strip().isdigit():
                    continue                                  # header line or junk
                rows.append((symbol, to_ts(rec[0]), float(rec[4]), float(rec[7]), float(rec[10]), int(rec[8])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--start", default="2021-07")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"))
    args = ap.parse_args()

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS taker_flow (
                symbol text NOT NULL, ts timestamptz NOT NULL,
                close double precision, quote_volume double precision,
                taker_buy_quote double precision, trades integer,
                PRIMARY KEY (symbol, ts))""")
    conn.commit()

    total = 0
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        n_sym, missing = 0, []
        for ym in months(args.start, args.end):
            with tempfile.TemporaryDirectory(prefix="binance_kl_") as tmp:      # its own empty directory
                r = requests.get(BASE.format(sym=sym, ym=ym), timeout=60)
                if r.status_code == 404:
                    missing.append(ym)
                    continue
                r.raise_for_status()
                (Path(tmp) / f"{sym}-{ym}.zip").write_bytes(r.content)
                rows = parse_zip((Path(tmp) / f"{sym}-{ym}.zip").read_bytes(), sym)
            if rows:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        INSERT INTO taker_flow (symbol, ts, close, quote_volume, taker_buy_quote, trades)
                        VALUES %s ON CONFLICT (symbol, ts) DO UPDATE SET
                          close = EXCLUDED.close, quote_volume = EXCLUDED.quote_volume,
                          taker_buy_quote = EXCLUDED.taker_buy_quote, trades = EXCLUDED.trades""", rows)
                conn.commit()
                n_sym += len(rows)
        total += n_sym
        print(f"{sym}: {n_sym} часовых свечей" + (f"; нет файлов за {', '.join(missing)}" if missing else ""))
    conn.close()
    print(f"Всего записано: {total}")


if __name__ == "__main__":
    main()
