"""
One-time (or re-runnable) migration: copies just enough LIVE state from the
local Postgres into the Supabase instance that GitHub Actions will read from
- see db/init/01_schema_supabase.sql for why this is a deliberate subset,
not a full replica, of the local DB.

Critically copies testnet_state as-is (open positions, entry prices,
cooldowns) so switching infra doesn't lose track of positions already open
- this is NOT a fresh install, it's a handoff.

Lookback windows are sized to what the live rules' classifiers actually
need (fixed thresholds, not recomputed quantiles - verified against
classify_deribit_dvol.py / classify_traditional_markets.py before writing
this), not the full backtest history:
  - ohlcv: 45 days (covers the 720h/30d trend_filter_sma with margin)
  - traditional_markets: 30 days (covers the 7d DXY change calc with margin)
  - deribit_dvol: 10 days (covers the 1d change calc with margin)
  - fear_greed_index: all of it - the whole table is a few thousand rows,
    materially smaller than the free-tier limit even copied in full
  - events: 10 days (covers technical_hype's 4h window with a lot of margin,
    kept a bit longer so classify_events.py's dedup index has something to
    dedup against on the first live run)

Requires SUPABASE_DB_URL in .env (the connection string from the Supabase
project's Settings -> Database page) in addition to the existing local
POSTGRES_* vars that db.py already reads.

Usage:
    python scripts/seed_supabase.py                 # copy live-state subset
    python scripts/seed_supabase.py --schema-only    # just apply the schema, no data
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from db import get_connection  # local DB, unchanged behavior

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "db" / "init" / "01_schema_supabase.sql"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def get_supabase_connection():
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        print("ERROR: SUPABASE_DB_URL not set in .env - copy the connection string from "
              "Supabase's Settings -> Database page first.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(url)


def apply_schema(sb_conn):
    sql = SCHEMA_FILE.read_text()
    with sb_conn.cursor() as cur:
        cur.execute(sql)
    sb_conn.commit()
    print(f"Schema applied from {SCHEMA_FILE.name}")


def copy_table(local_conn, sb_conn, table: str, query: str, params: dict | None = None):
    df = pd.read_sql(query, local_conn, params=params or {})
    if df.empty:
        print(f"  {table}: 0 rows (nothing to copy)")
        return
    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    pk_cols = PRIMARY_KEYS[table]
    update_cols = [c for c in cols if c not in pk_cols]
    conflict_clause = (
        f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        if update_cols else f"ON CONFLICT ({', '.join(pk_cols)}) DO NOTHING"
    )
    # execute_values batches many rows per round trip - a naive executemany()
    # was measured at ~108ms/row over the Supabase session pooler (network
    # round-trip bound, not data-volume bound) - ~13min just for the ohlcv
    # table at that rate. Batched, the whole migration is single-digit seconds.
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s {conflict_clause}"
    df = df.astype(object).where(df.notna(), None)  # pandas NaT/NaN -> SQL NULL (psycopg2 can't serialize NaT)
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    with sb_conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    sb_conn.commit()
    print(f"  {table}: {len(df)} rows copied")


PRIMARY_KEYS = {
    "testnet_state": ["symbol"],
    "ohlcv": ["symbol", "ts"],
    "fear_greed_index": ["ts"],
    "traditional_markets": ["ticker", "ts"],
    "deribit_dvol": ["symbol", "ts"],
    "events": ["id", "ts"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()

    local_conn = get_connection()
    sb_conn = get_supabase_connection()

    apply_schema(sb_conn)
    if args.schema_only:
        print("Schema-only run, skipping data copy.")
        return

    print("Copying live-state subset:")
    copy_table(local_conn, sb_conn, "testnet_state", "SELECT * FROM testnet_state")
    copy_table(local_conn, sb_conn, "ohlcv",
               "SELECT symbol, ts, open, high, low, close, volume, turnover FROM ohlcv "
               "WHERE symbol = ANY(%(symbols)s) AND ts >= now() - interval '45 days' ORDER BY ts",
               {"symbols": SYMBOLS})
    copy_table(local_conn, sb_conn, "fear_greed_index", "SELECT * FROM fear_greed_index ORDER BY ts")
    copy_table(local_conn, sb_conn, "traditional_markets",
               "SELECT * FROM traditional_markets WHERE ts >= now() - interval '30 days' ORDER BY ts")
    copy_table(local_conn, sb_conn, "deribit_dvol",
               "SELECT * FROM deribit_dvol WHERE ts >= now() - interval '10 days' ORDER BY ts")
    copy_table(local_conn, sb_conn, "events",
               "SELECT id, source, source_id, ts, author, title, content, url, category, sentiment, raw "
               "FROM events WHERE ts >= now() - interval '10 days' ORDER BY ts")

    local_conn.close()
    sb_conn.close()
    print("\nDone. testnet_run_log/testnet_trades start empty on Supabase - "
          "that's just the audit trail, not required for correct behavior.")


if __name__ == "__main__":
    main()
