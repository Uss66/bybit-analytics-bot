"""
Health check for the live paper-trading pipeline (2026-09-14). Run this
any time to get a quick PASS/FAIL read on whether the hourly bot is
actually alive and behaving sanely - answers "how do I know it's still
working" without having to remember which tables/logs to check by hand.

Usage:
    python scripts/health_check.py
Exit code 0 if all checks pass, 1 if any fail (so it can also be used in
a script/CI-style context, e.g. a daily check that alerts on failure).
"""
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

from db import get_connection

MAX_TICK_AGE_MINUTES = 90          # hourly cadence + generous buffer
# fear_greed_index/deribit_dvol/traditional_markets all update once per DAY
# (one row per UTC day), not hourly - only `ohlcv` is genuinely hourly.
# Verified 2026-09-14 by inspecting each table directly (deribit_dvol's
# rows are all at 00:00:00 UTC) rather than assuming - a wrong assumption
# here produced 2 false "FAIL"s on the first run of this script.
MAX_DATA_AGE_HOURS = {"ohlcv": 3, "fear_greed_index": 30, "deribit_dvol": 30, "traditional_markets": 30}
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    all_ok = True
    conn = get_connection()
    now = pd.read_sql("SELECT now() as n", conn).iloc[0]["n"]

    # 1. Is the bot ticking on schedule?
    last_tick = pd.read_sql("SELECT MAX(ts) as ts FROM testnet_run_log", conn).iloc[0]["ts"]
    age_min = (now - last_tick).total_seconds() / 60 if last_tick is not None else None
    all_ok &= check("Hourly tick is recent", age_min is not None and age_min <= MAX_TICK_AGE_MINUTES,
                     f"last tick {age_min:.0f} min ago (limit {MAX_TICK_AGE_MINUTES})" if age_min is not None else "no ticks found")

    # 2. Any errors in the last 24h?
    recent_errors = pd.read_sql(
        "SELECT ts, symbol, notes FROM testnet_run_log WHERE action_taken='error' AND ts > now() - interval '24 hours' ORDER BY ts DESC", conn)
    all_ok &= check("No errors in the last 24h", recent_errors.empty,
                     f"{len(recent_errors)} error(s), most recent: {recent_errors.iloc[0]['notes'] if not recent_errors.empty else ''}")

    # 3. Every symbol ticked in the last run batch (no symbol silently dropped)
    last_batch_ts = last_tick
    if last_batch_ts is not None:
        window_start = last_batch_ts - timedelta(minutes=5)
        ticked = pd.read_sql("SELECT DISTINCT symbol FROM testnet_run_log WHERE ts BETWEEN %(a)s AND %(b)s",
                              conn, params={"a": window_start, "b": last_batch_ts + timedelta(seconds=1)})
        missing = set(SYMBOLS) - set(ticked["symbol"])
        all_ok &= check("All 7 symbols present in the last tick batch", not missing, f"missing: {missing}" if missing else "")

    # 4. testnet_state sanity - no impossible half-open positions
    state = pd.read_sql("SELECT * FROM testnet_state", conn)
    bad_state = state[state["in_position"] & (state["entry_price"].isna() | state["entry_qty"].isna())]
    all_ok &= check("No corrupted position state (in_position=True but missing price/qty)",
                     bad_state.empty, f"symbols: {list(bad_state['symbol'])}" if not bad_state.empty else "")

    # 5. Data freshness for sources the active rules depend on
    for table, max_age_h in MAX_DATA_AGE_HOURS.items():
        r = pd.read_sql(f"SELECT MAX(ts) as ts FROM {table}", conn)
        last_ts = r.iloc[0]["ts"]
        age_h = (now - last_ts).total_seconds() / 3600 if last_ts is not None else None
        all_ok &= check(f"{table} data is fresh", age_h is not None and age_h <= max_age_h,
                         f"last update {age_h:.1f}h ago (limit {max_age_h}h)" if age_h is not None else "no data")

    # 6. Task Scheduler cadence sanity - compare against actual pipeline uptime,
    # not a fixed window (a system younger than the window would always "fail"
    # a hardcoded-48h check - caught on this script's own first run).
    first_tick = pd.read_sql("SELECT MIN(ts) as ts FROM testnet_run_log", conn).iloc[0]["ts"]
    uptime_hours = (now - first_tick).total_seconds() / 3600 if first_tick is not None else 0
    lookback_hours = min(48, max(1, uptime_hours))
    recent_ticks = pd.read_sql(
        "SELECT DISTINCT ts FROM testnet_run_log WHERE ts > now() - make_interval(hours => %(h)s)",
        conn, params={"h": lookback_hours})
    n_expected = int(lookback_hours)
    n_actual_hours = recent_ticks["ts"].dt.floor("h").nunique() if not recent_ticks.empty else 0
    all_ok &= check(f"Roughly hourly cadence over the last {lookback_hours:.0f}h (pipeline uptime-aware)",
                     n_actual_hours >= n_expected * 0.8,
                     f"{n_actual_hours} distinct hours ticked / ~{n_expected} expected")

    conn.close()
    print()
    print("=== OVERALL:", "HEALTHY" if all_ok else "NEEDS ATTENTION", "===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
