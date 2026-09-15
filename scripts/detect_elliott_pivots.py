"""
Detect ZigZag pivots (swing highs/lows) from 2-year hourly OHLCV data
and populate the elliott_pivots table for Elliott Wave analysis.

The algorithm:
  1. Resample hourly closes to daily (using the LAST close of each UTC day)
  2. Detect ZigZag pivots: track last confirmed pivot, track current extreme
     since that pivot. When price reverses by threshold_pct from the extreme
     in the opposite direction, confirm the extreme as a pivot and start a
     new extreme from there. Alternates between high and low pivots.
  3. Upsert all pivots (symbol, ts, price, pivot_type, threshold_pct) with
     conflict resolution to handle re-runs.

Usage:
    cd scripts && ../../.venv/Scripts/python.exe detect_elliott_pivots.py

Thresholds tested: [0.05, 0.08, 0.12] (5%, 8%, 12% minimum swing)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from psycopg2.extras import execute_values

from db import get_connection


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
# Expanded 2026-09-12 per user request to explore shorter-term (faster,
# smaller-degree) and longer-term (slower, bigger-degree) wave scales,
# beyond the original 5/8/12% range.
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]


def detect_pivots(daily_closes, threshold_pct):
    """
    Detect ZigZag pivots from a daily close series.

    Args:
        daily_closes: pandas Series with datetime index and close prices (sorted ascending)
        threshold_pct: minimum swing threshold (e.g., 0.05 for 5%)

    Returns:
        List of tuples: (timestamp, price, pivot_type, confirmed_ts) where
        pivot_type is 'high'/'low' and confirmed_ts is the timestamp at
        which the reversal past threshold_pct was actually observed - i.e.
        the earliest point this pivot could have been known in real time,
        as opposed to `timestamp` which is when the extreme itself occurred
        (usually earlier). This distinction is the fix for the repainting/
        look-ahead problem found in the 2026-09-12 Elliott Wave analysis -
        see project_elliott_wave_analysis memory: any signal built on this
        pivot must use confirmed_ts as its earliest possible entry point,
        never the pivot's own timestamp.
    """
    pivots = []

    if len(daily_closes) < 2:
        return pivots

    # Determine initial direction based on first two closes
    if daily_closes.iloc[1] > daily_closes.iloc[0]:
        # Starting from low, next pivot will be a high
        last_pivot_type = 'low'
        current_extreme_type = 'high'
    else:
        # Starting from high, next pivot will be a low
        last_pivot_type = 'high'
        current_extreme_type = 'low'

    # Record the first close as the first pivot (no confirmation lag concept
    # applies to this seed point - it's not a real detected reversal)
    last_pivot_price = daily_closes.iloc[0]
    last_pivot_ts = daily_closes.index[0]
    extreme_price = daily_closes.iloc[1]
    extreme_ts = daily_closes.index[1]

    pivots.append((last_pivot_ts, last_pivot_price, last_pivot_type, last_pivot_ts))

    # Iterate through remaining prices
    for i in range(2, len(daily_closes)):
        price = daily_closes.iloc[i]
        ts = daily_closes.index[i]

        if current_extreme_type == 'high':
            # Tracking a high
            if price > extreme_price:
                # New higher high
                extreme_price = price
                extreme_ts = ts
            elif price < extreme_price * (1 - threshold_pct):
                # Price has reversed down by threshold_pct from the high -
                # `ts` (today) is when this became knowable, NOT extreme_ts.
                pivots.append((extreme_ts, extreme_price, 'high', ts))
                last_pivot_price = extreme_price
                last_pivot_ts = extreme_ts
                last_pivot_type = 'high'
                extreme_price = price
                extreme_ts = ts
                current_extreme_type = 'low'
        else:
            # Tracking a low
            if price < extreme_price:
                # New lower low
                extreme_price = price
                extreme_ts = ts
            elif price > extreme_price * (1 + threshold_pct):
                # Price has reversed up by threshold_pct from the low -
                # `ts` (today) is when this became knowable, NOT extreme_ts.
                pivots.append((extreme_ts, extreme_price, 'low', ts))
                last_pivot_price = extreme_price
                last_pivot_ts = extreme_ts
                last_pivot_type = 'low'
                extreme_price = price
                extreme_ts = ts
                current_extreme_type = 'high'

    return pivots


def resample_to_daily(hourly_df):
    """
    Resample hourly OHLCV data to daily closes.

    Takes the 23:00 UTC close for each day (the last hourly candle of that
    day) - critically, the returned timestamp is the ACTUAL time that price
    occurred (23:00), not midnight.

    BUG FIXED (2026-09-12): the original implementation used
    `resample('D').last()`, which labels each daily bucket with its START
    (midnight) while the value is the LAST close in that bucket (~23:00) -
    a ~23-hour mislabeling between timestamp and value. This corrupted every
    downstream forward-return computation that looked up prices by
    timestamp against the hourly `ohlcv` table (e.g. elliott_wave_study_v2.py):
    querying "the price at this pivot's end_ts" via merge_asof returned the
    ACTUAL midnight price, not the labeled daily-close value, so a
    "forward return over the next 24h" window ended up re-capturing the
    SAME intraday move that defined the pivot's value in the first place -
    inflating t-stats dramatically (one signal showed t=10.6, several times
    stronger than anything else validated in this project, which is exactly
    the kind of result that should trigger suspicion rather than excitement).
    Fixed by selecting the actual 23:00 hourly row directly instead of
    resampling - timestamp and value now genuinely correspond.

    Args:
        hourly_df: DataFrame with columns 'ts' and 'close', sorted by ts

    Returns:
        pandas Series indexed by the actual 23:00 UTC timestamp of each day,
        with that hour's close price
    """
    hourly_df = hourly_df.copy()
    hourly_df['ts'] = pd.to_datetime(hourly_df['ts'], utc=True)
    daily_df = hourly_df[hourly_df['ts'].dt.hour == 23].set_index('ts')
    daily = daily_df['close'].dropna()

    return daily


def main():
    conn = get_connection()

    # Collect all results for summary printing
    summary_rows = []

    # Process each symbol
    for symbol in SYMBOLS:
        print(f"\n--- Processing {symbol} ---")

        # Load hourly OHLCV data
        query = "SELECT ts, close FROM ohlcv WHERE symbol = %s ORDER BY ts"
        hourly_df = pd.read_sql(query, conn, params=(symbol,))

        if hourly_df.empty:
            print(f"  No data found for {symbol}")
            continue

        print(f"  Loaded {len(hourly_df)} hourly candles")

        # Resample to daily closes
        daily_closes = resample_to_daily(hourly_df)
        print(f"  Resampled to {len(daily_closes)} daily candles")
        print(f"  Date range: {daily_closes.index[0].date()} to {daily_closes.index[-1].date()}")

        # Process each threshold
        for threshold_pct in THRESHOLDS:
            print(f"  Threshold {threshold_pct*100:.1f}%: ", end="", flush=True)

            # Detect pivots
            pivots = detect_pivots(daily_closes, threshold_pct)
            print(f"Found {len(pivots)} pivots")

            # Prepare rows for upsert
            if pivots:
                # Calculate days between consecutive pivots (for summary stats)
                pivot_dates = [p[0] for p in pivots]
                if len(pivot_dates) > 1:
                    days_between = [(pivot_dates[i+1] - pivot_dates[i]).days for i in range(len(pivot_dates)-1)]
                    avg_days_between = np.mean(days_between) if days_between else 0
                else:
                    avg_days_between = 0

                summary_rows.append({
                    'symbol': symbol,
                    'threshold_pct': threshold_pct,
                    'num_pivots': len(pivots),
                    'avg_days_between': avg_days_between,
                    'date_range_start': daily_closes.index[0].date(),
                    'date_range_end': daily_closes.index[-1].date(),
                })

                # Prepare values for execute_values
                values = [
                    (symbol, ts, float(price), pivot_type, threshold_pct, confirmed_ts)
                    for ts, price, pivot_type, confirmed_ts in pivots
                ]

                # Upsert into elliott_pivots table
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO elliott_pivots (symbol, ts, price, pivot_type, threshold_pct, confirmed_ts)
                        VALUES %s
                        ON CONFLICT (symbol, ts, threshold_pct) DO UPDATE SET
                            price = EXCLUDED.price,
                            pivot_type = EXCLUDED.pivot_type,
                            confirmed_ts = EXCLUDED.confirmed_ts
                        """,
                        values,
                    )
                conn.commit()
            else:
                summary_rows.append({
                    'symbol': symbol,
                    'threshold_pct': threshold_pct,
                    'num_pivots': 0,
                    'avg_days_between': 0,
                    'date_range_start': daily_closes.index[0].date() if len(daily_closes) > 0 else None,
                    'date_range_end': daily_closes.index[-1].date() if len(daily_closes) > 0 else None,
                })

    conn.close()

    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY: Elliott Wave Pivots Detected")
    print("="*120)
    print(f"{'Symbol':<12} {'Threshold':<12} {'Pivots':<10} {'Avg Days Between':<20} {'Date Range':<40}")
    print("-"*120)

    for row in summary_rows:
        threshold_str = f"{row['threshold_pct']*100:.1f}%"
        avg_days_str = f"{row['avg_days_between']:.1f}" if row['num_pivots'] > 1 else "N/A"
        date_range = f"{row['date_range_start']} to {row['date_range_end']}"
        print(f"{row['symbol']:<12} {threshold_str:<12} {row['num_pivots']:<10} {avg_days_str:<20} {date_range:<40}")

    print("="*120)
    print("Done. All pivots upserted to elliott_pivots table.")


if __name__ == "__main__":
    main()
