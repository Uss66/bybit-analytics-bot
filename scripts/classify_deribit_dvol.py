"""
Turns extreme 1-day moves in Deribit's DVOL (BTC options-implied volatility,
crypto's VIX equivalent) into synthetic rows in the `events` table
(source='dvol', category='dvol_spike_extreme' / 'dvol_drop_extreme') - same
pattern as classify_traditional_markets.py.

Thresholds (+-2.35%/-2.61% 1-day change) are the empirical top/bottom
quintile boundaries of DVOL_chg_1d over the OHLCV-overlapping history
(2024-08-27 onward) - fixed here so live classification doesn't depend on
recomputing quantiles over an ever-changing sample.

Usage:
    python scripts/classify_deribit_dvol.py
"""
import pandas as pd
from psycopg2.extras import execute_values

from db import get_connection

UP_THRESHOLD = 0.0235    # top quintile boundary - DVOL spiking (fear rising)
DOWN_THRESHOLD = -0.0261  # bottom quintile boundary - DVOL dropping (calm returning)


def main():
    conn = get_connection()
    dvol = pd.read_sql("SELECT ts, close FROM deribit_dvol WHERE symbol = 'BTC' ORDER BY ts", conn)
    dvol["ts"] = pd.to_datetime(dvol["ts"], utc=True)
    dvol["chg_1d"] = dvol["close"].pct_change(1)

    events = []
    for row in dvol.itertuples():
        if pd.isna(row.chg_1d):
            continue
        if row.chg_1d >= UP_THRESHOLD:
            category = "dvol_spike_extreme"
        elif row.chg_1d <= DOWN_THRESHOLD:
            category = "dvol_drop_extreme"
        else:
            continue
        events.append(("dvol", f"dvol_{row.ts.date()}", row.ts, category,
                        f"DVOL 1d change = {row.chg_1d:.4f}", 0.0))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO events (source, source_id, ts, category, content, sentiment)
            VALUES %s
            ON CONFLICT (source, source_id, ts) DO UPDATE SET category = EXCLUDED.category, content = EXCLUDED.content
            """,
            events,
        )
    conn.commit()
    conn.close()
    print(f"Upserted {len(events)} DVOL extreme-move events "
          f"({sum(1 for e in events if e[3]=='dvol_spike_extreme')} spike, "
          f"{sum(1 for e in events if e[3]=='dvol_drop_extreme')} drop).")


if __name__ == "__main__":
    main()
