"""
Turns extreme weekly moves in the US Dollar Index (DXY) into synthetic rows
in the `events` table (source='macro_market', category='dxy_up_extreme' /
'dxy_down_extreme'), so strategy.py's existing event-scoring mechanism can
pick them up - same pattern as classify_fear_greed.py.

Thresholds (+-0.8%/+0.9% over a trailing 7 days) are the empirical top/bottom
quintile boundaries of DXY_chg_7d over the full 2-year sample (see
traditional_markets_study.py / project_traditional_markets_findings memory) -
fixed here so live classification doesn't depend on recomputing quantiles
over an ever-changing sample.

Usage:
    python scripts/classify_traditional_markets.py
"""
from psycopg2.extras import execute_values

from db import get_connection

UP_THRESHOLD = 0.009    # top quintile boundary
DOWN_THRESHOLD = -0.008  # bottom quintile boundary


def main():
    conn = get_connection()
    dxy = None
    import pandas as pd
    dxy = pd.read_sql("SELECT ts, close FROM traditional_markets WHERE ticker = 'DXY' ORDER BY ts", conn)
    dxy["ts"] = pd.to_datetime(dxy["ts"], utc=True)
    dxy["chg_7d"] = dxy["close"].pct_change(7)

    events = []
    for row in dxy.itertuples():
        if pd.isna(row.chg_7d):
            continue
        if row.chg_7d >= UP_THRESHOLD:
            category = "dxy_up_extreme"
        elif row.chg_7d <= DOWN_THRESHOLD:
            category = "dxy_down_extreme"
        else:
            continue
        events.append(("macro_market", f"dxy_{row.ts.date()}", row.ts, category,
                        f"DXY 7d change = {row.chg_7d:.4f}", 0.0))

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
    print(f"Upserted {len(events)} DXY extreme-move events "
          f"({sum(1 for e in events if e[3]=='dxy_up_extreme')} up, "
          f"{sum(1 for e in events if e[3]=='dxy_down_extreme')} down).")


if __name__ == "__main__":
    main()
