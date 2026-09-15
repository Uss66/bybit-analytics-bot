"""
Turns daily Fear & Greed Index readings into synthetic rows in the `events`
table (source='fgi'), one per day the index is in a given zone, so
strategy.py's existing event-based scoring mechanism (built for
news/whale-alert events) can pick them up without any special-casing.

Only 'fgi_greed' (value 55-80) is validated - see memory / README: it's the
one signal that survived a train/holdout split on both 24h and 72h horizons,
across BTC/ETH/SOL, with consistent magnitude. extreme_fear/extreme_greed/
neutral/fear zones are inserted too (for completeness and future study) but
are NOT wired into strategy.py's RULES - don't add them there without
re-running the same train/holdout check first.

Usage:
    python scripts/classify_fear_greed.py
"""
from psycopg2.extras import execute_values

from db import get_connection


def zone_for(value: int) -> str:
    if value <= 20:
        return "fgi_extreme_fear"
    if value <= 45:
        return "fgi_fear"
    if value <= 55:
        return "fgi_neutral"
    if value <= 80:
        return "fgi_greed"
    return "fgi_extreme_greed"


def main():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT ts, value FROM fear_greed_index ORDER BY ts")
        rows = cur.fetchall()

    events = [
        ("fgi", str(ts.date()), ts, zone_for(value), f"Fear & Greed Index = {value}", 0.0)
        for ts, value in rows
    ]
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
    print(f"Upserted {len(events)} Fear & Greed zone events.")


if __name__ == "__main__":
    main()
