"""
Fetch aggregate USD-pegged stablecoin market cap from DefiLlama - free, no
key, no rate limit, daily, history back to 2017. Proxy for new liquidity
entering crypto (rising stablecoin supply = dry powder / buying capacity).

Usage:
    python scripts/fetch_defillama_stablecoins.py
"""
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def fetch() -> list[tuple]:
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    rows = []
    for pt in resp.json():
        total = pt.get("totalCirculatingUSD", {}).get("peggedUSD")
        if total is None:
            continue
        ts = datetime.fromtimestamp(int(pt["date"]), tz=timezone.utc)
        rows.append((ts, float(total)))
    return rows


def upsert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO defillama_stablecoins (ts, total_usd) VALUES %s
            ON CONFLICT (ts) DO UPDATE SET total_usd = EXCLUDED.total_usd
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    conn = get_connection()
    rows = fetch()
    n = upsert(conn, rows)
    conn.close()
    print(f"{n} rows written ({rows[0][0].date()} -> {rows[-1][0].date()})")


if __name__ == "__main__":
    main()
