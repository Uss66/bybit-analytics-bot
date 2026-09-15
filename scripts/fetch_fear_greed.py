"""
Fetch the full history of the Crypto Fear & Greed Index (alternative.me) -
free, daily, no API key. One call gets the entire series (limit=0).

Usage:
    python scripts/fetch_fear_greed.py
"""
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

URL = "https://api.alternative.me/fng/?limit=0&format=json"


def main():
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    rows = [
        (datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc), int(item["value"]), item["value_classification"])
        for item in data
    ]
    conn = get_connection()
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO fear_greed_index (ts, value, classification) VALUES %s
            ON CONFLICT (ts) DO UPDATE SET value = EXCLUDED.value, classification = EXCLUDED.classification
            """,
            rows,
        )
    conn.commit()
    conn.close()
    print(f"Upserted {len(rows)} days of Fear & Greed Index.")


if __name__ == "__main__":
    main()
