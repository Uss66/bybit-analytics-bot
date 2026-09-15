"""
Backfill historical crypto-related news events from the free GDELT DOC 2.0 API
and store them into the `events` table (source='gdelt').

GDELT's DOC API caps each response at 250 records, so we slice the requested
date range into daily windows to get full coverage.

NOTE: GDELT's DOC API is a small, slow, informally rate-limited server -
single requests routinely take 20-35s, and hammering it (parallel requests,
retries in quick succession) gets you HTTP 429. This is normal, not an
outage: pace requests generously (REQUEST_SLEEP below) and let 429s back off
hard (RATE_LIMIT_COOLDOWN) rather than retrying quickly.

Usage:
    python scripts/fetch_gdelt_events.py --query "bitcoin OR crypto" --days 730
"""
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
from psycopg2.extras import execute_values
from tenacity import retry, stop_after_attempt, wait_exponential

from db import get_connection

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SESSION = requests.Session()
REQUEST_SLEEP = 3.0
RATE_LIMIT_COOLDOWN = 60.0


class RateLimited(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=RATE_LIMIT_COOLDOWN),
)
def fetch_day(query: str, day_start: datetime, day_end: datetime):
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 250,
        "format": "json",
        "startdatetime": day_start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": day_end.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    # GDELT's DOC API is slow (30s+ per request is normal), not down - give it room.
    resp = SESSION.get(BASE_URL, params=params, timeout=90)
    if resp.status_code == 429:
        time.sleep(RATE_LIMIT_COOLDOWN)
        raise RateLimited("GDELT rate-limited us (429), cooled down and retrying")
    resp.raise_for_status()
    if not resp.text.strip():
        return []
    data = resp.json()
    return data.get("articles", [])


def upsert_events(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO events (source, source_id, ts, title, url, content, raw)
            VALUES %s
            ON CONFLICT (source, source_id, ts) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def get_completed_days(conn, source: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT period_key FROM fetch_log WHERE source = %s AND status = 'success'", (source,))
        return {row[0] for row in cur.fetchall()}


def log_day_result(conn, source: str, period_key: str, status: str, item_count: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fetch_log (source, period_key, status, item_count, fetched_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (source, period_key) DO UPDATE SET
                status = EXCLUDED.status, item_count = EXCLUDED.item_count, fetched_at = now()
            """,
            (source, period_key, status, item_count),
        )
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="bitcoin OR crypto OR cryptocurrency")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--force", action="store_true", help="Refetch days already marked successful in fetch_log")
    args = parser.parse_args()
    source = "gdelt"

    end_dt = datetime.now(timezone.utc)
    conn = get_connection()

    completed = set() if args.force else get_completed_days(conn, source)
    if completed:
        print(f"Resuming: {len(completed)} day(s) already fetched successfully, skipping those.")

    total = 0
    skipped = 0
    for i in range(args.days):
        day_end = end_dt - timedelta(days=i)
        day_start = day_end - timedelta(days=1)
        period_key = day_start.date().isoformat()
        if period_key in completed:
            skipped += 1
            continue
        try:
            articles = fetch_day(args.query, day_start, day_end)
        except Exception as e:
            print(f"{period_key}: FAILED ({e})", flush=True)
            log_day_result(conn, source, period_key, "failed", 0)
            time.sleep(REQUEST_SLEEP)
            continue
        rows = []
        for a in articles:
            url = a.get("url")
            seendate = a.get("seendate")
            try:
                ts = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                ts = day_end
            rows.append(("gdelt", url, ts, a.get("title"), url, a.get("domain"), None))
        n = upsert_events(conn, rows)
        log_day_result(conn, source, period_key, "success", n)
        total += n
        print(f"{period_key}: {n} articles", flush=True)
        time.sleep(REQUEST_SLEEP)
    conn.close()
    print(f"Done. Skipped (already done): {skipped}. Total articles upserted this run: {total}")
    print("Re-run the same command any time to resume - it will retry failed/unattempted days and skip successful ones.")


if __name__ == "__main__":
    main()
