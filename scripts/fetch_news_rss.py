"""
Fetch current crypto news RSS feeds and store items into the `events` table.

RSS feeds only expose recent items (no 2-year backfill) - this script is meant
to be run on a schedule going forward. Historical backfill for the 2-year
window relies on GDELT (see fetch_gdelt_events.py).

Usage:
    python scripts/fetch_news_rss.py
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
from psycopg2.extras import execute_values

from db import get_connection
from text_utils import strip_html

FEEDS = {
    "rss:cointelegraph": "https://cointelegraph.com/rss",
    "rss:coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss",
    "rss:decrypt": "https://decrypt.co/feed",
    "rss:theblock": "https://www.theblock.co/rss.xml",
    "rss:cryptoslate": "https://cryptoslate.com/feed/",
}


def parse_entry_ts(entry) -> datetime:
    if getattr(entry, "published", None):
        try:
            return parsedate_to_datetime(entry.published).astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def fetch_feed(source: str, url: str):
    parsed = feedparser.parse(url)
    items = []
    for entry in parsed.entries:
        items.append(
            (
                source,
                entry.get("id") or entry.get("link"),
                parse_entry_ts(entry),
                None,
                entry.get("title"),
                strip_html(entry.get("summary", "")),
                entry.get("link"),
            )
        )
    return items


def upsert_events(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO events (source, source_id, ts, author, title, content, url)
            VALUES %s
            ON CONFLICT (source, source_id, ts) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    conn = get_connection()
    total = 0
    for source, url in FEEDS.items():
        print(f"Fetching {source} ...", end=" ", flush=True)
        try:
            rows = fetch_feed(source, url)
            n = upsert_events(conn, rows)
            total += n
            print(f"{n} items")
        except Exception as e:
            print(f"FAILED: {e}")
    conn.close()
    print(f"Done. Total items upserted (attempted): {total}")


if __name__ == "__main__":
    main()
