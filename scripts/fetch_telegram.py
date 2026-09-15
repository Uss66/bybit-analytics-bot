"""
Backfill crypto-related Telegram channels via their public web preview
(t.me/s/<channel>) - no login, no bot token, no API id/hash needed since we
only read public channels.

The preview paginates backwards through history via ?before=<message_id>.
Progress per channel is saved to `telegram_cursor` so an interrupted/rerun
backfill continues deeper into history instead of restarting from the
latest message.

Usage:
    python scripts/fetch_telegram.py --days 730
    python scripts/fetch_telegram.py --channels whale_alert_io --days 90
"""
import argparse
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from psycopg2.extras import execute_values
from tenacity import retry, stop_after_attempt, wait_exponential

from db import get_connection
from text_utils import strip_html

DEFAULT_CHANNELS = [
    "whale_alert_io", "wublockchainenglish", "WatcherGuru",
    # geopolitics/regulation/tariffs, added 2026-08-28 at user request
    "walterbloomberg", "BNONews", "KobeissiLetter", "tass_agency",
]
BASE_URL = "https://t.me/s"
REQUEST_SLEEP = 1.5
SESSION = requests.Session()

_MESSAGE_SPLIT_RE = re.compile(r'(?=<div class="tgme_widget_message(?:\s|")[^>]*data-post=)')
_POST_ID_RE = re.compile(r'data-post="[^/"]+/(\d+)"')
_TEXT_RE = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.DOTALL)
_TIME_RE = re.compile(r'<time datetime="([^"]+)"')
_PREV_LINK_RE = re.compile(r'rel="prev" href="[^"]*\?before=(\d+)"')


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_page(channel: str, before: int | None):
    url = f"{BASE_URL}/{channel}"
    params = {"before": before} if before else {}
    resp = SESSION.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_page(html: str):
    messages = []
    for chunk in _MESSAGE_SPLIT_RE.split(html)[1:]:
        post_match = _POST_ID_RE.search(chunk)
        if not post_match:
            continue
        post_id = int(post_match.group(1))
        text_match = _TEXT_RE.search(chunk)
        text = strip_html(text_match.group(1)) if text_match else ""
        time_match = _TIME_RE.search(chunk)
        if not time_match:
            continue
        ts = datetime.fromisoformat(time_match.group(1))
        messages.append((post_id, text, ts))
    next_before = _PREV_LINK_RE.search(html)
    return messages, (int(next_before.group(1)) if next_before else None)


def upsert_events(conn, channel: str, messages):
    if not messages:
        return 0
    rows = [
        (
            f"telegram:{channel}",
            str(post_id),
            ts,
            channel,
            None,
            text,
            f"https://t.me/{channel}/{post_id}",
        )
        for post_id, text, ts in messages
    ]
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


def load_cursor(conn, channel: str):
    with conn.cursor() as cur:
        cur.execute("SELECT before_cursor FROM telegram_cursor WHERE channel = %s", (channel,))
        row = cur.fetchone()
        return row[0] if row else None


def save_cursor(conn, channel: str, before_cursor):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO telegram_cursor (channel, before_cursor, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (channel) DO UPDATE SET before_cursor = EXCLUDED.before_cursor, updated_at = now()
            """,
            (channel, before_cursor),
        )
    conn.commit()


def backfill_channel(conn, channel: str, cutoff: datetime, max_pages: int):
    before = load_cursor(conn, channel)
    resumed = before is not None
    print(f"{channel}: {'resuming from before=' + str(before) if resumed else 'starting from latest'}", flush=True)

    total = 0
    for page_num in range(max_pages):
        html = fetch_page(channel, before)
        messages, next_before = parse_page(html)
        if not messages:
            print(f"{channel}: no more messages, reached channel start.", flush=True)
            break

        n = upsert_events(conn, channel, messages)
        total += n
        oldest_ts = min(ts for _, _, ts in messages)
        before = next_before if next_before is not None else min(m[0] for m in messages) - 1
        save_cursor(conn, channel, before)
        print(f"{channel}: page {page_num + 1} -> {len(messages)} msgs, oldest={oldest_ts.date()}", flush=True)

        if oldest_ts < cutoff:
            print(f"{channel}: reached cutoff date.", flush=True)
            break
        if next_before is None:
            print(f"{channel}: no earlier page link, reached channel start.", flush=True)
            break
        time.sleep(REQUEST_SLEEP)
    else:
        print(f"{channel}: hit --max-pages limit ({max_pages}) for this run, rerun to continue deeper.", flush=True)

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=str, default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--max-pages", type=int, default=1000, help="Safety cap on pages fetched per channel per run")
    args = parser.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    conn = get_connection()
    grand_total = 0
    for channel in channels:
        try:
            grand_total += backfill_channel(conn, channel, cutoff, args.max_pages)
        except Exception as e:
            print(f"{channel}: FAILED ({e})", flush=True)
    conn.close()
    print(f"Done. Total messages upserted this run: {grand_total}")


if __name__ == "__main__":
    main()
