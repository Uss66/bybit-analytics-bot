"""
Fetch Polymarket's implied probability of a 25bps Fed rate cut at each FOMC
meeting, stitched into one continuous series across meetings (Polymarket
doesn't keep one market open for 2 years - each meeting gets its own
short-lived market, grouped by Polymarket into an "event").

Free, no key needed: Gamma API (https://gamma-api.polymarket.com) for market
discovery, CLOB API (https://clob.polymarket.com) for hourly price history.

Usage:
    python scripts/fetch_polymarket_fed.py
"""
import json
import re
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
SERIES_KEY = "fed_cut_25bps"

# Event-title patterns to EXCLUDE: multi-meeting combos, dissent-count
# markets, and anything not a single specific meeting's rate decision.
EXCLUDE_PATTERNS = [
    r"dissent", r"decisions? \(", r"combo", r"three fed decisions",
]

session = requests.Session()


def discover_meeting_events() -> list[dict]:
    seen = {}
    for query in ["Fed decision", "Fed interest rates"]:
        resp = session.get(f"{GAMMA_URL}/public-search", params={"q": query, "limit_per_type": 100}, timeout=20)
        resp.raise_for_status()
        for ev in resp.json().get("events", []):
            title = ev.get("title") or ""
            if not re.search(r"fed decision|fed interest rates", title, re.I):
                continue
            if any(re.search(p, title, re.I) or re.search(p, ev.get("slug", ""), re.I) for p in EXCLUDE_PATTERNS):
                continue
            seen[ev["slug"]] = ev
    return list(seen.values())


def get_cut_market(event_slug: str) -> dict | None:
    resp = session.get(f"{GAMMA_URL}/events", params={"slug": event_slug}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    ev = data[0]
    for m in ev.get("markets", []):
        q = m.get("question") or ""
        if re.search(r"decreases? interest rates by 25 bps", q, re.I) and "50" not in q and "75" not in q:
            return m
    return None


def fetch_price_history(clob_token_id: str, start_ts: int) -> list[tuple]:
    # NOTE: passing both startTs and endTs errors with "interval is too long"
    # beyond a few months - startTs alone returns the market's full lifetime
    # at the requested fidelity, so that's what we use.
    resp = session.get(
        f"{CLOB_URL}/prices-history",
        params={"market": clob_token_id, "startTs": start_ts, "fidelity": 60},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    return [(pt["t"], pt["p"]) for pt in resp.json().get("history", [])]


def upsert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO polymarket_odds (series_key, ts, probability, market_slug, question)
            VALUES %s
            ON CONFLICT (series_key, ts, market_slug) DO UPDATE SET probability = EXCLUDED.probability
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main():
    conn = get_connection()
    events = discover_meeting_events()
    print(f"Found {len(events)} candidate meeting events")

    total = 0
    for ev in sorted(events, key=lambda e: e.get("startDate") or ""):
        slug = ev["slug"]
        market = get_cut_market(slug)
        if market is None:
            print(f"[{slug}] no 25bps-cut market found, skipping")
            continue
        clob_ids_raw = market.get("clobTokenIds")
        if not clob_ids_raw:
            print(f"[{slug}] no clobTokenIds, skipping")
            continue
        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
        yes_token = clob_ids[0]  # outcomes[0] == "Yes" confirmed structurally

        start_raw = market.get("startDate") or market.get("createdAt")
        if not start_raw:
            print(f"[{slug}] no start date on market, skipping")
            continue
        start_ts = int(datetime.fromisoformat(start_raw.replace("Z", "+00:00")).timestamp())
        history = fetch_price_history(yes_token, start_ts)
        if not history:
            print(f"[{slug}] no price history, skipping")
            continue

        rows = [
            (SERIES_KEY, datetime.fromtimestamp(t, tz=timezone.utc), p, slug, market.get("question"))
            for t, p in history
        ]
        n = upsert(conn, rows)
        total += n
        print(f"[{slug}] {market.get('question')}: {n} points")

    conn.close()
    print(f"\nDone. {total} total points written to polymarket_odds (series_key='{SERIES_KEY}').")


if __name__ == "__main__":
    main()
