"""
Fetch Polymarket's implied probability of a spot ETF being approved, per
coin - unlike the Fed rate-cut series (scripts/fetch_polymarket_fed.py),
this is NOT a recurring cyclical event: each coin gets one long-running
market (or a couple of overlapping ones with different deadlines) that
tracks evolving approval odds until the question resolves.

One canonical market per coin is hand-picked below (the broadest-coverage,
highest-volume one - see README for the discovery process), rather than
auto-discovered, since deadline-bound markets ("approved by July 31") can
silently collapse to 0 after their deadline passes even if the coin's ETF
is approved later - "approved in <year>" markets avoid that trap.

Free, no key needed: Gamma API for market metadata, CLOB API for hourly
price history.

Usage:
    python scripts/fetch_polymarket_etf.py
"""
import json
from datetime import datetime, timezone

import requests
from psycopg2.extras import execute_values

from db import get_connection

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# symbol -> canonical market slug (picked for broadest time coverage / no
# premature deadline-driven collapse to 0)
MARKETS = {
    "BTCUSDT": "bitcoin-etf-approved-by-jan-15",
    "ETHUSDT": "ethereum-etf-approved-by-may-31",
    "SOLUSDT": "solana-etf-approved-in-2025",
    "XRPUSDT": "ripple-etf-approved-in-2025",
    "DOGEUSDT": "doge-etf-approved-in-2025",
    "LTCUSDT": "litecoin-etf-approved-in-2025",
}

session = requests.Session()


def fetch_market(event_slug: str) -> dict | None:
    resp = session.get(f"{GAMMA_URL}/events", params={"slug": event_slug}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    markets = data[0].get("markets", [])
    return markets[0] if markets else None


def fetch_price_history(clob_token_id: str, start_ts: int) -> list[tuple]:
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
    total = 0
    for symbol, slug in MARKETS.items():
        market = fetch_market(slug)
        if market is None:
            print(f"[{symbol}] market {slug!r} not found, skipping")
            continue
        clob_ids_raw = market.get("clobTokenIds")
        if not clob_ids_raw:
            print(f"[{symbol}] no clobTokenIds, skipping")
            continue
        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
        yes_token = clob_ids[0]  # outcomes[0] == "Yes"

        start_raw = market.get("startDate") or market.get("createdAt")
        start_ts = int(datetime.fromisoformat(start_raw.replace("Z", "+00:00")).timestamp())
        history = fetch_price_history(yes_token, start_ts)
        if not history:
            print(f"[{symbol}] no price history, skipping")
            continue

        series_key = f"etf_approval_{symbol}"
        rows = [
            (series_key, datetime.fromtimestamp(t, tz=timezone.utc), p, slug, market.get("question"))
            for t, p in history
        ]
        n = upsert(conn, rows)
        total += n
        print(f"[{symbol}] {market.get('question')}: {n} points")

    conn.close()
    print(f"\nDone. {total} total points written to polymarket_odds.")


if __name__ == "__main__":
    main()
