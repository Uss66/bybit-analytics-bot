"""
Structured parser for Whale Alert Telegram messages (source='telegram:whale_alert_io').
The generic keyword classifier (classify_events.py) can't make sense of these -
they're template-generated alerts, not prose - so this pulls out coin, USD
value, and transfer direction, and overrides category/sentiment with a
much more informative label than the generic "general" bucket.

Direction heuristic: an endpoint is "known" if it's a named/tagged entity
(exchange, treasury, custodian, etc.) rather than literal "unknown wallet".
    known -> unknown   = exchange_outflow  (leaving custody, bullish lean)
    unknown -> known    = exchange_inflow   (entering custody, bearish lean)
    known -> known       = exchange_to_exchange (neutral)
    unknown -> unknown   = wallet_to_wallet (neutral, whale repositioning)
    minted                = stablecoin_mint (bullish lean - new liquidity)
    burned                = stablecoin_burn (bearish lean)
    dormant address woke  = dormant_activation (ambiguous, no lean)
This is a heuristic, not certainty - "known" entities aren't always classic
exchanges (e.g. "Tether Treasury"), but it's a standard proxy whale-watchers
use for custodial vs self-custody movement.

Usage:
    python scripts/classify_whale_alerts.py            # only unparsed rows
    python scripts/classify_whale_alerts.py --recompute # reparse everything
"""
import argparse
import json
import re

from psycopg2.extras import execute_values

from db import get_connection

SOURCE = "telegram:whale_alert_io"

_TRANSFER_RE = re.compile(
    r"[\d,]+\s*[#$](\w+)\s*\(([\d,]+)\s*USD\)\s*transferred from\s+(.+?)\s+to\s+(.+?)(?:\s+Details)?$",
    re.IGNORECASE,
)
_MINT_BURN_RE = re.compile(
    r"[\d,]+\s*[#$](\w+)\s*\(([\d,]+)\s*USD\)\s*(minted|burned)\s+at\s+(.+?)(?:\s+Details)?$",
    re.IGNORECASE,
)
_DORMANT_RE = re.compile(
    r"dormant address containing\s+[\d,]+\s*[#$]?(\w+)\s*\(([\d,]+)\s*USD\)\s*has just been activated",
    re.IGNORECASE,
)


def _is_known_entity(text: str) -> bool:
    return text.strip().lower() != "unknown wallet"


def parse_message(text: str):
    """Returns (category, sentiment, extra_dict) or None if unparseable."""
    text = text.strip()

    m = _DORMANT_RE.search(text)
    if m:
        coin, usd = m.group(1), int(m.group(2).replace(",", ""))
        return "dormant_activation", 0.0, {"coin": coin, "usd_value": usd}

    m = _MINT_BURN_RE.search(text)
    if m:
        coin, usd, action, entity = m.group(1), int(m.group(2).replace(",", "")), m.group(3).lower(), m.group(4).strip()
        category = "stablecoin_mint" if action == "minted" else "stablecoin_burn"
        sentiment = 1.0 if action == "minted" else -1.0
        return category, sentiment, {"coin": coin, "usd_value": usd, "entity": entity}

    m = _TRANSFER_RE.search(text)
    if m:
        coin, usd, frm, to = m.group(1), int(m.group(2).replace(",", "")), m.group(3).strip(), m.group(4).strip()
        frm_known, to_known = _is_known_entity(frm), _is_known_entity(to)
        if frm_known and not to_known:
            category, sentiment = "exchange_outflow", 1.0
        elif not frm_known and to_known:
            category, sentiment = "exchange_inflow", -1.0
        elif frm_known and to_known:
            category, sentiment = "exchange_to_exchange", 0.0
        else:
            category, sentiment = "wallet_to_wallet", 0.0
        return category, sentiment, {"coin": coin, "usd_value": usd, "from": frm, "to": to}

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    conn = get_connection()
    with conn.cursor() as cur:
        if args.recompute:
            cur.execute("SELECT id, ts, content FROM events WHERE source = %s", (SOURCE,))
        else:
            cur.execute("SELECT id, ts, content FROM events WHERE source = %s AND raw IS NULL", (SOURCE,))
        rows = cur.fetchall()

    print(f"Parsing {len(rows)} whale alert message(s)...")
    updates = []
    category_counts: dict[str, int] = {}
    unparsed = 0
    for event_id, ts, content in rows:
        parsed = parse_message(content or "")
        if parsed is None:
            unparsed += 1
            continue
        category, sentiment, extra = parsed
        category_counts[category] = category_counts.get(category, 0) + 1
        updates.append((event_id, ts, category, sentiment, json.dumps(extra)))

    if updates:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                UPDATE events AS e SET category = data.category, sentiment = data.sentiment, raw = data.raw
                FROM (VALUES %s) AS data (id, ts, category, sentiment, raw)
                WHERE e.id = data.id AND e.ts = data.ts
                """,
                updates,
                template="(%s, %s, %s, %s::double precision, %s::jsonb)",
            )
        conn.commit()
    conn.close()

    print("Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    print(f"  (unparsed, left as-is): {unparsed}")
    print("Done.")


if __name__ == "__main__":
    main()
