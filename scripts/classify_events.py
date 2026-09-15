"""
Classify events (news/RSS/GDELT, later social posts) into a fixed taxonomy of
event types relevant to crypto price movement, and score sentiment.

Deliberately cheap and local: keyword rules for category + VADER for
sentiment, no Claude/LLM API calls per item. This keeps classifying
thousands of historical items free and fast. Ambiguous/low-confidence cases
can be sampled and reviewed manually (or spot-checked with Claude) later -
see README.

Usage:
    python scripts/classify_events.py            # classify only unclassified rows
    python scripts/classify_events.py --recompute # reclassify everything
"""
import argparse
import re

from psycopg2.extras import execute_values
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from db import get_connection

# Ordered (category -> keywords) - first matching category wins.
# Keep categories few and meaningful for an event study, not exhaustive taxonomy.
# Keywords are matched as whole words/phrases (regex word boundaries), so
# short fragments like "bill" or "sue" don't false-positive on "billion" or
# "issue".
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("regulation", [
        "sec", "cftc", "regulat\\w*", "lawsuit", "sues", "sue", "sued", "legal battle",
        "court", "ruling", "license", "compliance", "\\bbill\\b", "congress", "senate",
        "ban", "banned", "illegal", "clarity act", "framework",
    ]),
    ("etf", [
        "etf", "spot etf", "inflow", "outflow", "blackrock", "grayscale", "ishares",
        "fidelity", "vaneck",
    ]),
    ("macro", [
        "federal reserve", "\\bfed\\b", "fomc", "interest rate", "inflation", "cpi",
        "recession", "jobs report", "nasdaq", "s&p", "stocks", "nvidia", "rate cut",
        "rate hike", "treasury yield",
    ]),
    ("exchange_security", [
        "hack\\w*", "exploit", "breach", "stolen", "rug pull", "outage",
        "delisting", "delist", "bankruptcy", "insolvent", "frozen funds", "exit scam",
    ]),
    ("institutional_adoption", [
        "acquir\\w*", "acquisition", "treasury", "adopts", "partnership", "raises \\$",
        "capital raise", "funding round", "invests in", "buys bitcoin", "microstrategy",
    ]),
    ("key_person_statement", [
        "trump", "elon musk", "musk", "powell", "saylor", "zhao", "\\bcz\\b", "said", "says",
        "tweet", "posts on x", "truth social",
    ]),
    ("technical", [
        "halving", "upgrade", "hard fork", "mainnet", "lightning", "protocol update",
    ]),
    ("price_action", [
        "surge", "plunge", "rally", "crash", "steadies", "tests its", "resistance",
        "support level", "all-time high", "record high", "\\bprice\\b",
    ]),
]

_COMPILED_RULES = [
    (category, re.compile(r"\b(?:" + "|".join(keywords) + r")\b", re.IGNORECASE))
    for category, keywords in CATEGORY_RULES
]


def classify_category(text: str) -> str:
    for category, pattern in _COMPILED_RULES:
        if pattern.search(text):
            return category
    return "general"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Reclassify rows that already have a category")
    args = parser.parse_args()

    analyzer = SentimentIntensityAnalyzer()
    conn = get_connection()
    with conn.cursor() as cur:
        if args.recompute:
            cur.execute("SELECT id, ts, title, content FROM events")
        else:
            cur.execute("SELECT id, ts, title, content FROM events WHERE category IS NULL")
        rows = cur.fetchall()

    print(f"Classifying {len(rows)} event(s)...")
    updates = []
    category_counts: dict[str, int] = {}
    for event_id, ts, title, content in rows:
        text = " ".join(filter(None, [title, content]))
        category = classify_category(text)
        sentiment = analyzer.polarity_scores(text)["compound"] if text.strip() else 0.0
        category_counts[category] = category_counts.get(category, 0) + 1
        updates.append((event_id, ts, category, sentiment))

    if updates:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                UPDATE events AS e SET category = data.category, sentiment = data.sentiment
                FROM (VALUES %s) AS data (id, ts, category, sentiment)
                WHERE e.id = data.id AND e.ts = data.ts
                """,
                updates,
                template="(%s, %s, %s, %s::double precision)",
            )
        conn.commit()
    conn.close()

    print("Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    print("Done.")


if __name__ == "__main__":
    main()
