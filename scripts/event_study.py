"""
Event study: for every classified event, compute the forward return of one
or more symbols at several horizons, then aggregate by category (and by
sentiment sign) to see which event types actually correlate with price
moves - plus a time-of-day/day-of-week seasonality check that's independent
of events entirely.

Methodology notes (read before trusting a number):
  - Entry price = nearest ohlcv close to the event timestamp; exit price =
    nearest ohlcv close to (event timestamp + horizon). With 1h candles the
    "nearest" match is within ~30min, fine for hour-to-day-scale horizons.
  - No look-ahead: category/sentiment are derived from the event's own text,
    known at event time - the forward return is the label we're testing
    against, not an input to it.
  - t-stat here is a plain one-sample mean/(std/sqrt(n)) versus zero, no
    scipy dependency. |t|>~2 is the rough "probably not noise" threshold
    for n>=30; below n=30 treat everything as anecdotal.
  - This is in-sample, exploratory, and dominated by whale_alert_io volume
    (it's ~2/3 of all events) - don't read it as proof, read it as "what's
    worth building a strategy rule around vs what's still just noise."

Usage:
    python scripts/event_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 1,4,24
"""
import argparse

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from db import get_connection

MIN_N_FOR_CONFIDENCE = 30

# Whale-alert categories that name a specific coin in raw.coin - a $50M BTC
# transfer says nothing directly about SOL, so these should only be tested
# against the matching symbol. stablecoin_mint/burn are deliberately excluded
# - a USDC mint/burn is a market-wide liquidity event, not a directional bet
# on one non-stable coin, so it stays unfiltered (applies to every symbol).
COIN_SPECIFIC_CATEGORIES = {
    "exchange_inflow", "exchange_outflow", "wallet_to_wallet",
    "exchange_to_exchange", "dormant_activation",
}


def base_coin(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def filter_for_symbol(events: pd.DataFrame, symbol: str, coin_match: bool = True) -> pd.DataFrame:
    if not coin_match:
        return events
    coin = base_coin(symbol)
    is_coin_specific = events["category"].isin(COIN_SPECIFIC_CATEGORIES)
    event_coin = events["raw"].apply(lambda r: (r or {}).get("coin") if isinstance(r, dict) else None)
    matches = ~is_coin_specific | (event_coin == coin)
    return events[matches]


def load_ohlcv(conn, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT ts, close FROM ohlcv WHERE symbol = %(symbol)s ORDER BY ts",
        conn,
        params={"symbol": symbol},
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_events(conn, min_usd_value: float = 0) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT id, ts, source, category, sentiment, raw
        FROM events
        WHERE category IS NOT NULL AND category NOT IN ('general')
          AND (raw->>'usd_value' IS NULL OR (raw->>'usd_value')::numeric >= %(min_usd)s)
        ORDER BY ts
        """,
        conn,
        params={"min_usd": min_usd_value},
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def compute_forward_returns(events: pd.DataFrame, ohlcv: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    if ohlcv.empty or events.empty:
        return pd.DataFrame()
    entry = pd.merge_asof(events[["id", "ts"]], ohlcv, on="ts", direction="nearest", tolerance=pd.Timedelta("2h"))
    entry = entry.rename(columns={"close": "entry_price"})

    rows = []
    for h in horizons:
        target = events[["id", "ts"]].copy()
        target["target_ts"] = target["ts"] + pd.Timedelta(hours=h)
        exit_ = pd.merge_asof(
            target[["id", "target_ts"]].rename(columns={"target_ts": "ts"}),
            ohlcv,
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )
        merged = entry[["id", "ts", "entry_price"]].merge(
            exit_[["id", "close"]].rename(columns={"close": "exit_price"}), on="id"
        )
        merged["horizon_hours"] = h
        merged["fwd_return"] = (merged["exit_price"] - merged["entry_price"]) / merged["entry_price"]
        rows.append(merged[["id", "ts", "horizon_hours", "fwd_return"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def persist_returns(conn, symbol: str, returns: pd.DataFrame):
    if returns.empty:
        return
    rows = [
        (int(r.id), r.ts.to_pydatetime(), symbol, int(r.horizon_hours), None if pd.isna(r.fwd_return) else float(r.fwd_return))
        for r in returns.itertuples()
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO event_returns (event_id, event_ts, symbol, horizon_hours, fwd_return)
            VALUES %s
            ON CONFLICT (event_id, event_ts, symbol, horizon_hours) DO UPDATE SET fwd_return = EXCLUDED.fwd_return
            """,
            rows,
        )
    conn.commit()


def summarize(events: pd.DataFrame, returns: pd.DataFrame, symbol: str) -> pd.DataFrame:
    merged = returns.merge(events[["id", "category", "sentiment"]], on="id")
    merged["sentiment_sign"] = np.sign(merged["sentiment"]).map({-1.0: "negative", 0.0: "neutral", 1.0: "positive"})

    def agg(group):
        n = group["fwd_return"].count()
        mean = group["fwd_return"].mean()
        std = group["fwd_return"].std()
        t_stat = mean / (std / np.sqrt(n)) if n > 1 and std and not np.isnan(std) else np.nan
        return pd.Series({"n": n, "mean_return": mean, "std": std, "t_stat": t_stat})

    summary = (
        merged.groupby(["category", "sentiment_sign", "horizon_hours"])
        .apply(agg, include_groups=False)
        .reset_index()
    )
    summary["symbol"] = symbol
    summary["confident"] = summary["n"] >= MIN_N_FOR_CONFIDENCE
    return summary.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)


def seasonality(ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = ohlcv.copy()
    df["ret"] = df["close"].pct_change()
    df["hour_utc"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.day_name()
    hourly = df.groupby("hour_utc")["ret"].agg(["mean", "std", "count"]).reset_index()
    hourly["symbol"] = symbol
    hourly["dim"] = "hour_utc"
    hourly = hourly.rename(columns={"hour_utc": "bucket"})
    daily = df.groupby("weekday")["ret"].agg(["mean", "std", "count"]).reset_index()
    daily["symbol"] = symbol
    daily["dim"] = "weekday"
    daily = daily.rename(columns={"weekday": "bucket"})
    return pd.concat([hourly, daily], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--horizons", type=str, default="1,4,24")
    parser.add_argument("--top", type=int, default=15, help="How many top |t-stat| rows to print per symbol")
    parser.add_argument("--min-usd-value", type=float, default=0, help="Drop whale-alert-style events below this USD magnitude (non-whale events unaffected)")
    parser.add_argument("--no-coin-match", action="store_true", help="Disable filtering whale-transfer categories to the symbol's own coin (raw.coin)")
    parser.add_argument("--start-date", type=str, default=None, help="Only use events from this date on (YYYY-MM-DD) - for out-of-sample checks")
    parser.add_argument("--end-date", type=str, default=None, help="Only use events up to this date (YYYY-MM-DD) - for out-of-sample checks")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()
    events = load_events(conn, min_usd_value=args.min_usd_value)
    if args.start_date:
        events = events[events["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        events = events[events["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
    print(f"Events in scope (category != general, usd_value >= {args.min_usd_value}): {len(events)}")

    all_summaries = []
    all_seasonality = []
    for symbol in symbols:
        ohlcv = load_ohlcv(conn, symbol)
        if ohlcv.empty:
            print(f"{symbol}: no ohlcv data, skipping")
            continue
        symbol_events = filter_for_symbol(events, symbol, coin_match=not args.no_coin_match)
        returns = compute_forward_returns(symbol_events, ohlcv, horizons)
        persist_returns(conn, symbol, returns)
        summary = summarize(symbol_events, returns, symbol)
        all_summaries.append(summary)
        all_seasonality.append(seasonality(ohlcv, symbol))

        print(f"\n=== {symbol}: top event correlations by |t-stat| ===")
        top = summary.head(args.top)
        with pd.option_context("display.width", 140, "display.max_columns", 10):
            print(top.to_string(index=False))

    conn.close()

    if all_summaries:
        out = pd.concat(all_summaries, ignore_index=True)
        out.to_csv("event_study_summary.csv", index=False)
        print("\nFull summary written to event_study_summary.csv")
    if all_seasonality:
        out2 = pd.concat(all_seasonality, ignore_index=True)
        out2.to_csv("seasonality_summary.csv", index=False)
        print("Seasonality written to seasonality_summary.csv")


if __name__ == "__main__":
    main()
