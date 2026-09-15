"""
Quantile study: does funding rate / open-interest level or change predict
forward spot returns? Unlike event_study.py (discrete categorical events),
these are continuous signals, so we bucket them into quintiles and compare
forward returns across buckets - same t-stat approach, same "check on a
holdout split before trusting it" discipline as strategy.py, since round 2
of the news-event research showed full-period-only results are not enough.

Usage:
    python scripts/funding_oi_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 24,168
    python scripts/funding_oi_study.py --symbols BTCUSDT --end-date 2025-11-30   # train
    python scripts/funding_oi_study.py --symbols BTCUSDT --start-date 2025-12-01 # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection


def load_series(conn, symbol: str) -> pd.DataFrame:
    ohlcv = pd.read_sql(
        "SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol}
    )
    funding = pd.read_sql(
        "SELECT ts, funding_rate FROM funding_rate WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol}
    )
    oi = pd.read_sql(
        "SELECT ts, open_interest FROM open_interest WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol}
    )
    for df in (ohlcv, funding, oi):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df = pd.merge_asof(ohlcv, funding, on="ts", direction="backward")
    df = pd.merge_asof(df, oi, on="ts", direction="backward")
    df["funding_rate_24h_avg"] = df["funding_rate"].rolling(24, min_periods=8).mean()
    df["oi_change_24h"] = df["open_interest"].pct_change(24)
    df["oi_change_168h"] = df["open_interest"].pct_change(168)
    df["price_change_24h"] = df["close"].pct_change(24)
    return df


def price_oi_divergence_study(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Classic technical-analysis interaction signal: price direction x OI
    direction over the trailing 24h, four quadrants instead of independent
    univariate quantiles.
      price up + OI up    -> trend confirmed by fresh positioning
      price up + OI down  -> rally on short-covering / closing longs, weak
      price down + OI up  -> trend confirmed by fresh short positioning
      price down + OI down -> capitulation / position unwind, could be near exhaustion
    """
    valid = df.dropna(subset=["price_change_24h", "oi_change_24h"]).copy()
    if len(valid) < 200:
        return pd.DataFrame()
    valid["quadrant"] = np.select(
        [
            (valid["price_change_24h"] > 0) & (valid["oi_change_24h"] > 0),
            (valid["price_change_24h"] > 0) & (valid["oi_change_24h"] <= 0),
            (valid["price_change_24h"] <= 0) & (valid["oi_change_24h"] > 0),
            (valid["price_change_24h"] <= 0) & (valid["oi_change_24h"] <= 0),
        ],
        ["price_up_oi_up", "price_up_oi_down", "price_down_oi_up", "price_down_oi_down"],
        default="",
    )
    rows = []
    for h in horizons:
        fwd_ret = valid["close"].shift(-h) / valid["close"] - 1
        tmp = pd.DataFrame({"bucket": valid["quadrant"], "fwd_ret": fwd_ret}).dropna()
        agg = tmp.groupby("bucket", observed=True)["fwd_ret"].agg(["count", "mean", "std"])
        agg["t_stat"] = agg["mean"] / (agg["std"] / np.sqrt(agg["count"]))
        agg["horizon_hours"] = h
        agg["feature"] = "price_oi_divergence"
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quantile_study(df: pd.DataFrame, feature: str, horizons: list[int], n_buckets: int = 5) -> pd.DataFrame:
    rows = []
    valid = df.dropna(subset=[feature])
    if len(valid) < n_buckets * 20:
        return pd.DataFrame()
    try:
        buckets = pd.qcut(valid[feature], n_buckets, labels=[f"q{i+1}" for i in range(n_buckets)], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    for h in horizons:
        fwd_ret = valid["close"].shift(-h) / valid["close"] - 1
        tmp = pd.DataFrame({"bucket": buckets, "fwd_ret": fwd_ret}).dropna()
        agg = tmp.groupby("bucket", observed=True)["fwd_ret"].agg(["count", "mean", "std"])
        agg["t_stat"] = agg["mean"] / (agg["std"] / np.sqrt(agg["count"]))
        agg["horizon_hours"] = h
        agg["feature"] = feature
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--horizons", type=str, default="24,168")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()
    all_results = []
    for symbol in symbols:
        df = load_series(conn, symbol)
        if args.start_date:
            df = df[df["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            df = df[df["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]

        print(f"\n=== {symbol} ({len(df)} candles) ===")
        for feature in ["funding_rate", "funding_rate_24h_avg", "oi_change_24h", "oi_change_168h"]:
            res = quantile_study(df, feature, horizons)
            if res.empty:
                continue
            res["symbol"] = symbol
            all_results.append(res)
            print(f"\n-- {feature} --")
            with pd.option_context("display.width", 140):
                print(res.sort_values(["horizon_hours", "bucket"]).to_string(index=False))

        div = price_oi_divergence_study(df, horizons)
        if not div.empty:
            div["symbol"] = symbol
            all_results.append(div)
            print("\n-- price_oi_divergence --")
            with pd.option_context("display.width", 140):
                print(div.sort_values(["horizon_hours", "bucket"]).to_string(index=False))
    conn.close()

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("funding_oi_summary.csv", index=False)
        print("\nFull summary written to funding_oi_summary.csv")


if __name__ == "__main__":
    main()
