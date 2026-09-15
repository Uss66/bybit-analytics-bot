"""
Does the Fear & Greed Index level, its rate of change, or its "extreme"
readings (classic contrarian zones) predict forward spot returns? Same
quantile + train/holdout discipline as funding_oi_study.py, after three
independent funding/OI slicings all failed the same test.

Usage:
    python scripts/fear_greed_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 24,72,168
    python scripts/fear_greed_study.py --symbols BTCUSDT --end-date 2025-11-30    # train
    python scripts/fear_greed_study.py --symbols BTCUSDT --start-date 2025-12-01  # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection


def load_series(conn, symbol: str) -> pd.DataFrame:
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    fgi = pd.read_sql("SELECT ts, value FROM fear_greed_index ORDER BY ts", conn)
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    fgi["ts"] = pd.to_datetime(fgi["ts"], utc=True)

    df = pd.merge_asof(ohlcv, fgi, on="ts", direction="backward")
    df["fgi_change_1d"] = df["value"].diff(24)
    df["fgi_change_7d"] = df["value"].diff(168)
    return df


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


def extreme_zone_study(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Classic fixed bands: <=20 extreme fear, 20-45 fear, 45-55 neutral, 55-80 greed, >80 extreme greed."""
    valid = df.dropna(subset=["value"]).copy()
    valid["zone"] = pd.cut(
        valid["value"], bins=[-1, 20, 45, 55, 80, 101],
        labels=["extreme_fear", "fear", "neutral", "greed", "extreme_greed"],
    )
    rows = []
    for h in horizons:
        fwd_ret = valid["close"].shift(-h) / valid["close"] - 1
        tmp = pd.DataFrame({"bucket": valid["zone"], "fwd_ret": fwd_ret}).dropna()
        agg = tmp.groupby("bucket", observed=True)["fwd_ret"].agg(["count", "mean", "std"])
        agg["t_stat"] = agg["mean"] / (agg["std"] / np.sqrt(agg["count"]))
        agg["horizon_hours"] = h
        agg["feature"] = "fgi_zone"
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--horizons", type=str, default="24,72,168")
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
        for feature in ["value", "fgi_change_1d", "fgi_change_7d"]:
            res = quantile_study(df, feature, horizons)
            if res.empty:
                continue
            res["symbol"] = symbol
            all_results.append(res)
            print(f"\n-- {feature} --")
            with pd.option_context("display.width", 140):
                print(res.sort_values(["horizon_hours", "bucket"]).to_string(index=False))

        zones = extreme_zone_study(df, horizons)
        if not zones.empty:
            zones["symbol"] = symbol
            all_results.append(zones)
            print("\n-- fgi_zone --")
            with pd.option_context("display.width", 140):
                print(zones.sort_values(["horizon_hours", "bucket"]).to_string(index=False))
    conn.close()

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("fear_greed_summary.csv", index=False)
        print("\nFull summary written to fear_greed_summary.csv")


if __name__ == "__main__":
    main()
