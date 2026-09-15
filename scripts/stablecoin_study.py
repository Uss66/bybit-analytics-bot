"""
Does the growth rate of aggregate stablecoin supply (DefiLlama) - new
liquidity entering crypto - predict forward crypto returns? Same quantile +
train/holdout discipline as other *_study.py scripts, daily-resampled by
default to avoid the autocorrelation trap.

NOTE: only % CHANGE features are tested here, not the raw level - total
stablecoin supply has grown ~3000x since 2017 (a strong secular trend), so
quantile-bucketing the raw level would just bucket by time period and
trivially correlate with anything else that trended over the same years.
Growth rate is the stationary, meaningful feature.

Usage:
    python scripts/stablecoin_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 24,72,168
    python scripts/stablecoin_study.py --symbols BTCUSDT --end-date 2025-11-30    # train
    python scripts/stablecoin_study.py --symbols BTCUSDT --start-date 2025-12-01  # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

FEATURES = ["chg_1d", "chg_3d", "chg_7d", "chg_30d"]


def load_series(conn, symbol: str) -> pd.DataFrame:
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)

    sc = pd.read_sql("SELECT ts, total_usd FROM defillama_stablecoins ORDER BY ts", conn)
    sc["ts"] = pd.to_datetime(sc["ts"], utc=True)
    sc["chg_1d"] = sc["total_usd"].pct_change(1)
    sc["chg_3d"] = sc["total_usd"].pct_change(3)
    sc["chg_7d"] = sc["total_usd"].pct_change(7)
    sc["chg_30d"] = sc["total_usd"].pct_change(30)

    df = pd.merge_asof(ohlcv, sc[["ts"] + FEATURES], on="ts", direction="backward")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--horizons", type=str, default="24,72,168")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--hourly", action="store_true",
                         help="Use every hourly candle instead of daily resampling. WARNING: "
                              "reintroduces the autocorrelation trap - only for exploratory peeking.")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons_hours = [int(h) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()
    all_results = []
    for symbol in symbols:
        df = load_series(conn, symbol)
        if args.start_date:
            df = df[df["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            df = df[df["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]

        if args.hourly:
            horizons = horizons_hours
        else:
            df = df[df["ts"].dt.hour == 0].reset_index(drop=True)
            horizons = sorted({max(1, round(h / 24)) for h in horizons_hours})

        symbol_results = []
        for feature in FEATURES:
            res = quantile_study(df, feature, horizons)
            if res.empty:
                continue
            if not args.hourly:
                res["horizon_hours"] = res["horizon_hours"] * 24
            res["symbol"] = symbol
            symbol_results.append(res)
        if not symbol_results:
            print(f"\n=== {symbol}: no data ===")
            continue
        combined = pd.concat(symbol_results, ignore_index=True)
        all_results.append(combined)
        print(f"\n=== {symbol} ({len(df)} candles): top |t-stat| ===")
        top = combined.reindex(combined["t_stat"].abs().sort_values(ascending=False).index).head(args.top)
        with pd.option_context("display.width", 140):
            print(top.to_string(index=False))
    conn.close()

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("stablecoin_summary.csv", index=False)
        print("\nFull summary written to stablecoin_summary.csv")


if __name__ == "__main__":
    main()
