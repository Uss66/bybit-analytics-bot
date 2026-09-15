"""
Does Deribit's DVOL (options-implied volatility, crypto's VIX equivalent) -
its level, or how fast it's moving - predict forward crypto returns? Same
quantile + train/holdout discipline as fear_greed_study.py /
traditional_markets_study.py, daily-resampled by default to avoid the
autocorrelation trap (hourly-resampling a once-daily feature inflates
t-stats - see project_traditional_markets_findings memory).

DVOL only exists for BTC and ETH on Deribit - tested against all requested
symbols anyway (BTC's DVOL is a market-wide vol proxy, may still say
something about SOL/etc, though coin-matching isn't possible the way it was
for whale-alert data).

Usage:
    python scripts/dvol_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 24,72,168
    python scripts/dvol_study.py --symbols BTCUSDT --end-date 2025-11-30    # train
    python scripts/dvol_study.py --symbols BTCUSDT --start-date 2025-12-01  # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

FEATURES = ["dvol", "chg_1d", "chg_3d", "chg_7d"]


def load_series(conn, symbol: str, dvol_symbol: str) -> pd.DataFrame:
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)

    dvol = pd.read_sql(
        "SELECT ts, close AS dvol FROM deribit_dvol WHERE symbol = %(s)s ORDER BY ts",
        conn, params={"s": dvol_symbol},
    )
    dvol["ts"] = pd.to_datetime(dvol["ts"], utc=True)
    dvol["chg_1d"] = dvol["dvol"].pct_change(1)
    dvol["chg_3d"] = dvol["dvol"].pct_change(3)
    dvol["chg_7d"] = dvol["dvol"].pct_change(7)

    df = pd.merge_asof(ohlcv, dvol[["ts"] + FEATURES], on="ts", direction="backward")
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
    parser.add_argument("--dvol-symbol", type=str, default="BTC", choices=["BTC", "ETH"])
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
        df = load_series(conn, symbol, args.dvol_symbol)
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
        print(f"\n=== {symbol} vs DVOL-{args.dvol_symbol} ({len(df)} candles): top |t-stat| ===")
        top = combined.reindex(combined["t_stat"].abs().sort_values(ascending=False).index).head(args.top)
        with pd.option_context("display.width", 140):
            print(top.to_string(index=False))
    conn.close()

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("dvol_summary.csv", index=False)
        print("\nFull summary written to dvol_summary.csv")


if __name__ == "__main__":
    main()
