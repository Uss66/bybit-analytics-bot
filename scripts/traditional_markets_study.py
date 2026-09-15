"""
Does the recent move in a traditional market (S&P500, Nasdaq, gold, DXY,
VIX, 10Y yield) predict forward crypto returns? Same quantile + train/holdout
discipline as fear_greed_study.py / funding_oi_study.py.

Traditional markets are daily and only trade on weekdays - their last close
is forward-filled onto crypto's continuous hourly timeline (same technique
used for the Fear & Greed Index, which is also daily).

Usage:
    python scripts/traditional_markets_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --horizons 24,72,168
    python scripts/traditional_markets_study.py --symbols BTCUSDT --end-date 2025-11-30    # train
    python scripts/traditional_markets_study.py --symbols BTCUSDT --start-date 2025-12-01  # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

TICKERS = ["SPX", "NDX", "GOLD", "DXY", "VIX", "UST10Y", "OIL", "SILVER", "PLATINUM", "COPPER", "NVDA"]


def load_series(conn, symbol: str) -> pd.DataFrame:
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    df = ohlcv
    for ticker in TICKERS:
        tm = pd.read_sql(
            "SELECT ts, close FROM traditional_markets WHERE ticker = %(t)s ORDER BY ts", conn, params={"t": ticker}
        )
        tm["ts"] = pd.to_datetime(tm["ts"], utc=True)
        tm[f"{ticker}_chg_1d"] = tm["close"].pct_change(1)
        tm[f"{ticker}_chg_3d"] = tm["close"].pct_change(3)
        tm[f"{ticker}_chg_7d"] = tm["close"].pct_change(7)
        cols = ["ts", f"{ticker}_chg_1d", f"{ticker}_chg_3d", f"{ticker}_chg_7d"]
        df = pd.merge_asof(df, tm[cols], on="ts", direction="backward")
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
    parser.add_argument("--horizons", type=str, default="24,72,168", help="In hours - converted to whole days internally when --daily (the default) is active")
    parser.add_argument("--windows", type=str, default="1d,3d,7d")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--hourly", action="store_true",
                         help="Use every hourly crypto candle instead of resampling to one row/day. "
                              "WARNING: this reintroduces the autocorrelation trap documented in "
                              "project_traditional_markets_findings memory - consecutive hourly rows "
                              "share nearly the same daily-change feature value, which inflates t-stats "
                              "a lot (measured: up to t=26 hourly vs t<6 daily on the same data). Only "
                              "use this for exploratory peeking, never to validate a rule.")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons_hours = [int(h) for h in args.horizons.split(",") if h.strip()]
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]

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
            horizons = sorted({max(1, round(h / 24)) for h in horizons_hours})  # rows are now 1/day

        symbol_results = []
        for ticker in TICKERS:
            for w in windows:
                res = quantile_study(df, f"{ticker}_chg_{w}", horizons)
                if res.empty:
                    continue
                if not args.hourly:
                    res["horizon_hours"] = res["horizon_hours"] * 24  # report back in hours for consistency
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
        pd.concat(all_results, ignore_index=True).to_csv("traditional_markets_summary.csv", index=False)
        print("\nFull summary written to traditional_markets_summary.csv")


if __name__ == "__main__":
    main()
