"""
Redesign attempt for the parked whale rules: instead of "did at least one
$Nm+ transfer happen in the last week" (which turned out to be active
97-100% of the time for BTC/ETH - see strategy.py's module docstring and
project_strategy_scoring_bug memory), build a continuous net-flow series -
sum(outflow_usd) - sum(inflow_usd) for the matching coin over a trailing
window - and quantile-bucket that against forward returns. A continuous
aggregate measure can't get stuck at "always active"; it's always defined,
and its quantiles are by construction spread across the whole timeline.

Usage:
    python scripts/whale_netflow_study.py --symbols BTCUSDT,ETHUSDT --windows 24,72,168 --horizons 24,72,168
    python scripts/whale_netflow_study.py --symbols BTCUSDT --end-date 2025-11-30   # train
    python scripts/whale_netflow_study.py --symbols BTCUSDT --start-date 2025-12-01 # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection


def base_coin(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def load_flow_events(conn, coin: str) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT ts, category, (raw->>'usd_value')::numeric AS usd_value
        FROM events
        WHERE category IN ('exchange_inflow', 'exchange_outflow') AND raw->>'coin' = %(coin)s
        ORDER BY ts
        """,
        conn,
        params={"coin": coin},
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["signed_usd"] = np.where(df["category"] == "exchange_outflow", df["usd_value"], -df["usd_value"])
    return df


def build_netflow_series(flow_events: pd.DataFrame, ohlcv: pd.DataFrame, window_hours: int) -> pd.Series:
    """Rolling sum of signed USD flow over the trailing window_hours, aligned to ohlcv."""
    if flow_events.empty:
        return pd.Series(0.0, index=ohlcv.index)
    hourly = flow_events.set_index("ts")["signed_usd"].resample("1h").sum()
    hourly = hourly.reindex(pd.date_range(ohlcv["ts"].min(), ohlcv["ts"].max(), freq="1h", tz="UTC"), fill_value=0.0)
    rolling = hourly.rolling(window_hours, min_periods=1).sum()
    rolling_df = rolling.reset_index()
    rolling_df.columns = ["ts", "netflow"]
    merged = pd.merge_asof(ohlcv[["ts"]], rolling_df, on="ts", direction="backward")
    return merged["netflow"].fillna(0.0)


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
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--windows", type=str, default="24,72,168", help="Trailing net-flow windows in hours")
    parser.add_argument("--horizons", type=str, default="24,72,168")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()
    all_results = []
    for symbol in symbols:
        coin = base_coin(symbol)
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        flow_events = load_flow_events(conn, coin)

        df = ohlcv.copy()
        for w in windows:
            df[f"netflow_{w}h"] = build_netflow_series(flow_events, ohlcv, w)

        if args.start_date:
            df = df[df["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            df = df[df["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]

        print(f"\n=== {symbol} (coin={coin}, {len(df)} candles, {len(flow_events)} flow events total) ===")
        for w in windows:
            res = quantile_study(df, f"netflow_{w}h", horizons)
            if res.empty:
                continue
            res["symbol"] = symbol
            all_results.append(res)
            print(f"\n-- netflow_{w}h --")
            with pd.option_context("display.width", 140):
                print(res.sort_values(["horizon_hours", "bucket"]).to_string(index=False))
    conn.close()

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("whale_netflow_summary.csv", index=False)
        print("\nFull summary written to whale_netflow_summary.csv")


if __name__ == "__main__":
    main()
