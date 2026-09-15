"""
Elliott Wave forward-return event study: for every rule-valid wave, compute the
underlying symbol's forward return at several horizons, then group by wave type
and volume behavior to see which patterns actually correlate with price moves.

CRITICAL DATA QUALITY CHECK FIRST: Due to sliding-window pivot generation,
overlapping candidate sequences can share the same end_ts under DIFFERENT
wave_labels. This makes forward-return samples grouped by wave_label NOT
independent observations - a real statistical risk. Before trusting any
t-stats, this script reports the overlap ratio (rows / distinct end_ts) for
both impulse and correction waves. Ratio near 1.0 = good (mostly independent),
much higher = flag this clearly (heavy overlap - same end_ts reused across
different wave_labels).

Methodology notes (read before trusting a number):
  - Entry price = nearest ohlcv close to the wave's end_ts; exit price =
    nearest ohlcv close to (end_ts + horizon). With 1h candles the "nearest"
    match is within ~30min, fine for hour-to-day-scale horizons.
  - Forward return = (exit - entry) / entry, computed for [24, 72, 168] hours.
  - t-stat is a one-sample mean/(std/sqrt(n)) versus zero; |t|>~2 is the rough
    "probably not noise" threshold for n>=30; below n=30 treat as anecdotal.
  - This is in-sample, exploratory. Don't read it as proof, read it as "what's
    worth building a wave-strategy rule around vs what's still just noise."

Usage:
    python scripts/elliott_wave_study.py --threshold 0.08 --horizons 24,72,168
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

MIN_N_FOR_CONFIDENCE = 30
IMPULSE_WAVES = {"1", "3", "5"}
CORRECTION_WAVES = {"2", "4", "A", "B", "C"}
TRAIN_CUTOFF = pd.Timestamp("2025-12-01", tz="UTC")


def load_waves(conn, threshold_pct: float, rule_valid_only: bool = True) -> pd.DataFrame:
    """Load Elliott wave sequences from the database."""
    sql = """
    SELECT
        symbol, wave_label, sequence_id, start_ts, end_ts, start_price, end_price,
        direction, avg_volume, end_volume, volume_vs_prior_wave, volume_trend
    FROM elliott_waves
    WHERE threshold_pct = %(threshold)s
    """
    params = {"threshold": threshold_pct}

    if rule_valid_only:
        sql += " AND rule_valid = TRUE"

    sql += " ORDER BY end_ts"

    df = pd.read_sql(sql, conn, params=params)
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
    return df


def load_ohlcv(conn, symbol: str) -> pd.DataFrame:
    """Load hourly OHLCV closes for a symbol."""
    df = pd.read_sql(
        "SELECT ts, close FROM ohlcv WHERE symbol = %(symbol)s ORDER BY ts",
        conn,
        params={"symbol": symbol},
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def check_overlap(waves: pd.DataFrame) -> dict:
    """
    For impulse and correction waves separately, compute:
    - total number of valid wave rows
    - number of distinct end_ts values
    - ratio = rows / distinct end_ts

    A ratio near 1.0 means little overlap (good, mostly independent).
    A ratio much higher than 1.0 means heavy overlap (flag clearly).
    """
    impulse = waves[waves["wave_label"].isin(IMPULSE_WAVES)]
    correction = waves[waves["wave_label"].isin(CORRECTION_WAVES)]

    results = {}

    for wave_type, wave_df in [("impulse", impulse), ("correction", correction)]:
        if len(wave_df) == 0:
            results[wave_type] = {
                "total_rows": 0,
                "distinct_end_ts": 0,
                "ratio": np.nan,
            }
        else:
            total_rows = len(wave_df)
            distinct_end_ts = wave_df["end_ts"].nunique()
            ratio = total_rows / distinct_end_ts if distinct_end_ts > 0 else np.nan

            results[wave_type] = {
                "total_rows": total_rows,
                "distinct_end_ts": distinct_end_ts,
                "ratio": ratio,
            }

    return results


def compute_forward_returns(waves: pd.DataFrame, ohlcv_by_symbol: dict, horizons: list[int]) -> pd.DataFrame:
    """
    For each wave row, compute forward returns at the specified horizons.
    Uses merge_asof to find entry price at end_ts and exit price at end_ts + horizon.
    """
    rows = []

    for _, wave in waves.iterrows():
        symbol = wave["symbol"]
        if symbol not in ohlcv_by_symbol:
            continue

        ohlcv = ohlcv_by_symbol[symbol]
        if ohlcv.empty:
            continue

        # Find entry price at wave end_ts
        entry_df = pd.DataFrame({"ts": [wave["end_ts"]]})
        entry_merged = pd.merge_asof(
            entry_df, ohlcv, on="ts", direction="nearest", tolerance=pd.Timedelta("2h")
        )

        if entry_merged.empty or pd.isna(entry_merged["close"].iloc[0]):
            continue

        entry_price = entry_merged["close"].iloc[0]

        # Find exit prices at end_ts + horizon
        for h in horizons:
            target_ts = wave["end_ts"] + pd.Timedelta(hours=h)
            exit_df = pd.DataFrame({"ts": [target_ts]})
            exit_merged = pd.merge_asof(
                exit_df, ohlcv, on="ts", direction="nearest", tolerance=pd.Timedelta("2h")
            )

            if exit_merged.empty or pd.isna(exit_merged["close"].iloc[0]):
                continue

            exit_price = exit_merged["close"].iloc[0]
            fwd_ret = (exit_price - entry_price) / entry_price

            rows.append({
                "symbol": symbol,
                "wave_label": wave["wave_label"],
                "volume_trend": wave["volume_trend"],
                "direction": wave["direction"],
                "end_ts": wave["end_ts"],
                "horizon_hours": h,
                "fwd_return": fwd_ret,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarize(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Group forward returns by (wave_label, volume_trend, horizon_hours, period)
    and compute: count, mean, std, t-stat for groups with n >= 5.
    """
    returns["period"] = returns["end_ts"].apply(
        lambda ts: "train" if ts < TRAIN_CUTOFF else "holdout"
    )

    rows = []
    for (wave_label, volume_trend, horizon_hours, period), group in returns.groupby(
        ["wave_label", "volume_trend", "horizon_hours", "period"]
    ):
        n = len(group)
        if n < 5:
            continue

        mean = group["fwd_return"].mean()
        std = group["fwd_return"].std()
        t_stat = mean / (std / np.sqrt(n)) if n > 1 and std and not np.isnan(std) else np.nan

        # Get direction from first row in group
        direction = group.iloc[0]["direction"]

        rows.append({
            "wave_label": wave_label,
            "volume_trend": volume_trend,
            "horizon_hours": horizon_hours,
            "period": period,
            "count": n,
            "mean_return": mean,
            "std": std,
            "t_stat": t_stat,
            "direction": direction,
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    return summary.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)


def find_robust_signals(summary: pd.DataFrame) -> list:
    """
    Find (wave_label, volume_trend) combinations where BOTH train and holdout
    show the same sign with |t-stat| > 2 in both periods.
    """
    signals = []

    for (wave_label, volume_trend), group in summary.groupby(["wave_label", "volume_trend"]):
        # Split by period
        train = group[group["period"] == "train"]
        holdout = group[group["period"] == "holdout"]

        if train.empty or holdout.empty:
            continue

        # Check if we have data at any horizon
        for horizon in group["horizon_hours"].unique():
            train_h = train[train["horizon_hours"] == horizon]
            holdout_h = holdout[holdout["horizon_hours"] == horizon]

            if train_h.empty or holdout_h.empty:
                continue

            train_row = train_h.iloc[0]
            holdout_row = holdout_h.iloc[0]

            train_t = train_row["t_stat"]
            holdout_t = holdout_row["t_stat"]

            # Same sign and |t| > 2 in both periods
            if (not pd.isna(train_t) and not pd.isna(holdout_t) and
                abs(train_t) > 2 and abs(holdout_t) > 2 and
                np.sign(train_t) == np.sign(holdout_t)):

                signals.append({
                    "wave_label": wave_label,
                    "volume_trend": volume_trend,
                    "horizon_hours": horizon,
                    "train_count": train_row["count"],
                    "train_mean_return": train_row["mean_return"],
                    "train_t_stat": train_t,
                    "holdout_count": holdout_row["count"],
                    "holdout_mean_return": holdout_row["mean_return"],
                    "holdout_t_stat": holdout_t,
                    "direction": train_row["direction"],
                })

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.08, help="Pivot detection threshold (pct)")
    parser.add_argument("--horizons", type=str, default="24,72,168", help="Forward-return horizons (hours)")
    parser.add_argument("--top", type=int, default=25, help="Top N rows by |t-stat| to print")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()

    # ========== STEP 1: DATA QUALITY CHECK ==========
    print("\n" + "="*80)
    print("STEP 1: DATA QUALITY CHECK - Overlap/Pseudo-replication Risk")
    print("="*80)

    waves = load_waves(conn, args.threshold, rule_valid_only=True)
    overlap_check = check_overlap(waves)

    print(f"\nFor threshold_pct = {args.threshold} and rule_valid = TRUE:")
    print(f"\nIMPULSE WAVES (1, 3, 5):")
    impulse_info = overlap_check["impulse"]
    print(f"  Total rows: {impulse_info['total_rows']}")
    print(f"  Distinct end_ts values: {impulse_info['distinct_end_ts']}")
    print(f"  Ratio (rows/distinct_end_ts): {impulse_info['ratio']:.3f}" if not np.isnan(impulse_info["ratio"]) else "  Ratio: N/A")
    if impulse_info['ratio'] > 1.5:
        print(f"  [OVERLAP DETECTED] High ratio suggests heavy reuse of end_ts across different wave_labels")
    elif impulse_info['ratio'] > 1.1:
        print(f"  [Some overlap present] moderate reuse of end_ts")
    else:
        print(f"  [Good] minimal overlap, mostly independent observations")

    print(f"\nCORRECTION WAVES (2, 4, A, B, C):")
    correction_info = overlap_check["correction"]
    print(f"  Total rows: {correction_info['total_rows']}")
    print(f"  Distinct end_ts values: {correction_info['distinct_end_ts']}")
    print(f"  Ratio (rows/distinct_end_ts): {correction_info['ratio']:.3f}" if not np.isnan(correction_info["ratio"]) else "  Ratio: N/A")
    if correction_info['ratio'] > 1.5:
        print(f"  [OVERLAP DETECTED] High ratio suggests heavy reuse of end_ts across different wave_labels")
    elif correction_info['ratio'] > 1.1:
        print(f"  [Some overlap present] moderate reuse of end_ts")
    else:
        print(f"  [Good] minimal overlap, mostly independent observations")

    print(f"\nTOTAL WAVES ANALYZED: {len(waves)} rows, {waves['end_ts'].nunique()} distinct end_ts values")

    # ========== STEP 2: FORWARD RETURN EVENT STUDY ==========
    print("\n" + "="*80)
    print("STEP 2: FORWARD RETURN ANALYSIS")
    print("="*80)

    # Load OHLCV for all symbols present in waves
    symbols = waves["symbol"].unique()
    ohlcv_by_symbol = {}
    for symbol in symbols:
        ohlcv_by_symbol[symbol] = load_ohlcv(conn, symbol)

    print(f"\nComputing forward returns for {len(symbols)} symbols at horizons: {horizons}")
    returns = compute_forward_returns(waves, ohlcv_by_symbol, horizons)

    if returns.empty:
        print("ERROR: No forward returns computed. Check data availability.")
        conn.close()
        return

    print(f"Forward returns computed: {len(returns)} observations")

    summary = summarize(returns)

    if summary.empty:
        print("ERROR: No summary statistics computed (no groups with n >= 5)")
        conn.close()
        return

    print(f"Summary statistics computed: {len(summary)} groups with n >= 5")

    # Write full results to CSV
    output_file = "elliott_wave_study_summary.csv"
    summary_for_csv = summary.copy()
    summary_for_csv = summary_for_csv.sort_values(
        "t_stat", key=lambda s: s.abs(), ascending=False
    )
    summary_for_csv.to_csv(output_file, index=False)
    print(f"\nFull summary written to {output_file}")

    # ========== STEP 3: REPORT ==========
    print("\n" + "="*80)
    print("STEP 3: TOP 25 RESULTS BY |t-stat|")
    print("="*80)

    top = summary.head(args.top)
    with pd.option_context("display.width", 180, "display.max_columns", 15):
        print("\n", top.to_string(index=False))

    # Look for robust signals
    print("\n" + "="*80)
    print("ROBUST SIGNALS (|t-stat| > 2 in both train AND holdout, same sign)")
    print("="*80)

    signals = find_robust_signals(summary)

    if signals:
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values(
            "train_t_stat", key=lambda s: s.abs(), ascending=False
        )
        print(f"\nFound {len(signals)} robust signal(s):")
        with pd.option_context("display.width", 180, "display.max_columns", 15):
            print(signals_df.to_string(index=False))
    else:
        print("\nNo robust signals found (no wave_label+volume_trend combos with |t-stat| > 2 in both train and holdout).")

    conn.close()
    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)


if __name__ == "__main__":
    main()
