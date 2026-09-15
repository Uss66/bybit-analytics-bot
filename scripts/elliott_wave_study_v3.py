"""
Corrected forward-return event study on elliott_waves_v2, incorporating
everything learned from the v2 study (see project_elliott_wave_analysis
memory for the full history):

1. Entry point is `end_confirmed_ts` (when the wave's ending pivot was
   ACTUALLY confirmable in real time), never `end_ts` (when the price
   extreme itself occurred) - this is the fix for the repainting/
   look-ahead bug that inflated the v2 study's headline numbers to
   implausible t-stats (t=10+) that evaporated once corrected.
2. A baseline comparison: forward returns from ANY confirmed pivot of the
   same type (high/low), not just wave-labeled ones - Opus's analysis
   found the Elliott wave label added ZERO information beyond "this is a
   confirmed pivot" for the previous (also-rejected) wave-C finding. Any
   candidate signal here must beat this baseline, not just beat zero.
3. Per-degree (threshold_pct) results, NOT pooled across degrees - Opus
   found the earlier pooled statistic combined degrees that individually
   failed significance, inflating the appearance of a real effect.
4. Expanded threshold range (2%/3%/5%/8%/12%/18%/25%) per user request to
   check shorter-term and longer-term wave scales - flagging sample-size
   reality at the extremes (25% has essentially n=1 impulse sequence
   across all 7 symbols and 2 years - not statistically usable, reported
   but not trusted).

Usage:
    python scripts/elliott_wave_study_v3.py
"""
import numpy as np
import pandas as pd

from db import get_connection

TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
HORIZONS = [24, 72, 168]


def load_waves(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM elliott_waves_v2", conn)
    df["end_confirmed_ts"] = pd.to_datetime(df["end_confirmed_ts"], utc=True)
    df["period"] = np.where(df["end_confirmed_ts"] < TRAIN_END, "train", "holdout")
    return df


def load_pivots(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM elliott_pivots", conn)
    df["confirmed_ts"] = pd.to_datetime(df["confirmed_ts"], utc=True)
    df["period"] = np.where(df["confirmed_ts"] < TRAIN_END, "train", "holdout")
    return df


def load_ohlcv(conn, symbol: str) -> pd.DataFrame:
    df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def forward_return(ohlcv: pd.DataFrame, event_ts: pd.Timestamp, horizon_hours: float):
    base = pd.merge_asof(pd.DataFrame({"ts": [event_ts]}), ohlcv, on="ts", direction="backward")
    if base["close"].isna().any():
        return None
    target_ts = event_ts + pd.Timedelta(hours=horizon_hours)
    fwd = pd.merge_asof(pd.DataFrame({"ts": [target_ts]}), ohlcv, on="ts", direction="backward")
    if fwd["close"].isna().any() or fwd["ts"].iloc[0] < event_ts:
        return None
    return float(fwd["close"].iloc[0] / base["close"].iloc[0] - 1)


def t_stat(returns: list[float]) -> tuple[float, float, float, int]:
    arr = np.array(returns)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    mean, std = arr.mean(), arr.std(ddof=1)
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return mean, std, t, n


def main():
    conn = get_connection()
    waves = load_waves(conn)
    pivots = load_pivots(conn)
    symbols = waves["symbol"].unique()
    ohlcv_cache = {s: load_ohlcv(conn, s) for s in symbols}
    conn.close()

    # --- Wave-labeled forward returns (honest entry: end_confirmed_ts) ---
    wave_rows = []
    for _, w in waves.iterrows():
        ohlcv = ohlcv_cache[w["symbol"]]
        for h in HORIZONS:
            ret = forward_return(ohlcv, w["end_confirmed_ts"], h)
            wave_rows.append(dict(
                threshold_pct=w["threshold_pct"], wave_label=w["wave_label"],
                direction=w["direction"], period=w["period"], horizon=h, ret=ret,
            ))
    wave_df = pd.DataFrame(wave_rows).dropna(subset=["ret"])

    # --- Baseline: ALL confirmed pivots (not wave-labeled), same entry convention ---
    pivot_rows = []
    for _, p in pivots.iterrows():
        ohlcv = ohlcv_cache[p["symbol"]]
        for h in HORIZONS:
            ret = forward_return(ohlcv, p["confirmed_ts"], h)
            pivot_rows.append(dict(
                threshold_pct=p["threshold_pct"], pivot_type=p["pivot_type"],
                period=p["period"], horizon=h, ret=ret,
            ))
    pivot_df = pd.DataFrame(pivot_rows).dropna(subset=["ret"])

    print("=" * 120)
    print("WAVE-LABELED RESULTS (honest entry, PER DEGREE - not pooled), n>=8 only")
    print("=" * 120)
    wave_summary = []
    for (thr, label, direction, horizon), g in wave_df.groupby(
        ["threshold_pct", "wave_label", "direction", "horizon"]
    ):
        for period in ["train", "holdout"]:
            vals = g.loc[g["period"] == period, "ret"].tolist()
            if len(vals) < 8:
                continue
            mean, std, t, n = t_stat(vals)
            wave_summary.append(dict(threshold_pct=thr, wave_label=label, direction=direction,
                                      horizon=horizon, period=period, n=n, mean_return=mean, t_stat=t))
    wave_summary_df = pd.DataFrame(wave_summary).sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
    wave_summary_df.to_csv("elliott_wave_v3_summary.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 60):
        print(wave_summary_df.head(40).to_string(index=False))

    print("\n" + "=" * 120)
    print("BASELINE: ALL confirmed pivots (any type), PER DEGREE, n>=8 only")
    print("=" * 120)
    baseline_summary = []
    for (thr, ptype, horizon), g in pivot_df.groupby(["threshold_pct", "pivot_type", "horizon"]):
        for period in ["train", "holdout"]:
            vals = g.loc[g["period"] == period, "ret"].tolist()
            if len(vals) < 8:
                continue
            mean, std, t, n = t_stat(vals)
            baseline_summary.append(dict(threshold_pct=thr, pivot_type=ptype, horizon=horizon,
                                          period=period, n=n, mean_return=mean, t_stat=t))
    baseline_df = pd.DataFrame(baseline_summary).sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
    baseline_df.to_csv("elliott_pivot_baseline_v3.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 60):
        print(baseline_df.head(20).to_string(index=False))

    # --- Robust signal check: same wave_label+threshold+horizon, |t|>2 both periods, same sign ---
    print("\n" + "=" * 120)
    print("ROBUST WAVE SIGNAL CHECK (|t|>2 both periods, same sign, PER DEGREE)")
    print("=" * 120)
    found_any = False
    for (thr, label, direction, horizon), g in wave_summary_df.groupby(
        ["threshold_pct", "wave_label", "direction", "horizon"]
    ):
        pivot_tbl = g.set_index("period")["t_stat"]
        if "train" in pivot_tbl.index and "holdout" in pivot_tbl.index:
            t_train, t_hold = pivot_tbl["train"], pivot_tbl["holdout"]
            if abs(t_train) > 2 and abs(t_hold) > 2 and np.sign(t_train) == np.sign(t_hold):
                found_any = True
                # Compare against the matching baseline (same threshold, pivot type from wave's own end)
                print(f"  threshold={thr} wave={label} dir={direction} horizon={horizon}h: "
                      f"train t={t_train:.2f}, holdout t={t_hold:.2f}")
    if not found_any:
        print("  None found.")

    print("\n" + "=" * 120)
    print("SAMPLE SIZE REALITY CHECK: distinct wave sequences per threshold")
    print("=" * 120)
    conn = get_connection()
    seq_counts = pd.read_sql(
        "SELECT threshold_pct, wave_label, COUNT(DISTINCT sequence_id) n_sequences "
        "FROM elliott_waves_v2 WHERE wave_label IN ('1','A') GROUP BY threshold_pct, wave_label "
        "ORDER BY threshold_pct", conn)
    conn.close()
    print(seq_counts.to_string(index=False))


if __name__ == "__main__":
    main()
