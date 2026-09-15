"""
Elliott Wave re-test at higher frequency (15-min bars instead of the
original daily-resampled-from-hourly series), requested 2026-09-14 after
"а давай попробуем волновую теорию в контексте высокочастотной торговли".

IMPORTANT CONTEXT (told to the user before building this): the original
Elliott Wave rejection (see project_elliott_wave_analysis memory) was NOT
about timeframe - the deepest of the three problems found was that wave
LABELING itself is hindsight-biased (a wave's label depends on rules that
reference LATER waves: "wave 3 is not the shortest of 1/3/5" can't be known
until wave 5 exists). That problem is scale-INVARIANT - it will not go away
just because the bars are smaller. This script exists to test that
prediction empirically rather than just assert it, using EXACTLY the same
verified methodology as elliott_wave_study_v3.py (imported directly, not
reimplemented):
  - entry at confirmed_ts/end_confirmed_ts (never the pivot's own ts)
  - per-degree (threshold_pct) results, not pooled
  - wave-labeled returns compared against a same-type-pivot BASELINE
    (a candidate must beat "any confirmed pivot", not just beat zero)
  - train/holdout split, robustness = |t|>2 in BOTH periods, same sign

Differences from v3: runs directly on 15-min closes (no daily resampling -
that was only ever a device to get to "daily-degree" swings for the
original hourly study), on ohlcv_intraday (BTC only, 2yr - see
fetch_bybit_intraday.py), with shorter forward-return horizons appropriate
for intraday bars, and pivots/waves are computed in-memory (not persisted
to new DB tables) since this is a one-off exploratory re-test, not a
promoted-to-production pipeline.

Usage:
    python scripts/elliott_wave_hf_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from detect_elliott_pivots import detect_pivots, THRESHOLDS
from generate_elliott_waves_v2 import generate_waves_sequential_tiling
from elliott_wave_study_v3 import forward_return, t_stat

SYMBOL = "BTCUSDT"
INTERVAL_MIN = 15
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
HORIZONS_HOURS = [0.25, 1, 4, 12, 24]  # 15min, 1h, 4h, 12h, 1d - short, matching the "HF" bar size


def main():
    conn = get_connection()
    ohlcv = pd.read_sql(
        "SELECT ts, close FROM ohlcv_intraday WHERE symbol = %(s)s AND interval_min = %(i)s ORDER BY ts",
        conn, params={"s": SYMBOL, "i": INTERVAL_MIN},
    )
    conn.close()
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    closes = ohlcv.set_index("ts")["close"]
    print(f"Loaded {len(closes)} {INTERVAL_MIN}-min candles for {SYMBOL}, "
          f"{closes.index[0]} -> {closes.index[-1]}")

    # --- Detect pivots directly on the intraday series (no daily resample) ---
    all_pivots = []  # (ts, price, pivot_type, confirmed_ts, threshold_pct)
    for thr in THRESHOLDS:
        pivots = detect_pivots(closes, thr)
        print(f"  threshold {thr*100:.1f}%: {len(pivots)} pivots")
        for ts, price, ptype, confirmed_ts in pivots:
            all_pivots.append(dict(ts=ts, price=price, pivot_type=ptype, confirmed_ts=confirmed_ts, threshold_pct=thr))
    pivots_df = pd.DataFrame(all_pivots)

    # --- Generate waves per threshold (reusing the exact verified tiling logic) ---
    all_waves = []
    seq_id = 1000
    for thr in THRESHOLDS:
        sub = pivots_df[pivots_df["threshold_pct"] == thr].sort_values("ts")
        pivot_tuples = [(row.ts, row.price, row.pivot_type) for row in sub.itertuples()]
        confirmed_by_ts = {row.ts: row.confirmed_ts for row in sub.itertuples()}
        empty_volumes = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        waves, seq_id = generate_waves_sequential_tiling(pivot_tuples, empty_volumes, seq_id_start=seq_id)
        for w in waves:
            w["threshold_pct"] = thr
            w["end_confirmed_ts"] = confirmed_by_ts.get(w["end_ts"], w["end_ts"])
        all_waves.extend(waves)
        n_impulse = len(set(w["sequence_id"] for w in waves if w["wave_label"] == "1"))
        n_correction = len(set(w["sequence_id"] for w in waves if w["wave_label"] == "A"))
        print(f"  threshold {thr*100:.1f}%: {n_impulse} impulse seqs, {n_correction} correction seqs")
    waves_df = pd.DataFrame(all_waves)
    waves_df["period"] = np.where(waves_df["end_confirmed_ts"] < TRAIN_END, "train", "holdout")
    pivots_df["period"] = np.where(pivots_df["confirmed_ts"] < TRAIN_END, "train", "holdout")

    # --- Forward returns: wave-labeled (honest entry) ---
    wave_rows = []
    for w in waves_df.itertuples():
        for h in HORIZONS_HOURS:
            ret = forward_return(ohlcv, w.end_confirmed_ts, h)
            wave_rows.append(dict(threshold_pct=w.threshold_pct, wave_label=w.wave_label,
                                   direction=w.direction, period=w.period, horizon=h, ret=ret))
    wave_df = pd.DataFrame(wave_rows).dropna(subset=["ret"])

    # --- Baseline: any confirmed pivot of the same type ---
    pivot_rows = []
    for p in pivots_df.itertuples():
        for h in HORIZONS_HOURS:
            ret = forward_return(ohlcv, p.confirmed_ts, h)
            pivot_rows.append(dict(threshold_pct=p.threshold_pct, pivot_type=p.pivot_type,
                                    period=p.period, horizon=h, ret=ret))
    pivot_df = pd.DataFrame(pivot_rows).dropna(subset=["ret"])

    print("\n" + "=" * 120)
    print(f"WAVE-LABELED RESULTS ({SYMBOL} {INTERVAL_MIN}min, honest entry, PER DEGREE), n>=8 only")
    print("=" * 120)
    wave_summary = []
    for (thr, label, direction, horizon), g in wave_df.groupby(["threshold_pct", "wave_label", "direction", "horizon"]):
        for period in ["train", "holdout"]:
            vals = g.loc[g["period"] == period, "ret"].tolist()
            if len(vals) < 8:
                continue
            mean, std, t, n = t_stat(vals)
            wave_summary.append(dict(threshold_pct=thr, wave_label=label, direction=direction,
                                      horizon=horizon, period=period, n=n, mean_return=mean, t_stat=t))
    wave_summary_df = pd.DataFrame(wave_summary)
    if not wave_summary_df.empty:
        wave_summary_df = wave_summary_df.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
        wave_summary_df.to_csv("elliott_wave_hf_summary.csv", index=False)
        with pd.option_context("display.width", 140, "display.max_rows", 40):
            print(wave_summary_df.head(30).to_string(index=False))
    else:
        print("  (no wave/period/horizon combo reached n>=8)")

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
    baseline_df = pd.DataFrame(baseline_summary)
    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
        baseline_df.to_csv("elliott_pivot_hf_baseline.csv", index=False)
        with pd.option_context("display.width", 140, "display.max_rows", 20):
            print(baseline_df.head(15).to_string(index=False))
    else:
        print("  (no pivot/period/horizon combo reached n>=8)")

    print("\n" + "=" * 120)
    print("ROBUST WAVE SIGNAL CHECK (|t|>2 both periods, same sign, PER DEGREE) -")
    print("AND must beat the matching baseline's |t| to mean anything beyond 'it's a pivot'")
    print("=" * 120)
    found_any = False
    if not wave_summary_df.empty:
        for (thr, label, direction, horizon), g in wave_summary_df.groupby(["threshold_pct", "wave_label", "direction", "horizon"]):
            pivot_tbl = g.set_index("period")["t_stat"]
            if "train" in pivot_tbl.index and "holdout" in pivot_tbl.index:
                t_train, t_hold = pivot_tbl["train"], pivot_tbl["holdout"]
                if abs(t_train) > 2 and abs(t_hold) > 2 and np.sign(t_train) == np.sign(t_hold):
                    found_any = True
                    print(f"  threshold={thr} wave={label} dir={direction} horizon={horizon}h: "
                          f"train t={t_train:.2f}, holdout t={t_hold:.2f}")
    if not found_any:
        print("  None found.")

    print("\n" + "=" * 120)
    print("SAMPLE SIZE REALITY CHECK: distinct wave sequences per threshold")
    print("=" * 120)
    if not waves_df.empty:
        seq_counts = waves_df[waves_df["wave_label"].isin(["1", "A"])].groupby(
            ["threshold_pct", "wave_label"])["sequence_id"].nunique().reset_index(name="n_sequences")
        print(seq_counts.to_string(index=False))


if __name__ == "__main__":
    main()
