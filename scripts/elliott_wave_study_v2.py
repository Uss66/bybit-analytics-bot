"""
Forward-return event study on the CLEAN, non-overlapping elliott_waves_v2
data (see project_elliott_wave_analysis memory for why v1 was discarded -
severe pseudo-replication from overlapping sliding-window candidates).

Two horizon schemes, both reported side by side:
  - Fixed horizons [24, 72, 168]h, for comparability with the rest of the
    project's event studies.
  - Degree-appropriate ("fractal") horizon: the empirical median duration
    of the NEXT wave in the Elliott sequence at that threshold (degree) -
    e.g. after wave 2 ends (entering wave 3), the horizon is wave 3's
    median duration at that threshold, not a fixed number. This directly
    operationalizes "expected trade duration should scale with the wave's
    degree" per the user's request.

Written directly (not via subagent) given two real bugs were already
caught in earlier subagent-built pieces of this pipeline - this is the
step that feeds Opus's analysis, worth extra care.

Usage:
    python scripts/elliott_wave_study_v2.py
"""
import numpy as np
import pandas as pd

from db import get_connection

TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
FIXED_HORIZONS = [24, 72, 168]

# Which wave typically follows which, for the "next wave" fractal horizon.
# Wave 5 end and wave C end both signal a new cycle starting - use wave A's
# median duration (a fresh corrective/initial move) as a reasonable proxy
# horizon in both cases, since we don't know yet if the next cycle will be
# impulsive or corrective.
NEXT_WAVE = {"1": "2", "2": "3", "3": "4", "4": "5", "5": "A", "A": "B", "B": "C", "C": "A"}


def load_waves(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM elliott_waves_v2", conn)
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
    df["period"] = np.where(df["end_ts"] < TRAIN_END, "train", "holdout")
    return df


def load_ohlcv(conn, symbol: str) -> pd.DataFrame:
    df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def forward_return(ohlcv: pd.DataFrame, event_ts: pd.Timestamp, horizon_hours: float) -> float | None:
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
    mean, std = arr.mean(), arr.std(ddof=1) if n > 1 else 0.0
    t = mean / (std / np.sqrt(n)) if std > 0 and n > 1 else float("nan")
    return mean, std, t, n


def median_duration_by_threshold(waves: pd.DataFrame) -> dict:
    """{(threshold_pct, wave_label): median_duration_hours}"""
    return waves.groupby(["threshold_pct", "wave_label"])["duration_hours"].median().to_dict()


def main():
    conn = get_connection()
    waves = load_waves(conn)
    durations = median_duration_by_threshold(waves)

    symbols = waves["symbol"].unique()
    ohlcv_cache = {s: load_ohlcv(conn, s) for s in symbols}
    conn.close()

    rows = []
    for _, w in waves.iterrows():
        ohlcv = ohlcv_cache[w["symbol"]]
        next_label = NEXT_WAVE[w["wave_label"]]
        fractal_horizon_h = durations.get((w["threshold_pct"], next_label))

        entry = {
            "threshold_pct": w["threshold_pct"], "wave_label": w["wave_label"],
            "direction": w["direction"], "volume_trend": w["volume_trend"], "period": w["period"],
        }
        for h in FIXED_HORIZONS:
            entry[f"ret_fixed_{h}h"] = forward_return(ohlcv, w["end_ts"], h)
        if fractal_horizon_h and fractal_horizon_h > 0:
            entry["ret_fractal"] = forward_return(ohlcv, w["end_ts"], fractal_horizon_h)
            entry["fractal_horizon_h"] = fractal_horizon_h
        else:
            entry["ret_fractal"] = None
            entry["fractal_horizon_h"] = None
        rows.append(entry)

    results = pd.DataFrame(rows)
    results.to_csv("elliott_wave_v2_returns_raw.csv", index=False)

    print("=" * 110)
    print("FIXED-HORIZON RESULTS (pooled across all symbols, by wave_label x period)")
    print("=" * 110)
    summary_rows = []
    for h in FIXED_HORIZONS:
        col = f"ret_fixed_{h}h"
        for wave_label, group in results.groupby("wave_label"):
            for period in ["train", "holdout"]:
                vals = group.loc[group["period"] == period, col].dropna().tolist()
                if len(vals) < 5:
                    continue
                mean, std, t, n = t_stat(vals)
                summary_rows.append(dict(horizon=f"{h}h_fixed", wave_label=wave_label, period=period,
                                          n=n, mean_return=mean, t_stat=t))

    print("=" * 110)
    print("FRACTAL-HORIZON RESULTS (horizon = median duration of the NEXT wave at that degree)")
    print("=" * 110)
    for wave_label, group in results.groupby("wave_label"):
        for period in ["train", "holdout"]:
            vals = group.loc[group["period"] == period, "ret_fractal"].dropna().tolist()
            if len(vals) < 5:
                continue
            mean, std, t, n = t_stat(vals)
            avg_h = group.loc[group["period"] == period, "fractal_horizon_h"].dropna().mean()
            summary_rows.append(dict(horizon=f"fractal(~{avg_h:.0f}h)", wave_label=wave_label, period=period,
                                      n=n, mean_return=mean, t_stat=t))

    summary = pd.DataFrame(summary_rows).sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
    summary.to_csv("elliott_wave_v2_study_summary.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(summary.to_string(index=False))

    # Robust-signal check: same wave_label, same sign, |t|>2 in BOTH train and holdout
    print("\n" + "=" * 110)
    print("ROBUST SIGNAL CHECK (|t|>2 in both train and holdout, same sign, same horizon scheme)")
    print("=" * 110)
    found_any = False
    for horizon, hgroup in summary.groupby("horizon"):
        pivot = hgroup.pivot_table(index="wave_label", columns="period", values="t_stat")
        if "train" not in pivot.columns or "holdout" not in pivot.columns:
            continue
        for wave_label, row in pivot.iterrows():
            t_train, t_hold = row.get("train"), row.get("holdout")
            if pd.notna(t_train) and pd.notna(t_hold) and abs(t_train) > 2 and abs(t_hold) > 2 \
               and np.sign(t_train) == np.sign(t_hold):
                found_any = True
                print(f"  {horizon} / wave {wave_label}: train t={t_train:.2f}, holdout t={t_hold:.2f}")
    if not found_any:
        print("  None found.")


if __name__ == "__main__":
    main()
