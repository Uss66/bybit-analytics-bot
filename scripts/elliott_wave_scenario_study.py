"""
Elliott Wave, tested the way it's actually practiced (per user's sources,
2026-09-14): NOT as a retrospective, fully-confirmed label (that's what
elliott_wave_study_v3.py / elliott_wave_hf_study.py tested and rejected -
those only recorded "wave 4" for sequences where wave 5 ALSO later
validated rule 2, which needs future information), but as a live,
point-in-time SCENARIO with a hard invalidation level:

  - A "candidate wave-4-end" is flagged using ONLY rules checkable with
    pivots P0..P4 (already occurred): rule 1 (wave 2 doesn't fully retrace
    wave 1) and rule 3 (wave 4 doesn't overlap wave 1's territory). Rule 2
    ("wave 3 not shortest of 1/3/5") is dropped at entry time since it
    needs wave 5, which hasn't happened yet - exactly the fix the user's
    sources describe (trade the "wave 5 continuation" scenario without
    knowing if it will fully validate).
  - Entry: at P4's confirmed_ts (never P4's own ts - same repainting fix as
    before), targeting continuation in the wave 1-3 direction.
  - Invalidation level (the stop, per the sources' own "invalidation"
    principle): P1's price. If price breaks back past P1, the wave-4/5
    count is dead by the theory's own rule 3 - this is BOTH the correct
    live stop-loss AND fully known at entry time, no lookahead.
  - Baseline: forward returns from ANY confirmed pivot of the matching
    type (same convention as v3), to test whether the rule-1+3 filter adds
    anything beyond "entered after a swing pivot".

Two return measures per candidate:
  - plain forward return at a fixed horizon (comparable to the old study)
  - invalidation-stopped return: exits at the first bar the invalidation
    level is breached within the horizon, else at the horizon (mirrors
    strategy.py's own stop_loss mechanics, just with a price LEVEL instead
    of a fixed percentage)

Runs on BOTH datasets:
  (a) "classic": all 7 symbols, daily-resampled-from-hourly pivots already
      in `elliott_pivots` (the same dataset every prior Elliott Wave
      verdict in this project was based on)
  (b) "hf": BTC 15-min bars from `ohlcv_intraday` (see
      elliott_wave_hf_study.py), pivots recomputed in-memory

Usage:
    python scripts/elliott_wave_scenario_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE
from detect_elliott_pivots import detect_pivots, THRESHOLDS

TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")


def find_candidates(pivots: list[tuple], threshold_pct: float) -> tuple[list[dict], list[dict]]:
    """pivots: chronological list of (ts, price, pivot_type, confirmed_ts).
    Returns (candidates, all_pivots_as_dicts) - candidates are P4-end
    points where ONLY rule1+rule3 (checkable without future pivots) pass."""
    candidates = []
    for i in range(len(pivots) - 4):
        p0, p1, p2, p3, p4 = pivots[i:i + 5]
        direction = "up" if p0[2] == "low" else "down"
        if direction == "up":
            rule1 = p2[1] > p0[1]
            rule3 = p4[1] > p1[1]
        else:
            rule1 = p2[1] < p0[1]
            rule3 = p4[1] < p1[1]
        if rule1 and rule3:
            candidates.append(dict(
                threshold_pct=threshold_pct, direction=direction,
                entry_ts=p4[3], invalidation_price=p1[1], p4_ts=p4[0],
            ))
    return candidates


def stopped_return(ohlcv: pd.DataFrame, entry_ts, invalidation_price: float, direction: str,
                    horizon_hours: float, fee: float = SPOT_TAKER_FEE):
    """Mirrors strategy.py's stop_loss mechanics: exit at the first bar the
    invalidation level is breached within [entry_ts, entry_ts+horizon],
    else exit at the horizon end. Returns (plain_ret, stopped_ret) or
    (None, None) if data doesn't cover the window."""
    base = pd.merge_asof(pd.DataFrame({"ts": [entry_ts]}), ohlcv, on="ts", direction="backward")
    if base["close"].isna().any():
        return None, None
    entry_price = float(base["close"].iloc[0])
    end_ts = entry_ts + pd.Timedelta(hours=horizon_hours)
    window = ohlcv[(ohlcv["ts"] >= entry_ts) & (ohlcv["ts"] <= end_ts)]
    if window.empty or window["ts"].iloc[-1] < end_ts - pd.Timedelta(hours=horizon_hours * 0.5):
        return None, None  # not enough forward data to trust this window

    plain_exit_price = float(window["close"].iloc[-1])
    if direction == "up":
        breach = window[window["close"] <= invalidation_price]
    else:
        breach = window[window["close"] >= invalidation_price]
    stopped_exit_price = float(breach["close"].iloc[0]) if not breach.empty else plain_exit_price

    if direction == "up":
        plain_ret = (plain_exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
        stopped_ret = (stopped_exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
    else:
        plain_ret = (entry_price * (1 - fee)) / (plain_exit_price * (1 + fee)) - 1
        stopped_ret = (entry_price * (1 - fee)) / (stopped_exit_price * (1 + fee)) - 1
    return plain_ret, stopped_ret


def forward_return_typed(ohlcv, event_ts, horizon_hours, direction, fee=SPOT_TAKER_FEE):
    base = pd.merge_asof(pd.DataFrame({"ts": [event_ts]}), ohlcv, on="ts", direction="backward")
    if base["close"].isna().any():
        return None
    target_ts = event_ts + pd.Timedelta(hours=horizon_hours)
    fwd = pd.merge_asof(pd.DataFrame({"ts": [target_ts]}), ohlcv, on="ts", direction="backward")
    if fwd["close"].isna().any() or fwd["ts"].iloc[0] < event_ts:
        return None
    entry, exit_ = float(base["close"].iloc[0]), float(fwd["close"].iloc[0])
    if direction == "up":
        return (exit_ * (1 - fee)) / (entry * (1 + fee)) - 1
    return (entry * (1 - fee)) / (exit_ * (1 + fee)) - 1


def t_stat(returns):
    arr = np.array(returns)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan"), n
    mean, std = arr.mean(), arr.std(ddof=1)
    t = mean / (std / np.sqrt(n)) if std > 0 else float("nan")
    return mean, t, n


def run_study(label: str, pivots_by_threshold: dict, ohlcv: pd.DataFrame, horizons: list[float]):
    print("\n" + "#" * 120)
    print(f"# {label}")
    print("#" * 120)

    all_candidates = []
    all_baseline = []
    for thr, pivots in pivots_by_threshold.items():
        cands = find_candidates(pivots, thr)
        all_candidates.extend(cands)
        for ts, price, ptype, confirmed_ts in pivots:
            all_baseline.append(dict(threshold_pct=thr, pivot_type=ptype, entry_ts=confirmed_ts))
        n_up = sum(1 for c in cands if c["direction"] == "up")
        n_down = sum(1 for c in cands if c["direction"] == "down")
        print(f"  threshold {thr*100:.1f}%: {len(pivots)} pivots -> {len(cands)} candidate wave-4-ends "
              f"({n_up} up / {n_down} down)")

    cand_rows = []
    for c in all_candidates:
        period = "train" if c["entry_ts"] < TRAIN_END else "holdout"
        for h in horizons:
            plain, stopped = stopped_return(ohlcv, c["entry_ts"], c["invalidation_price"], c["direction"], h)
            if plain is None:
                continue
            cand_rows.append(dict(threshold_pct=c["threshold_pct"], direction=c["direction"], horizon=h,
                                   period=period, plain_ret=plain, stopped_ret=stopped))
    cand_df = pd.DataFrame(cand_rows)

    base_rows = []
    for b in all_baseline:
        period = "train" if b["entry_ts"] < TRAIN_END else "holdout"
        direction = "up" if b["pivot_type"] == "low" else "down"
        for h in horizons:
            ret = forward_return_typed(ohlcv, b["entry_ts"], h, direction)
            if ret is None:
                continue
            base_rows.append(dict(threshold_pct=b["threshold_pct"], pivot_type=b["pivot_type"], horizon=h,
                                   period=period, ret=ret))
    base_df = pd.DataFrame(base_rows)

    print(f"\n{'thr':>6} {'dir':>5} {'hz':>6} {'period':>8} {'n':>5} "
          f"{'plain_mean':>11} {'plain_t':>8} {'stop_mean':>10} {'stop_t':>8} "
          f"{'base_mean':>10} {'base_t':>8}  beats_base")
    results = []
    if not cand_df.empty:
        for (thr, direction, h), g in cand_df.groupby(["threshold_pct", "direction", "horizon"]):
            for period in ["train", "holdout"]:
                sub = g[g["period"] == period]
                if len(sub) < 8:
                    continue
                plain_mean, plain_t, n = t_stat(sub["plain_ret"].tolist())
                stop_mean, stop_t, _ = t_stat(sub["stopped_ret"].tolist())

                ptype = "low" if direction == "up" else "high"
                base_sub = base_df[(base_df["threshold_pct"] == thr) & (base_df["pivot_type"] == ptype) &
                                    (base_df["horizon"] == h) & (base_df["period"] == period)]
                if len(base_sub) >= 8:
                    base_mean, base_t, _ = t_stat(base_sub["ret"].tolist())
                else:
                    base_mean, base_t = float("nan"), float("nan")

                beats_base = abs(stop_t) > abs(base_t) if pd.notna(base_t) else None
                results.append(dict(threshold_pct=thr, direction=direction, horizon=h, period=period, n=n,
                                     plain_mean=plain_mean, plain_t=plain_t, stop_mean=stop_mean, stop_t=stop_t,
                                     base_mean=base_mean, base_t=base_t, beats_base=beats_base))
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("stop_t", key=lambda s: s.abs(), ascending=False)
        for r in results_df.head(25).itertuples():
            print(f"{r.threshold_pct*100:>5.1f}% {r.direction:>5} {r.horizon:>6.2f} {r.period:>8} {r.n:>5} "
                  f"{r.plain_mean:>11.4f} {r.plain_t:>8.2f} {r.stop_mean:>10.4f} {r.stop_t:>8.2f} "
                  f"{r.base_mean:>10.4f} {r.base_t:>8.2f}  {r.beats_base}")

        print("\n  ROBUST CHECK (|stop_t|>2 both periods, same sign, AND beats baseline in both):")
        found = False
        for (thr, direction, h), g in results_df.groupby(["threshold_pct", "direction", "horizon"]):
            pv = g.set_index("period")
            if "train" in pv.index and "holdout" in pv.index:
                t_tr, t_ho = pv.loc["train", "stop_t"], pv.loc["holdout", "stop_t"]
                b_tr, b_ho = pv.loc["train", "beats_base"], pv.loc["holdout", "beats_base"]
                if abs(t_tr) > 2 and abs(t_ho) > 2 and np.sign(t_tr) == np.sign(t_ho) and b_tr and b_ho:
                    found = True
                    print(f"    threshold={thr} dir={direction} horizon={h}h: train t={t_tr:.2f}, holdout t={t_ho:.2f}")
        if not found:
            print("    None found.")
    else:
        print("  (no combo reached n>=8)")

    return results_df


def main():
    conn = get_connection()

    # --- (a) classic: 7 symbols, daily-resampled pivots already in DB ---
    pivots_all = pd.read_sql("SELECT symbol, ts, price, pivot_type, threshold_pct, confirmed_ts FROM elliott_pivots ORDER BY symbol, threshold_pct, ts", conn)
    pivots_all["ts"] = pd.to_datetime(pivots_all["ts"], utc=True)
    pivots_all["confirmed_ts"] = pd.to_datetime(pivots_all["confirmed_ts"], utc=True)
    symbols = pivots_all["symbol"].unique()

    classic_results = []
    for symbol in symbols:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        sym_pivots = pivots_all[pivots_all["symbol"] == symbol]
        pivots_by_threshold = {}
        for thr in THRESHOLDS:
            sub = sym_pivots[sym_pivots["threshold_pct"] == thr]
            pivots_by_threshold[thr] = [(r.ts, r.price, r.pivot_type, r.confirmed_ts) for r in sub.itertuples()]
        res = run_study(f"CLASSIC - {symbol} (daily-resampled)", pivots_by_threshold, ohlcv,
                         horizons=[72, 168, 336, 720])
        if not res.empty:
            res["symbol"] = symbol
            classic_results.append(res)
    conn.close()

    if classic_results:
        pd.concat(classic_results, ignore_index=True).to_csv("elliott_wave_scenario_classic.csv", index=False)

    # --- (b) HF: BTC 15-min, recompute pivots in-memory ---
    conn = get_connection()
    ohlcv_hf = pd.read_sql(
        "SELECT ts, close FROM ohlcv_intraday WHERE symbol = 'BTCUSDT' AND interval_min = 15 ORDER BY ts", conn)
    conn.close()
    ohlcv_hf["ts"] = pd.to_datetime(ohlcv_hf["ts"], utc=True)
    closes = ohlcv_hf.set_index("ts")["close"]
    pivots_by_threshold_hf = {thr: detect_pivots(closes, thr) for thr in THRESHOLDS}
    hf_res = run_study("HF - BTCUSDT 15min", pivots_by_threshold_hf, ohlcv_hf, horizons=[1, 4, 12, 24, 72])
    if not hf_res.empty:
        hf_res.to_csv("elliott_wave_scenario_hf.csv", index=False)


if __name__ == "__main__":
    main()
