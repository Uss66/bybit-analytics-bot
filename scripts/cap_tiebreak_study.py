"""
Coin-diversification study (2026-09-15): the portfolio cap (MAX_POSITIONS=3,
live since 2026-09-13) currently breaks ties among simultaneously-firing
symbols by `entry_score` - but score is a weak differentiator since all 3
active rules (fgi_greed/DXY/DVOL) are global/market-wide, so score
differences mostly just reflect which subset of rules happens to apply to
a symbol, not a real per-symbol signal (see strategy.py's
simulate_portfolio docstring, and Opus's original critique in
project_technical_indicators memory).

Tests 3 alternative tiebreak criteria against the live baseline (score),
using the exact same verified segment-based portfolio simulation as the
live cap - reuses compute_candidate_segments/simulate_portfolio directly
via import, no reimplementation:

  1. entry_score          - BASELINE (what's live now)
  2. trend_strength        - (price/SMA720 - 1) at entry: among symbols
                              whose trend filter already passes, prefer
                              the one furthest above its own trend, not
                              just barely passing
  3. cross_sectional_rank  - trailing 168h return rank among the 7 symbols
                              (the same feature tested and REJECTED as an
                              ENTRY signal in project_technical_indicators
                              - this tests it in a DIFFERENT role, ranking
                              among already-qualified candidates rather
                              than triggering entries on its own, so the
                              earlier rejection doesn't directly apply)
  4. inv_volatility        - inverse of trailing 30-day realized volatility
                              (prefer the calmer coin among candidates -
                              classic risk-parity style diversification
                              logic, capital efficiency argument)

Usage:
    python scripts/cap_tiebreak_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from strategy import (
    load_events_with_returns, generate_signals, compute_candidate_segments, simulate_portfolio,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
MAX_POSITIONS = 3
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)

    closes_by_symbol = {}
    ohlcv_by_symbol = {}
    for sym in SYMBOLS:
        df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": sym})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        ohlcv_by_symbol[sym] = df
        closes_by_symbol[sym] = df.set_index("ts")["close"]
    conn.close()

    # --- cross-sectional 168h-return rank matrix, aligned to BTC's hourly timeline ---
    common_ts = ohlcv_by_symbol["BTCUSDT"]["ts"]
    ret_matrix = pd.DataFrame({sym: s.pct_change(168) for sym, s in closes_by_symbol.items()})
    ret_matrix = ret_matrix.reindex(common_ts).ffill()
    rank_matrix = ret_matrix.rank(axis=1, ascending=False)  # 1 = highest return
    cross_sectional_strength = {sym: (8 - rank_matrix[sym]) for sym in SYMBOLS}  # higher = better rank

    # --- trailing 30d realized volatility per symbol ---
    vol_by_symbol = {}
    for sym in SYMBOLS:
        df = ohlcv_by_symbol[sym].copy()
        df["hret"] = df["close"].pct_change()
        df["vol30d"] = df["hret"].rolling(30 * 24, min_periods=30 * 24 // 3).std()
        vol_by_symbol[sym] = df.set_index("ts")["vol30d"]

    all_segments = []
    for sym in SYMBOLS:
        signals = generate_signals(events, sym)
        segs = compute_candidate_segments(sym, ohlcv_by_symbol[sym], signals, stop_loss=STOP_LOSS,
                                           trend_filter_sma=TREND_FILTER_SMA)
        cs_series = cross_sectional_strength[sym]
        vol_series = vol_by_symbol[sym]
        for seg in segs:
            entry_ts = seg["entry_ts"]
            cs_val = cs_series.asof(entry_ts) if entry_ts >= cs_series.index[0] else None
            vol_val = vol_series.asof(entry_ts) if entry_ts >= vol_series.index[0] else None
            seg["cross_sectional_strength"] = float(cs_val) if pd.notna(cs_val) else None
            seg["inv_volatility"] = float(1.0 / vol_val) if pd.notna(vol_val) and vol_val > 0 else None
        all_segments.extend(segs)

    criteria = ["entry_score", "trend_strength", "cross_sectional_strength", "inv_volatility"]
    print(f"{'tiebreak':<26}{'period':>8}{'n':>6}{'total_ret':>11}{'max_dd':>9}{'ratio':>8}{'rejected':>10}")
    for crit in criteria:
        for period_label, seg_filter in [
            ("train", lambda s: s["entry_ts"] < TRAIN_END),
            ("holdout", lambda s: s["entry_ts"] >= TRAIN_END),
        ]:
            period_segments = [s for s in all_segments if seg_filter(s)]
            trades_df, rejection_log = simulate_portfolio(period_segments, MAX_POSITIONS,
                                                            cooldown_after_losses=COOLDOWN_AFTER_LOSSES,
                                                            tiebreak_key=crit)
            if trades_df.empty:
                print(f"{crit:<26}{period_label:>8}{'0':>6}{'n/a':>11}{'n/a':>9}{'n/a':>8}{len(rejection_log):>10}")
                continue
            trades_df = trades_df.sort_values("exit_ts")
            per_slot_capital = 10000.0 / len(SYMBOLS)
            equity_curve = [per_slot_capital * MAX_POSITIONS]
            for r in trades_df["ret"]:
                equity_curve.append(equity_curve[-1] + per_slot_capital * r)
            equity_series = pd.Series(equity_curve)
            total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            running_peak = equity_series.cummax()
            max_dd = ((equity_series - running_peak) / running_peak).min()
            ratio = total_return / abs(max_dd) if max_dd != 0 else float("nan")
            print(f"{crit:<26}{period_label:>8}{len(trades_df):>6}{total_return:>11.2%}{max_dd:>9.2%}"
                  f"{ratio:>8.2f}{len(rejection_log):>10}")


if __name__ == "__main__":
    main()
