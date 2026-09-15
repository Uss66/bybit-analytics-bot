"""
Honest test of a volatility-scaled stop-loss (ATR-style) instead of the
fixed -8% used for every symbol regardless of its own typical volatility.

At each entry, compute the symbol's trailing 30-day realized volatility
(std of hourly returns, scaled to a daily-equivalent via sqrt(24)), and
set that trade's stop_loss = -k * daily_vol_at_entry. k is calibrated so
that at BTC's own AVERAGE historical daily-equivalent volatility, the
stop comes out to approximately -8% (matching the current baseline for
continuity) - so BTC's stops stay close to today's, while more volatile
alts (SOL, DOGE, etc.) get proportionally wider stops and less volatile
periods get tighter ones.

Usage:
    python scripts/vol_stop_study.py --end-date 2025-11-30   # train
    python scripts/vol_stop_study.py --start-date 2025-12-01 # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

FIXED_STOP = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
VOL_WINDOW_HOURS = 720  # 30 days
STOP_K = None  # calibrated below from BTC's own history


def calibrate_k(conn):
    """k such that -k * BTC's mean historical daily-equivalent vol ~= -0.08."""
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol='BTCUSDT' ORDER BY ts", conn)
    ret = ohlcv["close"].pct_change()
    daily_vol = ret.rolling(VOL_WINDOW_HOURS).std() * np.sqrt(24)
    mean_vol = daily_vol.mean()
    k = 0.08 / mean_vol
    print(f"Calibration: BTC mean 30d daily-equivalent vol = {mean_vol:.4f}, k = {k:.4f}")
    return k


def backtest_vol_stop(symbol, full_ohlcv, signals, use_vol_stop, k, start_date=None, end_date=None,
                       fee=SPOT_TAKER_FEE):
    # compute score/SMA/volatility on the FULL series first, so rows near the
    # start of a requested train/holdout window still have valid rolling
    # lookback data instead of NaN from a truncated history, then slice.
    ohlcv = full_ohlcv.sort_values("ts").reset_index(drop=True)
    ohlcv["score"] = compute_score_series(signals, ohlcv)
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()
    ret = ohlcv["close"].pct_change()
    ohlcv["daily_vol"] = ret.rolling(VOL_WINDOW_HOURS).std() * np.sqrt(24)

    if start_date:
        ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(end_date, tz="UTC")]
    ohlcv = ohlcv.reset_index(drop=True)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()

    trades = []
    consecutive_losses = 0
    cooldown_until_ts = None
    for seg_id, seg in ohlcv[ohlcv["is_long"]].groupby("seg_id"):
        entry_idx = seg.index[0]
        natural_exit_idx = min(seg.index[-1] + 1, len(ohlcv) - 1)
        entry_price = ohlcv["close"].iloc[entry_idx]
        entry_ts_candidate = ohlcv["ts"].iloc[entry_idx]
        if cooldown_until_ts is not None and entry_ts_candidate < cooldown_until_ts:
            continue
        sma = ohlcv["sma"].iloc[entry_idx]
        if pd.notna(sma) and entry_price < sma:
            continue

        if use_vol_stop:
            vol_at_entry = ohlcv["daily_vol"].iloc[entry_idx]
            stop_loss = -k * vol_at_entry if pd.notna(vol_at_entry) else FIXED_STOP
            stop_loss = max(stop_loss, -0.30)  # sanity cap - never absurdly wide
        else:
            stop_loss = FIXED_STOP

        exit_idx = natural_exit_idx
        path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
        hit = path[path / entry_price - 1 <= stop_loss]
        if not hit.empty:
            exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
        p2 = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (p2 / entry_price - 1).min()
        trades.append(dict(entry_ts=entry_ts, exit_ts=exit_ts, stop_used=stop_loss, ret=net_ret,
                            worst_intratrade_dd=worst_intratrade_dd))

        if net_ret < 0:
            consecutive_losses += 1
            if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                cooldown_until_ts = exit_ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                consecutive_losses = 0
        else:
            consecutive_losses = 0

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {}

    capital = 10_000.0
    equity_curve = [capital]
    for r in trades_df["ret"]:
        equity_curve.append(equity_curve[-1] * (1 + r))
    equity = equity_curve[-1]
    total_return = equity / capital - 1
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    trade_level_max_drawdown = ((equity_series - running_peak) / running_peak).min()

    stats = dict(symbol=symbol, n_trades=len(trades_df), win_rate=(trades_df["ret"] > 0).mean(),
                 avg_stop=trades_df["stop_used"].mean(), strategy_total_return_net=total_return,
                 trade_level_max_drawdown=trade_level_max_drawdown)
    return trades_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    events = load_events_with_returns(conn)
    k = calibrate_k(conn)

    print(f"{'Symbol':<10} {'variant':<10} {'n_trades':>9} {'win_rate':>9} {'avg_stop':>9} "
          f"{'total_ret':>10} {'max_dd':>8}")
    pooled = {"fixed": [], "vol_scaled": []}
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        signals = generate_signals(events, symbol)

        for variant, use_vol in [("fixed", False), ("vol_scaled", True)]:
            _, stats = backtest_vol_stop(symbol, ohlcv, signals, use_vol, k,
                                          start_date=args.start_date, end_date=args.end_date)
            if stats:
                print(f"{symbol:<10} {variant:<10} {stats['n_trades']:>9} {stats['win_rate']:>9.1%} "
                      f"{stats['avg_stop']:>9.1%} {stats['strategy_total_return_net']:>10.1%} "
                      f"{stats['trade_level_max_drawdown']:>8.1%}")
                pooled[variant].append(stats["strategy_total_return_net"])
        print()
    conn.close()

    print("=" * 60)
    print("POOLED (mean total return across 7 symbols)")
    print("=" * 60)
    for variant in ["fixed", "vol_scaled"]:
        if pooled[variant]:
            print(f"  {variant}: mean total return = {sum(pooled[variant])/len(pooled[variant]):.1%}")


if __name__ == "__main__":
    main()
