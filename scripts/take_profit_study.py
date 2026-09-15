"""
Honest test of adding a fixed take-profit target (exit at +X% from entry,
locking in gains) alongside the existing stop-loss and natural (score-
based) exit - whichever of the three fires first wins. Currently the
strategy has no take-profit at all; a winning position rides until either
the score flips or price craters.

Tests candidate take-profit levels [None (baseline), 10%, 15%, 20%, 30%]
on the full extended history, train/holdout split, same risk-management
defaults as the live strategy otherwise (stop_loss -8%, trend_filter_sma
720, cooldown_after_losses 3).

Usage:
    python scripts/take_profit_study.py --end-date 2025-11-30   # train
    python scripts/take_profit_study.py --start-date 2025-12-01 # holdout
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def backtest_take_profit(symbol, ohlcv, signals, take_profit, fee=SPOT_TAKER_FEE):
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    ohlcv["score"] = compute_score_series(signals, ohlcv)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()

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

        exit_idx = natural_exit_idx
        exit_reason = "natural"
        path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
        rel = path / entry_price - 1

        stop_hit = rel[rel <= STOP_LOSS]
        tp_hit = rel[rel >= take_profit] if take_profit is not None else pd.Series(dtype=float)

        candidates = []
        if not stop_hit.empty:
            candidates.append((stop_hit.index[0], "stop_loss"))
        if not tp_hit.empty:
            candidates.append((tp_hit.index[0], "take_profit"))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            exit_idx, exit_reason = candidates[0]

        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
        p2 = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (p2 / entry_price - 1).min()
        trades.append(dict(entry_ts=entry_ts, exit_ts=exit_ts, exit_reason=exit_reason,
                            hours_held=(exit_ts - entry_ts).total_seconds() / 3600,
                            gross_ret=gross_ret, ret=net_ret, worst_intratrade_dd=worst_intratrade_dd))

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

    stats = dict(
        symbol=symbol, n_trades=len(trades_df), win_rate=(trades_df["ret"] > 0).mean(),
        pct_tp_exits=(trades_df["exit_reason"] == "take_profit").mean(),
        strategy_total_return_net=total_return, trade_level_max_drawdown=trade_level_max_drawdown,
    )
    return trades_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    events = load_events_with_returns(conn)

    candidates = [None, 0.10, 0.15, 0.20, 0.30]
    print(f"{'Symbol':<10} {'take_profit':>11} {'n_trades':>9} {'win_rate':>9} {'pct_tp':>7} "
          f"{'total_ret':>10} {'max_dd':>8}")
    pooled = {c: [] for c in candidates}
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)

        for c in candidates:
            _, stats = backtest_take_profit(symbol, ohlcv, signals, c)
            if stats:
                label = "none" if c is None else f"{c:.0%}"
                print(f"{symbol:<10} {label:>11} {stats['n_trades']:>9} {stats['win_rate']:>9.1%} "
                      f"{stats['pct_tp_exits']:>7.1%} {stats['strategy_total_return_net']:>10.1%} "
                      f"{stats['trade_level_max_drawdown']:>8.1%}")
                pooled[c].append(stats["strategy_total_return_net"])
        print()
    conn.close()

    print("=" * 60)
    print("POOLED (mean total return across 7 symbols, by take-profit level)")
    print("=" * 60)
    for c in candidates:
        label = "none" if c is None else f"{c:.0%}"
        if pooled[c]:
            print(f"  take_profit={label:>6}: mean total return = {sum(pooled[c])/len(pooled[c]):.1%}")


if __name__ == "__main__":
    main()
