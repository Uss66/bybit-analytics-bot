"""
Honest test of adding a MINIMUM cooldown after EVERY exit (not just after
a losing streak, which is what strategy.py's cooldown_after_losses already
does), to see if it reduces rapid re-entry churning without hurting
returns. Prompted by the user noticing several exit/re-entry cycles within
hours of each other in a scenario check.

Tests candidate values [0 (baseline), 6, 12, 24, 48] hours. The new
min-cooldown and the existing loss-streak cooldown both apply - whichever
lockout ends later wins, i.e. this is purely additive, never removes the
existing protection.

Usage:
    python scripts/min_cooldown_study.py --end-date 2025-11-30   # train
    python scripts/min_cooldown_study.py --start-date 2025-12-01 # holdout
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
LOSS_COOLDOWN_HOURS = 168
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def backtest_min_cooldown(symbol, ohlcv, signals, min_cooldown_hours, fee=SPOT_TAKER_FEE):
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
        if STOP_LOSS is not None:
            path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
            hit = path[path / entry_price - 1 <= STOP_LOSS]
            if not hit.empty:
                exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
        path = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (path / entry_price - 1).min()
        trades.append(dict(entry_ts=entry_ts, exit_ts=exit_ts, hours_held=(exit_ts - entry_ts).total_seconds() / 3600,
                            gross_ret=gross_ret, ret=net_ret, worst_intratrade_dd=worst_intratrade_dd))

        # NEW: minimum cooldown after EVERY exit, regardless of win/loss
        min_cd_until = exit_ts + pd.Timedelta(hours=min_cooldown_hours) if min_cooldown_hours else None

        if net_ret < 0:
            consecutive_losses += 1
            if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                loss_cd_until = exit_ts + pd.Timedelta(hours=LOSS_COOLDOWN_HOURS)
                consecutive_losses = 0
            else:
                loss_cd_until = None
        else:
            consecutive_losses = 0
            loss_cd_until = None

        candidates = [t for t in [min_cd_until, loss_cd_until] if t is not None]
        cooldown_until_ts = max(candidates) if candidates else None

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
        pct_short_trades=(trades_df["hours_held"] <= 6).mean(),
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

    candidate_hours = [0, 6, 12, 24, 48]
    print(f"{'Symbol':<10} {'min_cd_h':>9} {'n_trades':>9} {'win_rate':>9} {'pct<=6h':>8} "
          f"{'total_ret':>10} {'max_dd':>8}")
    pooled = {h: [] for h in candidate_hours}
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)

        for h in candidate_hours:
            _, stats = backtest_min_cooldown(symbol, ohlcv, signals, h)
            if stats:
                print(f"{symbol:<10} {h:>9} {stats['n_trades']:>9} {stats['win_rate']:>9.1%} "
                      f"{stats['pct_short_trades']:>8.1%} {stats['strategy_total_return_net']:>10.1%} "
                      f"{stats['trade_level_max_drawdown']:>8.1%}")
                pooled[h].append(stats["strategy_total_return_net"])
        print()
    conn.close()

    print("=" * 60)
    print("POOLED (mean total return across 7 symbols, by min-cooldown)")
    print("=" * 60)
    for h in candidate_hours:
        if pooled[h]:
            print(f"  min_cooldown={h:>3}h: mean total return = {sum(pooled[h])/len(pooled[h]):.1%}")


if __name__ == "__main__":
    main()
