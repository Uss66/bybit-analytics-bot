"""
Honest test of scaling position size with signal strength (score
magnitude at entry) instead of always deploying 100% of capital.

IMPORTANT design note (fixed after an initial flawed version diluted a
fixed $100 against a separate $10,000 bankroll, which mathematically
swamped any real difference between variants): this must stay consistent
with how strategy.py's own backtest() works - 100% of a symbol's OWN
compounding capital deployed on every trade, no leverage (spot-only, can't
deploy more than 100%). So "size up on conviction" here means: deploy a
FRACTION of capital scaled by score, with the remainder sitting idle
(uninvested) for that trade - never more than 100%. Rule tested:
fraction = min(1.0, score / 5.0), i.e. full size only once score reaches
5 (e.g. fgi_greed + dvol both firing), partial size (40%) at the minimum
entry score of 2 (a single dvol rule alone).

Usage:
    python scripts/position_sizing_study.py --end-date 2025-11-30   # train
    python scripts/position_sizing_study.py --start-date 2025-12-01 # holdout
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
SCORE_FOR_FULL_SIZE = 5.0


def backtest_sized(symbol, ohlcv, signals, scale_by_score, capital: float, fee=SPOT_TAKER_FEE):
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
        path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
        hit = path[path / entry_price - 1 <= STOP_LOSS]
        if not hit.empty:
            exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1

        if scale_by_score:
            score_at_entry = ohlcv["score"].iloc[entry_idx]
            fraction = min(1.0, score_at_entry / SCORE_FOR_FULL_SIZE)
        else:
            fraction = 1.0

        # only the deployed FRACTION of capital compounds this trade; the rest
        # sits idle (0% return) - no leverage, capital can't exceed 100% deployed
        trade_ret_on_full_capital = fraction * net_ret

        p2 = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (p2 / entry_price - 1).min()
        trades.append(dict(entry_ts=entry_ts, exit_ts=exit_ts, fraction=fraction, ret=net_ret,
                            capital_ret=trade_ret_on_full_capital, worst_intratrade_dd=worst_intratrade_dd))

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

    equity_curve = [capital]
    for r in trades_df["capital_ret"]:
        equity_curve.append(equity_curve[-1] * (1 + r))
    equity = equity_curve[-1]
    total_return = equity / capital - 1
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    trade_level_max_drawdown = ((equity_series - running_peak) / running_peak).min()

    stats = dict(symbol=symbol, n_trades=len(trades_df), win_rate=(trades_df["ret"] > 0).mean(),
                 avg_fraction=trades_df["fraction"].mean(), strategy_total_return_net=total_return,
                 trade_level_max_drawdown=trade_level_max_drawdown)
    return trades_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    events = load_events_with_returns(conn)

    print(f"{'Symbol':<10} {'variant':<12} {'n_trades':>9} {'win_rate':>9} {'avg_fraction':>12} "
          f"{'total_ret':>10} {'max_dd':>8}")
    pooled = {"fixed": [], "scaled": []}
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)

        for variant, scale in [("fixed", False), ("scaled", True)]:
            _, stats = backtest_sized(symbol, ohlcv, signals, scale, capital=10_000.0)
            if stats:
                print(f"{symbol:<10} {variant:<12} {stats['n_trades']:>9} {stats['win_rate']:>9.1%} "
                      f"{stats['avg_fraction']:>12.1%} {stats['strategy_total_return_net']:>10.1%} "
                      f"{stats['trade_level_max_drawdown']:>8.1%}")
                pooled[variant].append(stats["strategy_total_return_net"])
        print()
    conn.close()

    print("=" * 60)
    print("POOLED (mean total return across 7 symbols)")
    print("=" * 60)
    for variant in ["fixed", "scaled"]:
        if pooled[variant]:
            print(f"  {variant}: mean total return = {sum(pooled[variant])/len(pooled[variant]):.1%}")


if __name__ == "__main__":
    main()
