"""
Honest test of position pyramiding (adding to a winning position while
still held) vs the current single-entry-per-trade behavior.

Rule tested: while in a position, if price has risen `PYRAMID_THRESHOLD`
from the price of the LAST add (or the original entry, for the first add),
buy another $100 tranche - up to `PYRAMID_MAX_ADDS` additional tranches.
Stop-loss and the natural (score-based) exit apply to the WHOLE position
using its average cost basis, not per-tranche - the simplest, most common
convention. Uses the exact same signals as strategy.py (generate_signals +
compute_score_series) and the same risk-management defaults (stop_loss
-8%, trend_filter_sma 720, cooldown_after_losses 3) - only the position-
sizing behavior differs from strategy.py's own backtest().

Usage:
    python scripts/pyramid_study.py --symbols BTCUSDT,ETHUSDT,... --end-date 2025-11-30
    python scripts/pyramid_study.py --symbols ... --start-date 2025-12-01
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import (
    SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series, backtest,
)

TRANCHE_USDT = 100.0
PYRAMID_THRESHOLD = 0.05
PYRAMID_MAX_ADDS = 2
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168


def backtest_pyramid(symbol: str, ohlcv: pd.DataFrame, signals: pd.DataFrame, fee: float = SPOT_TAKER_FEE):
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

        # Walk the segment hour by hour, tracking tranches and average cost basis
        tranche_usdt = [TRANCHE_USDT]
        tranche_qty = [TRANCHE_USDT / entry_price]
        last_add_price = entry_price
        exit_idx = natural_exit_idx
        stop_hit = False
        for idx in range(entry_idx + 1, natural_exit_idx + 1):
            price = ohlcv["close"].iloc[idx]
            avg_cost = sum(tranche_usdt) / sum(tranche_qty)
            if price / avg_cost - 1 <= STOP_LOSS:
                exit_idx = idx
                stop_hit = True
                break
            if len(tranche_usdt) - 1 < PYRAMID_MAX_ADDS and price / last_add_price - 1 >= PYRAMID_THRESHOLD:
                tranche_usdt.append(TRANCHE_USDT)
                tranche_qty.append(TRANCHE_USDT / price)
                last_add_price = price
        if not stop_hit:
            exit_idx = natural_exit_idx

        exit_price = ohlcv["close"].iloc[exit_idx]
        total_usdt_in = sum(tranche_usdt) * (1 + fee)  # each tranche pays a buy fee
        total_qty = sum(tranche_qty)
        proceeds = total_qty * exit_price * (1 - fee)  # one sell fee on the whole position
        net_ret = proceeds / total_usdt_in - 1
        avg_cost = sum(tranche_usdt) / sum(tranche_qty)
        gross_ret = (exit_price - avg_cost) / avg_cost

        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        path = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (path / avg_cost - 1).min()
        trades.append(dict(
            entry_ts=entry_ts, exit_ts=exit_ts, n_tranches=len(tranche_usdt),
            avg_cost=avg_cost, exit_price=exit_price,
            hours_held=(exit_ts - entry_ts).total_seconds() / 3600,
            gross_ret=gross_ret, ret=net_ret, worst_intratrade_dd=worst_intratrade_dd,
        ))
        if cooldown_after_losses := COOLDOWN_AFTER_LOSSES:
            if net_ret < 0:
                consecutive_losses += 1
                if consecutive_losses >= cooldown_after_losses:
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

    start_price = ohlcv["close"].iloc[0]
    end_price = ohlcv["close"].iloc[-1]
    buy_hold_return = end_price / start_price - 1

    stats = dict(
        symbol=symbol, n_trades=len(trades_df),
        avg_tranches=trades_df["n_tranches"].mean(),
        pct_trades_pyramided=(trades_df["n_tranches"] > 1).mean(),
        win_rate=(trades_df["ret"] > 0).mean(),
        avg_trade_return_net=trades_df["ret"].mean(),
        worst_intratrade_drawdown=trades_df["worst_intratrade_dd"].min(),
        trade_level_max_drawdown=trade_level_max_drawdown,
        strategy_total_return_net=total_return,
        buy_hold_return=buy_hold_return,
    )
    return trades_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str,
                         default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    conn = get_connection()
    events = load_events_with_returns(conn)

    print(f"{'Symbol':<10} {'Variant':<10} {'n_trades':>9} {'avg_tranches':>13} {'win_rate':>9} "
          f"{'total_ret':>10} {'max_dd':>8} {'buy_hold':>9}")
    for symbol in symbols:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)

        # baseline: strategy.py's own single-entry backtest, same risk params
        _, base_stats = backtest(symbol, ohlcv, signals, capital=10_000.0,
                                  stop_loss=STOP_LOSS, trend_filter_sma=TREND_FILTER_SMA,
                                  cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
        if base_stats:
            print(f"{symbol:<10} {'baseline':<10} {base_stats['n_trades']:>9} {'1.00':>13} "
                  f"{base_stats['win_rate']:>9.1%} {base_stats['strategy_total_return_net']:>10.1%} "
                  f"{base_stats['trade_level_max_drawdown']:>8.1%} {base_stats['buy_hold_return']:>9.1%}")

        _, pyr_stats = backtest_pyramid(symbol, ohlcv, signals)
        if pyr_stats:
            print(f"{symbol:<10} {'pyramid':<10} {pyr_stats['n_trades']:>9} "
                  f"{pyr_stats['avg_tranches']:>13.2f} {pyr_stats['win_rate']:>9.1%} "
                  f"{pyr_stats['strategy_total_return_net']:>10.1%} "
                  f"{pyr_stats['trade_level_max_drawdown']:>8.1%} {pyr_stats['buy_hold_return']:>9.1%}")
        print()
    conn.close()


if __name__ == "__main__":
    main()
