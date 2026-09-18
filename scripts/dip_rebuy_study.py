"""
Honest test of a DIFFERENT pyramiding hypothesis than the one already
rejected (see project_pyramiding_findings memory / scripts/pyramid_study.py,
which added tranches on continued STRENGTH - price +5% from the last add -
and was rejected decisively, worse in all 14 symbol x period combinations).

User's new hypothesis (2026-09-18): add to a held position on a CONFIRMED
BOUNCE after a dip, not on strength. Mechanistically this is a different
risk profile from the rejected version: the original rule added at a
HIGHER price than the running average, which pulls the average cost basis
UP and therefore tightens the -8% stop's effective distance from the
current price (the actual failure mechanism found). Adding on a dip adds
at a LOWER price than the average, pulling the average DOWN and therefore
WIDENING the stop's effective distance - the opposite mechanical effect,
so the prior rejection does not mechanically transfer to this variant.
Worth testing on its own merits, not assumed to work OR assumed to fail.

Rule tested: while in a position, track the running low since entry (or
since the last add). If that running low has dropped at least
DIP_THRESHOLD below the position's average cost basis, AND price has
since recovered at least REBOUND_THRESHOLD off that running low (a
confirmed bounce, not just "still falling"), add a $100 tranche - up to
MAX_ADDS extra tranches. Stop-loss/exit apply to the whole position at
its average cost basis, same convention as pyramid_study.py, for a fair
comparison. Same signals/risk defaults as the live strategy.

Usage:
    python scripts/dip_rebuy_study.py --end-date 2025-11-30   # train
    python scripts/dip_rebuy_study.py --start-date 2025-12-01 # holdout
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series, backtest

TRANCHE_USDT = 100.0
MAX_ADDS = 2
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168

# Modest grid, not an exhaustive tuning sweep (this project's own standing
# practice against multiple-testing/overfitting - see min_cooldown_study.py)
PARAM_GRID = [
    dict(dip=-0.03, rebound=0.02, label="dip3_bounce2"),
    dict(dip=-0.05, rebound=0.03, label="dip5_bounce3"),
]


def backtest_dip_rebuy(symbol: str, ohlcv: pd.DataFrame, signals: pd.DataFrame,
                        dip_threshold: float, rebound_threshold: float, fee: float = SPOT_TAKER_FEE):
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

        tranche_usdt = [TRANCHE_USDT]
        tranche_qty = [TRANCHE_USDT / entry_price]
        running_low = entry_price  # resets after each add - look for a FRESH dip+bounce before adding again
        exit_idx = natural_exit_idx
        stop_hit = False
        for idx in range(entry_idx + 1, natural_exit_idx + 1):
            price = ohlcv["close"].iloc[idx]
            avg_cost = sum(tranche_usdt) / sum(tranche_qty)
            if price / avg_cost - 1 <= STOP_LOSS:
                exit_idx = idx
                stop_hit = True
                break
            running_low = min(running_low, price)
            dip_pct = running_low / avg_cost - 1
            bounce_pct = price / running_low - 1
            if (len(tranche_usdt) - 1 < MAX_ADDS and dip_pct <= dip_threshold
                    and bounce_pct >= rebound_threshold):
                tranche_usdt.append(TRANCHE_USDT)
                tranche_qty.append(TRANCHE_USDT / price)
                running_low = price  # fresh tracking window for the next potential add
        if not stop_hit:
            exit_idx = natural_exit_idx

        exit_price = ohlcv["close"].iloc[exit_idx]
        total_usdt_in = sum(tranche_usdt) * (1 + fee)
        total_qty = sum(tranche_qty)
        proceeds = total_qty * exit_price * (1 - fee)
        net_ret = proceeds / total_usdt_in - 1
        avg_cost = sum(tranche_usdt) / sum(tranche_qty)

        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        path = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (path / avg_cost - 1).min()
        trades.append(dict(
            entry_ts=entry_ts, exit_ts=exit_ts, n_tranches=len(tranche_usdt),
            avg_cost=avg_cost, exit_price=exit_price,
            hours_held=(exit_ts - entry_ts).total_seconds() / 3600,
            ret=net_ret, worst_intratrade_dd=worst_intratrade_dd,
        ))
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
        symbol=symbol, n_trades=len(trades_df),
        avg_tranches=trades_df["n_tranches"].mean(),
        pct_trades_added=(trades_df["n_tranches"] > 1).mean(),
        win_rate=(trades_df["ret"] > 0).mean(),
        worst_intratrade_drawdown=trades_df["worst_intratrade_dd"].min(),
        trade_level_max_drawdown=trade_level_max_drawdown,
        strategy_total_return_net=total_return,
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

    print(f"{'Symbol':<10} {'Variant':<14} {'n_trades':>9} {'avg_tranches':>13} {'win_rate':>9} "
          f"{'total_ret':>10} {'max_dd':>8}")
    summary_rows = []
    for symbol in symbols:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)

        _, base_stats = backtest(symbol, ohlcv, signals, capital=10_000.0,
                                  stop_loss=STOP_LOSS, trend_filter_sma=TREND_FILTER_SMA,
                                  cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
        if base_stats:
            print(f"{symbol:<10} {'baseline':<14} {base_stats['n_trades']:>9} {'1.00':>13} "
                  f"{base_stats['win_rate']:>9.1%} {base_stats['strategy_total_return_net']:>10.1%} "
                  f"{base_stats['trade_level_max_drawdown']:>8.1%}")
            summary_rows.append(dict(symbol=symbol, variant="baseline",
                                      total_ret=base_stats['strategy_total_return_net'],
                                      max_dd=base_stats['trade_level_max_drawdown']))

        for params in PARAM_GRID:
            _, stats = backtest_dip_rebuy(symbol, ohlcv, signals, params["dip"], params["rebound"])
            if stats:
                print(f"{symbol:<10} {params['label']:<14} {stats['n_trades']:>9} "
                      f"{stats['avg_tranches']:>13.2f} {stats['win_rate']:>9.1%} "
                      f"{stats['strategy_total_return_net']:>10.1%} "
                      f"{stats['trade_level_max_drawdown']:>8.1%}")
                summary_rows.append(dict(symbol=symbol, variant=params["label"],
                                          total_ret=stats['strategy_total_return_net'],
                                          max_dd=stats['trade_level_max_drawdown']))
        print()
    conn.close()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("dip_rebuy_summary.csv", index=False)
    print("Summary written to dip_rebuy_summary.csv")


if __name__ == "__main__":
    main()
