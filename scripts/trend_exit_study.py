"""
Should a coin be sold on ITS OWN trend breaking, instead of waiting for
the market-wide signal to die? (2026-09-30, user's hypothesis)

The observation behind it is structurally correct. Every rule that can
END a position is macro:

    fgi_greed          +3   Fear & Greed index      all 7 symbols
    dvol_drop_extreme  +2   Deribit implied vol     all 7
    dxy_up_extreme     +2   dollar index            ETH, SOL
    dvol_spike_extreme -2   Deribit implied vol     all 7
    dxy_down_extreme   -2   dollar index            ETH, SOL
    technical_hype     -1   news sentiment          BTC, ETH, SOL

So when the score dies it dies for everything at once, and the whole book
closes together regardless of how each individual coin was behaving. The
one genuinely per-symbol input in the system - SMA720, price against its
own 30-day trend - is used ONLY as an entry gate (strategy.py:443-445).
A position is held all the way down through its own trend as long as the
macro signal holds.

This is NOT the rejected rotation study. That evicted the weakest holding
by SCORE, which is a market-wide quantity and therefore a weak
cross-sectional signal (project_rotation_study). This exits on the coin's
OWN price versus its OWN trend - the per-symbol dimension the strategy
currently ignores after entry.

Variants (small grid on purpose):
  baseline        - today: exit on the -8% stop or score <= 0
  trend_exit      - also exit the moment close < SMA720
  trend_24h       - also exit after close has been under SMA720 for 24
                    consecutive hours (ignores a one-candle dip)
  trend_buffer    - also exit when close < SMA720 * 0.98, a 2% tolerance
                    band so the exit is not triggered by hugging the line

A freed slot is not wasted: the cap lets another symbol take it, which is
the actual upside being measured - capital leaving a coin that broke down
for one still in trend. Re-entry into the exited coin is blocked by the
same SMA720 gate until its price recovers above the trend, so there is no
churn loop.

Engine is slot_cap_study's cash-based portfolio walk, untouched otherwise:
cap 3, cash/free-slots sizing, dip-rebuy dip3_bounce2, 168h cooldown
after 3 losses, taker fee both legs, hourly mark-to-market.

Usage:
    python scripts/trend_exit_study.py --end-date 2025-11-30    # train
    python scripts/trend_exit_study.py --start-date 2025-12-01  # holdout
"""
import argparse
import os

import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import (
    SYMBOLS, STOP_LOSS, COOLDOWN_AFTER_LOSSES, COOLDOWN_HOURS,
    DIP_REBUY_THRESHOLD, DIP_REBUY_REBOUND, DIP_REBUY_MAX_ADDS,
    load_panel,
)
from slot_cap_study import CashPosition, STARTING_CASH
from strategy import SPOT_TAKER_FEE

MAX_POSITIONS = 3

VARIANTS = {
    "baseline (сейчас)": dict(mode="none"),
    "выход под трендом": dict(mode="instant", buffer=0.0),
    "под трендом 24ч": dict(mode="confirmed", hours=24, buffer=0.0),
    "под трендом -2%": dict(mode="instant", buffer=0.02),
}


def simulate(close, score, sma, variant, fee=SPOT_TAKER_FEE, starting_cash=STARTING_CASH):
    cash = starting_cash
    open_positions: dict[str, CashPosition] = {}
    below_since: dict[str, int] = {}      # hours the symbol has been under its trend
    cooldown_until = {s: None for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    trades, equity_curve = [], []

    col = {s: i for i, s in enumerate(SYMBOLS)}
    close_v, score_v, sma_v = close.values, score.values, sma.values
    mark_v = close.ffill().values
    mode = variant["mode"]
    buffer = variant.get("buffer", 0.0)
    need_hours = variant.get("hours", 0)

    for row, ts in enumerate(close.index):
        # how long each symbol has been trading under its own trend
        for symbol in SYMBOLS:
            i = col[symbol]
            price, trend = close_v[row, i], sma_v[row, i]
            if np.isnan(price) or np.isnan(trend):
                continue
            below_since[symbol] = below_since.get(symbol, 0) + 1 if price < trend * (1 - buffer) else 0

        for symbol in list(open_positions):
            price = close_v[row, col[symbol]]
            if np.isnan(price):
                continue
            pos = open_positions[symbol]

            hit_stop = price / pos.avg_cost - 1 <= STOP_LOSS
            signal_dead = not score_v[row, col[symbol]] > 0
            broke_trend = False
            if mode == "instant":
                broke_trend = below_since.get(symbol, 0) >= 1
            elif mode == "confirmed":
                broke_trend = below_since.get(symbol, 0) >= need_hours

            if hit_stop or signal_dead or broke_trend:
                proceeds = pos.proceeds(price, fee)
                cash += proceeds
                ret = proceeds / pos.spent - 1
                reason = "stop" if hit_stop else ("signal" if signal_dead else "trend")
                trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=ts, spent=pos.spent,
                                   ret=ret, pnl=proceeds - pos.spent, exit_reason=reason,
                                   hours=(ts - pos.entry_ts).total_seconds() / 3600))
                del open_positions[symbol]
                if ret < 0:
                    consecutive_losses[symbol] += 1
                    if consecutive_losses[symbol] >= COOLDOWN_AFTER_LOSSES:
                        cooldown_until[symbol] = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                        consecutive_losses[symbol] = 0
                else:
                    consecutive_losses[symbol] = 0
                continue

            pos.running_low = min(pos.running_low, price)
            if pos.n_tranches - 1 < DIP_REBUY_MAX_ADDS and cash >= pos.tranche_cash:
                dip = pos.running_low / pos.avg_cost - 1
                bounce = price / pos.running_low - 1
                if dip <= DIP_REBUY_THRESHOLD and bounce >= DIP_REBUY_REBOUND:
                    pos.add(price, pos.tranche_cash, fee)
                    cash -= pos.tranche_cash

        candidates = []
        for symbol in SYMBOLS:
            if symbol in open_positions:
                continue
            i = col[symbol]
            price, sc, trend = close_v[row, i], score_v[row, i], sma_v[row, i]
            if np.isnan(price) or not sc > 0:
                continue
            cd = cooldown_until[symbol]
            if cd is not None and ts < cd:
                continue
            if not np.isnan(trend) and price < trend:
                continue      # same gate as live: never enter under the trend
            candidates.append((sc, symbol, price))
        candidates.sort(key=lambda c: -c[0])

        for sc, symbol, price in candidates:
            free_slots = MAX_POSITIONS - len(open_positions)
            if free_slots <= 0:
                break
            stake = cash / free_slots
            if stake < 10:
                break
            open_positions[symbol] = CashPosition(symbol, ts, price, stake, fee)
            cash -= stake

        book = sum(p.qty * mark_v[row, col[s]] for s, p in open_positions.items()
                   if not np.isnan(mark_v[row, col[s]]))
        equity_curve.append(cash + book)

    last = close.index[-1]
    for symbol, pos in list(open_positions.items()):
        price = close[symbol].dropna().iloc[-1]
        proceeds = pos.proceeds(price, fee)
        trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=last, spent=pos.spent,
                           ret=proceeds / pos.spent - 1, pnl=proceeds - pos.spent,
                           exit_reason="open_at_end",
                           hours=(last - pos.entry_ts).total_seconds() / 3600))

    equity = pd.Series(equity_curve, index=close.index)
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    t = pd.DataFrame(trades)
    total_ret = equity.iloc[-1] / starting_cash - 1
    by_reason = t["exit_reason"].value_counts().to_dict() if len(t) else {}
    return dict(
        n_trades=len(t),
        by_trend=by_reason.get("trend", 0),
        by_stop=by_reason.get("stop", 0),
        win_rate=(t["ret"] > 0).mean() if len(t) else float("nan"),
        total_ret=total_ret,
        max_dd=max_dd,
        ratio=total_ret / abs(max_dd) if max_dd < 0 else float("inf"),
        avg_hours=t["hours"].mean() if len(t) else float("nan"),
    ), t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    close, score, sma = load_panel(conn, args.start_date, args.end_date)
    conn.close()
    period = f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}"
    print(f"Период: {period} ({len(close)} часов), кэп {MAX_POSITIONS}\n")

    print(f"{'вариант':>20} {'сделок':>7} {'по тренду':>10} {'по стопу':>9} {'win':>6} "
          f"{'доходность':>11} {'max_dd':>8} {'ret/dd':>7} {'ср.часов':>9}")
    rows = []
    for label, variant in VARIANTS.items():
        stats, _ = simulate(close, score, sma, variant)
        rows.append(dict(period=period, variant=label, **stats))
        print(f"{label:>20} {stats['n_trades']:>7d} {stats['by_trend']:>10d} {stats['by_stop']:>9d} "
              f"{stats['win_rate']:>6.1%} {stats['total_ret']:>11.1%} {stats['max_dd']:>8.1%} "
              f"{stats['ratio']:>7.2f} {stats['avg_hours']:>9.0f}")

    pd.DataFrame(rows).to_csv("trend_exit_summary.csv", mode="a", index=False,
                              header=not os.path.exists("trend_exit_summary.csv"))
    print("\nДописано в trend_exit_summary.csv")


if __name__ == "__main__":
    main()
