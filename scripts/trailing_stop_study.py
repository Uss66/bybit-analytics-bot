"""
Should the stop follow the price up? (2026-09-22, user's hypothesis)

The observation that prompted it is concrete and correct: BTC sits at
86,400 against an average cost of 70,280, so its -8%-from-average stop is
25% BELOW the current price. A crash would hand back the entire +23%
before the stop even starts to protect anything. On a strategy that
takes few trades and holds them for weeks, a stop anchored to the entry
falls further and further behind the position it is supposed to guard.

This is NOT one of the stop variants already tested:
  * adaptive_stop_study.py pegged the stop to a confirmed swing low,
  * vol_stop_study.py scaled the stop's WIDTH by realized volatility,
  * take-profit (four_hypotheses, rejected) exits at a fixed target and
    gives up the rest of the move.
All three answer "where should the stop start". This one answers
"should it RATCHET", which nothing here has measured.

Variants (deliberately few - the project's standing defence against
fitting a grid, see min_cooldown_study.py):
  fixed        - today's rule: -8% from average cost, never moves up
  trail8       - stop = max(avg cost, highest close since entry) * 0.92
  trail12      - same, but 12% below the peak: looser, gives the position
                 more room to breathe before the ratchet bites
  breakeven10  - fixed -8% until the position is +10% up, then the stop
                 moves to the average cost and stays there. The cheapest
                 possible version of the idea: it only ever removes the
                 possibility of turning a winner into a loser.

Everything else is identical to the live system: same three scoring
rules, SMA720 trend filter, dip-rebuy dip3_bounce2, 3-position cap,
168h cooldown after 3 losses, taker fee on both legs, capital sized as
cash/free-slots and marked to market hourly.

A trailing stop is not free in live trading: every new high means
cancelling and re-placing the resting exchange order. That costs no fee
on Bybit, but it is more moving parts, so the improvement has to be
real to be worth it.

Usage:
    python scripts/trailing_stop_study.py --end-date 2025-11-30    # train
    python scripts/trailing_stop_study.py --start-date 2025-12-01  # holdout
"""
import argparse

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
    "fixed (сейчас)": dict(kind="fixed"),
    "trail 8% от пика": dict(kind="trail", give_back=0.08),
    "trail 12% от пика": dict(kind="trail", give_back=0.12),
    "в безубыток при +10%": dict(kind="breakeven", trigger=0.10),
    # Точная формулировка гипотезы пользователя: не трогать стоп, пока
    # позиция не разогналась, и только потом подтягивать. Бьёт ровно по
    # тем 9% сделок (train) / 2% (holdout), которые вообще поднимаются на
    # +20% над средней, и не трогает остальные 90%, где трейлингу нечего
    # защищать и он лишь ловит шум.
    "trail 10% после +20%": dict(kind="late_trail", trigger=0.20, give_back=0.10),
}


def stop_price(pos, peak, variant):
    """Where this position's stop sits right now, under the given rule.
    Never below the fixed -8% floor: the ratchet may only ever move the
    stop UP, which is the entire point."""
    floor = pos.avg_cost * (1 + STOP_LOSS)
    kind = variant["kind"]
    if kind == "fixed":
        return floor
    if kind == "trail":
        return max(floor, peak * (1 - variant["give_back"]))
    if kind == "breakeven":
        return max(floor, pos.avg_cost) if peak >= pos.avg_cost * (1 + variant["trigger"]) else floor
    if kind == "late_trail":
        if peak < pos.avg_cost * (1 + variant["trigger"]):
            return floor
        return max(floor, peak * (1 - variant["give_back"]))
    raise ValueError(kind)


def simulate(close, score, sma, variant, fee=SPOT_TAKER_FEE, starting_cash=STARTING_CASH):
    cash = starting_cash
    open_positions: dict[str, CashPosition] = {}
    peaks: dict[str, float] = {}
    cooldown_until = {s: None for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    trades, equity_curve = [], []

    col = {s: i for i, s in enumerate(SYMBOLS)}
    close_v, score_v, sma_v = close.values, score.values, sma.values
    mark_v = close.ffill().values

    for row, ts in enumerate(close.index):
        for symbol in list(open_positions):
            price = close_v[row, col[symbol]]
            if np.isnan(price):
                continue
            pos = open_positions[symbol]
            peaks[symbol] = max(peaks[symbol], price)

            hit_stop = price <= stop_price(pos, peaks[symbol], variant)
            if hit_stop or not score_v[row, col[symbol]] > 0:
                proceeds = pos.proceeds(price, fee)
                cash += proceeds
                ret = proceeds / pos.spent - 1
                trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=ts, spent=pos.spent,
                                   ret=ret, pnl=proceeds - pos.spent,
                                   peak_gain=peaks[symbol] / pos.avg_cost - 1,
                                   hours=(ts - pos.entry_ts).total_seconds() / 3600,
                                   exit_reason="stop" if hit_stop else "natural"))
                del open_positions[symbol]
                del peaks[symbol]
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
                continue
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
            peaks[symbol] = price
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
                           peak_gain=peaks[symbol] / pos.avg_cost - 1,
                           hours=(last - pos.entry_ts).total_seconds() / 3600,
                           exit_reason="open_at_end"))

    equity = pd.Series(equity_curve, index=close.index)
    peak_eq = equity.cummax()
    max_dd = ((equity - peak_eq) / peak_eq).min()
    t = pd.DataFrame(trades)
    total_ret = equity.iloc[-1] / starting_cash - 1
    return dict(
        n_trades=len(t),
        stopped=int((t["exit_reason"] == "stop").sum()) if len(t) else 0,
        win_rate=(t["ret"] > 0).mean() if len(t) else float("nan"),
        total_ret=total_ret,
        max_dd=max_dd,
        ratio=total_ret / abs(max_dd) if max_dd < 0 else float("inf"),
        avg_hours=t["hours"].mean() if len(t) else float("nan"),
        # сколько прибыли позиция отдавала от своего пика до выхода -
        # ровно та величина, которую трейлинг должен уменьшать
        give_back=(t["peak_gain"] - t["ret"]).mean() if len(t) else float("nan"),
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
    print(f"Период: {period} ({len(close)} часов)\n")

    print(f"{'вариант':>22} {'сделок':>7} {'стопов':>7} {'win':>6} {'доходность':>11} "
          f"{'max_dd':>8} {'ret/dd':>7} {'отдано с пика':>14} {'ср.часов':>9}")
    rows = []
    for label, variant in VARIANTS.items():
        stats, trades = simulate(close, score, sma, variant)
        rows.append(dict(period=period, variant=label, **stats))
        print(f"{label:>22} {stats['n_trades']:>7d} {stats['stopped']:>7d} {stats['win_rate']:>6.1%} "
              f"{stats['total_ret']:>11.1%} {stats['max_dd']:>8.1%} {stats['ratio']:>7.2f} "
              f"{stats['give_back']:>14.1%} {stats['avg_hours']:>9.0f}")

    import os
    pd.DataFrame(rows).to_csv("trailing_stop_summary.csv", mode="a", index=False,
                              header=not os.path.exists("trailing_stop_summary.csv"))
    print("\nДописано в trailing_stop_summary.csv")


if __name__ == "__main__":
    main()
