"""
Would the strategy gain from reacting to the SPEED of a move? (2026-10-15,
user's hypothesis, prompted by the 5-14% drop of 2026-10-07..14)

    "exit on sharp drops, add on sharp rises"

The strategy today reacts to price in exactly two ways: the -8% stop on
the average cost, and the SMA720 gate at ENTRY only. Neither looks at how
fast price is moving, so a position is held straight through a 10% slide
while the Fear & Greed score stays at 3.0.

What has and has not been tested before - this is not a re-run:
  * add on strength (pyramiding): REJECTED, worse in 14/14 symbol x period
    combos. The trigger there was "price above entry". Here it is a rise
    of at least X% over the last H hours, which is a different trigger but
    the same economic act (buy higher, average cost up, stop closer).
  * exit on the coin's own trend break (SMA720): REJECTED, train-best was
    holdout-worst. That is a LEVEL (price vs 30-day mean). A crash exit is
    a RATE (X% in H hours), so it can fire while price is still far above
    the trend - the 2026-10-08 BNB/SOL drops did exactly that.
  * trailing / ratcheting stop (4 variants): REJECTED. That follows the
    peak; a crash exit does not care where the peak was.

Variants - a small grid fixed BEFORE looking at any result:

  crash_4h_5    exit when the last 4h return <= -5%
  crash_24h_5   exit when the last 24h return <= -5%
  crash_24h_8   exit when the last 24h return <= -8%
  surge_24h_5   dip-independent add when the last 24h return >= +5% (max 2)
  surge_24h_8   same at +8%
  crash+surge   crash_24h_5 together with surge_24h_5

After a crash exit the symbol is blocked from re-entry for 24h; without
that the still-positive score would buy it straight back on the next hour
(a churn loop the SMA gate prevented in trend_exit_study, but cannot here
because a crash can happen far above the trend).

Engine is slot_cap_study's cash-based portfolio walk, untouched otherwise:
cap 3, cash / free-slots sizing, dip-rebuy dip3_bounce2, 168h cooldown
after 3 losses, taker fee both legs, hourly mark-to-market. Run at BOTH
the fee the backtests have always assumed (0.1%) and the 0.18% measured on
the real account: a rate-based rule trades more, so it is the variant most
exposed to the difference.

Usage:
    python scripts/momentum_study.py --end-date 2025-11-30     # train
    python scripts/momentum_study.py --start-date 2025-12-01   # holdout
    python scripts/momentum_study.py --start-date 2025-12-01 --fee 0.0018
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
SURGE_MAX_ADDS = 2          # same ceiling the dip-rebuy has
CRASH_REENTRY_BLOCK_H = 24

VARIANTS = {
    "baseline (сейчас)": dict(),
    "обвал 4ч -5%": dict(drop_h=4, drop_x=0.05),
    "обвал 24ч -5%": dict(drop_h=24, drop_x=0.05),
    "обвал 24ч -8%": dict(drop_h=24, drop_x=0.08),
    "рывок 24ч +5%": dict(surge_h=24, surge_x=0.05),
    "рывок 24ч +8%": dict(surge_h=24, surge_x=0.08),
    "обвал 24ч -5% + рывок +5%": dict(drop_h=24, drop_x=0.05, surge_h=24, surge_x=0.05),
}


def momentum(close, hours):
    """N-hour return on forward-filled closes. A missing candle must not turn
    into a fake -100% 'crash'; a stale price simply reads as no movement."""
    cf = close.ffill()
    return (cf / cf.shift(hours) - 1).values


def simulate(close, score, sma, variant, fee=SPOT_TAKER_FEE, starting_cash=STARTING_CASH):
    cash = starting_cash
    open_positions: dict[str, CashPosition] = {}
    cooldown_until = {s: None for s in SYMBOLS}
    crash_block = {s: None for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    trades, equity_curve = [], []
    surge_adds_total = 0

    col = {s: i for i, s in enumerate(SYMBOLS)}
    close_v, score_v, sma_v = close.values, score.values, sma.values
    mark_v = close.ffill().values
    drop_v = momentum(close, variant["drop_h"]) if "drop_h" in variant else None
    surge_v = momentum(close, variant["surge_h"]) if "surge_h" in variant else None

    for row, ts in enumerate(close.index):
        for symbol in list(open_positions):
            i = col[symbol]
            price = close_v[row, i]
            if np.isnan(price):
                continue
            pos = open_positions[symbol]

            hit_stop = price / pos.avg_cost - 1 <= STOP_LOSS
            signal_dead = not score_v[row, i] > 0
            crash = False
            if drop_v is not None and not np.isnan(drop_v[row, i]):
                crash = drop_v[row, i] <= -variant["drop_x"]

            if hit_stop or signal_dead or crash:
                proceeds = pos.proceeds(price, fee)
                cash += proceeds
                ret = proceeds / pos.spent - 1
                reason = "stop" if hit_stop else ("signal" if signal_dead else "crash")
                trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=ts, spent=pos.spent,
                                   ret=ret, pnl=proceeds - pos.spent, exit_reason=reason,
                                   hours=(ts - pos.entry_ts).total_seconds() / 3600))
                del open_positions[symbol]
                if reason == "crash":
                    crash_block[symbol] = ts + pd.Timedelta(hours=CRASH_REENTRY_BLOCK_H)
                if ret < 0:
                    consecutive_losses[symbol] += 1
                    if consecutive_losses[symbol] >= COOLDOWN_AFTER_LOSSES:
                        cooldown_until[symbol] = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                        consecutive_losses[symbol] = 0
                else:
                    consecutive_losses[symbol] = 0
                continue

            pos.running_low = min(pos.running_low, price)
            surge_adds = getattr(pos, "surge_adds", 0)
            dip_adds = pos.n_tranches - 1 - surge_adds
            if dip_adds < DIP_REBUY_MAX_ADDS and cash >= pos.tranche_cash:
                dip = pos.running_low / pos.avg_cost - 1
                bounce = price / pos.running_low - 1
                if dip <= DIP_REBUY_THRESHOLD and bounce >= DIP_REBUY_REBOUND:
                    pos.add(price, pos.tranche_cash, fee)
                    cash -= pos.tranche_cash
                    continue
            if (surge_v is not None and surge_adds < SURGE_MAX_ADDS and cash >= pos.tranche_cash
                    and not np.isnan(surge_v[row, i]) and surge_v[row, i] >= variant["surge_x"]):
                pos.add(price, pos.tranche_cash, fee)
                pos.surge_adds = surge_adds + 1
                cash -= pos.tranche_cash
                surge_adds_total += 1

        candidates = []
        for symbol in SYMBOLS:
            if symbol in open_positions:
                continue
            i = col[symbol]
            price, sc, trend = close_v[row, i], score_v[row, i], sma_v[row, i]
            if np.isnan(price) or not sc > 0:
                continue
            for block in (cooldown_until[symbol], crash_block[symbol]):
                if block is not None and ts < block:
                    break
            else:
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
        by_crash=by_reason.get("crash", 0),
        by_stop=by_reason.get("stop", 0),
        surge_adds=surge_adds_total,
        win_rate=(t["ret"] > 0).mean() if len(t) else float("nan"),
        total_ret=total_ret,
        max_dd=max_dd,
        ratio=total_ret / abs(max_dd) if max_dd < 0 else float("inf"),
    ), t, equity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--fee", type=float, default=SPOT_TAKER_FEE)
    args = parser.parse_args()

    conn = get_connection()
    close, score, sma = load_panel(conn, args.start_date, args.end_date)
    conn.close()
    period = f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}"
    print(f"Период: {period} ({len(close)} часов), кэп {MAX_POSITIONS}, комиссия {args.fee:.2%} за ногу\n")

    print(f"{'вариант':>28} {'сделок':>7} {'обвалов':>8} {'стопов':>7} {'рывков':>7} {'win':>6} "
          f"{'доходность':>11} {'max_dd':>8} {'ret/dd':>7}")
    rows = []
    for label, variant in VARIANTS.items():
        stats, _, _ = simulate(close, score, sma, variant, fee=args.fee)
        rows.append(dict(period=period, fee=args.fee, variant=label, **stats))
        print(f"{label:>28} {stats['n_trades']:>7d} {stats['by_crash']:>8d} {stats['by_stop']:>7d} "
              f"{stats['surge_adds']:>7d} {stats['win_rate']:>6.1%} {stats['total_ret']:>11.1%} "
              f"{stats['max_dd']:>8.1%} {stats['ratio']:>7.2f}")

    pd.DataFrame(rows).to_csv("momentum_summary.csv", mode="a", index=False,
                              header=not os.path.exists("momentum_summary.csv"))
    print("\nДописано в momentum_summary.csv")


if __name__ == "__main__":
    main()
