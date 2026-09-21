"""
Does the 3-position cap still make sense once position size is derived
from the REAL balance (2026-09-22)?

The cap was validated in portfolio_exposure_study.py under FIXED notional
per slot: every trade got the same $X, so a lower cap meant less capital
in the market. Under that accounting cap=3 won because it capped
exposure. The live sizing rule chosen on 2026-09-22 is different - free
USDT divided by free slots - so the capital is fully deployed at ANY cap,
and the cap now decides something else entirely: how many pieces the same
money is split into. Three concentrated positions versus seven small
ones, not "some money in" versus "more money in".

That is a genuinely different question, so it gets a genuinely different
simulation: this one carries cash, marks the book to market every hour,
compounds, and sizes each entry as cash/(free slots) exactly like the
live rule would.

Engine is the tick-by-tick portfolio walk from rotation_study.py (same
rules: 3 live scoring rules, SMA720 trend filter, -8% stop on average
cost, dip-rebuy dip3_bounce2, 168h cooldown after 3 losses, taker fee
both legs). A dip-rebuy add reuses the position's original tranche size
and is skipped if cash is short - the same thing that would happen on a
real account.

Usage:
    python scripts/slot_cap_study.py --end-date 2025-11-30    # train
    python scripts/slot_cap_study.py --start-date 2025-12-01  # holdout
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
from strategy import SPOT_TAKER_FEE

STARTING_CASH = 10_000.0
CAPS = [1, 2, 3, 4, 5, 7]  # 7 = no cap at all, one slot per tradable symbol


class CashPosition:
    """A position that knows what it cost in CASH, not in notional units."""

    def __init__(self, symbol, entry_ts, price, tranche_cash, fee):
        self.symbol = symbol
        self.entry_ts = entry_ts
        self.tranche_cash = tranche_cash          # reused for any dip-rebuy add
        self.spent = tranche_cash                 # cash out of the wallet, fee included
        self.qty = tranche_cash * (1 - fee) / price
        self.n_tranches = 1
        self.running_low = price

    @property
    def avg_cost(self):
        return self.spent / self.qty

    def add(self, price, cash, fee):
        self.spent += cash
        self.qty += cash * (1 - fee) / price
        self.n_tranches += 1
        self.running_low = price

    def proceeds(self, price, fee):
        return self.qty * price * (1 - fee)


def simulate(close, score, sma, max_positions, fee=SPOT_TAKER_FEE, starting_cash=STARTING_CASH):
    cash = starting_cash
    open_positions: dict[str, CashPosition] = {}
    cooldown_until = {s: None for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    trades, equity_curve = [], []

    col = {s: i for i, s in enumerate(SYMBOLS)}
    close_v, score_v, sma_v = close.values, score.values, sma.values
    # Marking the book to market uses forward-filled prices; DECISIONS still
    # use the raw close and simply do nothing on a missing candle. Without
    # this, a single gap in one symbol's history valued a held position at
    # zero for that hour, which showed up as a fake -100% drawdown at the
    # concentrated caps (caught 2026-09-22 - a -8% stop cannot lose 100%).
    mark_v = close.ffill().values
    cap = max_positions or len(SYMBOLS)

    for row, ts in enumerate(close.index):
        for symbol in list(open_positions):
            price = close_v[row, col[symbol]]
            if np.isnan(price):
                continue
            pos = open_positions[symbol]
            hit_stop = price / pos.avg_cost - 1 <= STOP_LOSS
            if hit_stop or not score_v[row, col[symbol]] > 0:
                proceeds = pos.proceeds(price, fee)
                cash += proceeds
                ret = proceeds / pos.spent - 1
                trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=ts,
                                   spent=pos.spent, ret=ret, pnl=proceeds - pos.spent,
                                   n_tranches=pos.n_tranches,
                                   exit_reason="stop_loss" if hit_stop else "natural"))
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
                continue
            candidates.append((sc, symbol, price))
        candidates.sort(key=lambda c: -c[0])

        for sc, symbol, price in candidates:
            free_slots = cap - len(open_positions)
            if free_slots <= 0:
                break
            # THE live sizing rule: split the free cash across the slots
            # still open, so the last slot gets whatever is left rather
            # than a fixed amount that may no longer fit.
            stake = cash / free_slots
            if stake < 10:  # nothing meaningful left to deploy
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
        cash += proceeds
        trades.append(dict(symbol=symbol, entry_ts=pos.entry_ts, exit_ts=last, spent=pos.spent,
                           ret=proceeds / pos.spent - 1, pnl=proceeds - pos.spent,
                           n_tranches=pos.n_tranches, exit_reason="open_at_end"))

    equity = pd.Series(equity_curve, index=close.index)
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    trades_df = pd.DataFrame(trades)
    return dict(
        n_trades=len(trades_df),
        win_rate=(trades_df["ret"] > 0).mean() if len(trades_df) else float("nan"),
        total_ret=equity.iloc[-1] / starting_cash - 1,
        max_dd=max_dd,
        ratio=(equity.iloc[-1] / starting_cash - 1) / abs(max_dd) if max_dd < 0 else float("inf"),
        avg_stake=trades_df["spent"].mean() if len(trades_df) else float("nan"),
    ), equity, trades_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    close, score, sma = load_panel(conn, args.start_date, args.end_date)
    conn.close()
    print(f"Period: {close.index[0]:%Y-%m-%d} .. {close.index[-1]:%Y-%m-%d}  "
          f"({len(close)} hourly rows), capital {STARTING_CASH:,.0f} USDT fully deployed\n")

    print(f"{'cap':>5} {'trades':>7} {'win':>7} {'return':>10} {'max_dd':>8} {'ret/dd':>7} {'avg stake':>10}")
    rows = []
    for cap in CAPS:
        stats, equity, trades = simulate(close, score, sma, cap)
        rows.append(dict(cap=cap, **stats))
        print(f"{cap:>5} {stats['n_trades']:>7d} {stats['win_rate']:>7.1%} {stats['total_ret']:>10.1%} "
              f"{stats['max_dd']:>8.1%} {stats['ratio']:>7.2f} {stats['avg_stake']:>10,.0f}")

    out = pd.DataFrame(rows)
    out.insert(0, "period", f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}")
    import os
    out.to_csv("slot_cap_summary.csv", mode="a", index=False,
               header=not os.path.exists("slot_cap_summary.csv"))
    print("\nAppended to slot_cap_summary.csv")


if __name__ == "__main__":
    main()
