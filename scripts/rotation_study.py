"""
Hypothesis (raised 2026-09-22, after noticing the live bot sitting at
capacity with ETH/SOL scoring 5.0 while three of the four HELD positions
scored 3.0): when the portfolio is full and a better-scoring candidate
appears, should the weakest held position be evicted to make room?

Today the cap is strictly first-come-first-served - `skip_capacity` just
drops the candidate, and it only gets in if a slot frees up on its own
(project_four_hypotheses_2026_09_13). Rotation is the obvious-looking
improvement, and obvious-looking improvements in this project have a
strong track record of dying in holdout, so it gets the same treatment as
every other candidate rule: train/holdout split, fees on both legs, and
no partial credit for looking good on one period.

Why this needs a new engine instead of strategy.py's simulate_portfolio():
that function arbitrates PRE-COMPUTED segments whose exit_ts and return
are already fixed. Rotation cuts a held position short at an arbitrary
hour, which changes its return - so the simulation has to walk the
timeline tick by tick, exactly like testnet_trader.py does live.

What is modeled (all of it identical between the arms, so the only
difference measured is rotation itself):
  - the 3 live scoring rules via compute_score_series (same as live)
  - SMA720 trend filter at entry, -8% stop-loss on AVERAGE cost,
    natural exit when the score stops being positive
  - the validated dip-rebuy add-on (dip3_bounce2, max 2 adds)
  - 3-position cap, contested slots go to the highest score
  - 168h cooldown after 3 consecutive losing trades on a symbol
  - taker fee on every buy and every sell, including rotation's extra
    round trip - the cost that has to be earned back for rotation to pay

Usage:
    python scripts/rotation_study.py --end-date 2025-11-30    # train
    python scripts/rotation_study.py --start-date 2025-12-01  # holdout
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168
MAX_POSITIONS = 3
DIP_REBUY_THRESHOLD = -0.03
DIP_REBUY_REBOUND = 0.02
DIP_REBUY_MAX_ADDS = 2

# Fixed notional per tranche, non-compounding - same accounting style as
# portfolio_exposure_study.py, so a trade's P&L is proportional to the
# capital it actually tied up (a 3-tranche position risks 3x a 1-tranche
# one, and rotating out of one costs 3 tranches' worth of exit fee).
TRANCHE_NOTIONAL = 1_000.0
STARTING_CAPITAL = MAX_POSITIONS * (1 + DIP_REBUY_MAX_ADDS) * TRANCHE_NOTIONAL

# Deliberately two variants, not a sweep: "any better candidate wins" and
# "only a clearly better one wins". Live scores are small integers (1/3/5),
# so a margin of 1.0 is the minimum meaningful difference and 2.0 demands
# a real gap. Grid kept tiny on purpose - see min_cooldown_study.py for
# why this project does not go fishing across parameter space.
ROTATION_MARGINS = [1.0, 2.0]


class Position:
    """One symbol's open position - possibly several dip-rebuy tranches."""

    def __init__(self, symbol, entry_ts, price, score):
        self.symbol = symbol
        self.entry_ts = entry_ts
        self.entry_score = score
        self.usdt = [TRANCHE_NOTIONAL]
        self.qty = [TRANCHE_NOTIONAL / price]
        self.running_low = price
        self.worst_dd = 0.0

    @property
    def avg_cost(self):
        return sum(self.usdt) / sum(self.qty)

    @property
    def invested(self):
        return sum(self.usdt)

    def add_tranche(self, price):
        self.usdt.append(TRANCHE_NOTIONAL)
        self.qty.append(TRANCHE_NOTIONAL / price)
        self.running_low = price  # fresh window before another add is allowed

    def close(self, exit_ts, price, reason, fee):
        cost = self.invested * (1 + fee)
        proceeds = sum(self.qty) * price * (1 - fee)
        return dict(
            symbol=self.symbol, entry_ts=self.entry_ts, exit_ts=exit_ts,
            n_tranches=len(self.usdt), invested=self.invested,
            avg_cost=self.avg_cost, exit_price=price, exit_reason=reason,
            hours_held=(exit_ts - self.entry_ts).total_seconds() / 3600,
            ret=proceeds / cost - 1, pnl=proceeds - cost, worst_dd=self.worst_dd,
        )


def load_panel(conn, start_date=None, end_date=None):
    """One aligned frame per input series: close / score / SMA720, indexed
    by the union of all symbols' hourly timestamps. The SMA is computed on
    the FULL history and only then trimmed to the requested window, so a
    holdout run's trend filter is not crippled by a missing warm-up."""
    events = load_events_with_returns(conn)
    close, score, sma = {}, {}, {}
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql(
            "SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
            conn, params={"s": symbol},
        )
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        ohlcv["score"] = compute_score_series(generate_signals(events, symbol), ohlcv)
        ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()
        ohlcv = ohlcv.set_index("ts")
        close[symbol], score[symbol], sma[symbol] = ohlcv["close"], ohlcv["score"], ohlcv["sma"]

    close, score, sma = pd.DataFrame(close), pd.DataFrame(score), pd.DataFrame(sma)
    idx = close.index
    if start_date:
        idx = idx[idx >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        idx = idx[idx <= pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)]
    return close.loc[idx], score.loc[idx], sma.loc[idx]


def simulate(close, score, sma, rotation_margin=None, fee=SPOT_TAKER_FEE):
    """Walks the timeline hour by hour. rotation_margin=None reproduces
    today's live behaviour (a full portfolio simply refuses candidates);
    a float allows evicting the weakest-scoring holding when a candidate
    beats it by at least that margin."""
    open_positions = {}
    cooldown_until = {s: None for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    trades = []
    rotations = 0

    timestamps = close.index
    close_v, score_v, sma_v = close.values, score.values, sma.values
    col = {s: i for i, s in enumerate(SYMBOLS)}

    for row, ts in enumerate(timestamps):
        # --- exits and dip-rebuy adds on everything currently held ---
        for symbol in list(open_positions):
            price = close_v[row, col[symbol]]
            if np.isnan(price):
                continue
            pos = open_positions[symbol]
            pos.worst_dd = min(pos.worst_dd, price / pos.avg_cost - 1)

            hit_stop = price / pos.avg_cost - 1 <= STOP_LOSS
            still_long = score_v[row, col[symbol]] > 0
            if hit_stop or not still_long:
                trade = pos.close(ts, price, "stop_loss" if hit_stop else "natural", fee)
                trades.append(trade)
                del open_positions[symbol]
                if trade["ret"] < 0:
                    consecutive_losses[symbol] += 1
                    if consecutive_losses[symbol] >= COOLDOWN_AFTER_LOSSES:
                        cooldown_until[symbol] = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                        consecutive_losses[symbol] = 0
                else:
                    consecutive_losses[symbol] = 0
                continue

            pos.running_low = min(pos.running_low, price)
            if len(pos.usdt) - 1 < DIP_REBUY_MAX_ADDS:
                dip = pos.running_low / pos.avg_cost - 1
                bounce = price / pos.running_low - 1
                if dip <= DIP_REBUY_THRESHOLD and bounce >= DIP_REBUY_REBOUND:
                    pos.add_tranche(price)

        # --- who wants in this hour ---
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
                continue  # below its own trend - same filter as live
            candidates.append((sc, symbol, price))
        if not candidates:
            continue
        candidates.sort(key=lambda c: -c[0])

        # --- fill genuinely free slots first ---
        while candidates and len(open_positions) < MAX_POSITIONS:
            sc, symbol, price = candidates.pop(0)
            open_positions[symbol] = Position(symbol, ts, price, sc)

        # --- rotation: evict the weakest holding for a clearly better candidate ---
        if rotation_margin is None or not candidates:
            continue
        while candidates:
            sc, symbol, price = candidates[0]
            weakest = min(open_positions.values(), key=lambda p: (score_v[row, col[p.symbol]], p.entry_ts))
            weakest_score = score_v[row, col[weakest.symbol]]
            if sc - weakest_score < rotation_margin:
                break  # candidates are sorted, so nothing behind it can qualify either
            evicted_price = close_v[row, col[weakest.symbol]]
            if np.isnan(evicted_price):
                break
            trade = weakest.close(ts, evicted_price, "rotated_out", fee)
            trades.append(trade)
            del open_positions[weakest.symbol]
            rotations += 1
            if trade["ret"] < 0:
                consecutive_losses[weakest.symbol] += 1
                if consecutive_losses[weakest.symbol] >= COOLDOWN_AFTER_LOSSES:
                    cooldown_until[weakest.symbol] = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                    consecutive_losses[weakest.symbol] = 0
            else:
                consecutive_losses[weakest.symbol] = 0
            candidates.pop(0)
            open_positions[symbol] = Position(symbol, ts, price, sc)

    # positions still open at the end are marked to market, not discarded -
    # dropping them would quietly favour whichever arm happens to be holding
    # a loser when the data runs out
    last_ts = timestamps[-1]
    for symbol, pos in open_positions.items():
        price = close[symbol].dropna().iloc[-1]
        trades.append(pos.close(last_ts, price, "open_at_end", fee))

    return pd.DataFrame(trades), rotations


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n_trades=0, win_rate=float("nan"), total_ret=float("nan"),
                    max_dd=float("nan"), ratio=float("nan"), avg_hours=float("nan"))
    trades = trades.sort_values("exit_ts")
    equity = STARTING_CAPITAL + trades["pnl"].cumsum()
    equity = pd.concat([pd.Series([STARTING_CAPITAL]), equity], ignore_index=True)
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    total_ret = equity.iloc[-1] / STARTING_CAPITAL - 1
    return dict(
        n_trades=len(trades),
        win_rate=(trades["ret"] > 0).mean(),
        total_ret=total_ret,
        max_dd=max_dd,
        ratio=total_ret / abs(max_dd) if max_dd < 0 else float("inf"),
        avg_hours=trades["hours_held"].mean(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    close, score, sma = load_panel(conn, args.start_date, args.end_date)
    conn.close()
    period = f"{close.index[0]:%Y-%m-%d} .. {close.index[-1]:%Y-%m-%d}"
    print(f"Period: {period}  ({len(close)} hourly rows, {len(SYMBOLS)} symbols)\n")

    rows = []
    for label, margin in [("baseline (no rotation)", None),
                          *[(f"rotate if score +{m:.0f}", m) for m in ROTATION_MARGINS]]:
        trades, rotations = simulate(close, score, sma, rotation_margin=margin)
        stats = summarize(trades)
        rows.append(dict(variant=label, rotations=rotations, **stats))

    out = pd.DataFrame(rows)
    print(f"{'variant':>24} {'trades':>7} {'rot':>5} {'win':>7} {'return':>9} "
          f"{'max_dd':>8} {'ret/dd':>7} {'avg_h':>7}")
    for _, r in out.iterrows():
        print(f"{r['variant']:>24} {r['n_trades']:>7.0f} {r['rotations']:>5.0f} {r['win_rate']:>7.1%} "
              f"{r['total_ret']:>9.1%} {r['max_dd']:>8.1%} {r['ratio']:>7.2f} {r['avg_hours']:>7.0f}")

    out.insert(0, "period", period)
    out.to_csv("rotation_summary.csv", mode="a", index=False,
               header=not __import__("os").path.exists("rotation_summary.csv"))
    print("\nAppended to rotation_summary.csv")


if __name__ == "__main__":
    main()
