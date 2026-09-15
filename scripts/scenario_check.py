"""
One-off scenario check: given a position manually opened at a specific
past date/amount (independent of whether the strategy's own signal said
to enter exactly then), what would our CURRENT strategy logic (stop-loss,
natural exit, re-entry) have done with it through to now?

REWRITTEN 2026-09-13 after finding two bugs in the original version by
cross-checking a related script (portfolio_exposure_study.py) against
strategy.py's own backtest() - see project_four_hypotheses_2026-09-13
memory for the full story:
  1. An hour-by-hour "in position? then check exit; else check entry"
     state machine allows immediate re-entry right after a stop-loss exit
     if score is still positive - backtest() never does this. Once a
     segment (contiguous score>0 run) is cut short by a stop, no more
     trades happen until score genuinely drops to <=0 and rises again
     (a NEW segment).
  2. Cooldown/trend-filter must be evaluated ONLY at a segment's start and,
     if blocked, skip that ENTIRE segment permanently - never re-check
     mid-segment once the block condition clears.

Fixed by reusing the same verified `compute_candidate_segments`-style
logic as portfolio_exposure_study.py for every segment AFTER the user's
manual first entry closes. The manual first entry itself is a special
case (it doesn't necessarily start exactly at a segment boundary, since
the user picked an arbitrary calendar date) - its own exit (stop-loss or
first score<=0 after entry) is computed directly, then normal segment-
based behavior resumes for everything after.

Usage:
    python scripts/scenario_check.py --symbol BTCUSDT --entry-date 2026-08-20 --capital 700
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168


def compute_all_segments(ohlcv: pd.DataFrame) -> list[dict]:
    """Every contiguous score>0 run, with its own predetermined
    entry/exit/ret/trend_ok - exactly mirrors backtest()'s per-segment
    logic, unconditional on cooldown (cooldown is applied later, in
    chronological order, since it depends on realized trade history)."""
    ohlcv = ohlcv.reset_index(drop=True)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()

    segments = []
    for seg_id, seg in ohlcv[ohlcv["is_long"]].groupby("seg_id"):
        entry_idx = seg.index[0]
        natural_exit_idx = min(seg.index[-1] + 1, len(ohlcv) - 1)
        entry_price = ohlcv["close"].iloc[entry_idx]
        entry_ts = ohlcv["ts"].iloc[entry_idx]
        sma = ohlcv["sma"].iloc[entry_idx]
        trend_ok = not (pd.notna(sma) and entry_price < sma)

        exit_idx = natural_exit_idx
        path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
        hit = path[path / entry_price - 1 <= STOP_LOSS]
        if not hit.empty:
            exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        exit_ts = ohlcv["ts"].iloc[exit_idx]
        net_ret = (exit_price * (1 - SPOT_TAKER_FEE)) / (entry_price * (1 + SPOT_TAKER_FEE)) - 1

        segments.append(dict(entry_ts=entry_ts, entry_price=entry_price, exit_ts=exit_ts,
                              exit_price=exit_price, trend_ok=trend_ok, ret=net_ret))
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--entry-date", type=str, required=True)
    parser.add_argument("--capital", type=float, required=True)
    args = parser.parse_args()

    conn = get_connection()
    events = load_events_with_returns(conn)
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                         conn, params={"s": args.symbol})
    conn.close()
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)

    signals = generate_signals(events, args.symbol)
    ohlcv["score"] = compute_score_series(signals, ohlcv)
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()

    entry_ts_target = pd.Timestamp(args.entry_date, tz="UTC")
    start_idx = ohlcv[ohlcv["ts"] >= entry_ts_target].index[0]
    manual_entry_price = ohlcv["close"].iloc[start_idx]
    manual_entry_ts = ohlcv["ts"].iloc[start_idx]

    print(f"Scenario: bought {args.symbol} with {args.capital:.2f} USD at {manual_entry_ts} "
          f"(price {manual_entry_price:.2f})")
    initial_qty = args.capital / (manual_entry_price * (1 + SPOT_TAKER_FEE))
    print(f"Initial qty: {initial_qty:.8f} {args.symbol.replace('USDT', '')}")
    print("-" * 100)

    # 1) the manual first position: exit at the first stop-loss breach or
    # first score<=0 after entry, whichever comes first - computed directly
    # since it doesn't necessarily start at a segment boundary.
    future = ohlcv.iloc[start_idx:].reset_index(drop=True)
    rel = future["close"] / manual_entry_price - 1
    stop_hit = rel[rel <= STOP_LOSS]
    natural_hit = future.index[future["score"] <= 0]
    natural_hit = natural_hit[natural_hit > 0]  # score is >0 at entry by construction of "already holding"; look forward
    candidates = []
    if not stop_hit.empty:
        candidates.append((stop_hit.index[0], "stop_loss"))
    if len(natural_hit) > 0:
        candidates.append((natural_hit[0], "natural_exit"))
    if candidates:
        candidates.sort()
        exit_idx_rel, exit_reason = candidates[0]
    else:
        exit_idx_rel, exit_reason = len(future) - 1, "still_open"

    exit_price = future["close"].iloc[exit_idx_rel]
    exit_ts = future["ts"].iloc[exit_idx_rel]
    proceeds = initial_qty * exit_price * (1 - SPOT_TAKER_FEE)
    ret = proceeds / (initial_qty * manual_entry_price * (1 + SPOT_TAKER_FEE)) - 1

    events_log = []
    cash = args.capital
    in_position = True
    consecutive_losses = 0
    cooldown_until_ts = None

    if exit_reason != "still_open":
        events_log.append(f"{exit_ts}: EXIT ({exit_reason}) at {exit_price:.2f} -> {proceeds:.2f} USD "
                           f"(trade ret {ret:+.2%})")
        cash = proceeds
        in_position = False
        if ret < 0:
            consecutive_losses = 1
            if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                cooldown_until_ts = exit_ts + pd.Timedelta(hours=COOLDOWN_HOURS)
                consecutive_losses = 0
        resume_from_ts = exit_ts
    else:
        resume_from_ts = None  # still in the manual position, nothing more to simulate

    # 2) after the manual position closes (if it does), resume NORMAL
    # segment-based behavior using the exact same verified logic as
    # portfolio_exposure_study.py's simulate_capped (single symbol, no cap).
    if resume_from_ts is not None:
        remaining = ohlcv[ohlcv["ts"] > resume_from_ts].reset_index(drop=True)
        segments = compute_all_segments(remaining)
        last_ts, last_score = ohlcv["ts"].iloc[-1], ohlcv["score"].iloc[-1]
        qty = 0.0
        for i, seg in enumerate(segments):
            if not seg["trend_ok"]:
                continue
            if cooldown_until_ts is not None and seg["entry_ts"] < cooldown_until_ts:
                continue
            qty = cash / (seg["entry_price"] * (1 + SPOT_TAKER_FEE))
            events_log.append(f"{seg['entry_ts']}: RE-ENTER at {seg['entry_price']:.2f} with "
                               f"{cash:.2f} USD -> qty {qty:.8f}")

            # the LAST segment's "exit" is only genuine if it's an actual
            # stop-loss/natural-exit, not just where our price data happens
            # to end - compute_all_segments always assigns SOME exit_idx
            # (defaulting to the final available row) even when score is
            # still positive and no stop was hit, i.e. the position is
            # still genuinely open, not closed.
            is_last = (i == len(segments) - 1)
            still_open = is_last and seg["exit_ts"] == last_ts and last_score > 0 and \
                (ohlcv["close"].iloc[-1] / seg["entry_price"] - 1) > STOP_LOSS
            if still_open:
                in_position = True
                break

            proceeds2 = qty * seg["exit_price"] * (1 - SPOT_TAKER_FEE)
            events_log.append(f"{seg['exit_ts']}: EXIT at {seg['exit_price']:.2f} -> {proceeds2:.2f} USD "
                               f"(trade ret {seg['ret']:+.2%})")
            cash = proceeds2
            in_position = False
            if seg["ret"] < 0:
                consecutive_losses += 1
                if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                    cooldown_until_ts = seg["exit_ts"] + pd.Timedelta(hours=COOLDOWN_HOURS)
                    consecutive_losses = 0
            else:
                consecutive_losses = 0

    for line in events_log:
        print(line)
    print("-" * 100)

    latest_price = ohlcv["close"].iloc[-1]
    latest_ts = ohlcv["ts"].iloc[-1]
    if in_position:
        # qty of the currently-open position is whatever the last RE-ENTER line computed,
        # or the original initial_qty if we never exited the manual position at all.
        current_qty = initial_qty if resume_from_ts is None else qty
        current_value = current_qty * latest_price
        print(f"Still IN POSITION as of {latest_ts}: {current_qty:.8f} {args.symbol.replace('USDT', '')} "
              f"@ {latest_price:.2f} = {current_value:.2f} USD (mark-to-market, no exit fee deducted yet)")
    else:
        current_value = cash
        print(f"Currently FLAT (in cash) as of {latest_ts}: {current_value:.2f} USD")

    bh_qty = args.capital / (manual_entry_price * (1 + SPOT_TAKER_FEE))
    bh_value = bh_qty * latest_price
    print(f"\nFor comparison, simple buy & hold (no strategy, just held): {bh_value:.2f} USD "
          f"({(bh_value/args.capital-1):+.2%})")
    print(f"Our strategy result: {current_value:.2f} USD ({(current_value/args.capital-1):+.2%})")


if __name__ == "__main__":
    main()
