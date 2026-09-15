"""
Quantifies the fee-saving opportunity from direct crypto-to-crypto swaps
during portfolio rotation (2026-09-15) - user's request: "keep the exact
same signal/risk-management logic, just save on fees when one position
closes and another opens around the same time, by swapping directly
(e.g. BTC->XRP) instead of routing through USDT (BTC->USDT->XRP, 2 fee
legs instead of 1)."

Direct pairs actually available on Bybit spot among our 7 symbols
(checked via /v5/market/instruments-info, 2026-09-15): ONLY
ETH/BTC, SOL/BTC, XRP/BTC exist. No DOGE/BNB/LINK pairs against BTC or
against each other - rotations involving those 3 symbols cannot benefit,
they still route through USDT regardless.

Method: run the EXACT unmodified cap=3 portfolio simulation (same
signals, same stop-loss/trend-filter/cooldown - reuses
compute_candidate_segments/simulate_portfolio unchanged, no behavior
change to the strategy itself), then scan the REALIZED trade list for
"genuine rotations" - an exit of symbol A and an entry of symbol B close
enough in time that B's admission was actually enabled by A's slot
freeing up (not just coincidental timing when a slot was already free).
For each such rotation where {A, B} intersect a direct-pair-eligible set
(BTC paired with one of ETH/SOL/XRP), compute the fee saved: 1 fee leg
(0.1%) instead of 2 (0.2%), on the rotated position's dollar size.

Usage:
    python scripts/rotation_fee_study.py
"""
import pandas as pd

from db import get_connection
from strategy import (
    load_events_with_returns, generate_signals, compute_candidate_segments, simulate_portfolio,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
MAX_POSITIONS = 3
ROTATION_WINDOW_HOURS = 24 * 7  # generous on purpose - finds an UPPER BOUND on the opportunity,
# see project_rotation_fee_study memory: even this generous window found the effect negligible
# (0.355% of total P&L), so a tighter/more realistic window would only look smaller still.
DIRECT_PAIR_COINS = {"ETHUSDT", "SOLUSDT", "XRPUSDT"}  # each has a direct X/BTC pair on Bybit
PER_SLOT_CAPITAL = 10000.0 / len(SYMBOLS)
ONE_LEG_FEE = 0.001  # one taker fee, saved on the "middle" USDT leg when swapping directly


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)
    all_segments = []
    for sym in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": sym})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        signals = generate_signals(events, sym)
        segs = compute_candidate_segments(sym, ohlcv, signals, stop_loss=STOP_LOSS, trend_filter_sma=TREND_FILTER_SMA)
        all_segments.extend(segs)
    conn.close()

    trades_df, rejection_log = simulate_portfolio(all_segments, MAX_POSITIONS, cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
    trades_df = trades_df.sort_values("entry_ts").reset_index(drop=True)
    print(f"Total realized trades (cap={MAX_POSITIONS}, unmodified strategy): {len(trades_df)}")

    # Replay portfolio occupancy to find exits that were genuinely "at
    # capacity" right before freeing a slot (so the next entry actually
    # depended on that exit, not just happened to be nearby in time).
    events_list = []
    for _, t in trades_df.iterrows():
        events_list.append(dict(ts=t["entry_ts"], kind="entry", symbol=t["symbol"]))
        events_list.append(dict(ts=t["exit_ts"], kind="exit", symbol=t["symbol"]))
    events_list.sort(key=lambda e: (e["ts"], 0 if e["kind"] == "exit" else 1))

    open_count = 0
    exits_log = []  # (exit_ts, symbol, was_at_capacity_before)
    for e in events_list:
        if e["kind"] == "exit":
            exits_log.append(dict(ts=e["ts"], symbol=e["symbol"], was_full=(open_count >= MAX_POSITIONS)))
            open_count -= 1
        else:
            open_count += 1

    exits_df = pd.DataFrame(exits_log)
    entries_df = trades_df[["symbol", "entry_ts"]].rename(columns={"symbol": "entry_symbol"})

    rotations = []
    for _, ex in exits_df[exits_df["was_full"]].iterrows():
        window_end = ex["ts"] + pd.Timedelta(hours=ROTATION_WINDOW_HOURS)
        candidates = entries_df[(entries_df["entry_ts"] >= ex["ts"]) & (entries_df["entry_ts"] <= window_end) &
                                 (entries_df["entry_symbol"] != ex["symbol"])]
        if not candidates.empty:
            entry_symbol = candidates.iloc[0]["entry_symbol"]
            rotations.append(dict(exit_ts=ex["ts"], exit_symbol=ex["symbol"], entry_symbol=entry_symbol))

    rotations_df = pd.DataFrame(rotations)
    print(f"\nGenuine rotations found (exit freed a slot at capacity, new entry within {ROTATION_WINDOW_HOURS}h): {len(rotations_df)}")

    if rotations_df.empty:
        print("No rotations - nothing to save on.")
        return

    def is_direct_pair_eligible(row):
        pair = {row["exit_symbol"], row["entry_symbol"]}
        return "BTCUSDT" in pair and (pair - {"BTCUSDT"}).pop() in DIRECT_PAIR_COINS

    rotations_df["direct_pair_eligible"] = rotations_df.apply(is_direct_pair_eligible, axis=1)
    eligible = rotations_df[rotations_df["direct_pair_eligible"]]
    print(f"Of those, eligible for a direct swap (one side is BTC, other is ETH/SOL/XRP): {len(eligible)}")

    print("\nEligible rotation pairs:")
    print(eligible[["exit_ts", "exit_symbol", "entry_symbol"]].to_string(index=False))

    total_fee_saved = len(eligible) * PER_SLOT_CAPITAL * ONE_LEG_FEE
    print(f"\nFee saved per eligible rotation: {PER_SLOT_CAPITAL:.2f} USD slot x {ONE_LEG_FEE:.1%} = "
          f"{PER_SLOT_CAPITAL * ONE_LEG_FEE:.2f} USD")
    print(f"Total fee saved across all eligible rotations (full backtest history, ${PER_SLOT_CAPITAL:.0f}/slot): "
          f"${total_fee_saved:.2f}")

    # context: total backtest return in $ terms, to judge materiality
    equity_curve = [PER_SLOT_CAPITAL * MAX_POSITIONS]
    for r in trades_df.sort_values("exit_ts")["ret"]:
        equity_curve.append(equity_curve[-1] + PER_SLOT_CAPITAL * r)
    total_pnl = equity_curve[-1] - equity_curve[0]
    print(f"Total portfolio $ P&L over the same period: ${total_pnl:.2f}")
    print(f"Fee savings as a fraction of total P&L: {total_fee_saved / total_pnl:.3%}")


if __name__ == "__main__":
    main()
