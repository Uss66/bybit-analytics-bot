"""
Honest test of capping simultaneous portfolio exposure (max N symbols in
a position at once, prioritizing the highest-score candidates when more
than N want to enter at the same hour) vs the current behavior (every
symbol trades fully independently, so fgi_greed/DVOL being market-wide
signals can put all 7 symbols in a position simultaneously).

REWRITTEN after two rounds of bugs found by comparing this script's
DOGEUSDT output against strategy.py's own backtest() (they must produce
IDENTICAL trades when max_positions=None, since that's the same rule -
this check caught both bugs):
  1. An hour-by-hour "is this symbol flat and score>0" state machine
     allows a symbol to enter PARTWAY into an already-started signal
     segment right when its cooldown lifts. backtest() never does this -
     cooldown/trend-filter are checked ONLY at a segment's start, and if
     blocked, that ENTIRE segment is permanently skipped, never revisited
     mid-segment. Same subtlety would apply to a capacity check.
  2. Computing indicators (SMA) on the full history and filtering
     afterward differs from strategy.py main()'s own convention (filter
     dates first, then compute indicators inside backtest()) - the two
     give different trend-filter behavior for roughly the first 10 days of
     any window, because a truncated series has insufficient SMA lookback
     there (NaN is treated as "trend filter passes"). This is a pre-
     existing, project-wide quirk of every backtest number ever quoted in
     this project, not something to fix silently in just one script.

REWRITTEN AGAIN (2026-09-13): the verified compute_candidate_segments()/
simulate_capped() logic below was promoted into strategy.py itself (as
compute_candidate_segments()/simulate_portfolio()) so the live trader
(testnet_trader.py) and this study both use the exact same, single
implementation - no risk of the two copies silently drifting apart, which
is exactly the kind of bug this project has been repeatedly bitten by.
This script now just imports and drives it. One behavioral change:
strategy.py's simulate_portfolio() breaks same-timestamp ties by highest
score rather than processing order (see its docstring for why) - so
numbers here will differ slightly from the pre-2026-09-13 version of this
file, though the qualitative conclusion (moderate cap 3-4 improves
risk-adjusted returns) is unchanged.

Usage:
    python scripts/portfolio_exposure_study.py --end-date 2025-11-30   # train
    python scripts/portfolio_exposure_study.py --start-date 2025-12-01 # holdout
"""
import argparse

import pandas as pd

from db import get_connection
from strategy import load_events_with_returns, generate_signals, compute_candidate_segments, simulate_portfolio

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def simulate_capped(all_segments: list[dict], max_positions: int | None):
    trades_df, rejection_log = simulate_portfolio(all_segments, max_positions, cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
    return trades_df, [r["symbol"] for r in rejection_log]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    events = load_events_with_returns(conn)

    all_segments = []
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        signals = generate_signals(events, symbol)
        all_segments.extend(compute_candidate_segments(symbol, ohlcv, signals, stop_loss=STOP_LOSS,
                                                         trend_filter_sma=TREND_FILTER_SMA))
    conn.close()

    per_slot_capital = 10_000.0 / len(SYMBOLS)
    print(f"{'max_positions':>14} {'n_trades':>9} {'win_rate':>9} {'total_ret':>10} {'max_dd':>8}")
    trades_by_cap = {}
    rejections_by_cap = {}
    for cap in [None, 5, 4, 3, 2, 1]:
        trades_df, rejection_log = simulate_capped(all_segments, cap)
        trades_by_cap[cap] = trades_df
        rejections_by_cap[cap] = rejection_log
        if trades_df.empty:
            continue
        trades_df = trades_df.sort_values("exit_ts")
        starting_capital = per_slot_capital * (cap or len(SYMBOLS))
        equity_curve = [starting_capital]
        for r in trades_df["ret"]:
            equity_curve.append(equity_curve[-1] + per_slot_capital * r)
        equity = equity_curve[-1]
        total_return = equity / starting_capital - 1
        equity_series = pd.Series(equity_curve)
        running_peak = equity_series.cummax()
        max_dd = ((equity_series - running_peak) / running_peak).min()
        win_rate = (trades_df["ret"] > 0).mean()
        label = "none (all 7)" if cap is None else str(cap)
        print(f"{label:>14} {len(trades_df):>9} {win_rate:>9.1%} {total_return:>10.1%} {max_dd:>8.1%}")

    print("\n" + "=" * 70)
    print("PER-SYMBOL TRADE COUNT AND MEAN RETURN, by cap (crowding-out check)")
    print("=" * 70)
    for cap in [None, 3, 2, 1]:
        df = trades_by_cap[cap]
        if df.empty:
            continue
        label = "none" if cap is None else str(cap)
        print(f"\n--- cap={label} ---")
        by_symbol = df.groupby("symbol").agg(n=("ret", "size"), win_rate=("ret", lambda x: (x > 0).mean()),
                                              mean_ret=("ret", "mean"))
        print(by_symbol.to_string())

    print("\n" + "=" * 70)
    print("REJECTION COUNTS by symbol, by cap")
    print("=" * 70)
    for cap in [5, 4, 3, 2, 1]:
        rej = rejections_by_cap[cap]
        if not rej:
            continue
        counts = pd.Series(rej).value_counts()
        print(f"\n--- cap={cap} (total rejections: {len(rej)}) ---")
        print(counts.to_string())


if __name__ == "__main__":
    main()
