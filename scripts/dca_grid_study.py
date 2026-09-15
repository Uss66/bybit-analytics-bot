"""
Stress-test of "Vitali's averaging strategy" (course conspect, module 5,
lesson 5 - see project_igorcrypto_course_review memory) as a standalone
money-management mechanism, independent of any entry-timing signal.

Grid (the course's own "uptrend" example, $20k reference deposit, rescaled
to whatever --capital is passed):
  entry:  10% of capital
  TP:     +1.1% from average entry (course says 0.7-1.5%, using midpoint)
  DCA 1:  -3% from average entry  -> add 20% of capital
  DCA 2:  -7% from (new) average  -> add 30% of capital
  DCA 3:  -17% from (new) average -> add 45% of capital
  No further averaging past level 3 (course: "in 98% of cases we don't
  reach the 3rd averaging" - implying this is treated as close to a floor,
  not that a 4th level exists). NO stop-loss anywhere - explicit in the
  course ("we trade WITHOUT stops - we use averaging").

This is a martingale-style mechanism: position size increases as price
moves further against you, funded entirely by hoping for eventual mean
reversion, with no forced exit if that reversion doesn't come. The
concerning scenario isn't the common case (small pullback, TP hits,
grid recycles) - it's a sustained one-directional move that exhausts all
three averaging levels and then just sits, marked deeply underwater,
waiting indefinitely for a recovery. This script measures exactly that:
how often the grid blows through all 3 levels, how deep the resulting
drawdown gets, and whether/when it recovers - both over full history and
specifically over the 2022 bear market (Luna/FTX), which this project
already measured for the live cap=3 strategy and for buy&hold - see
project_2022_bear_market_test memory (+14.3% for the live strategy vs
-34%...-94% buy&hold across the 7 symbols) - so this run is directly
comparable.

No entry-timing signal is applied here on purpose - a new grid opens
immediately (next bar) after the previous one closes at TP, for the
entire window. This isolates the money-management mechanism's own risk
profile from any question of signal quality.

Usage:
    python scripts/dca_grid_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT --capital 10000
    python scripts/dca_grid_study.py --symbols SOLUSDT --start-date 2022-01-01 --end-date 2022-12-31
"""
import argparse

import pandas as pd

from db import get_connection

TP_PCT = 0.011  # midpoint of the course's stated 0.7%-1.5% range
DCA_LEVELS = [
    dict(trigger_pct=-0.03, add_frac=0.20),
    dict(trigger_pct=-0.07, add_frac=0.30),
    dict(trigger_pct=-0.17, add_frac=0.45),
]
ENTRY_FRAC = 0.10


def simulate(ohlcv: pd.DataFrame, capital: float) -> tuple[pd.DataFrame, dict]:
    prices = ohlcv["close"].values
    ts = ohlcv["ts"].values
    n = len(prices)

    cycles = []
    i = 0
    max_open_dd_series = []  # (ts, mark_to_market_pct) while a grid is open, for the whole run
    while i < n:
        entry_price = prices[i]
        entry_ts = ts[i]
        invested = capital * ENTRY_FRAC
        shares = invested / entry_price
        avg_price = entry_price
        level_reached = 0
        worst_mtm = 0.0
        worst_mtm_ts = entry_ts
        j = i
        closed = False
        while j < n:
            price = prices[j]
            mtm = price / avg_price - 1
            if mtm < worst_mtm:
                worst_mtm = mtm
                worst_mtm_ts = ts[j]
            max_open_dd_series.append((ts[j], mtm))
            if mtm >= TP_PCT:
                closed = True
                break
            if level_reached < len(DCA_LEVELS):
                lvl = DCA_LEVELS[level_reached]
                if mtm <= lvl["trigger_pct"]:
                    add_amount = capital * lvl["add_frac"]
                    add_shares = add_amount / price
                    invested += add_amount
                    shares += add_shares
                    avg_price = invested / shares
                    level_reached += 1
                    continue  # re-check TP/next level against new avg_price same bar next loop
            j += 1
        exit_j = j if closed else n - 1
        exit_price = prices[exit_j]
        final_value = shares * exit_price
        cycles.append(dict(
            entry_ts=entry_ts, exit_ts=ts[exit_j], entry_price=entry_price, exit_price=exit_price,
            invested=invested, final_value=final_value, ret=final_value / invested - 1,
            levels_reached=level_reached, worst_mtm=worst_mtm, worst_mtm_ts=worst_mtm_ts,
            closed_at_tp=closed, hours_open=(pd.Timestamp(ts[exit_j]) - pd.Timestamp(entry_ts)).total_seconds() / 3600,
        ))
        i = exit_j + 1 if closed else n  # if never closed, this cycle runs off the end of the data

    cycles_df = pd.DataFrame(cycles)
    stats = dict(
        n_cycles=len(cycles_df),
        n_blew_through_all_3=int((cycles_df["levels_reached"] == 3).sum()) if not cycles_df.empty else 0,
        worst_single_cycle_mtm=cycles_df["worst_mtm"].min() if not cycles_df.empty else None,
        pct_cycles_closed_at_tp=cycles_df["closed_at_tp"].mean() if not cycles_df.empty else None,
        avg_hours_per_cycle=cycles_df["hours_open"].mean() if not cycles_df.empty else None,
        max_hours_stuck_in_one_cycle=cycles_df["hours_open"].max() if not cycles_df.empty else None,
    )
    return cycles_df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str,
                         default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    conn = get_connection()
    all_cycles = []
    for symbol in symbols:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        ohlcv = ohlcv.reset_index(drop=True)
        if len(ohlcv) < 10:
            print(f"\n{symbol}: not enough data in window")
            continue

        cycles_df, stats = simulate(ohlcv, args.capital)
        buy_hold_ret = ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1

        # portfolio-level: sum of realized cycle P&L + mark-to-market of any still-open cycle at the end
        equity = args.capital
        for _, c in cycles_df.iterrows():
            equity += (c["final_value"] - c["invested"]) if c["closed_at_tp"] else 0
        last_cycle = cycles_df.iloc[-1] if not cycles_df.empty else None
        still_open = last_cycle is not None and not last_cycle["closed_at_tp"]
        if still_open:
            equity += last_cycle["final_value"] - last_cycle["invested"]  # marked at window's last price
        total_return = equity / args.capital - 1

        print(f"\n=== {symbol} ({len(ohlcv)} candles, {ohlcv['ts'].iloc[0].date()} -> {ohlcv['ts'].iloc[-1].date()}) ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"  grid_total_return: {total_return:.4f}")
        print(f"  buy_hold_return: {buy_hold_ret:.4f}")
        print(f"  still_open_at_window_end: {still_open}")
        if still_open:
            print(f"  final_open_cycle: entered {last_cycle['entry_ts']}, levels_reached={last_cycle['levels_reached']}, "
                  f"mark_to_market={last_cycle['ret']:.4f}")

        cycles_df["symbol"] = symbol
        all_cycles.append(cycles_df)
    conn.close()

    if all_cycles:
        pd.concat(all_cycles, ignore_index=True).to_csv("dca_grid_cycles.csv", index=False)
        print("\nFull cycle log written to dca_grid_cycles.csv")


if __name__ == "__main__":
    main()
