"""
Leverage/futures study (2026-09-15) - user wants to explore a MORE risky
variant using leveraged perpetual futures instead of spot, explicitly
WITHOUT touching the current live spot strategy ("не разрушая текущую") -
this is a pure research/backtest track, nothing here executes real
trades or changes testnet_trader.py.

Scope limitation, reported honestly: funding-rate data (the new cost
factor perpetual futures have that spot doesn't) only exists for
BTC/ETH/SOL, only from 2024-08-28 onward (~2 years) - see `funding_rate`
table, already collected for the earlier (rejected) funding/OI signal
study. DOGE/BNB/LINK/XRP and any period before 2024-08-28 (including the
2022 bear-market stress test) CANNOT be included in a funding-cost-aware
leverage simulation with current data.

Uses the EXACT SAME validated entries/exits as the live spot strategy
(compute_candidate_segments, unchanged signals/stop-loss/trend-filter) -
leverage only changes position sizing and adds funding cost, never the
entry/exit timing itself.

LIQUIDATION RISK CHECK (the most important part - not just a return
number): for each historical trade, checks whether its
worst_intratrade_dd (the worst mark-to-market drawdown actually observed
during the hold - i.e. what really happened intra-trade, not just the
close-to-close numbers) would have breached an approximate liquidation
threshold at each tested leverage BEFORE the -8% stop-loss could have
executed. A leverage level is only meaningful to discuss further if this
comes back clean.

Usage:
    python scripts/leverage_study.py
"""
import pandas as pd

from db import get_connection
from strategy import load_events_with_returns, generate_signals, backtest, SPOT_TAKER_FEE

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # only ones with funding_rate data
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
FUNDING_DATA_START = pd.Timestamp("2024-08-28", tz="UTC")
LEVERAGES = [2, 3, 5, 10]
# Approx isolated-margin liquidation threshold: roughly -(1/leverage) plus
# a maintenance-margin buffer (~0.5% typical on Bybit for majors) - a
# simplification (real formula also nets in unrealized PnL/fees) but
# conservative enough to flag genuine danger zones.
MAINTENANCE_MARGIN = 0.005


def liquidation_threshold(leverage: float) -> float:
    return -(1 / leverage) + MAINTENANCE_MARGIN


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)
    funding = pd.read_sql("SELECT symbol, ts, funding_rate FROM funding_rate ORDER BY symbol, ts", conn)
    funding["ts"] = pd.to_datetime(funding["ts"], utc=True)

    all_trades = []
    for sym in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s AND ts >= %(start)s ORDER BY ts",
                             conn, params={"s": sym, "start": FUNDING_DATA_START})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        signals = generate_signals(events, sym)
        trades_df, stats = backtest(sym, ohlcv, signals, capital=10000, fee=SPOT_TAKER_FEE,
                                     stop_loss=STOP_LOSS, trend_filter_sma=TREND_FILTER_SMA,
                                     cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
        if trades_df.empty:
            continue
        trades_df["symbol"] = sym
        all_trades.append(trades_df)
    conn.close()

    trades = pd.concat(all_trades, ignore_index=True)
    print(f"Trades since {FUNDING_DATA_START.date()} (BTC/ETH/SOL only, spot entries/exits unchanged): {len(trades)}")
    print(f"Worst single-trade intratrade drawdown observed: {trades['worst_intratrade_dd'].min():.2%}")
    print(f"5 worst intratrade drawdowns: {sorted(trades['worst_intratrade_dd'])[:5]}")

    print("\n" + "=" * 100)
    print("LIQUIDATION RISK CHECK (would any historical trade have breached the liquidation")
    print("threshold before the -8% stop could execute?)")
    print("=" * 100)
    for lev in LEVERAGES:
        thresh = liquidation_threshold(lev)
        breaches = trades[trades["worst_intratrade_dd"] <= thresh]
        flag = "DANGER" if not breaches.empty else "clean"
        print(f"  {lev}x leverage: liquidation ~{thresh:.1%} - {len(breaches)} historical trade(s) "
              f"would have breached it  [{flag}]")
        if not breaches.empty:
            print(breaches[["symbol", "entry_ts", "worst_intratrade_dd", "ret"]].to_string(index=False))

    def net_leveraged_ret(t, lev):
        fseries = funding[(funding["symbol"] == t.symbol) & (funding["ts"] >= t.entry_ts) & (funding["ts"] < t.exit_ts)]
        funding_cost = fseries["funding_rate"].sum() * lev  # long pays when rate>0, receives when <0; scales with leveraged notional
        fee_cost = SPOT_TAKER_FEE * 2 * lev  # round-trip fee also scales with notional
        return t.gross_ret * lev - funding_cost - fee_cost

    print("\n" + "=" * 100)
    print("LEVERAGED vs SPOT: per-symbol compounded total return, same entries/exits, funding-cost-aware")
    print("=" * 100)
    print(f"{'symbol':<10}{'n':>5}{'spot (live, ref)':>18}" + "".join(f"{f'{lev}x':>14}" for lev in LEVERAGES))
    for sym in SYMBOLS:
        sym_trades = trades[trades["symbol"] == sym].sort_values("exit_ts")
        spot_equity = 10000.0
        for r in sym_trades["ret"]:
            spot_equity *= (1 + r)
        row = f"{sym:<10}{len(sym_trades):>5}{spot_equity / 10000 - 1:>18.2%}"
        for lev in LEVERAGES:
            lev_equity = 10000.0
            for t in sym_trades.itertuples():
                lev_equity *= (1 + net_leveraged_ret(t, lev))
            row += f"{lev_equity / 10000 - 1:>14.2%}"
        print(row)

    print("\nTotal funding cost paid, by symbol and leverage (positive = cost, negative = you were paid):")
    for sym in SYMBOLS:
        sym_trades = trades[trades["symbol"] == sym]
        line = f"  {sym:<10}"
        for lev in LEVERAGES:
            total_funding = sum(
                funding[(funding["symbol"] == t.symbol) & (funding["ts"] >= t.entry_ts) & (funding["ts"] < t.exit_ts)]["funding_rate"].sum() * lev
                for t in sym_trades.itertuples()
            )
            line += f"  {lev}x: {total_funding:+.2%}"
        print(line)


if __name__ == "__main__":
    main()
