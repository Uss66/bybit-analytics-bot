"""
Confluence-gate study (2026-09-15): tests requiring >=2 simultaneously-
active BUY rules (fgi_greed/dxy_up_extreme/dvol_drop_extreme) instead of
the current OR-gate (any single active rule makes score>0) - see
strategy.py's compute_confluence_score_series() docstring and
project_confluence_gate_study memory for the motivation (Opus's critique
of the additive scoring as structurally weak once multiple rules can fire).

Important asymmetry to report honestly: dxy_up_extreme only applies to
ETH/SOL. For the other 5 symbols (BTC/XRP/DOGE/BNB/LINK), only 2 BUY
rules exist at all (fgi_greed, dvol_drop_extreme) - so "confluence>=2"
there means "require BOTH simultaneously", a much stricter bar than for
ETH/SOL (any 2-of-3). This changes what's actually being tested per
symbol and must not be glossed over when interpreting results.

Reuses the EXACT verified segment/stop-loss/trend-filter/cooldown/cap
logic via strategy.py's score_series override - no reimplementation.

Usage:
    python scripts/confluence_gate_study.py
"""
import pandas as pd

from db import get_connection
from strategy import (
    load_events_with_returns, generate_signals, compute_score_series,
    compute_confluence_score_series, compute_buy_rule_count_series,
    compute_candidate_segments, simulate_portfolio,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
MAX_POSITIONS = 3
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")


def portfolio_stats(trades_df, rejection_log):
    if trades_df.empty:
        return None
    trades_df = trades_df.sort_values("exit_ts")
    per_slot_capital = 10000.0 / len(SYMBOLS)
    equity_curve = [per_slot_capital * MAX_POSITIONS]
    for r in trades_df["ret"]:
        equity_curve.append(equity_curve[-1] + per_slot_capital * r)
    equity_series = pd.Series(equity_curve)
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    running_peak = equity_series.cummax()
    max_dd = ((equity_series - running_peak) / running_peak).min()
    ratio = total_return / abs(max_dd) if max_dd != 0 else float("nan")
    return dict(n=len(trades_df), win_rate=(trades_df["ret"] > 0).mean(), total_return=total_return,
                max_dd=max_dd, ratio=ratio, rejected=len(rejection_log))


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)
    ohlcv_by_symbol = {}
    for sym in SYMBOLS:
        df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": sym})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        ohlcv_by_symbol[sym] = df
    conn.close()

    # "mode": "full" = confluence required for the whole segment (REJECTED,
    # see project_confluence_gate_study memory); "entry" = confluence
    # checked ONLY at the segment's own start, baseline score still
    # defines segment boundaries/exit - the follow-up variant proposed
    # after "full" truncated trade duration to the shortest rule's window.
    variants = {
        "baseline (OR-gate, live)": (None, "full"),
        "confluence>=2 (full-segment)": (2, "full"),
        "confluence>=2 (entry-only)": (2, "entry"),
    }

    print(f"{'variant':<30}{'period':>8}{'n':>6}{'win_rate':>10}{'total_ret':>11}{'max_dd':>9}{'ratio':>8}{'rejected':>10}")
    for label, (min_buy, mode) in variants.items():
        all_segments = []
        for sym in SYMBOLS:
            signals = generate_signals(events, sym)
            ohlcv = ohlcv_by_symbol[sym]
            if min_buy is None:
                segs = compute_candidate_segments(sym, ohlcv, signals, stop_loss=STOP_LOSS,
                                                   trend_filter_sma=TREND_FILTER_SMA)
            elif mode == "full":
                score_series = compute_confluence_score_series(signals, ohlcv, min_buy_rules=min_buy)
                segs = compute_candidate_segments(sym, ohlcv, signals, stop_loss=STOP_LOSS,
                                                   trend_filter_sma=TREND_FILTER_SMA, score_series=score_series)
            else:  # entry-only: baseline segments, filtered by confluence AT ENTRY
                buy_count, _avoid = compute_buy_rule_count_series(signals, ohlcv)
                segs = compute_candidate_segments(sym, ohlcv, signals, stop_loss=STOP_LOSS,
                                                   trend_filter_sma=TREND_FILTER_SMA,
                                                   min_buy_rules_at_entry=min_buy, buy_count_series=buy_count)
            all_segments.extend(segs)

        for period_label, seg_filter in [
            ("train", lambda s: s["entry_ts"] < TRAIN_END),
            ("holdout", lambda s: s["entry_ts"] >= TRAIN_END),
        ]:
            period_segments = [s for s in all_segments if seg_filter(s)]
            trades_df, rejection_log = simulate_portfolio(period_segments, MAX_POSITIONS,
                                                            cooldown_after_losses=COOLDOWN_AFTER_LOSSES)
            stats = portfolio_stats(trades_df, rejection_log)
            if stats is None:
                print(f"{label:<28}{period_label:>8}{'0':>6}   (no trades)")
                continue
            print(f"{label:<28}{period_label:>8}{stats['n']:>6}{stats['win_rate']:>10.1%}"
                  f"{stats['total_return']:>11.2%}{stats['max_dd']:>9.2%}{stats['ratio']:>8.2f}{stats['rejected']:>10}")

    # --- per-symbol duty-cycle sanity check for the confluence variant ---
    print("\n" + "=" * 90)
    print("Per-symbol confluence>=2 duty cycle (sanity check - how often does this even fire)")
    print("=" * 90)
    for sym in SYMBOLS:
        signals = generate_signals(events, sym)
        ohlcv = ohlcv_by_symbol[sym]
        conf_score = compute_confluence_score_series(signals, ohlcv, min_buy_rules=2)
        base_score = compute_score_series(signals, ohlcv)
        n_rules_applicable = signals[signals["action"] == "buy"]["rule"].nunique()
        print(f"  {sym:<10} buy-rules applicable: {n_rules_applicable}  "
              f"confluence-active: {(conf_score > 0).mean():.2%}   baseline-active: {(base_score > 0).mean():.2%}")


if __name__ == "__main__":
    main()
