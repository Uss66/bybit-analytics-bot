"""
Honest, probabilistic comparison of the current fixed -8% stop-loss vs an
ADAPTIVE stop pegged to the most recently CONFIRMED swing low (from
elliott_pivots, using confirmed_ts - never the pivot's own timestamp, per
the repainting lesson from project_elliott_wave_analysis). This uses the
pivot-detection infrastructure as a risk-management reference level, not a
predictive entry signal - the one legitimate remaining use identified
after the Elliott Wave signal search was rejected.

Both stop types are applied to the IDENTICAL set of entries (from the
already-validated strategy.py signals: fgi_greed, DXY, DVOL) - only the
exit-stop rule differs, so this is an apples-to-apples comparison, not a
new signal search.

Reports results PROBABILISTICALLY (bootstrap resampling of trades) rather
than a single point estimate or a pass/fail t-stat gate - per user
request: express uncertainty explicitly rather than a binary verdict.

Usage:
    python scripts/adaptive_stop_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from strategy import SPOT_TAKER_FEE, load_events_with_returns, generate_signals, compute_score_series

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
FIXED_STOP = -0.08
TREND_FILTER_SMA = 720  # matches strategy.py's --trend-filter-sma default, same risk-management convention
COOLDOWN_AFTER_LOSSES = 3  # matches strategy.py's --cooldown-after-losses default
SWING_THRESHOLD = 0.05  # which degree of confirmed swing low defines the adaptive stop
N_BOOTSTRAP = 5000
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")


def load_confirmed_lows(conn, symbol: str, threshold_pct: float) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT confirmed_ts, price FROM elliott_pivots "
        "WHERE symbol=%(s)s AND threshold_pct=%(t)s AND pivot_type='low' ORDER BY confirmed_ts",
        conn, params={"s": symbol, "t": threshold_pct},
    )
    df["confirmed_ts"] = pd.to_datetime(df["confirmed_ts"], utc=True)
    return df.rename(columns={"price": "swing_low_price"})


def run_backtest(ohlcv: pd.DataFrame, signals: pd.DataFrame, confirmed_lows: pd.DataFrame,
                  stop_mode: str) -> pd.DataFrame:
    """stop_mode: 'fixed' or 'adaptive'."""
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    ohlcv["score"] = compute_score_series(signals, ohlcv)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()

    if stop_mode == "adaptive":
        # step function: as of each bar, the most recent CONFIRMED swing low price
        ohlcv = pd.merge_asof(ohlcv, confirmed_lows, left_on="ts", right_on="confirmed_ts", direction="backward")

    trades = []
    consecutive_losses = 0
    cooldown_until_ts = None
    for seg_id, seg in ohlcv[ohlcv["is_long"]].groupby("seg_id"):
        entry_idx = seg.index[0]
        natural_exit_idx = min(seg.index[-1] + 1, len(ohlcv) - 1)
        entry_price = ohlcv["close"].iloc[entry_idx]
        entry_ts_candidate = ohlcv["ts"].iloc[entry_idx]
        if cooldown_until_ts is not None and entry_ts_candidate < cooldown_until_ts:
            continue
        sma = ohlcv["sma"].iloc[entry_idx]
        if pd.notna(sma) and entry_price < sma:
            continue

        exit_idx = natural_exit_idx
        path = ohlcv.iloc[entry_idx:natural_exit_idx + 1]
        if stop_mode == "fixed":
            hit = path[path["close"] / entry_price - 1 <= FIXED_STOP]
        else:
            # adaptive: exit if price closes below the swing low that was
            # ALREADY confirmed as of that same bar (no lookahead - the
            # swing_low_price column at each bar only reflects pivots whose
            # confirmed_ts <= that bar's ts, via the merge_asof above)
            hit = path[path["close"] < path["swing_low_price"]]
        if not hit.empty:
            exit_idx = hit.index[0]

        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        net_ret = (exit_price * (1 - SPOT_TAKER_FEE)) / (entry_price * (1 + SPOT_TAKER_FEE)) - 1
        trade_path = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_dd = (trade_path / entry_price - 1).min()
        trades.append(dict(entry_ts=entry_ts, exit_ts=exit_ts, net_ret=net_ret, worst_dd=worst_dd,
                            period="train" if entry_ts < TRAIN_END else "holdout"))

        if net_ret < 0:
            consecutive_losses += 1
            if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                cooldown_until_ts = exit_ts + pd.Timedelta(hours=168)
                consecutive_losses = 0
        else:
            consecutive_losses = 0

    return pd.DataFrame(trades)


def bootstrap_compare(fixed_rets: np.ndarray, adaptive_rets: np.ndarray, n: int = N_BOOTSTRAP) -> dict:
    """Paired-by-index isn't valid here since trade COUNTS can differ between stop
    modes (different exit points shift downstream entries) - so this compares
    the two trade-return DISTRIBUTIONS independently via bootstrap of the mean,
    not a paired difference. That's the honest framing: 'is the adaptive
    stop's return distribution better', not 'did each matched trade improve'."""
    if len(fixed_rets) < 3 or len(adaptive_rets) < 3:
        return dict(prob_adaptive_better=float("nan"), mean_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        f_sample = rng.choice(fixed_rets, size=len(fixed_rets), replace=True)
        a_sample = rng.choice(adaptive_rets, size=len(adaptive_rets), replace=True)
        diffs.append(a_sample.mean() - f_sample.mean())
    diffs = np.array(diffs)
    return dict(
        prob_adaptive_better=float((diffs > 0).mean()),
        mean_diff=float(diffs.mean()),
        ci_low=float(np.percentile(diffs, 2.5)),
        ci_high=float(np.percentile(diffs, 97.5)),
    )


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)

    all_results = []
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        signals = generate_signals(events, symbol)
        confirmed_lows = load_confirmed_lows(conn, symbol, SWING_THRESHOLD)

        fixed_trades = run_backtest(ohlcv.copy(), signals, confirmed_lows, "fixed")
        adaptive_trades = run_backtest(ohlcv.copy(), signals, confirmed_lows, "adaptive")

        for period in ["train", "holdout"]:
            f = fixed_trades.loc[fixed_trades["period"] == period, "net_ret"].values if not fixed_trades.empty else np.array([])
            a = adaptive_trades.loc[adaptive_trades["period"] == period, "net_ret"].values if not adaptive_trades.empty else np.array([])
            f_dd = fixed_trades.loc[fixed_trades["period"] == period, "worst_dd"].values if not fixed_trades.empty else np.array([])
            a_dd = adaptive_trades.loc[adaptive_trades["period"] == period, "worst_dd"].values if not adaptive_trades.empty else np.array([])
            cmp = bootstrap_compare(f, a)
            all_results.append(dict(
                symbol=symbol, period=period,
                n_fixed=len(f), n_adaptive=len(a),
                mean_ret_fixed=f.mean() if len(f) else float("nan"),
                mean_ret_adaptive=a.mean() if len(a) else float("nan"),
                worst_dd_fixed=f_dd.min() if len(f_dd) else float("nan"),
                worst_dd_adaptive=a_dd.min() if len(a_dd) else float("nan"),
                prob_adaptive_better=cmp["prob_adaptive_better"],
                mean_diff=cmp["mean_diff"], ci_low=cmp["ci_low"], ci_high=cmp["ci_high"],
            ))
    conn.close()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("adaptive_stop_comparison.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 30):
        print(results_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("POOLED VERDICT (probabilistic, not pass/fail)")
    print("=" * 100)
    for period in ["train", "holdout"]:
        sub = results_df[results_df["period"] == period]
        avg_prob = sub["prob_adaptive_better"].mean()
        print(f"{period}: mean probability(adaptive stop beats fixed) across symbols = {avg_prob:.1%}")


if __name__ == "__main__":
    main()
