"""
Cheap redundancy pre-check (2026-09-14, per Opus's suggestion during the
"experienced trader"/technical-indicator expansion, see
project_technical_indicators memory): before spending a full event-study +
train/holdout cycle on a new candidate indicator, correlate its activation
series against the 3 already-active rules (fgi_greed, dvol_drop_extreme,
dvol_spike_extreme, dxy_up_extreme). |r|>0.6 against an existing rule means
near-zero expected marginal value even with a great standalone t-stat -
the exact failure mode that already hurt key_person_statement (looked real
standalone, was actually redundant/harmful once combined). This kills bad
candidates for the price of a correlation, not a full validation cycle.

Three candidates checked (Opus's top picks):
  1. Cross-sectional relative strength: is BTC in the top-2 or bottom-2 of
     the 7 symbols by trailing 168h return?
  2. Volume z-score: is BTC's volume in the top quintile of its trailing
     30-day distribution?
  3. RSI(14) on DAILY bars (not hourly - Opus's explicit point: hourly RSI
     is mostly microstructure noise): is BTC's daily RSI <30 or >70?

Usage:
    python scripts/indicator_correlation_precheck.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from strategy import load_events_with_returns, generate_signals
from detect_elliott_pivots import resample_to_daily

REF_SYMBOL = "BTCUSDT"   # fgi_greed + dvol rules are keyed off BTC's own signal series
DXY_SYMBOL = "ETHUSDT"   # dxy_up/down_extreme only apply to ETH/SOL, not BTC
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def rule_active_series(signals: pd.DataFrame, ohlcv: pd.DataFrame, rule_name: str) -> pd.Series:
    """Same interval-counting logic as strategy.py's compute_score_series,
    isolated to a single rule, returning a 0/1 series aligned to ohlcv['ts']."""
    group = signals[signals["rule"] == rule_name]
    if group.empty:
        return pd.Series(0.0, index=ohlcv.index)
    starts = pd.DataFrame({"ts": group["ts"], "delta": 1})
    ends = pd.DataFrame({"ts": group["ts"] + pd.to_timedelta(group["hold_hours"], unit="h"), "delta": -1})
    changes = pd.concat([starts, ends], ignore_index=True).groupby("ts", as_index=False)["delta"].sum().sort_values("ts")
    changes["active_count"] = changes["delta"].cumsum()
    merged = pd.merge_asof(ohlcv[["ts"]], changes[["ts", "active_count"]], on="ts", direction="backward")
    return (merged["active_count"].fillna(0) > 0).astype(float)


def compute_rsi(daily_closes: pd.Series, period: int = 14) -> pd.Series:
    delta = daily_closes.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def main():
    conn = get_connection()
    events = load_events_with_returns(conn)

    ohlcv_ref = pd.read_sql("SELECT ts, close, volume FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": REF_SYMBOL})
    ohlcv_ref["ts"] = pd.to_datetime(ohlcv_ref["ts"], utc=True)
    ohlcv_dxy = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": DXY_SYMBOL})
    ohlcv_dxy["ts"] = pd.to_datetime(ohlcv_dxy["ts"], utc=True)

    closes_by_symbol = {}
    for sym in ALL_SYMBOLS:
        df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": sym})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        closes_by_symbol[sym] = df.set_index("ts")["close"]
    conn.close()

    signals_ref = generate_signals(events, REF_SYMBOL)
    signals_dxy = generate_signals(events, DXY_SYMBOL)

    active_rules = {
        "fgi_greed": rule_active_series(signals_ref, ohlcv_ref, "fgi_greed"),
        "dvol_drop_extreme": rule_active_series(signals_ref, ohlcv_ref, "dvol_drop_extreme"),
        "dvol_spike_extreme": rule_active_series(signals_ref, ohlcv_ref, "dvol_spike_extreme"),
        "dxy_up_extreme (on ETH)": rule_active_series(signals_dxy, ohlcv_dxy, "dxy_up_extreme"),
    }

    # --- Candidate 1: cross-sectional relative strength (BTC's own rank) ---
    ret_matrix = pd.DataFrame({sym: s.pct_change(168) for sym, s in closes_by_symbol.items()})
    ret_matrix = ret_matrix.reindex(ohlcv_ref["ts"]).ffill()
    rank_matrix = ret_matrix.rank(axis=1, ascending=False)  # 1 = highest return
    btc_top2 = (rank_matrix["BTCUSDT"] <= 2).astype(float).reset_index(drop=True)
    btc_bottom2 = (rank_matrix["BTCUSDT"] >= 6).astype(float).reset_index(drop=True)

    # --- Candidate 2: volume z-score, BTC top quintile ---
    ohlcv_ref["vol_ma30d"] = ohlcv_ref["volume"].rolling(30 * 24, min_periods=30 * 24 // 3).mean()
    ohlcv_ref["vol_std30d"] = ohlcv_ref["volume"].rolling(30 * 24, min_periods=30 * 24 // 3).std()
    ohlcv_ref["vol_z"] = (ohlcv_ref["volume"] - ohlcv_ref["vol_ma30d"]) / ohlcv_ref["vol_std30d"]
    vol_high = (ohlcv_ref["vol_z"] > ohlcv_ref["vol_z"].quantile(0.8)).astype(float)

    # --- Candidate 3: RSI(14) daily, BTC ---
    daily_closes = resample_to_daily(ohlcv_ref[["ts", "close"]])
    rsi = compute_rsi(daily_closes, 14)
    rsi_df = pd.DataFrame({"ts": rsi.index, "oversold": (rsi < 30).astype(float).values,
                            "overbought": (rsi > 70).astype(float).values})
    rsi_hourly = pd.merge_asof(ohlcv_ref[["ts"]], rsi_df, on="ts", direction="backward")

    candidates = {
        "cross_sect_top2_BTC": btc_top2,
        "cross_sect_bottom2_BTC": btc_bottom2,
        "volume_zscore_top20pct_BTC": vol_high.reset_index(drop=True),
        "rsi14_daily_oversold_BTC": rsi_hourly["oversold"],
        "rsi14_daily_overbought_BTC": rsi_hourly["overbought"],
    }

    print(f"{'candidate':<28}{'duty_cycle':>12}" + "".join(f"{name:>26}" for name in active_rules))
    for cname, cseries in candidates.items():
        cseries = pd.Series(cseries).reset_index(drop=True)
        duty = cseries.mean()
        row = f"{cname:<28}{duty:>12.1%}"
        for rname, rseries in active_rules.items():
            rseries = pd.Series(rseries).reset_index(drop=True)
            n = min(len(cseries), len(rseries))
            valid = pd.DataFrame({"c": cseries.values[:n], "r": rseries.values[:n]}).dropna()
            r = np.corrcoef(valid["c"], valid["r"])[0, 1] if len(valid) > 1 and valid["c"].std() > 0 and valid["r"].std() > 0 else float("nan")
            row += f"{r:>26.3f}"
        print(row)

    print("\nFlagging: |r| > 0.6 against any active rule = expect near-zero marginal value, deprioritize.")


if __name__ == "__main__":
    main()
