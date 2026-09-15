"""
Full event study (train/holdout, per-symbol) for the 3 candidate technical
indicators that passed the cheap redundancy pre-check
(indicator_correlation_precheck.py, all |r|<0.3 vs active rules) - see
project_technical_indicators memory for the Opus/Fable design consult this
follows.

Horizons are PRE-REGISTERED (one per indicator/direction, per Opus's own
suggestion) to avoid multiple-testing inflation from horizon-shopping:
  - cross-sectional relative strength: 72h (Opus's spec)
  - volume confirmation (z-score, split by concurrent-return sign): 24h
    (Opus's spec, also matches DVOL's already-validated 24h convention)
  - RSI(14) DAILY, not hourly: 72h (Opus's spec)

For each, tests EVERY symbol independently (this project's convention -
FGI/DVOL were validated per-symbol, not pooled), reports duty cycle,
train t-stat, holdout t-stat, and flags n<30 as below this project's
standing confidence bar.

Usage:
    python scripts/technical_indicators_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from detect_elliott_pivots import resample_to_daily
from elliott_wave_study_v3 import forward_return, t_stat

TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
CONF_N = 30  # this project's standing confidence bar for holdout n


def compute_rsi(daily_closes: pd.Series, period: int = 14) -> pd.Series:
    delta = daily_closes.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def episode_starts(is_active: pd.Series, ts: pd.Series) -> pd.Series:
    """Collapses a boolean condition into only the START of each contiguous
    True-run (state transition False->True) - mirrors strategy.py's own
    segment construction (seg_id = is_long.ne(is_long.shift()).cumsum()).

    CRITICAL FIX (2026-09-14): the first pass of this study fed EVERY
    hourly timestamp where a slow-moving condition (168h trailing rank,
    30d trailing volume z-score) held true directly into forward_return()
    - thousands of highly overlapping, non-independent observations of the
    SAME underlying multi-day regime, each measuring an overlapping chunk
    of the same forward window. This is the exact autocorrelation-inflation
    trap already caught once in this project (see
    project_traditional_markets_findings - the first DXY pass had the same
    flaw). It produced t-stats up to 14.7 with n in the thousands - an
    order of magnitude beyond anything else validated in this project,
    which alone should have triggered suspicion per this project's own
    standing practice. Fixed by testing only genuinely independent episode-
    start events, same as every segment-based signal elsewhere in this
    project."""
    is_active = is_active.astype(bool).reset_index(drop=True)
    ts = ts.reset_index(drop=True)
    starts_mask = is_active & ~is_active.shift(fill_value=False)
    return ts[starts_mask.values]


def event_rows(events_ts: pd.Series, ohlcv: pd.DataFrame, horizon: float):
    rows = []
    for ts in events_ts:
        ret = forward_return(ohlcv, ts, horizon)
        if ret is not None:
            period = "train" if ts < TRAIN_END else "holdout"
            rows.append(dict(ts=ts, period=period, ret=ret))
    return pd.DataFrame(rows)


def report(label, df):
    if df.empty:
        print(f"  {label:<45} (no events)")
        return
    for period in ["train", "holdout"]:
        sub = df[df["period"] == period]
        if len(sub) < 5:
            print(f"  {label:<45} {period:>8}: n={len(sub):<4} (too few)")
            continue
        mean, std, t, n = t_stat(sub["ret"].tolist())
        flag = "" if n >= CONF_N or period == "train" else "  [n<30, below confidence bar]"
        print(f"  {label:<45} {period:>8}: n={n:<4} mean={mean:+.4f} t={t:+.2f}{flag}")


def main():
    conn = get_connection()
    closes_by_symbol = {}
    ohlcv_by_symbol = {}
    for sym in ALL_SYMBOLS:
        df = pd.read_sql("SELECT ts, close, volume FROM ohlcv WHERE symbol=%(s)s ORDER BY ts", conn, params={"s": sym})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        ohlcv_by_symbol[sym] = df
        closes_by_symbol[sym] = df.set_index("ts")["close"]
    conn.close()

    common_ts = ohlcv_by_symbol["BTCUSDT"]["ts"]
    ret_matrix = pd.DataFrame({sym: s.pct_change(168) for sym, s in closes_by_symbol.items()})
    ret_matrix = ret_matrix.reindex(common_ts).ffill()
    rank_matrix = ret_matrix.rank(axis=1, ascending=False)

    print("=" * 100)
    print("CANDIDATE 1: Cross-sectional relative strength (168h return rank among 7 symbols), 72h horizon")
    print("=" * 100)
    for sym in ALL_SYMBOLS:
        ohlcv = ohlcv_by_symbol[sym]
        is_top2 = rank_matrix[sym] <= 2
        is_bottom2 = rank_matrix[sym] >= 6
        top2_ts = episode_starts(is_top2, common_ts)
        bottom2_ts = episode_starts(is_bottom2, common_ts)
        print(f" {sym}: {is_top2.sum()} hourly-true / {len(top2_ts)} episodes (top2); "
              f"{is_bottom2.sum()} hourly-true / {len(bottom2_ts)} episodes (bottom2)")
        report("  top2 (continuation-long hyp.)", event_rows(top2_ts, ohlcv, 72))
        report("  bottom2 (continuation-short/reversal-long hyp.)", event_rows(bottom2_ts, ohlcv, 72))

    print("\n" + "=" * 100)
    print("CANDIDATE 2: Volume z-score top quintile (30d trailing), split by concurrent 24h return sign, 24h horizon")
    print("=" * 100)
    for sym in ALL_SYMBOLS:
        ohlcv = ohlcv_by_symbol[sym].copy()
        ohlcv["vol_ma"] = ohlcv["volume"].rolling(30 * 24, min_periods=30 * 24 // 3).mean()
        ohlcv["vol_std"] = ohlcv["volume"].rolling(30 * 24, min_periods=30 * 24 // 3).std()
        ohlcv["vol_z"] = (ohlcv["volume"] - ohlcv["vol_ma"]) / ohlcv["vol_std"]
        ohlcv["ret24"] = ohlcv["close"].pct_change(24)
        q80 = ohlcv["vol_z"].quantile(0.8)
        high_vol = ohlcv["vol_z"] > q80
        advance = (high_vol & (ohlcv["ret24"] > 0)).reset_index(drop=True)
        decline = (high_vol & (ohlcv["ret24"] < 0)).reset_index(drop=True)
        advance_ts = episode_starts(advance, ohlcv["ts"])
        decline_ts = episode_starts(decline, ohlcv["ts"])
        print(f" {sym}: {advance.sum()} hourly-true / {len(advance_ts)} episodes (advance); "
              f"{decline.sum()} hourly-true / {len(decline_ts)} episodes (decline)")
        report("  high-vol advance (continuation hyp.)", event_rows(advance_ts, ohlcv, 24))
        report("  high-vol decline (capitulation-reversal hyp.)", event_rows(decline_ts, ohlcv, 24))

    print("\n" + "=" * 100)
    print("CANDIDATE 3: RSI(14) on DAILY bars, <30 oversold / >70 overbought, 72h horizon")
    print("=" * 100)
    for sym in ALL_SYMBOLS:
        ohlcv = ohlcv_by_symbol[sym]
        daily_closes = resample_to_daily(ohlcv[["ts", "close"]])
        rsi = compute_rsi(daily_closes, 14)
        rsi_ts = pd.Series(rsi.index)
        oversold_ts = episode_starts(pd.Series((rsi < 30).values), rsi_ts)
        overbought_ts = episode_starts(pd.Series((rsi > 70).values), rsi_ts)
        print(f" {sym}: {(rsi < 30).sum()} daily-true / {len(oversold_ts)} episodes (oversold); "
              f"{(rsi > 70).sum()} daily-true / {len(overbought_ts)} episodes (overbought)")
        report("  oversold (mean-reversion-long hyp.)", event_rows(pd.Series(oversold_ts), ohlcv, 72))
        report("  overbought (mean-reversion-short hyp.)", event_rows(pd.Series(overbought_ts), ohlcv, 72))


if __name__ == "__main__":
    main()
