"""
CCI(20) < -180 (1h timeframe) as a long entry trigger - candidate rule
sourced from a paid trading course's money-management lesson ("Vitali's
strategy" conspect, module 5). See project_igorcrypto_course_review memory
for the full course review; this tests ONLY the one genuinely formalizable,
non-proprietary piece (CCI threshold) - the course's other conditions for
this same entry (Elliott wave count, the closed-source "Traffic Light"
indicator, liquidation cascades) can't be coded up, either because they're
subjective (wave counting - already rejected project-wide, see
project_elliott_wave_analysis) or genuinely inaccessible (closed-source
paid indicator).

Course only specifies the threshold and timeframe (1h or 7m) - CCI's own
period isn't stated, so this uses the standard/default CCI(20). Only 1h is
testable (project has no sub-hourly OHLCV history - see cci_study.py's
sibling scripts, all hourly).

Autocorrelation discipline (this project's own repeated bug class, see
project_technical_indicators / project_traditional_markets_findings
memory): CCI is highly autocorrelated hour-to-hour, so treating every hour
CCI<-180 as an independent sample would massively inflate t-stats purely
from overlap. This instead counts only the ONSET of each oversold episode
(the hour CCI first crosses below -180 after being >=-180) as one event,
exactly like event_study.py treats discrete news events - forward return
measured from that single onset hour.

Usage:
    python scripts/cci_study.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

CCI_PERIOD = 20
CCI_THRESHOLD = -180
HOLDOUT_START = pd.Timestamp("2025-12-01", tz="UTC")


def compute_cci(ohlcv: pd.DataFrame, period: int = CCI_PERIOD) -> pd.Series:
    tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def onset_events(cci: pd.Series, threshold: float = CCI_THRESHOLD) -> pd.Series:
    """Boolean mask: True only on the hour CCI first crosses below threshold
    (was >=threshold or NaN the previous hour) - one event per episode, not
    every hour spent below it."""
    below = cci < threshold
    prev_below = below.shift(1, fill_value=False)
    return below & ~prev_below


def study_symbol(conn, symbol: str, horizons: list[int]) -> pd.DataFrame:
    ohlcv = pd.read_sql(
        "SELECT ts, open, high, low, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
        conn, params={"s": symbol},
    )
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    ohlcv["cci"] = compute_cci(ohlcv)
    ohlcv["onset"] = onset_events(ohlcv["cci"])

    rows = []
    for period_name, mask in [
        ("full", pd.Series(True, index=ohlcv.index)),
        ("train", ohlcv["ts"] < HOLDOUT_START),
        ("holdout", ohlcv["ts"] >= HOLDOUT_START),
    ]:
        events = ohlcv[ohlcv["onset"] & mask]
        for h in horizons:
            fwd_ret = ohlcv["close"].shift(-h) / ohlcv["close"] - 1
            sample = fwd_ret.loc[events.index].dropna()
            if len(sample) < 3:
                rows.append(dict(symbol=symbol, period=period_name, horizon_hours=h,
                                  n=len(sample), mean_ret=np.nan, t_stat=np.nan))
                continue
            mean_ret = sample.mean()
            t_stat = mean_ret / (sample.std(ddof=1) / np.sqrt(len(sample))) if sample.std(ddof=1) > 0 else np.nan
            rows.append(dict(symbol=symbol, period=period_name, horizon_hours=h,
                              n=len(sample), mean_ret=mean_ret, t_stat=t_stat))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str,
                         default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT")
    parser.add_argument("--horizons", type=str, default="24,72,168")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    conn = get_connection()
    all_results = []
    for symbol in symbols:
        res = study_symbol(conn, symbol, horizons)
        all_results.append(res)
        print(f"\n=== {symbol}: CCI(20) onset below {CCI_THRESHOLD} ===")
        with pd.option_context("display.width", 140):
            print(res.to_string(index=False))
    conn.close()

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("cci_study_summary.csv", index=False)
    print("\nFull summary written to cci_study_summary.csv")


if __name__ == "__main__":
    main()
