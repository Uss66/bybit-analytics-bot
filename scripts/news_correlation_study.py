"""
Does the currently-active strategy (fgi_greed + DXY + DVOL) correlate with
the original news-background data this project started from (whale alerts,
key_person_statement, macro/regulation events)?

Tests same-day and lag-1 Pearson correlation between daily news-volume
counts and (a) |Deribit BTC DVOL 1-day change|, (b) Fear & Greed Index
value - split into train (<=2025-11-30) and holdout (>=2025-12-01) to match
the project's standard validation convention, so a correlation that only
shows up in one period (overfitting/noise) can be told apart from one that
holds in both (real, stable relationship).

Usage:
    python scripts/news_correlation_study.py
"""
import math

import numpy as np
import pandas as pd

from db import get_connection

TRAIN_END = pd.Timestamp("2025-11-30", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-12-01", tz="UTC")


def pearson_with_p(x: pd.Series, y: pd.Series):
    """Pearson r plus a two-tailed p-value via the t-distribution
    approximation (no scipy dependency available in this environment)."""
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 10:
        return float("nan"), float("nan"), n
    r = np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    if abs(r) >= 1.0:
        return r, 0.0, n
    t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
    # two-tailed p-value from the t-distribution CDF via the regularized
    # incomplete beta function (avoids a scipy dependency)
    dof = n - 2
    x_beta = dof / (dof + t_stat ** 2)
    p = _betainc(dof / 2, 0.5, x_beta)
    return r, p, n


def _betainc(a, b, x, iters=200):
    """Regularized incomplete beta function via continued fraction
    (Numerical Recipes' betacf), enough precision for a p-value display."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x, iters) / a
    return 1.0 - front * _betacf(b, a, 1 - x, iters) / b


def _betacf(a, b, x, iters):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h


def main():
    conn = get_connection()

    whale = pd.read_sql("""
        SELECT date_trunc('day', ts) AS day, COUNT(*) n_whale_50m
        FROM events WHERE source='telegram:whale_alert_io'
          AND (raw->>'usd_value')::numeric >= 50000000
        GROUP BY day
    """, conn)

    kps = pd.read_sql("""
        SELECT date_trunc('day', ts) AS day, COUNT(*) n_kps
        FROM events WHERE category='key_person_statement'
        GROUP BY day
    """, conn)

    macro = pd.read_sql("""
        SELECT date_trunc('day', ts) AS day, COUNT(*) n_macro_reg
        FROM events WHERE category IN ('macro','regulation')
        GROUP BY day
    """, conn)

    dvol = pd.read_sql("SELECT ts AS day, close FROM deribit_dvol WHERE symbol='BTC' ORDER BY ts", conn)
    fgi = pd.read_sql("SELECT ts AS day, value FROM fear_greed_index ORDER BY ts", conn)
    conn.close()

    for df in (whale, kps, macro, dvol, fgi):
        df["day"] = pd.to_datetime(df["day"], utc=True).dt.floor("D")

    dvol["dvol_chg_1d"] = dvol["close"].pct_change(1).abs()

    merged = dvol[["day", "dvol_chg_1d"]].merge(fgi[["day", "value"]], on="day", how="outer")
    merged = merged.merge(whale, on="day", how="left").merge(kps, on="day", how="left").merge(macro, on="day", how="left")
    merged[["n_whale_50m", "n_kps", "n_macro_reg"]] = merged[["n_whale_50m", "n_kps", "n_macro_reg"]].fillna(0)
    merged = merged.sort_values("day").reset_index(drop=True)

    for col in ["n_whale_50m", "n_kps", "n_macro_reg"]:
        merged[f"{col}_lag1"] = merged[col].shift(1)

    news_cols = ["n_whale_50m", "n_kps", "n_macro_reg"]
    lag_cols = [f"{c}_lag1" for c in news_cols]

    periods = {
        "TRAIN (<=2025-11-30)": merged[merged["day"] <= TRAIN_END],
        "HOLDOUT (>=2025-12-01)": merged[merged["day"] >= HOLDOUT_START],
    }

    for label, df in periods.items():
        print("=" * 70)
        print(f"{label}  (n days = {len(df)})")
        print("=" * 70)

        print("\nSame-day Pearson correlation with |DVOL 1d change|:")
        for col in news_cols:
            r, p, n = pearson_with_p(df[col], df["dvol_chg_1d"])
            print(f"  {col:14s}: r={r:+.4f}  p={p:.4f}  n={n}")

        print("\nSame-day Pearson correlation with FGI value:")
        for col in news_cols:
            r, p, n = pearson_with_p(df[col], df["value"])
            print(f"  {col:14s}: r={r:+.4f}  p={p:.4f}  n={n}")

        print("\nLagged: yesterday's news count vs TODAY's |DVOL 1d change|:")
        for col in lag_cols:
            r, p, n = pearson_with_p(df[col], df["dvol_chg_1d"])
            print(f"  {col:18s}: r={r:+.4f}  p={p:.4f}  n={n}")
        print()


if __name__ == "__main__":
    main()
