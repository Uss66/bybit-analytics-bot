"""
Does order-flow imbalance predict the next hours? (2026-10-15, candidate 3 of
the reading-list note)

Imbalance of a window = (aggressive buys - aggressive sells) / total turnover,
from Binance's hourly klines (taker_buy_quote and quote_volume):
        imb = (2 * taker_buy_quote - quote_volume) / quote_volume   in [-1, 1]

Not the volume z-score the project already rejected: that measures HOW MUCH
traded, this measures WHO crossed the spread.

Honest timing, spelled out:
  * a candle is stamped with its OPEN; its flow is known at open + 1h;
  * the feature at time T uses candles that opened before T, i.e. the last
    W hours ending exactly at T, and the forward return runs from T to T+k,
    both measured on the candle CLOSES (price at T = close of the candle that
    opened at T - 1h). Nothing in the window overlaps the forward return.
  * as a sanity check the SAME-hour correlation (feature vs the return of the
    window it was built from) is printed: order flow and price move together
    in the hour, trivially. The question is only the next hours.

Design, fixed before any result:
  * windows W = 1, 4, 24 hours of cumulative imbalance;
  * the feature is turned into a percentile against the PREVIOUS 365 days of
    the same coin (no look-ahead); "high" = top quintile, "low" = bottom;
  * forward horizons k = 1, 4, 24 hours; samples taken every k hours so that
    windows do not overlap;
  * the seven coins move together (0.8+), so at each sampled time the coins
    are averaged first and t is taken over TIMES, not over coin-times;
  * train < 2025-12-01 <= holdout; quintile spread = mean forward return of
    the high group minus the low group.

Predeclared bar (all of it): spread t >= 3 in absolute value on train, the same
sign with |t| >= 2 on holdout, AND the spread larger than the 0.36% a round
trip costs (otherwise it cannot be traded even if real). 3 x 3 = 9 cells
x 2 periods; with that many looks a few |t| > 2 are chance, hence the
two-period requirement.

Usage:
    python scripts/taker_flow_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
WINDOWS = (1, 4, 24)
HORIZONS = (1, 4, 24)
COST = 0.0036


def load():
    conn = get_connection()
    df = pd.read_sql("SELECT symbol, ts, close, quote_volume, taker_buy_quote FROM taker_flow ORDER BY ts", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def t_of(v):
    v = pd.Series(v).dropna()
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 10 and v.std(ddof=1) > 0 else np.nan


def main():
    df = load()
    close = df.pivot(index="ts", columns="symbol", values="close")[SYMBOLS]
    qv = df.pivot(index="ts", columns="symbol", values="quote_volume")[SYMBOLS]
    tb = df.pivot(index="ts", columns="symbol", values="taker_buy_quote")[SYMBOLS]
    full = pd.date_range(close.index.min(), close.index.max(), freq="h")
    close, qv, tb = close.reindex(full), qv.reindex(full), tb.reindex(full)
    print(f"Данные Binance: {full[0]:%Y-%m-%d}..{full[-1]:%Y-%m-%d}, {len(SYMBOLS)} монет, "
          f"пропусков в часовых свечах {close.isna().mean().mean():.2%}\n")

    net = 2 * tb - qv                                         # net aggressive buying in quote currency
    print("=== 1. СВЯЗЬ ТОГО ЖЕ ЧАСА (проверка здравого смысла): корреляция дисбаланса часа с доходностью этого же часа ===")
    ret1 = close / close.shift(1) - 1                        # return of the candle that OPENED at ts-1h... see below
    # candle opened at t has close at t+1h; return over [t, t+1h] = close[t] / close[t-1h] - 1
    imb1 = net / qv
    same = pd.concat([imb1.stack(), ret1.stack()], axis=1, keys=["imb", "ret"]).dropna()
    print(f"  по всем монето-часам: {same['imb'].corr(same['ret']):+.2f}  (ожидаемо сильно положительная)\n")

    # forward returns: price at T = close of the candle opened at T-1h  =>  here the series index IS the open time,
    # so the price at T is close.shift(-0) of index T-1h. Build the price series on the 'instant' axis.
    px = close.copy()
    px.index = px.index + pd.Timedelta(hours=1)               # instant at which each close is known/observed

    rows = []
    for W in WINDOWS:
        feat = (net.rolling(W).sum() / qv.rolling(W).sum())   # candles opened in [T-W, T) -> known at T
        feat.index = feat.index + pd.Timedelta(hours=1)
        # percentile of today's value among the previous 365 days of the same coin (no look-ahead)
        pct = feat.rolling(24 * 365, min_periods=24 * 90).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
        for k in HORIZONS:
            fwd = px.shift(-k) / px - 1
            sel = np.arange(len(feat)) % k == 0               # non-overlapping samples
            hi_b = (pct >= 0.8) & fwd.notna()
            lo_b = (pct <= 0.2) & fwd.notna()
            fv = fwd.values
            rec = []
            for i in np.where(sel)[0]:
                h, l = hi_b.values[i], lo_b.values[i]
                if h.sum() >= 2 and l.sum() >= 2:
                    mh, ml = fv[i][h].mean(), fv[i][l].mean()
                    rec.append((feat.index[i], mh - ml, mh, ml))
            ev = pd.DataFrame(rec, columns=["ts", "spread", "hi", "lo"]).set_index("ts")
            for label, part in (("train", ev[ev.index < SPLIT]), ("holdout", ev[ev.index >= SPLIT])):
                rows.append(dict(W=W, k=k, period=label, n=len(part), spread=part["spread"].mean(),
                                 hi=part["hi"].mean(), lo=part["lo"].mean(), t=t_of(part["spread"])))
    res = pd.DataFrame(rows)

    print("=== 2. РАЗНИЦА «много агрессивных покупок» минус «много продаж» В СЛЕДУЮЩИЕ k ЧАСОВ ===")
    print(f"{'окно':>5} {'k':>4} | {'период':>8} {'моментов':>9} | {'разница':>8} {'t':>6} | {'верх. квинт.':>12} {'нижн.':>8}")
    for _, r in res.iterrows():
        print(f"{r.W:>4}ч {r.k:>3}ч | {r.period:>8} {r.n:>9d} | {r.spread:>+8.3%} {r.t:>6.1f} | {r.hi:>+12.3%} {r.lo:>+8.3%}")

    print("\n=== 3. ЗАРАНЕЕ ЗАДАННАЯ ПЛАНКА ===")
    passed = []
    for (W, k), g in res.groupby(["W", "k"]):
        tr, ho = g[g.period == "train"].iloc[0], g[g.period == "holdout"].iloc[0]
        ok = (abs(tr.t) >= 3 and np.sign(ho.spread) == np.sign(tr.spread) and abs(ho.t) >= 2
              and abs(tr.spread) > COST and abs(ho.spread) > COST)
        if ok:
            passed.append((W, k))
        flag = "ПРОШЛА" if ok else ("train |t|>=3, не подтверждено" if abs(tr.t) >= 3 else "нет")
        print(f"  окно {W:>2}ч, горизонт {k:>2}ч: train {tr.spread:+.3%} (t {tr.t:.1f}), holdout {ho.spread:+.3%} (t {ho.t:.1f}) -> {flag}")
    print(f"\nПрошли: {len(passed)} из {len(res) // 2}. Порог издержек {COST:.2%}: разница меньше него нельзя торговать, даже если она настоящая.")
    res.to_csv("taker_flow_summary.csv", index=False)


if __name__ == "__main__":
    main()
