"""
Do slow on-chain valuation / activity measures predict BTC and ETH returns?
(2026-10-15, candidate 6) Coin Metrics community data, free, BTC and ETH only.

Features, fixed in advance:
  MVRV pct   percentile of MVRV (market value / realised value; the ratio behind
             NUPL) against the previous 4 years (min 1 year): high = historically
             expensive, low = historically cheap
  MVRV chg   30-day change of MVRV
  ACT        30-day mean of active addresses / 365-day mean - 1 (activity growth)
  HR         BTC only: hash rate 30d mean / 60d mean - 1 (the "hash ribbon":
             negative = miners under stress)
Targets: the next 7, 30 and 90 days of the same asset, sampled without overlap.

HONEST TIMING is established before any test: whether a row stamped D describes
the day that starts at D or the one that ends there is not documented in the
data. It is measured by comparing Coin Metrics' own PriceUSD with our hourly
prices - the stamp whose daily return matches ours tells which day the row is
about. Whatever the answer, the feature is used only from 00:00 UTC of D+1, the
conservative choice (it can only waste a day, never peek).

An important limit, stated rather than hidden: the number of INDEPENDENT
observations is small. A 90-day horizon gives ~18 non-overlapping samples in the
train window; a 30-day one ~53. A slow valuation signal cannot be validated the
way an hourly one can, and |t| = 3 on 18 points is not the same evidence as on
2000. The sample sizes are printed beside every result.

Predeclared bar: |t| >= 3 on train, same sign with |t| >= 2 on holdout, and the
top-minus-bottom-quintile spread larger than the 0.36% a round trip costs.

Usage:
    python scripts/onchain_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
HORIZONS = (7, 30, 90)
COST = 0.0036


def load():
    conn = get_connection()
    oc = pd.read_sql("SELECT asset, ts, metric, value FROM onchain_daily ORDER BY ts", conn)
    hourly = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        hourly[s] = d.set_index("ts")["close"]
    conn.close()
    oc["ts"] = pd.to_datetime(oc["ts"], utc=True)
    wide = {a: g.pivot(index="ts", columns="metric", values="value") for a, g in oc.groupby("asset")}
    hourly = pd.DataFrame(hourly)
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")).ffill(limit=3)
    return wide, hourly


def t_of(v):
    v = pd.Series(v).dropna()
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 5 and v.std(ddof=1) > 0 else np.nan


def main():
    wide, hourly = load()
    # price at instant D 00:00 = close of the candle opened at (D-1) 23:00
    p00 = hourly[hourly.index.hour == 23].copy()
    p00.index = p00.index.normalize() + pd.Timedelta(days=1)        # instant index
    day_ret = p00 / p00.shift(1) - 1                                # return of the day that ENDS at that instant
    # day_ret at index X = return over the day [X-1d, X). A row stamped D is "about day D" if it matches day_ret at D+1.

    print("=== 1. К КАКОМУ ДНЮ ОТНОСИТСЯ СТРОКА С МЕТКОЙ D? (сверка PriceUSD Coin Metrics с нашими ценами) ===")
    for asset, sym in (("btc", "BTCUSDT"), ("eth", "ETHUSDT")):
        cm = wide[asset]["PriceUSD"]
        cm_ret = cm / cm.shift(1) - 1
        cm_ret.index = cm_ret.index.normalize()
        ours_same = day_ret[sym].copy(); ours_same.index = ours_same.index - pd.Timedelta(days=1)    # day D, stamped D
        ours_prev = day_ret[sym].copy(); ours_prev.index = ours_prev.index - pd.Timedelta(days=2)    # day D-1, stamped D
        c_same = pd.concat([cm_ret, ours_same], axis=1, join="inner").dropna().corr().iloc[0, 1]
        c_prev = pd.concat([cm_ret, ours_prev], axis=1, join="inner").dropna().corr().iloc[0, 1]
        verdict = "метка D = КОНЕЦ дня D (значение известно только после его конца)" if c_same > c_prev else "метка D = конец дня D-1"
        print(f"  {asset.upper()}: корреляция с доходностью дня D {c_same:+.3f}, дня D-1 {c_prev:+.3f} -> {verdict}")
    print("  Во всех случаях признак используется с 00:00 UTC дня D+1 (консервативно).\n")

    rows = []
    for asset, sym in (("btc", "BTCUSDT"), ("eth", "ETHUSDT")):
        w = wide[asset]
        w = w.copy(); w.index = w.index.normalize()
        feats = {"MVRV pct": w["CapMVRVCur"].rolling(1460, min_periods=365).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True),
                 "MVRV chg": w["CapMVRVCur"] / w["CapMVRVCur"].shift(30) - 1,
                 "ACT": w["AdrActCnt"].rolling(30).mean() / w["AdrActCnt"].rolling(365, min_periods=180).mean() - 1}
        if asset == "btc" and "HashRate" in w:
            feats["HR"] = w["HashRate"].rolling(30).mean() / w["HashRate"].rolling(60).mean() - 1
        # the feature of day D is usable at instant D+1 00:00; forward returns start there
        price = p00[sym]                                           # indexed by instant
        for h in HORIZONS:
            fwd = price.shift(-h) / price - 1                      # instant X -> X + h days
            fwd.index = fwd.index - pd.Timedelta(days=1)           # re-index by D (feature day), so fwd[D] starts at D+1 00:00
            for fname, f in feats.items():
                lo_e = f.rolling(1460, min_periods=365).quantile(0.2).shift(1)
                hi_e = f.rolling(1460, min_periods=365).quantile(0.8).shift(1)
                d = pd.concat([f, fwd, lo_e, hi_e], axis=1, keys=["f", "r", "lo", "hi"]).dropna()
                d = d.iloc[::h]
                for label, part in (("train", d[d.index < SPLIT]), ("holdout", d[d.index >= SPLIT])):
                    hi, lo = part[part["f"] >= part["hi"]]["r"], part[part["f"] <= part["lo"]]["r"]
                    if len(hi) > 3 and len(lo) > 3:
                        sp = hi.mean() - lo.mean()
                        se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
                        t = sp / se if se > 0 else np.nan
                    else:
                        sp = t = np.nan
                    rows.append(dict(asset=asset.upper(), feat=fname, h=h, period=label, n=len(part), nh=len(hi), nl=len(lo),
                                     corr=part["f"].corr(part["r"]), spread=sp, t=t))
    res = pd.DataFrame(rows)

    print("=== 2. РАЗНИЦА ДОХОДНОСТЕЙ: верхний квинтиль признака минус нижний (и число независимых выборок) ===")
    print(f"{'актив':>5} {'признак':>9} {'h':>4} | {'train: n':>8} {'разница':>9} {'t':>6} | {'holdout: n':>10} {'разница':>9} {'t':>6} | вердикт")
    passed = []
    for (a, f, h), g in res.groupby(["asset", "feat", "h"], sort=False):
        tr, ho = g[g.period == "train"].iloc[0], g[g.period == "holdout"].iloc[0]
        ok = (not np.isnan(tr.t) and abs(tr.t) >= 3 and not np.isnan(ho.t) and np.sign(ho.spread) == np.sign(tr.spread)
              and abs(ho.t) >= 2 and abs(tr.spread) > COST and abs(ho.spread) > COST)
        if ok:
            passed.append((a, f, h))
        note = "ПРОШЛА" if ok else ("train |t|>=3, holdout не подтвердил" if (not np.isnan(tr.t) and abs(tr.t) >= 3) else "")
        print(f"{a:>5} {f:>9} {h:>3}д | {int(tr.n):>8d} {tr.spread:>+9.1%} {tr.t:>6.1f} | {int(ho.n):>10d} {ho.spread:>+9.1%} {ho.t:>6.1f} | {note}")
    print(f"\nПрошли заранее заданную планку: {len(passed)} из {len(res) // 2}. {passed if passed else ''}")
    print("Напоминание: при h = 90 дней на train всего около 18 независимых точек; |t| на такой выборке слабое доказательство.")
    res.to_csv("onchain_summary.csv", index=False)


if __name__ == "__main__":
    main()
