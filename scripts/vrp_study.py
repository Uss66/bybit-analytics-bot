"""
Does the volatility risk premium predict returns? (2026-10-15, candidate 4)

VRP = implied volatility minus the volatility that then actually happens. Here
the trailing version, which is what can be observed: DVOL (Deribit's 30-day
implied vol index, annualised %) minus the realised volatility of the last 30
days, annualised. A high VRP means options are priced richer than recent
movement; the equity-market literature treats it as a premium for bearing risk
and as a predictor of returns. Not tested in this project before: DVOL was only
ever used as a 1-day CHANGE (the leaky rule) and, later, as a LEVEL percentile
(a regime gate that did not help).

Honest timing:
  * the DVOL bar of day D closes at the end of D: usable from 00:00 UTC of D+1;
  * the realised volatility is computed from hourly closes up to exactly that
    instant (price at 00:00 of D+1 = close of the candle that opened at 23:00);
  * the forward return starts at that same instant, D+1 00:00.
  So nothing in the feature overlaps the return it is asked to predict.

Features, fixed in advance:
  VRP      = DVOL_close(D) - realised_vol_30d(D)                (vol points)
  VRP_z    = VRP standardised against its previous 365 days     (no look-ahead)
  DVOL_pct = percentile of DVOL against the previous 365 days
Assets: BTC with BTC DVOL; ETH with ETH DVOL; the 7-coin basket with BTC DVOL.
Horizons: next 1 day, 7 days, 30 days, sampled without overlap (every h days).
For each cell: correlation, and the spread between the top and bottom quintile
of the feature (quintile edges from the PREVIOUS 365 days, so no look-ahead).

Predeclared bar: |t| >= 3 on train, the same sign with |t| >= 2 on holdout, and
a quintile spread larger than the 0.36% a round trip costs. 3 features x 3
assets x 3 horizons x 2 periods = a large number of looks: a few |t| > 2 are
chance, which is why the second period is required.

Usage:
    python scripts/vrp_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
HORIZONS = (1, 7, 30)
COST = 0.0036


def load():
    conn = get_connection()
    hourly = {}
    for s in SYMBOLS:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        hourly[s] = d.set_index("ts")["close"]
    hourly = pd.DataFrame(hourly)
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")).ffill(limit=3)
    dv = pd.read_sql("SELECT symbol, ts, close FROM deribit_dvol ORDER BY ts", conn)
    conn.close()
    dv["ts"] = pd.to_datetime(dv["ts"], utc=True).dt.normalize()
    dvol = {a: g.drop_duplicates("ts", keep="last").set_index("ts")["close"] for a, g in dv.groupby("symbol")}
    return hourly, dvol


def t_of(v):
    v = pd.Series(v).dropna()
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 8 and v.std(ddof=1) > 0 else np.nan


def main():
    hourly, dvol = load()
    # Price at instant D+1 00:00 = close of the candle opened at D 23:00. Index the daily series by D.
    day_end = hourly[hourly.index.hour == 23]
    day_end.index = day_end.index.normalize()
    day_end = day_end.reindex(pd.date_range(day_end.index.min(), day_end.index.max(), freq="D")).ffill(limit=3)
    logret = np.log(day_end).diff()
    rv30 = logret.rolling(30, min_periods=25).std() * np.sqrt(365) * 100        # vol points, like DVOL

    assets = {"BTC (DVOL BTC)": ("BTCUSDT", "BTC"), "ETH (DVOL ETH)": ("ETHUSDT", "ETH"),
              "корзина 7 монет (DVOL BTC)": (None, "BTC")}
    print(f"Окно: {day_end.index[0]:%Y-%m-%d}..{day_end.index[-1]:%Y-%m-%d}; train < {SPLIT:%Y-%m-%d}\n")

    rows = []
    for aname, (sym, dv_key) in assets.items():
        dv = dvol[dv_key].reindex(day_end.index)
        rv = (rv30["BTCUSDT"] if dv_key == "BTC" and sym is None else rv30[sym if sym else "BTCUSDT"])
        vrp = dv - rv
        feats = {"VRP": vrp,
                 "VRP z": (vrp - vrp.rolling(365, min_periods=90).mean().shift(1)) / vrp.rolling(365, min_periods=90).std().shift(1),
                 "DVOL pct": dv.rolling(365, min_periods=90).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)}
        for h in HORIZONS:
            if sym:
                fwd = day_end[sym].shift(-h) / day_end[sym] - 1
            else:
                fwd = (day_end.shift(-h) / day_end - 1).mean(axis=1)
            for fname, f in feats.items():
                # quintile edges from the previous 365 days only
                lo_e = f.rolling(365, min_periods=90).quantile(0.2).shift(1)
                hi_e = f.rolling(365, min_periods=90).quantile(0.8).shift(1)
                d = pd.concat([f, fwd, lo_e, hi_e], axis=1, keys=["f", "r", "lo", "hi"]).dropna()
                d = d.iloc[::h]                                          # non-overlapping samples
                for label, part in (("train", d[d.index < SPLIT]), ("holdout", d[d.index >= SPLIT])):
                    hi = part[part["f"] >= part["hi"]]["r"]
                    lo = part[part["f"] <= part["lo"]]["r"]
                    corr = part["f"].corr(part["r"])
                    n = len(part)
                    t_corr = corr * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - corr ** 2)) if n > 10 else np.nan
                    # Welch t on the quintile spread
                    if len(hi) > 5 and len(lo) > 5:
                        sp = hi.mean() - lo.mean()
                        se = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
                        t_sp = sp / se if se > 0 else np.nan
                    else:
                        sp = t_sp = np.nan
                    rows.append(dict(asset=aname, feat=fname, h=h, period=label, n=n, nh=len(hi), nl=len(lo),
                                     corr=corr, t_corr=t_corr, spread=sp, t=t_sp))
    res = pd.DataFrame(rows)

    print("=== РАЗНИЦА СРЕДНЕЙ ДОХОДНОСТИ: верхний квинтиль признака минус нижний, и корреляция ===")
    print(f"{'актив':>26} {'признак':>9} {'h':>3} | {'train: разница':>14} {'t':>6} {'корр.':>6} | {'holdout: разница':>16} {'t':>6} {'корр.':>6}")
    passed = []
    for (a, f, h), g in res.groupby(["asset", "feat", "h"], sort=False):
        tr, ho = g[g.period == "train"].iloc[0], g[g.period == "holdout"].iloc[0]
        ok = (not np.isnan(tr.t) and abs(tr.t) >= 3 and not np.isnan(ho.t) and np.sign(ho.spread) == np.sign(tr.spread)
              and abs(ho.t) >= 2 and abs(tr.spread) > COST and abs(ho.spread) > COST)
        if ok:
            passed.append((a, f, h))
        mark = "  <-- ПРОШЛА" if ok else ""
        print(f"{a:>26} {f:>9} {h:>2}д | {tr.spread:>+14.2%} {tr.t:>6.1f} {tr['corr']:>+6.2f} | {ho.spread:>+16.2%} {ho.t:>6.1f} {ho['corr']:>+6.2f}{mark}")
    print(f"\nПрошли заранее заданную планку: {len(passed)} из {len(res) // 2} ячеек. {passed if passed else ''}")
    res.to_csv("vrp_summary.csv", index=False)


if __name__ == "__main__":
    main()
