"""
Do yesterday's traditional-market moves predict crypto TODAY? An honest test
(2026-10-15). The first version of this question (DXY, project_traditional_
markets_findings) was answered with a bar that was not yet known.

Timing, spelled out because it is the whole point:
  * traditional_markets rows are daily bars stamped with the session's OPEN
    (13:30 UTC for US indices, 04:00-05:00 UTC for futures and DXY); the
    `close` is the value at the end of that session, 7-17 hours later.
  * So the move of day D is known from 00:00 UTC of D+1 at the earliest (all
    closes are by 22:00 UTC). The forward window therefore starts at D+1 00:00
    and runs 24h.
  * Price "at 00:00 UTC" is the close of the 23:00 candle of the previous
    day (our candles are stamped with their OPEN).
  * The same-day relationship (window D 00:00 -> D+1 00:00) is printed beside
    it as a sanity check: US equities should move together with crypto the
    same day, and that is exactly the contemporaneous correlation the old
    backtest mistook for prediction.

Features per ticker, from the daily close: 1-day change and 5-day change
(the old DXY rule used a 7-day change; 5 trading days is the same week).
Targets: BTC, and the equal-weight basket of the 7 strategy coins.

Predeclared bar for "interesting": next-day |t| >= 3 on train AND the same
sign with |t| >= 2 on holdout. With 11 tickers x 2 features x 2 targets x 2
periods, a handful of |t| > 2 are expected by chance alone, which is why the
bar needs both periods.

Plus the specific old claim: the top quintile of DXY's 7-day change (cutoffs
from TRAIN only) measured from D 00:00 as the backtest did, and honestly
from D+1 00:00.

Usage:
    python scripts/macro_honest_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
FEATURES = {"1д": 1, "5д": 5}


def load():
    conn = get_connection()
    hourly = {}
    for s in SYMBOLS:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        hourly[s] = d.set_index("ts")["close"]
    hourly = pd.DataFrame(hourly)
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")).ffill(limit=3)
    macro = pd.read_sql("SELECT ticker, ts, close FROM traditional_markets ORDER BY ts", conn)
    conn.close()
    macro["ts"] = pd.to_datetime(macro["ts"], utc=True)
    macro["date"] = macro["ts"].dt.normalize()
    return hourly, macro


def price_at(hourly, when):
    """Price at the instant `when` (UTC) = close of the candle that OPENED an hour earlier."""
    idx = when - pd.Timedelta(hours=1)
    return hourly.reindex(idx).values


def corr_t(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 30:
        return np.nan, np.nan, len(d)
    r = d.iloc[:, 0].corr(d.iloc[:, 1])
    t = r * np.sqrt(len(d) - 2) / np.sqrt(max(1e-12, 1 - r * r))
    return r, t, len(d)


def main():
    hourly, macro = load()
    basket_cols = list(hourly.columns)
    dates = pd.date_range(hourly.index.min().normalize() + pd.Timedelta(days=1),
                          hourly.index.max().normalize() - pd.Timedelta(days=2), freq="D")

    def fwd(start_offset_days, hours=24):
        a = dates + pd.Timedelta(days=start_offset_days)
        b = a + pd.Timedelta(hours=hours)
        pa, pb = price_at(hourly, a), price_at(hourly, b)
        ret = pd.DataFrame(pb / pa - 1, index=dates, columns=basket_cols)
        return ret["BTCUSDT"], ret.mean(axis=1)

    nxt_btc, nxt_bsk = fwd(1)            # honest: D+1 00:00 -> D+2 00:00
    same_btc, same_bsk = fwd(0)          # contemporaneous: D 00:00 -> D+1 00:00

    print(f"Окно сравнения: {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; train < {SPLIT:%Y-%m-%d}\n")
    print("=== 1. КОРРЕЛЯЦИЯ ИЗМЕНЕНИЯ РЯДА (дня D) С ДОХОДНОСТЬЮ КРИПТЫ ===")
    print("    «в тот же день» = окно D 00:00 -> D+1 00:00;  «на следующий» = D+1 00:00 -> D+2 00:00 (честное)")
    print(f"{'тикер':>9} {'приз.':>5} {'цель':>7} | {'train: тот же':>13} {'след.':>7} {'t':>6} | {'holdout: след.':>14} {'t':>6} | вердикт")
    interesting = []
    for ticker, g in macro.groupby("ticker"):
        close = g.sort_values("ts").drop_duplicates("date", keep="last").set_index("date")["close"]
        for fname, n in FEATURES.items():
            feat = close.pct_change(n) if ticker != "UST10Y" else close.diff(n) / close.shift(n)
            feat = feat.reindex(dates)
            for tname, nxt, same in (("BTC", nxt_btc, same_btc), ("корзина", nxt_bsk, same_bsk)):
                tr = feat.index < SPLIT
                r_s, _, _ = corr_t(feat[tr], same[tr])
                r_n, t_n, n_n = corr_t(feat[tr], nxt[tr])
                r_h, t_h, n_h = corr_t(feat[~tr], nxt[~tr])
                ok = (not np.isnan(t_n)) and abs(t_n) >= 3 and (not np.isnan(t_h)) and np.sign(r_h) == np.sign(r_n) and abs(t_h) >= 2
                if ok:
                    interesting.append((ticker, fname, tname, r_n, t_n, r_h, t_h))
                flag = "ИНТЕРЕСНО" if ok else ("train |t|>=3, holdout не подтвердил" if (not np.isnan(t_n) and abs(t_n) >= 3) else "")
                print(f"{ticker:>9} {fname:>5} {tname:>7} | {r_s:>13.2f} {r_n:>+7.2f} {t_n:>6.1f} | {r_h:>+14.2f} {t_h:>6.1f} | {flag}")

    print(f"\nПрошли заранее заданную планку (train |t|>=3, holdout тот же знак и |t|>=2): {len(interesting)}")
    for row in interesting:
        print("   ", row)

    # ---- the specific old claim: DXY top quintile of the 7-day change ----
    print("\n=== 2. СТАРОЕ УТВЕРЖДЕНИЕ: верхний квинтиль 7-дневного изменения DXY ===")
    g = macro[macro["ticker"] == "DXY"].sort_values("ts").drop_duplicates("date", keep="last").set_index("date")["close"]
    chg7 = g.pct_change(7)
    cut_hi = chg7[chg7.index < SPLIT].quantile(0.8)
    cut_lo = chg7[chg7.index < SPLIT].quantile(0.2)
    print(f"  границы по train: верхний квинтиль >= {cut_hi:+.2%}, нижний <= {cut_lo:+.2%}")
    f7 = chg7.reindex(dates)
    for label, mask in (("верхний (DXY вырос)", f7 >= cut_hi), ("нижний (DXY упал)", f7 <= cut_lo)):
        for per, pm in (("train", dates < SPLIT), ("holdout", dates >= SPLIT)):
            m = mask & pm
            a_same, a_next = same_bsk[m].dropna(), nxt_bsk[m].dropna()
            ts_ = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 5 else np.nan
            print(f"  {label:20s} {per:8s} дней {len(a_next):>4d} | как в бэктесте (от D 00:00): {a_same.mean():>+6.2%} (t {ts_(a_same):>5.1f})"
                  f" | честно (от D+1 00:00): {a_next.mean():>+6.2%} (t {ts_(a_next):>5.1f})")


if __name__ == "__main__":
    main()
