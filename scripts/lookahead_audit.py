"""
Does the backtest know the future? (2026-10-15)

Found while checking parameter sensitivity: delaying the whole score by 24h
collapses even the CURRENT strategy (train +235% -> +19%, holdout +32% ->
-23%). A slowly varying signal does not behave like that. The suspicion is a
timestamp problem in the daily data.

The mechanism, from the code:
  * deribit_dvol / traditional_markets rows are DAILY bars stamped with the
    date, i.e. 00:00 UTC of day D, but their close is the end-of-day-D value.
  * classify_deribit_dvol.py and classify_traditional_markets.py turn
    "close_D / close_{D-1} - 1" into an event stamped at that SAME 00:00 UTC
    of day D (events.ts = row.ts).
  * compute_score_series makes the rule active from events.ts. So from the
    first hour of day D the strategy already knows how DVOL / DXY ended day D.
  * The live bot cannot: at any hour of day D the most recent completed bar
    is D-1.
  FGI is different on paper (the value for date D is published at the start
  of D from the previous 24h), so it is checked rather than assumed.

Three steps, so the conclusion rests on measurement:

  1. EVIDENCE   same-day vs next-day correlation of each series' daily
                change with BTC's return. If a series is truly known at 00:00
                of D it cannot correlate with the return DURING D. A large
                same-day correlation and ~0 next-day correlation is the
                signature of a series that is only known at the end of D.
  2. SIGNAL     the DVOL / DXY extreme events' forward 24h return measured
                (a) as the backtest does, from 00:00 of D, and (b) honestly,
                from 00:00 of D+1, when the bar is actually known.
  3. PORTFOLIO  the same engine and fee as every other study, with
                  A  events as built (what every number so far used)
                  B  DVOL and DXY events available from D+1 (what the live
                     bot really sees)
                  C  as B, and FGI delayed a day too (conservative)
                  D  DVOL and DXY removed altogether (FGI + news only)

Usage:
    python scripts/lookahead_audit.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS, TREND_FILTER_SMA
from strategy import load_events_with_returns, generate_signals, compute_score_series
from momentum_study import simulate
from regime_study import window

FEE = 0.0018
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
TRAIN_END = SPLIT - pd.Timedelta(hours=1)
DVOL_CATS = ["dvol_drop_extreme", "dvol_spike_extreme"]
DXY_CATS = ["dxy_up_extreme", "dxy_down_extreme"]
FGI_CATS = ["fgi_extreme_fear", "fgi_fear", "fgi_neutral", "fgi_greed", "fgi_extreme_greed"]


def hourly_close(conn):
    frames = {}
    for s in SYMBOLS:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        frames[s] = d.set_index("ts")["close"]
    return pd.DataFrame(frames)


def build_panel(conn, close_by_sym, events, shift_cats=(), drop_cats=(), shift_hours=24, include_lookahead=False):
    """include_lookahead=True brings back the four DVOL / DXY rules that were taken out of the live
    score (strategy.LOOKAHEAD_RULES) - needed to reproduce variants A-C. Since 2026-10-15 the default
    score (and so every study's load_panel) is the honest one, variant D."""
    ev = events[~events["category"].isin(drop_cats)].copy()
    m = ev["category"].isin(shift_cats)
    ev.loc[m, "ts"] = ev.loc[m, "ts"] + pd.Timedelta(hours=shift_hours)
    close, score, sma = {}, {}, {}
    for s in SYMBOLS:
        o = pd.DataFrame({"ts": close_by_sym[s].index, "close": close_by_sym[s].values})
        o["score"] = compute_score_series(generate_signals(ev, s, include_lookahead=include_lookahead), o)
        o["sma"] = o["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()
        o = o.set_index("ts")
        close[s], score[s], sma[s] = o["close"], o["score"], o["sma"]
    return pd.DataFrame(close), pd.DataFrame(score), pd.DataFrame(sma)


def split(panel, name):
    close, score, sma = panel
    m = close.index <= TRAIN_END if name == "train" else close.index >= SPLIT
    return close[m], score[m], sma[m]


def main():
    conn = get_connection()
    hourly = hourly_close(conn)
    close_by_sym = {s: hourly[s].dropna() for s in SYMBOLS}
    events = load_events_with_returns(conn)

    # ---------------- 1. evidence ----------------
    btc = hourly["BTCUSDT"].dropna()
    day_open = btc.resample("D").first()
    r_same = (day_open.shift(-1) / day_open - 1)         # return DURING day D (00:00 D -> 00:00 D+1)
    series = {}
    d = pd.read_sql("SELECT ts, close FROM deribit_dvol WHERE symbol='BTC' ORDER BY ts", conn)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    series["DVOL (смена за день)"] = d.set_index("ts")["close"].pct_change(1)
    x = pd.read_sql("SELECT ts, close FROM traditional_markets WHERE ticker='DXY' ORDER BY ts", conn)
    x["ts"] = pd.to_datetime(x["ts"], utc=True)
    series["DXY (смена за день)"] = x.set_index("ts")["close"].pct_change(1)
    f = pd.read_sql("SELECT ts, value FROM fear_greed_index ORDER BY ts", conn)
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    series["FGI (смена за день)"] = f.set_index("ts")["value"].astype(float).diff(1)

    print("=== 1. ДОКАЗАТЕЛЬСТВО: корреляция дневного изменения ряда с доходностью BTC ===")
    print("   (бар дня D стоит на 00:00 дня D; «в тот же день» = доходность 00:00 D -> 00:00 D+1,")
    print("    «на следующий» = доходность 00:00 D+1 -> 00:00 D+2)")
    print(f"{'ряд':>22} | {'период':>8} {'в тот же день':>14} {'на следующий':>13} {'за день до':>11} {'дней':>5}")
    for name, s in series.items():
        s = s.copy(); s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]        # a daily table may carry two rows for one date
        for label, mask in (("train", s.index < SPLIT), ("holdout", s.index >= SPLIT)):
            ss = s[mask]
            cs = pd.concat([ss, r_same], axis=1, join="inner").dropna().corr().iloc[0, 1]
            cn = pd.concat([ss, r_same.shift(-1)], axis=1, join="inner").dropna().corr().iloc[0, 1]
            cp = pd.concat([ss, r_same.shift(1)], axis=1, join="inner").dropna().corr().iloc[0, 1]
            n = len(pd.concat([ss, r_same], axis=1, join="inner").dropna())
            print(f"{name:>22} | {label:>8} {cs:>+14.2f} {cn:>+13.2f} {cp:>+11.2f} {n:>5d}")

    # ---------------- 2. signal ----------------
    print("\n=== 2. СИГНАЛ: что делают 7 монет в следующие 24 ч после события ===")
    print("   как строит бэктест: от 00:00 дня D;  честно: от 00:00 дня D+1 (когда бар реально известен)")
    print(f"{'событие':>20} | {'период':>8} {'событий':>8} | {'как в бэктесте':>16} {'t':>6} | {'честно':>9} {'t':>6}")
    for cat in DVOL_CATS + DXY_CATS:
        ev = events[events["category"] == cat]
        for label, mask in (("train", ev["ts"] < SPLIT), ("holdout", ev["ts"] >= SPLIT)):
            rows = []
            for ts in ev.loc[mask, "ts"]:
                a0, a1, b1 = ts, ts + pd.Timedelta(hours=24), ts + pd.Timedelta(hours=48)
                ra, rb = [], []
                for s in SYMBOLS:
                    c = hourly[s]
                    if a0 in c.index and a1 in c.index and not (np.isnan(c[a0]) or np.isnan(c[a1])):
                        ra.append(c[a1] / c[a0] - 1)
                    if a1 in c.index and b1 in c.index and not (np.isnan(c[a1]) or np.isnan(c[b1])):
                        rb.append(c[b1] / c[a1] - 1)
                if ra and rb:
                    rows.append((np.mean(ra), np.mean(rb)))
            if len(rows) < 5:
                continue
            arr = np.array(rows)
            t = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            print(f"{cat:>20} | {label:>8} {len(arr):>8d} | {arr[:, 0].mean():>+16.2%} {t(arr[:, 0]):>6.1f} | "
                  f"{arr[:, 1].mean():>+9.2%} {t(arr[:, 1]):>6.1f}")

    # ---------------- 3. portfolio ----------------
    print("\n=== 3. ПОРТФЕЛЬ (тот же движок, комиссия 0,18%): доходность / просадка / ret-dd ===")
    variants = {
        "A как в бэктесте": dict(include_lookahead=True),
        "B DVOL и DXY с D+1": dict(include_lookahead=True, shift_cats=DVOL_CATS + DXY_CATS),
        "C B и FGI с D+1": dict(include_lookahead=True, shift_cats=DVOL_CATS + DXY_CATS + FGI_CATS),
        "D без DVOL и DXY": dict(),          # since 2026-10-15 this is simply the live score
    }
    panels = {k: build_panel(conn, close_by_sym, events, **v) for k, v in variants.items()}
    conn.close()
    print(f"{'вариант':>22} | {'train':^28} | {'holdout':^28}")
    eqs = {}
    for k, p in panels.items():
        cells = []
        for name in ("train", "holdout"):
            c, s, m = split(p, name)
            st, _, _ = simulate(c, s, m, {}, fee=FEE)
            cells.append(f"{st['total_ret']:>+9.0%} {st['max_dd']:>7.0%} {st['ratio']:>7.2f}")
        print(f"{k:>22} | {cells[0]:^28} | {cells[1]:^28}")
        eqs[k] = simulate(*p, {}, fee=FEE)[2]

    print("\n=== по годам (один прогон на всей истории): доходность / просадка ===")
    t0, t1 = hourly.index[0], hourly.index[-1] + pd.Timedelta(hours=1)
    print(f"{'год':>6} " + " ".join(f"{k[:20]:>22}" for k in eqs))
    for year in range(t0.year, t1.year + 1):
        a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), t0)
        b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), t1)
        cells = []
        for eq in eqs.values():
            w = window(eq, a, b)
            cells.append("-".rjust(22) if w is None else f"{w[0]:>+8.0%} / {w[1]:>6.0%}".rjust(22))
        print(f"{year:>6} " + " ".join(cells))


if __name__ == "__main__":
    main()
