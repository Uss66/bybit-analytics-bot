"""
During a market fall, would switching from the coin that fell hardest into
one that fell less - or is already bouncing - pay? (2026-10-15, user's idea)

    "rotate into a coin whose fall is smaller or that bounces right now,
     without waiting for the bottom"

What is already known, so this is not a re-run:
  * rotation by SCORE (evict the weakest-scoring holding) - rejected; the score
    is market-wide (FGI), nearly identical across coins;
  * cross-sectional relative strength as an ENTRY factor and as a cap
    tie-break - rejected;
  * speed_event_study.py - after a sharp fall a coin BOUNCES on average, and
    the harder it fell the more it bounced.
That last one is a warning: if the hardest-hit coin rebounds most, then
leaving it for the one that fell least is selling the best bounce. Whether
the cross-section really behaves that way inside a market-wide fall is the
question here, measured directly:

  Event = the equal-weight market (mean of the 7 coins) is down at least x%
  over the last H hours; events at least 24h apart.
  At each event take the coin that fell hardest ("worst") and the target:
    A  least-fall    - the coin with the best trailing-H return
    B  bouncing now  - the coin with the best last-2h return
    C  off its low   - the coin furthest above its own 24h low
  Gain of switching worst -> target over the next k hours:
        F_k(target) - F_k(worst) - 2 x 0.18%        (sell one, buy one)
  A ONE-WAY switch: there is no plan to go back to the first coin, so the
  cost is the two legs of the switch only, and k is just the window over
  which the two holdings are compared (4h / 24h / 72h / 1 week).
  Also printed: what the worst coin does against the market average
  (F_k(worst) - F_k(avg)), which says directly whether falls continue or
  rebound inside the cross-section.

Honesty about the statistics: all 7 coins move together, so an event is a
MARKET event and is counted once; t is over events. k = 24h windows do not
overlap (24h dead time); k = 72h windows do, so treat their t as optimistic.
Train <= 2025-11-30, holdout >= 2025-12-01. 7 coins only - the wider
database holds only coins that survived to today's turnover ranking.

Usage:
    python scripts/fall_switch_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

FEE = 0.0018
COST = 2 * FEE
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
DEAD_HOURS = 24
FORWARD = (4, 24, 72, 168)
GRID = {4: (0.02, 0.03), 12: (0.03, 0.05), 24: (0.04, 0.06)}      # H -> market fall thresholds


def load_close():
    conn = get_connection()
    frames = {}
    for s in SYMBOLS:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        frames[s] = d.set_index("ts")["close"]
    conn.close()
    c = pd.DataFrame(frames)
    return c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="h")).ffill(limit=3)


def t_stat(v):
    v = pd.Series(v).dropna()
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 5 and v.std(ddof=1) > 0 else np.nan


def main():
    close = load_close()
    fwd = {k: close.shift(-k) / close - 1 for k in FORWARD}
    last2 = close / close.shift(2) - 1
    off_low = close / close.rolling(24).min() - 1
    print(f"Данные {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}, {len(SYMBOLS)} монет, "
          f"стоимость переключения {COST:.2%}, мёртвое время между событиями {DEAD_HOURS}ч\n")

    rows = []
    for H, levels in GRID.items():
        trailing = close / close.shift(H) - 1
        market = trailing.mean(axis=1)
        for x in levels:
            ev_ts, last = [], None
            for ts in market.index[(market <= -x).values]:
                if last is None or ts - last >= pd.Timedelta(hours=DEAD_HOURS):
                    ev_ts.append(ts); last = ts
            recs = []
            for ts in ev_ts:
                if trailing.loc[ts].isna().any() or any(fwd[k].loc[ts].isna().any() for k in FORWARD):
                    continue
                tr = trailing.loc[ts]
                worst = tr.idxmin()
                others = [s for s in SYMBOLS if s != worst]
                pick = {"A": tr[others].idxmax(),
                        "B": last2.loc[ts, others].idxmax(),
                        "C": off_low.loc[ts, others].idxmax()}
                rec = dict(ts=ts)
                for k in FORWARD:
                    f = fwd[k].loc[ts]
                    rec[f"cont{k}"] = f[worst] - f.mean()                    # worst vs market average
                    for name, tgt in pick.items():
                        rec[f"{name}{k}"] = f[tgt] - f[worst] - COST
                recs.append(rec)
            ev = pd.DataFrame(recs)
            for label, part in (("train", ev[ev["ts"] < SPLIT]), ("holdout", ev[ev["ts"] >= SPLIT])):
                if len(part) < 3:
                    continue
                r = dict(H=H, x=x, period=label, n=len(part))
                for k in FORWARD:
                    r[f"cont{k}"] = part[f"cont{k}"].mean()
                    for name in "ABC":
                        r[f"{name}{k}"] = part[f"{name}{k}"].mean()
                for name in "ABC":
                    r[f"{name}_t24"] = t_stat(part[f"{name}24"])
                r["cont_t24"] = t_stat(part["cont24"])
                rows.append(r)
    res = pd.DataFrame(rows)

    print("=== 1. ПРОДОЛЖАЕТ ЛИ САМАЯ УПАВШАЯ МОНЕТА ПАДАТЬ ОТНОСИТЕЛЬНО РЫНКА? (F_k(худшая) - среднее по монетам) ===")
    print(f"{'H':>3} {'порог':>6} | {'период':>8} {'событий':>8} | {'k=4ч':>8} {'k=24ч':>8} {'t(24ч)':>7} {'k=72ч':>8} {'k=168ч':>8}")
    for _, r in res.iterrows():
        print(f"{r.H:>2}ч {r.x:>5.0%} | {r.period:>8} {r.n:>8d} | {r.cont4:>+8.2%} {r.cont24:>+8.2%} {r.cont_t24:>7.1f} {r.cont72:>+8.2%} {r.cont168:>+8.2%}")

    for name, title in (("A", "A: на монету, которая упала МЕНЬШЕ всех"),
                        ("B", "B: на монету, которая отскакивает ПРЯМО СЕЙЧАС (лучший результат за 2ч)"),
                        ("C", "C: на монету, дальше всех ушедшую ВВЕРХ от собственного 24ч-минимума")):
        print(f"\n=== 2{name}. ПЕРЕКЛЮЧЕНИЕ {title}; чистая выгода за вычетом {COST:.2%} ===")
        print(f"{'H':>3} {'порог':>6} | {'период':>8} {'событий':>8} | {'k=4ч':>8} {'k=24ч':>8} {'t(24ч)':>7} {'k=72ч':>8} {'k=168ч':>8}")
        for _, r in res.iterrows():
            print(f"{r.H:>2}ч {r.x:>5.0%} | {r.period:>8} {r.n:>8d} | {r[name + '4']:>+8.2%} {r[name + '24']:>+8.2%} "
                  f"{r[name + '_t24']:>7.1f} {r[name + '72']:>+8.2%} {r[name + '168']:>+8.2%}")

    print("\n=== 3. СВОДКА: сколько из ячеек положительны на train И на holdout на горизонте 24 ч ===")
    piv = res.pivot_table(index=["H", "x"], columns="period", values=[f"{n}24" for n in "ABC"])
    for name in "ABC":
        both = ((piv[f"{name}24"]["train"] > 0) & (piv[f"{name}24"]["holdout"] > 0)).sum()
        pos_tr = (piv[f"{name}24"]["train"] > 0).sum()
        print(f"  {name}: положительно на train {pos_tr} из {len(piv)}, на обоих периодах {both} из {len(piv)}")
    res.to_csv("fall_switch_summary.csv", index=False)
    print("\nЗаписано в fall_switch_summary.csv")


if __name__ == "__main__":
    main()
