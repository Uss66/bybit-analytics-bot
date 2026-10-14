"""
Is the crash-exit result a plateau or a spike? (2026-10-15)

momentum_study.py found that exiting on a sharp 24h drop improved ret/dd on
both train and holdout. Before believing it, the checks the methodology
asks for and a 6-point grid cannot give:

  1. SURFACE  - horizon x threshold neighbourhood at the realistic 0.18% fee.
                A real effect is a smooth hill; a lucky pick is a spike with
                worse values on every side.
  2. BY YEAR  - one full-history run, equity sliced by calendar year. An
                effect carried by 2022 alone is a 2022 effect.
  3. BY COIN  - summed P&L per symbol, baseline vs variant. The project's
                bar is "most symbols, not the best one".

Run on the full history in one pass (positions carry across the train /
holdout boundary), so the train and holdout columns here are windows of
the same equity curve rather than independent runs - compare their
DIRECTION with momentum_study.py, not the exact figures.
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import load_panel, SYMBOLS
from momentum_study import simulate

FEE = 0.0018
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")


def window(eq, a, b):
    s = eq[(eq.index >= a) & (eq.index < b)]
    if len(s) < 24:
        return None
    ret = s.iloc[-1] / s.iloc[0] - 1
    dd = ((s - s.cummax()) / s.cummax()).min()
    return ret, dd, (ret / abs(dd) if dd < 0 else float("inf"))


def main():
    conn = get_connection()
    close, score, sma = load_panel(conn)
    conn.close()
    t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
    print(f"Период {t0:%Y-%m-%d}..{t1:%Y-%m-%d}, комиссия {FEE:.2%}\n")

    _, base_t, base_eq = simulate(close, score, sma, {}, fee=FEE)

    print("=== 1. ПОВЕРХНОСТЬ: ret/dd (train | holdout), обвал-выход, горизонт x порог ===")
    b_tr, b_ho = window(base_eq, t0, SPLIT), window(base_eq, SPLIT, t1)
    print(f"baseline: train {b_tr[2]:.2f}  holdout {b_ho[2]:.2f}\n")
    thresholds = [0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"{'горизонт':>9} " + " ".join(f"{int(x*100):>2d}%".rjust(13) for x in thresholds))
    keep = {}
    for h in (12, 24, 48):
        cells = []
        for x in thresholds:
            _, t, eq = simulate(close, score, sma, dict(drop_h=h, drop_x=x), fee=FEE)
            tr, ho = window(eq, t0, SPLIT), window(eq, SPLIT, t1)
            keep[(h, x)] = (t, eq)
            mark = "+" if tr[2] > b_tr[2] and ho[2] > b_ho[2] else " "
            cells.append(f"{tr[2]:5.2f}|{ho[2]:5.2f}{mark}")
        print(f"{h:>8}ч " + " ".join(c.rjust(13) for c in cells))
    print("  (+ = лучше baseline и на train, и на holdout)\n")

    print("=== 2. ПО ГОДАМ: доходность / просадка, baseline против обвал 24ч -5% и -8% ===")
    variants = {"baseline": base_eq, "24ч -5%": keep[(24, 0.05)][1], "24ч -8%": keep[(24, 0.08)][1]}
    print(f"{'год':>6} " + " ".join(f"{k:>22}" for k in variants))
    for year in range(t0.year, t1.year + 1):
        a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), t0)
        b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), t1)
        row = []
        for eq in variants.values():
            w = window(eq, a, b)
            row.append("            -" if w is None else f"{w[0]:>+8.1%} / {w[1]:>7.1%}")
        print(f"{year:>6} " + " ".join(c.rjust(22) for c in row))

    print("\n=== 3. ПО МОНЕТАМ: суммарный P&L (USDT на старте 1000), baseline против обвал 24ч -8% и -5% ===")
    def per_symbol(t):
        return t.groupby("symbol")["pnl"].sum()
    pb = per_symbol(base_t)
    for key, label in ((((24, 0.08)), "24ч -8%"), ((24, 0.05), "24ч -5%")):
        pv = per_symbol(keep[key][0])
        better = 0
        print(f"\n  {label}:")
        for s in SYMBOLS:
            d = pv.get(s, 0.0) - pb.get(s, 0.0)
            better += d > 0
            print(f"    {s:9s} baseline {pb.get(s, 0.0):>+9.0f}   вариант {pv.get(s, 0.0):>+9.0f}   разница {d:>+8.0f}")
        print(f"    лучше baseline: {better} из {len(SYMBOLS)} монет")


if __name__ == "__main__":
    main()
