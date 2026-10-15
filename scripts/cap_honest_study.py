"""
Is the position cap of 3 still right on the HONEST score? (2026-10-15)

slot_cap_study.py / project_slot_cap_real_sizing validated cap = 3 (and showed
that removing it is worst on holdout) - on a score that included the DVOL and
DXY rules, which looked a day into the future. The cap is a live parameter, so
it is re-run on the score the bot really uses (FGI + news; load_panel returns
it by default since 2026-10-15).

Everything else at today's values; the cap is swept 1..7; real 0.18% fee;
train and holdout run separately, calendar years from one full run. What
matters is the SHAPE: a plateau around 3, a spike at 3, or a clear slope.

Caveat carried from the parameter sweep: on the honest score the holdout holds
only 6 trades (the score is positive ~10% of the hours), so the holdout can
hardly distinguish caps; train (106 trades) carries the verdict.

Usage:
    python scripts/cap_honest_study.py
"""
import pandas as pd

from db import get_connection
from rotation_study import load_panel
from momentum_study import simulate
from regime_study import window
from param_sensitivity_study import run, CURRENT

CAPS = (1, 2, 3, 4, 5, 6, 7)
FEE = 0.0018


def main():
    conn = get_connection()
    panel = load_panel(conn)
    conn.close()
    sma_cache = {}
    close = panel[0]
    print(f"Честный счёт сигнала; комиссия {FEE:.2%}; период {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}\n")
    print(f"{'кэп':>4} | {'train: доходн.':>13} {'просадка':>9} {'сделок':>7} {'ret/dd':>7} | {'holdout: доходн.':>15} {'просадка':>9} {'сделок':>7} {'ret/dd':>7}")
    for cap in CAPS:
        params = dict(CURRENT, max_pos=cap)
        r = {p: run(panel, sma_cache, params, p) for p in ("train", "holdout")}
        t, h = r["train"], r["holdout"]
        mark = "  <-- сейчас" if cap == 3 else ""
        print(f"{cap:>4} | {t['total_ret']:>+13.0%} {t['max_dd']:>9.1%} {t['n_trades']:>7d} {t['ratio']:>7.2f} | "
              f"{h['total_ret']:>+15.1%} {h['max_dd']:>9.1%} {h['n_trades']:>7d} {h['ratio']:>7.2f}{mark}")

    print("\nПо годам (один прогон на всей истории): доходность / просадка")
    t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
    eqs = {}
    for cap in CAPS:
        variant = {k: v for k, v in dict(CURRENT, max_pos=cap).items() if k != "sma"}
        eqs[cap] = simulate(*panel, variant, fee=FEE)[2]
    print(f"{'год':>6} " + " ".join(f"{'кэп ' + str(c):>16}" for c in CAPS))
    for y in range(t0.year, t1.year + 1):
        a = max(pd.Timestamp(f"{y}-01-01", tz="UTC"), t0)
        b = min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), t1)
        cells = []
        for c in CAPS:
            w = window(eqs[c], a, b)
            cells.append("-".rjust(16) if w is None else f"{w[0]:>+7.0%} / {w[1]:>5.0%}".rjust(16))
        print(f"{y:>6} " + " ".join(cells))


if __name__ == "__main__":
    main()
