"""
How arbitrary are the strategy's parameters? (2026-10-15, user's suspicion)

Where today's values came from, from the project's own history:
  * stop -8%      the default when stop-loss was first added to backtest();
                  never swept
  * SMA 720       "30 days in hours", picked after a diagnosis that buying
                  below the trend loses; the window itself was never swept
  * cooldown 3 losses / 168h    chosen by eye (one week); only the idea of a
                  cooldown and a separate "min cooldown after every exit"
                  were tested
  * dip-rebuy -3% / +2% / 2 adds  picked from a small grid
  * cap 3         swept (slot_cap_study) - not repeated here

So the suspicion is half right. This does NOT try to find better values.
It asks what shape the result has around the current ones, because that
tells apart two very different situations:

  plateau  - neighbours are about as good: the exact value does not matter
             and the current one is as defensible as any;
  spike    - the current value is a lucky pick between worse neighbours;
  slope    - neighbours on one side are clearly better: a candidate, which
             then has to survive the same checks as any other (the opposite
             side's period, calendar years).

One parameter at a time, everything else at today's values, at the real
0.18% fee, train and holdout run separately. Then a 2-D table of the two
parameters that matter most (stop x SMA window). The agreement between
train and holdout across each sweep (Spearman) is printed too: a parameter
that really matters ranks its values the same way in both periods; one
that does not, ranks them differently.

Engine: momentum_study.simulate (cash-based walk, cap 3, cash / free-slots
sizing, dip-rebuy, cooldown, hourly mark-to-market).

Usage:
    python scripts/param_sensitivity_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import load_panel
from momentum_study import simulate
from regime_study import window, TRAIN_END, HOLD_START

FEE = 0.0018
CURRENT = {"stop": -0.08, "sma": 720, "dip_thr": -0.03, "dip_reb": 0.02, "max_adds": 2,
           "cd_losses": 3, "cd_hours": 168}
SWEEPS = {
    "стоп-лосс": ("stop", [-0.04, -0.06, -0.08, -0.10, -0.12, -0.15, -0.20, -0.99], lambda v: f"{v:.0%}" if v > -0.9 else "нет"),
    "окно SMA (часов)": ("sma", [168, 336, 504, 720, 1080, 1440, 0], lambda v: str(v) if v else "нет фильтра"),
    "докупка: просадка": ("dip_thr", [-0.02, -0.03, -0.04, -0.05, -0.07], lambda v: f"{v:.0%}"),
    "докупка: отскок": ("dip_reb", [0.01, 0.02, 0.03, 0.04], lambda v: f"{v:.0%}"),
    "докупок максимум": ("max_adds", [0, 1, 2, 3, 4], str),
    "кулдаун: убытков подряд": ("cd_losses", [2, 3, 4, 5, 99], lambda v: str(v) if v < 99 else "нет"),
    "кулдаун: часов": ("cd_hours", [24, 72, 168, 336, 720], str),
}
GRID_STOPS = [-0.06, -0.08, -0.10, -0.12, -0.15]
GRID_SMAS = [336, 504, 720, 1080]


def sma_for(close, w, cache):
    if w not in cache:
        cache[w] = (pd.DataFrame(np.nan, index=close.index, columns=close.columns) if not w
                    else close.rolling(w, min_periods=max(w // 3, 1)).mean())
    return cache[w]


def run(panel, sma_cache, params, period):
    close, score, _ = panel
    mask = (close.index <= pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(hours=23)) if period == "train" \
        else (close.index >= pd.Timestamp(HOLD_START, tz="UTC"))
    sma = sma_for(close, params["sma"], sma_cache)
    variant = {k: v for k, v in params.items() if k != "sma"}
    stats, _, eq = simulate(close[mask], score[mask], sma[mask], variant, fee=FEE)
    return stats


def honest_panel(conn):
    """The score WITHOUT the DVOL and DXY rules. Their daily bars are stamped
    00:00 of day D but close at the end of D, so the backtest 'knew' the day
    in advance (lookahead_audit.py); the live bot cannot. Dropping them is
    the cleanest honest baseline - FGI (checked: known at 00:00) and the
    news rule remain."""
    from lookahead_audit import (hourly_close, build_panel, DVOL_CATS, DXY_CATS)
    from strategy import load_events_with_returns
    hourly = hourly_close(conn)
    by_sym = {s: hourly[s].dropna() for s in hourly.columns}
    return build_panel(conn, by_sym, load_events_with_returns(conn), drop_cats=DVOL_CATS + DXY_CATS)


def main():
    import sys
    honest = "--honest" in sys.argv
    conn = get_connection()
    panel = honest_panel(conn) if honest else load_panel(conn)
    conn.close()
    print("СЧЁТ СИГНАЛА: " + ("ЧЕСТНЫЙ (без DVOL и DXY, у которых бар дня D известен только к концу D)"
                               if honest else "как в бэктесте (с DVOL и DXY, заглядывающими на сутки вперёд)"))
    sma_cache = {}
    close = panel[0]
    print(f"Период {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}, комиссия {FEE:.2%}; "
          f"train <= {TRAIN_END}, holdout >= {HOLD_START}\n")

    base = {p: run(panel, sma_cache, CURRENT, p) for p in ("train", "holdout")}
    print(f"текущие значения: train ret/dd {base['train']['ratio']:.2f} ({base['train']['total_ret']:+.0%}, "
          f"{base['train']['max_dd']:.0%}), holdout {base['holdout']['ratio']:.2f} "
          f"({base['holdout']['total_ret']:+.0%}, {base['holdout']['max_dd']:.0%})\n")

    summary, flagged = [], []
    for title, (key, values, fmt) in SWEEPS.items():
        print(f"=== {title} (текущее: {fmt(CURRENT[key])}) ===")
        print(f"{'значение':>12} | {'train: доходн.':>13} {'просадка':>9} {'ret/dd':>7} | "
              f"{'holdout: доходн.':>15} {'просадка':>9} {'ret/dd':>7} | {'vs текущее':>10}")
        tr, ho = {}, {}
        for v in values:
            params = dict(CURRENT, **{key: v})
            r = {p: run(panel, sma_cache, params, p) for p in ("train", "holdout")}
            tr[v], ho[v] = r["train"]["ratio"], r["holdout"]["ratio"]
            better = tr[v] > base["train"]["ratio"] and ho[v] > base["holdout"]["ratio"]
            if better and v != CURRENT[key]:
                flagged.append((title, key, v, fmt(v)))
            mark = "  <-- сейчас" if v == CURRENT[key] else ("  лучше на обоих" if better else "")
            print(f"{fmt(v):>12} | {r['train']['total_ret']:>+13.0%} {r['train']['max_dd']:>9.1%} {tr[v]:>7.2f} | "
                  f"{r['holdout']['total_ret']:>+15.0%} {r['holdout']['max_dd']:>9.1%} {ho[v]:>7.2f} |{mark}")
        s_tr, s_ho = pd.Series(tr), pd.Series(ho)
        rho = s_tr.rank().corr(s_ho.rank())          # Spearman = Pearson on ranks; avoids a scipy dependency
        spread = lambda s: (s.max() - s.min()) / s.mean()
        best_tr, best_ho = s_tr.idxmax(), s_ho.idxmax()
        summary.append((title, fmt(CURRENT[key]), fmt(best_tr), fmt(best_ho), rho, spread(s_tr), spread(s_ho)))
        print(f"   лучшее на train: {fmt(best_tr)}, на holdout: {fmt(best_ho)}; согласие train/holdout (Спирмен): {rho:+.2f}\n")

    print("=== ДВУМЕРНО: стоп × окно SMA, ret/dd (train | holdout) ===")
    print(f"{'стоп \\ SMA':>11} " + " ".join(f"{w:>13}" for w in GRID_SMAS))
    for stop in GRID_STOPS:
        cells = []
        for w in GRID_SMAS:
            r = {p: run(panel, sma_cache, dict(CURRENT, stop=stop, sma=w), p) for p in ("train", "holdout")}
            now = (stop == CURRENT["stop"] and w == CURRENT["sma"])
            cells.append(f"{r['train']['ratio']:5.2f}|{r['holdout']['ratio']:5.2f}{'*' if now else ' '}")
        print(f"{stop:>11.0%} " + " ".join(f"{c:>13}" for c in cells))
    print("  (* = текущее сочетание)\n")

    print("=== СВОДКА: насколько параметр вообще влияет ===")
    print(f"{'параметр':>26} {'сейчас':>8} {'лучш.train':>11} {'лучш.holdout':>13} {'Спирмен':>8} {'разброс train':>14} {'разброс holdout':>16}")
    for title, cur, btr, bho, rho, s1, s2 in summary:
        print(f"{title:>26} {cur:>8} {btr:>11} {bho:>13} {rho:>+8.2f} {s1:>14.0%} {s2:>16.0%}")

    if flagged:
        print("\n=== ЛУЧШЕ ТЕКУЩЕГО И НА TRAIN, И НА HOLDOUT - разрез по годам ===")
        t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
        eqs = {"текущее": simulate(close, panel[1], sma_for(close, 720, sma_cache), {}, fee=FEE)[2]}
        for title, key, v, label in flagged[:5]:
            params = dict(CURRENT, **{key: v})
            variant = {k: x for k, x in params.items() if k != "sma"}
            eqs[f"{title[:14]} {label}"] = simulate(close, panel[1], sma_for(close, params["sma"], sma_cache),
                                                    variant, fee=FEE)[2]
        print(f"{'год':>6} " + " ".join(f"{k[:22]:>22}" for k in eqs))
        for year in range(t0.year, t1.year + 1):
            a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), t0)
            b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), t1)
            cells = []
            for eq in eqs.values():
                w = window(eq, a, b)
                cells.append("-".rjust(22) if w is None else f"{w[0]:>+7.0%} / {w[1]:>6.0%}".rjust(22))
            print(f"{year:>6} " + " ".join(cells))
    else:
        print("\nНи одно значение не лучше текущего на обоих периодах одновременно.")


if __name__ == "__main__":
    main()
