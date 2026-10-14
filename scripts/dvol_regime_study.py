"""
Does the LEVEL of implied volatility work as a regime gate? (2026-10-15)

The strategy already trades DVOL's 1-DAY CHANGE (dvol_drop_extreme +2,
dvol_spike_extreme -2). What it has never looked at is where DVOL SITS:
a market priced for stress (options-implied vol in the top of its own
recent range) versus a calm one. That is a regime question, the same shape
as regime_study.py's breadth gate, which came back null because breadth is
"on" about half the time and the gate removed half the return. DVOL level
at a high percentile should be rarer, and it is a different information
source (options market, not the coins' own prices), so it is a fair second
attempt rather than a re-run.

Level = percentile rank of BTC's DVOL close among the previous 365 daily
closes (min 90). NO LOOK-AHEAD: the daily bar of day D closes at the end of
D, so every hour of day D uses the rank computed from day D-1's close.

Variants, fixed before any result - the same four actions as regime_study,
at percentile thresholds 70 / 80 / 90:
  no_entries  no NEW position while the regime is on
  no_adds     no dip-rebuy add while it is on
  both
  defensive   also sell everything while it is on

Same engine (momentum_study.simulate), same fees (0.18% real, 0.1% old),
same first filter (ret/dd beats baseline on BOTH train and holdout at the
real fee) and the same year breakdown for whatever passes.

Usage:
    python scripts/dvol_regime_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import load_panel
from momentum_study import simulate
from regime_study import window, TRAIN_END, HOLD_START, FEES

THRESHOLDS = (0.70, 0.80, 0.90)


def load_levels():
    conn = get_connection()
    d = pd.read_sql("SELECT ts, close FROM deribit_dvol WHERE symbol = 'BTC' ORDER BY ts", conn)
    conn.close()
    d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.normalize()
    close = d.set_index("ts")["close"]
    rank = close.rolling(365, min_periods=90).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    return close, rank


def regime_array(index, rank, thr):
    """Boolean per hour: yesterday's DVOL percentile >= thr. Shifted one day
    so an hour never sees a daily close that has not happened yet."""
    known = rank.shift(1)                                  # day D carries day D-1's rank
    hourly = known.reindex(index.floor("D"))
    vals = hourly.values
    return np.where(np.isnan(vals), False, vals >= thr - 1e-9)


def build_variants(index, rank):
    v = {"baseline (сейчас)": {}}
    for thr in THRESHOLDS:
        arr = regime_array(index, rank, thr)
        n = int(thr * 100)
        v[f"нет входов DVOL≥{n}п"] = dict(regime_arr=arr, no_entries=True)
        v[f"нет докупок DVOL≥{n}п"] = dict(regime_arr=arr, no_adds=True)
        v[f"входы+докупки DVOL≥{n}п"] = dict(regime_arr=arr, no_entries=True, no_adds=True)
        v[f"защитный выход DVOL≥{n}п"] = dict(regime_arr=arr, no_entries=True, no_adds=True, defensive=True)
    return v


def main():
    dvol, rank = load_levels()
    print(f"DVOL BTC: {dvol.index[0]:%Y-%m-%d}..{dvol.index[-1]:%Y-%m-%d}, {len(dvol)} дней; "
          f"ранг считается с {rank.first_valid_index():%Y-%m-%d}\n")

    conn = get_connection()
    panels = {"train": load_panel(conn, None, TRAIN_END), "holdout": load_panel(conn, HOLD_START, None)}
    full = load_panel(conn)
    conn.close()

    print("=== 1. КАК ЧАСТО РЕЖИМ ВКЛЮЧЁН (доля часов) ===")
    for label, (c, _, _) in panels.items():
        print(f"  {label:8s} " + "   ".join(
            f"≥{int(t * 100)}п: {regime_array(c.index, rank, t).mean():5.1%}" for t in THRESHOLDS))
    print()

    variants = {p: build_variants(panels[p][0].index, rank) for p in panels}
    results = {}
    for fee in FEES:
        print(f"=== 2. ДОХОДНОСТЬ / ПРОСАДКА / RET-DD, комиссия {fee:.2%} за ногу ===")
        print(f"{'вариант':>28} | {'train':^30} | {'holdout':^30} | выходов по режиму")
        for label in variants["train"]:
            row = {}
            for period in ("train", "holdout"):
                close, score, sma = panels[period]
                stats, trades, _ = simulate(close, score, sma, variants[period][label], fee=fee)
                row[period] = stats
                row[period + "_regime"] = int((trades["exit_reason"] == "regime").sum()) if len(trades) else 0
            results[(label, fee)] = row
            t, h = row["train"], row["holdout"]
            print(f"{label:>28} | {t['total_ret']:>+8.1%} {t['max_dd']:>7.1%} {t['ratio']:>6.2f}      | "
                  f"{h['total_ret']:>+8.1%} {h['max_dd']:>7.1%} {h['ratio']:>6.2f}      | "
                  f"{row['train_regime']:>4d} / {row['holdout_regime']:>3d}")
        print()

    base, b_old = results[("baseline (сейчас)", FEES[0])], results[("baseline (сейчас)", FEES[1])]
    print("=== 3. ЧТО ПРОШЛО ПЕРВЫЙ ФИЛЬТР (ret/dd лучше baseline и на train, и на holdout при 0,18%) ===")
    print(f"  baseline: train {base['train']['ratio']:.2f}, holdout {base['holdout']['ratio']:.2f}")
    flagged = []
    for label in variants["train"]:
        if label.startswith("baseline"):
            continue
        r, ro = results[(label, FEES[0])], results[(label, FEES[1])]
        ok = r["train"]["ratio"] > base["train"]["ratio"] and r["holdout"]["ratio"] > base["holdout"]["ratio"]
        ok_old = ro["train"]["ratio"] > b_old["train"]["ratio"] and ro["holdout"]["ratio"] > b_old["holdout"]["ratio"]
        dd_ok = r["train"]["max_dd"] >= base["train"]["max_dd"] and r["holdout"]["max_dd"] >= base["holdout"]["max_dd"]
        if ok:
            flagged.append(label)
        print(f"  {label:>28}: train {r['train']['ratio']:5.2f} holdout {r['holdout']['ratio']:5.2f}   "
              f"{'ПРОШЁЛ' if ok else 'нет   '}  (и при 0,1%: {'да' if ok_old else 'нет'}; просадка не хуже на обоих: {'да' if dd_ok else 'нет'})")
    print()

    if flagged:
        print("=== 4. ПО ГОДАМ (один прогон на всей истории, 0,18%): доходность / просадка ===")
        close, score, sma = full
        fv = build_variants(close.index, rank)
        t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
        eqs = {"baseline": simulate(close, score, sma, {}, fee=FEES[0])[2]}
        for label in flagged[:4]:
            eqs[label] = simulate(close, score, sma, fv[label], fee=FEES[0])[2]
        print(f"{'год':>6} " + " ".join(f"{k[:24]:>24}" for k in eqs))
        for year in range(t0.year, t1.year + 1):
            a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), t0)
            b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), t1)
            cells = []
            for eq in eqs.values():
                w = window(eq, a, b)
                cells.append("-".rjust(24) if w is None else f"{w[0]:>+8.1%} / {w[1]:>7.1%}".rjust(24))
            print(f"{year:>6} " + " ".join(cells))
    else:
        print("=== 4. Ни один вариант не прошёл первый фильтр - разрез по годам не нужен ===")

    rows = [dict(variant=l, fee=f, period=p, **r[p]) for (l, f), r in results.items() for p in ("train", "holdout")]
    pd.DataFrame(rows).to_csv("dvol_regime_summary.csv", index=False)
    print("\nЗаписано в dvol_regime_summary.csv")


if __name__ == "__main__":
    main()
