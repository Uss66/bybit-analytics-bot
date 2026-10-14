"""
Does a market-wide SMA720 regime help? (2026-10-15, user's request after the
week of 2026-10-07..14)

The thing to explain is the week itself: ~1295 USDT fully invested in two
coins while five of the seven coins sat below their own SMA720 and the
market fell 5-14%. The per-coin SMA720 gate already refuses to ENTER a coin
that is under its trend; what it cannot see is that the rest of the market
is under theirs. "Breadth" is that missing number:

    breadth = share of the 7 coins currently trading below their own SMA720

Earlier tests of reacting to price action (rotation, trailing stop, per-coin
trend exit, crash exit) all failed, and speed_event_study.py showed why for
the last one: sharp falls bounce. This is a different class - it does not
react to a move, it limits exposure while the whole market is in a downtrend
- so it is not covered by those rejections.

Variants, fixed BEFORE any result. b is a breadth threshold:
0.55 ~ 4 of 7, 0.70 ~ 5 of 7, 0.85 ~ 6 of 7 coins under their SMA720.

  no_entries  b   no NEW position while breadth >= b
  no_adds     b   no dip-rebuy add while breadth >= b
  both        b   both of the above
  defensive   b   also sell everything while breadth >= b (plus both)

Engine is momentum_study.simulate (slot_cap_study's cash-based walk: cap 3,
cash/free-slots sizing, dip-rebuy, 168h cooldown after 3 losses, hourly
mark-to-market). Run at the real 0.18% fee and at the old 0.1%.

Checks the methodology asks for, so a lucky pick is not mistaken for an
effect:
  * duty cycle (a rule that is on ~100% or ~0% of the time is not a signal)
  * train and holdout kept apart, both fees
  * a variant only counts as flagged if ret/dd beats baseline on BOTH
    periods at the real fee
  * flagged variants then get the calendar-year breakdown, because the
    last two studies were carried by one bull year

Usage:
    python scripts/regime_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import load_panel
from momentum_study import simulate, breadth_below

THRESHOLDS = (0.55, 0.70, 0.85)
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
TRAIN_END, HOLD_START = "2025-11-30", "2025-12-01"
FEES = (0.0018, 0.001)

VARIANTS = {"baseline (сейчас)": {}}
for b in THRESHOLDS:
    n = round(b * 7)
    VARIANTS[f"нет входов ≥{n}/7"] = dict(breadth_x=b, no_entries=True)
    VARIANTS[f"нет докупок ≥{n}/7"] = dict(breadth_x=b, no_adds=True)
    VARIANTS[f"входы+докупки ≥{n}/7"] = dict(breadth_x=b, no_entries=True, no_adds=True)
    VARIANTS[f"защитный выход ≥{n}/7"] = dict(breadth_x=b, no_entries=True, no_adds=True, defensive=True)


def window(eq, a, b):
    s = eq[(eq.index >= a) & (eq.index < b)]
    if len(s) < 24:
        return None
    ret = s.iloc[-1] / s.iloc[0] - 1
    dd = ((s - s.cummax()) / s.cummax()).min()
    return ret, dd, (ret / abs(dd) if dd < 0 else float("inf"))


def run_period(conn_panels, variant, fee, period):
    close, score, sma = conn_panels[period]
    stats, trades, _ = simulate(close, score, sma, variant, fee=fee)
    return stats, trades


def main():
    conn = get_connection()
    panels = {"train": load_panel(conn, None, TRAIN_END), "holdout": load_panel(conn, HOLD_START, None)}
    full = load_panel(conn)
    conn.close()

    # 1. duty cycle
    print("=== 1. КАК ЧАСТО РЕЖИМ ВКЛЮЧЁН (доля часов, когда доля монет под SMA720 >= порога) ===")
    for label, (c, _, sm) in (("train", panels["train"]), ("holdout", panels["holdout"])):
        br = breadth_below(c, sm)
        print(f"  {label:8s} " + "   ".join(f"≥{round(b * 7)}/7: {(br >= b - 1e-9).mean():5.1%}" for b in THRESHOLDS))
    print()

    results = {}
    for fee in FEES:
        print(f"=== 2. ДОХОДНОСТЬ / ПРОСАДКА / RET-DD, комиссия {fee:.2%} за ногу ===")
        print(f"{'вариант':>26} | {'train':^30} | {'holdout':^30} | выходов по режиму")
        for label, variant in VARIANTS.items():
            row = {}
            for period in ("train", "holdout"):
                stats, trades = run_period(panels, variant, fee, period)
                row[period] = stats
                row[period + "_regime"] = int((trades["exit_reason"] == "regime").sum()) if len(trades) else 0
            results[(label, fee)] = row
            t, h = row["train"], row["holdout"]
            print(f"{label:>26} | {t['total_ret']:>+8.1%} {t['max_dd']:>7.1%} {t['ratio']:>6.2f}      | "
                  f"{h['total_ret']:>+8.1%} {h['max_dd']:>7.1%} {h['ratio']:>6.2f}      | "
                  f"{row['train_regime']:>4d} / {row['holdout_regime']:>3d}")
        print()

    # 3. flagged variants at the real fee
    base = results[("baseline (сейчас)", FEES[0])]
    print("=== 3. ЧТО ПРОШЛО ПЕРВЫЙ ФИЛЬТР (ret/dd лучше baseline и на train, и на holdout при 0,18%) ===")
    print(f"  baseline: train {base['train']['ratio']:.2f}, holdout {base['holdout']['ratio']:.2f}")
    flagged = []
    for label in VARIANTS:
        if label.startswith("baseline"):
            continue
        r = results[(label, FEES[0])]
        r_old = results[(label, FEES[1])]
        ok_new = r["train"]["ratio"] > base["train"]["ratio"] and r["holdout"]["ratio"] > base["holdout"]["ratio"]
        b_old = results[("baseline (сейчас)", FEES[1])]
        ok_old = r_old["train"]["ratio"] > b_old["train"]["ratio"] and r_old["holdout"]["ratio"] > b_old["holdout"]["ratio"]
        dd_ok = r["train"]["max_dd"] >= base["train"]["max_dd"] and r["holdout"]["max_dd"] >= base["holdout"]["max_dd"]
        if ok_new:
            flagged.append(label)
        print(f"  {label:>26}: train {r['train']['ratio']:5.2f} holdout {r['holdout']['ratio']:5.2f}   "
              f"{'ПРОШЁЛ' if ok_new else 'нет   '}  (и при 0,1%: {'да' if ok_old else 'нет'}; просадка не хуже на обоих: {'да' if dd_ok else 'нет'})")
    print()

    # 4. calendar years for baseline + flagged
    if flagged:
        print("=== 4. ПО ГОДАМ (один прогон на всей истории, 0,18%): доходность / просадка ===")
        close, score, sma = full
        t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
        eqs = {"baseline": simulate(close, score, sma, {}, fee=FEES[0])[2]}
        for label in flagged[:4]:
            eqs[label] = simulate(close, score, sma, VARIANTS[label], fee=FEES[0])[2]
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

    rows = []
    for (label, fee), r in results.items():
        for period in ("train", "holdout"):
            rows.append(dict(variant=label, fee=fee, period=period, **r[period]))
    pd.DataFrame(rows).to_csv("regime_summary.csv", index=False)
    print("\nЗаписано в regime_summary.csv")


if __name__ == "__main__":
    main()
