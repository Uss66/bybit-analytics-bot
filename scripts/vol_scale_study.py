"""
Should the strategy's scales follow the market's own volatility?
(2026-10-15, user's question: "adjust the scale of the analysis to the overall
movement of the market")

On the HONEST score (no DVOL / DXY rules - see lookahead_audit.py), because
every earlier number used a score that knew the future.

Idea: a fixed -8% stop, a fixed -3% dip and a fixed stake mean different
things in a calm market and in a violent one. One market-wide gauge scales
them:

    r(t) = BTC realized volatility over the last 30 days
           / its own median over the last 365 days          (r > 1: noisy)

Only trailing data is used, so nothing here looks ahead.

Variants, fixed before any result:
  stake_0.5    stake = cash / free_slots * min(1, 1/r), never below 0.5
               (volatility targeting: smaller positions when the market is
               noisy; never larger than today)
  stake_0.25   same with a floor of 0.25
  stop_scaled  stop = -8% * clip(r, 0.75, 1.5): wider when noisy, tighter
               when calm
  stop+dip     the stop above and the dip-rebuy thresholds scaled the same

Honest caveat built into the output: the honest strategy sits in the market
only ~10% of the holdout's hours (6 trades), so the holdout can barely tell
variants apart. Train carries the verdict, calendar years show whether one
year carries it, and the stake variants are the only ones that move holdout
at all.

Usage:
    python scripts/vol_scale_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from strategy import load_events_with_returns
from momentum_study import simulate
from regime_study import window
from lookahead_audit import hourly_close, build_panel, split, DVOL_CATS, DXY_CATS, TRAIN_END, SPLIT

FEE = 0.0018
STOP = -0.08


def vol_ratio(btc):
    """Trailing 30d realized vol / trailing 365d median of it. 1.0 until there
    is enough history, i.e. 'no scaling' rather than a guess."""
    r = np.log(btc.ffill()).diff()
    rv = r.rolling(720, min_periods=360).std()
    med = rv.rolling(8760, min_periods=2000).median()
    return (rv / med).fillna(1.0).clip(0.2, 5.0)


def arrays(ratio, index):
    r = ratio.reindex(index).fillna(1.0).values
    return dict(
        stake_0_5=np.clip(1.0 / r, 0.5, 1.0),
        stake_0_25=np.clip(1.0 / r, 0.25, 1.0),
        stop=STOP * np.clip(r, 0.75, 1.5),
        dip=np.clip(r, 0.75, 1.5),
    )


def variants_for(ratio, index):
    a = arrays(ratio, index)
    return {
        "baseline (сейчас)": {},
        "размер ×1/r, не < 0,5": dict(stake_arr=a["stake_0_5"]),
        "размер ×1/r, не < 0,25": dict(stake_arr=a["stake_0_25"]),
        "стоп × r (0,75–1,5)": dict(stop_arr=a["stop"]),
        "стоп и докупка × r": dict(stop_arr=a["stop"], dip_scale_arr=a["dip"]),
    }


def main():
    conn = get_connection()
    hourly = hourly_close(conn)
    panel = build_panel(conn, {s: hourly[s].dropna() for s in hourly.columns},
                        load_events_with_returns(conn), drop_cats=DVOL_CATS + DXY_CATS)
    conn.close()
    close = panel[0]
    ratio = vol_ratio(hourly["BTCUSDT"])
    print(f"r (волатильность BTC / её годовая медиана): медиана {ratio.median():.2f}, "
          f"p10 {ratio.quantile(.1):.2f}, p90 {ratio.quantile(.9):.2f}\n")

    results = {}
    print("=== ДОХОДНОСТЬ / ПРОСАДКА / RET-DD на честном сигнале, комиссия 0,18% ===")
    print(f"{'вариант':>26} | {'train':^30} | {'holdout':^30}")
    for label in variants_for(ratio, close.index[:1]):
        row = {}
        for name in ("train", "holdout"):
            c, s, m = split(panel, name)
            v = variants_for(ratio, c.index)[label]
            st, _, _ = simulate(c, s, m, v, fee=FEE)
            row[name] = st
        results[label] = row
        t, h = row["train"], row["holdout"]
        print(f"{label:>26} | {t['total_ret']:>+8.0%} {t['max_dd']:>7.1%} {t['ratio']:>6.2f}      | "
              f"{h['total_ret']:>+8.1%} {h['max_dd']:>7.1%} {h['ratio']:>6.2f}")

    base = results["baseline (сейчас)"]
    print("\n=== ПО ГОДАМ (один прогон на всей истории): доходность / просадка ===")
    t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(hours=1)
    fv = variants_for(ratio, close.index)
    eqs = {k: simulate(*panel, v, fee=FEE)[2] for k, v in fv.items()}
    print(f"{'год':>6} " + " ".join(f"{k[:22]:>22}" for k in eqs))
    better_years = {k: 0 for k in eqs}
    for year in range(t0.year, t1.year + 1):
        a = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), t0)
        b = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), t1)
        cells, ws = [], {}
        for k, eq in eqs.items():
            w = window(eq, a, b)
            ws[k] = w
            cells.append("-".rjust(22) if w is None else f"{w[0]:>+8.0%} / {w[1]:>6.0%}".rjust(22))
        print(f"{year:>6} " + " ".join(cells))
        b0 = ws["baseline (сейчас)"]
        for k, w in ws.items():
            if k != "baseline (сейчас)" and w and b0 and w[2] > b0[2]:
                better_years[k] += 1
    n_years = t1.year - t0.year + 1
    print("\nлет, когда ret/dd лучше baseline: " + ", ".join(
        f"{k[:22]} {v}/{n_years}" for k, v in better_years.items() if not k.startswith("baseline")))

    print("\n=== ПЕРВЫЙ ФИЛЬТР (лучше baseline по ret/dd на train, и holdout не хуже) ===")
    for label, row in results.items():
        if label.startswith("baseline"):
            continue
        ok = row["train"]["ratio"] > base["train"]["ratio"] and row["holdout"]["ratio"] >= base["holdout"]["ratio"] - 1e-9
        print(f"  {label:>26}: train {row['train']['ratio']:.2f} vs {base['train']['ratio']:.2f}, "
              f"holdout {row['holdout']['ratio']:.2f} vs {base['holdout']['ratio']:.2f}  -> {'ПРОШЁЛ' if ok else 'нет'}")


if __name__ == "__main__":
    main()
