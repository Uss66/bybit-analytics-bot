"""
After a sharp move, does price keep going or bounce - and is it worth the fee?
(2026-10-15, follow-up to momentum_study.py on the user's suggestion to vary
the "steepness" and look at SHORT horizons)

momentum_study.py asked the question through the whole portfolio engine, so
the answer was tangled up with slot caps, dip-rebuy and cooldowns. This asks
it directly, with no strategy around it:

    when a coin has just fallen X% in H hours, what does the NEXT k hours
    look like - and would "sell now, buy back after k hours" have paid for
    its two legs?

For a sharp FALL the exit pays only if the following return is worse than
the cost of getting out and back in:

    gain of selling and re-entering after k hours  =  -F_k  -  2 * fee

(F_k is the plain forward return of the coin; no drift is subtracted,
because the alternative to selling really is just holding.) For a sharp RISE
the add pays only if  F_k  >  fee.

Two things the earlier studies taught that are built in here:
  * Autocorrelation inflation (the project's recurring trap): consecutive
    hours of one slide are one event, not twenty. Events are declustered
    per coin with a 24h dead time, and the t-statistic is computed over
    event-DAYS (coins that trigger on the same day are averaged first),
    because the seven coins move together (0.83 correlation) and would
    otherwise count one market move seven times.
  * Train and holdout are kept apart (<= 2025-11-30 / >= 2025-12-01).

k = 24h is the primary horizon: with a 24h dead time its forward windows do
not overlap, so its t-statistics are honest. k = 72h and 168h are shown for
context and overlap, so treat their t as optimistic.

Usage:
    python scripts/speed_event_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS

FEE = 0.0018                     # measured on the real account, per leg
ROUND_TRIP = 2 * FEE
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
DEAD_HOURS = 24
FORWARD = (24, 72, 168)

# horizon (hours) -> three steepness levels, in % over that horizon
GRID = {
    1: (2, 3, 4),
    2: (3, 4, 5),
    4: (3, 4, 5),
    8: (4, 5, 6),
    12: (4, 6, 8),
    24: (5, 8, 10),
}


def load_close():
    conn = get_connection()
    frames = {}
    for s in SYMBOLS:
        df = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                         conn, params={"s": s})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        frames[s] = df.set_index("ts")["close"]
    conn.close()
    close = pd.DataFrame(frames)
    return close.reindex(pd.date_range(close.index.min(), close.index.max(), freq="h"))


def declustered_events(trigger):
    """Boolean series -> list of timestamps, at most one per DEAD_HOURS."""
    out, last = [], None
    for ts in trigger.index[trigger.values]:
        if last is None or ts - last >= pd.Timedelta(hours=DEAD_HOURS):
            out.append(ts)
            last = ts
    return out


def day_t(rows, col):
    """Average coins within an event-day first, then t over the days."""
    if rows.empty:
        return float("nan"), 0
    per_day = rows.groupby(rows["ts"].dt.floor("D"))[col].mean()
    n = len(per_day)
    if n < 3 or per_day.std(ddof=1) == 0:
        return float("nan"), n
    return per_day.mean() / (per_day.std(ddof=1) / np.sqrt(n)), n


def study(close, direction):
    fwd = {k: close.shift(-k) / close - 1 for k in FORWARD}
    results = []
    for h, levels in GRID.items():
        ret_h = close / close.shift(h) - 1
        for x in levels:
            sign = -1 if direction == "fall" else 1
            events = []
            for s in SYMBOLS:
                trig = (ret_h[s] <= -x / 100) if direction == "fall" else (ret_h[s] >= x / 100)
                for ts in declustered_events(trig):
                    row = dict(ts=ts, symbol=s)
                    ok = True
                    for k in FORWARD:
                        v = fwd[k].at[ts, s] if ts in fwd[k].index else np.nan
                        if np.isnan(v):
                            ok = False
                        row[f"F{k}"] = v
                    if ok:
                        events.append(row)
            ev = pd.DataFrame(events)
            if ev.empty:
                continue
            # what the move is worth, net of costs
            for k in FORWARD:
                if direction == "fall":
                    ev[f"g{k}"] = -ev[f"F{k}"] - ROUND_TRIP        # sell, rebuy after k
                else:
                    ev[f"g{k}"] = ev[f"F{k}"] - FEE                # add, hold k
            for label, part in (("train", ev[ev["ts"] < SPLIT]), ("holdout", ev[ev["ts"] >= SPLIT])):
                if part.empty:
                    continue
                t24, days = day_t(part, "g24")
                coins_pos = (part.groupby("symbol")["g24"].mean() > 0).sum()
                coins_n = part["symbol"].nunique()
                results.append(dict(
                    direction=direction, h=h, x=x, period=label, n=len(part), days=days,
                    g24=part["g24"].mean(), g72=part["g72"].mean(), g168=part["g168"].mean(),
                    t24=t24, hit24=(part["g24"] > 0).mean(), coins=f"{coins_pos}/{coins_n}"))
    return pd.DataFrame(results)


def show(res, direction):
    title = {"fall": "РЕЗКОЕ ПАДЕНИЕ: выгода от «продать сейчас и купить назад через k часов» (уже за вычетом 0,36% за круг)",
             "rise": "РЕЗКИЙ РОСТ: выгода от «докупить и держать k часов» (за вычетом 0,18%)"}[direction]
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")
    print(f"{'гор.':>4} {'порог':>6} | {'период':>8} {'событий':>8} {'дней':>5} | "
          f"{'k=24ч':>8} {'t(24ч)':>7} {'доля+':>6} {'монет+':>7} | {'k=72ч':>8} {'k=168ч':>8}")
    for (h, x), grp in res[res["direction"] == direction].groupby(["h", "x"]):
        for _, r in grp.iterrows():
            print(f"{h:>3}ч {x:>5}% | {r['period']:>8} {r['n']:>8d} {r['days']:>5d} | "
                  f"{r['g24']:>+8.2%} {r['t24']:>7.2f} {r['hit24']:>6.0%} {r['coins']:>7} | "
                  f"{r['g72']:>+8.2%} {r['g168']:>+8.2%}")


def main():
    close = load_close()
    print(f"Данные: {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}, {len(SYMBOLS)} монет, "
          f"комиссия {FEE:.2%} за ногу, мёртвое время между событиями {DEAD_HOURS}ч")
    base = {k: (close.shift(-k) / close - 1).stack().mean() for k in FORWARD}
    print("Безусловный средний рост за k часов (для сравнения): "
          + ", ".join(f"{k}ч {v:+.2%}" for k, v in base.items()))
    res = pd.concat([study(close, "fall"), study(close, "rise")], ignore_index=True)
    show(res, "fall")
    show(res, "rise")
    res.to_csv("speed_event_summary.csv", index=False)
    print("\nЗаписано в speed_event_summary.csv")


if __name__ == "__main__":
    main()
