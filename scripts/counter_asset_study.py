"""
Is there anything that moves AGAINST the market, to rotate into during a
drawdown? (2026-10-15, user's question)

The strategy's seven coins correlate about 0.83 in the holdout (measured in
trend_exit_study), so rotating between THEM in a drawdown moves between
near-identical bets. Two honest questions:

  1. Among every coin the database has a long-enough history for, does any
     one behave differently when the market falls? "Differently" is
     measured two ways, because they answer different things:
       * downside correlation - correlation of daily returns on the days BTC
         fell 2% or more. A real counter-trend asset is negative here; a
         high-beta one is near 1.
       * what it does AFTER a drawdown starts - the next 7 days after BTC's
         7-day return reaches -5% (events at least 7 days apart, so one
         slide is one event), compared with BTC's own next 7 days.
  2. Does the ranking SURVIVE? Picking last period's "best defensive coin"
     only matters if it is still defensive afterwards. Train and holdout use
     the project split; the rank agreement across assets is printed. With
     this many assets, one of them will always look defensive in one
     period.

Gold (XAUTUSDT, Tether Gold - tradable spot on Bybit; PAXG is not listed)
is included as the one candidate that is a different asset class rather than
another crypto coin. Its history starts 2025-04, so its train window is
short; read its numbers with that in mind.

Costs: moving from a held coin into another is a sell and a buy, 2 x 0.18%.
The "net" column is the 7-day excess over BTC after that 0.36%.

Usage:
    python scripts/counter_asset_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection

FEE = 0.0018
ROTATE_COST = 2 * FEE
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
MIN_HOURS = 9000                 # >= ~375 days of hourly candles
EVENT_DROP = -0.05               # BTC 7-day return that starts a "drawdown"
DEAD_DAYS = 7
DOWN_DAY = -0.02                 # BTC daily return that defines a down day
STRATEGY_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]


def load_daily():
    conn = get_connection()
    syms = pd.read_sql("SELECT symbol FROM ohlcv GROUP BY symbol HAVING count(*) >= %(n)s ORDER BY symbol",
                       conn, params={"n": MIN_HOURS})["symbol"].tolist()
    frames = {}
    for s in syms:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        frames[s] = d.set_index("ts")["close"]
    conn.close()
    hourly = pd.DataFrame(frames)
    daily = hourly.resample("D").last()
    # Cut every series at the last date BTC has. XAUT was fetched later than
    # the rest and runs a month past them; left in, the last drawdown event
    # looked 7 days ahead into a BTC price that does not exist and put a NaN
    # into the gold average.
    daily = daily.loc[:daily["BTCUSDT"].last_valid_index()]
    return daily.loc[daily.index >= "2024-08-27"]       # common ground for the long-history coins


def drawdown_events(btc):
    """Dates when BTC's 7-day return first reaches EVENT_DROP, at least DEAD_DAYS apart."""
    ret7 = btc / btc.shift(7) - 1
    out, last = [], None
    for ts in ret7.index[ret7 <= EVENT_DROP]:
        if last is None or (ts - last).days >= DEAD_DAYS:
            out.append(ts)
            last = ts
    return out


def stats_for(daily, period_mask):
    d = daily[period_mask]
    ret = d.pct_change()
    btc = ret["BTCUSDT"]
    down = btc <= DOWN_DAY
    events = [e for e in drawdown_events(d["BTCUSDT"]) if e in d.index]
    rows = {}
    for s in d.columns:
        if s == "BTCUSDT" or d[s].notna().sum() < 120:
            continue
        both = pd.concat([ret[s], btc], axis=1, keys=["a", "b"]).dropna()
        dn = both[both["b"] <= DOWN_DAY]
        beta = both["a"].cov(both["b"]) / both["b"].var()
        fwd = []
        for e in events:
            i = d.index.get_loc(e)
            if (i + 7 < len(d) and not np.isnan(d[s].iloc[i]) and not np.isnan(d[s].iloc[i + 7])
                    and not np.isnan(d["BTCUSDT"].iloc[i + 7])):
                fwd.append((d[s].iloc[i + 7] / d[s].iloc[i] - 1) - (d["BTCUSDT"].iloc[i + 7] / d["BTCUSDT"].iloc[i] - 1))
        rows[s] = dict(
            n_days=len(both), beta=beta, corr=both["a"].corr(both["b"]),
            down_corr=dn["a"].corr(dn["b"]) if len(dn) >= 8 else np.nan,
            down_ret=dn["a"].mean() if len(dn) else np.nan, n_down=len(dn),
            n_ev=len(fwd), excess7=np.mean(fwd) if fwd else np.nan,
            net7=(np.mean(fwd) - ROTATE_COST) if fwd else np.nan)
    return pd.DataFrame(rows).T, len(events)


def main():
    daily = load_daily()
    print(f"Активы с историей >= {MIN_HOURS} часов: {len(daily.columns)}; дневные данные "
          f"{daily.index[0]:%Y-%m-%d}..{daily.index[-1]:%Y-%m-%d}")
    train_mask, hold_mask = daily.index < SPLIT, daily.index >= SPLIT
    tr, n_tr = stats_for(daily, train_mask)
    ho, n_ho = stats_for(daily, hold_mask)
    print(f"Просадок BTC (7д <= {EVENT_DROP:.0%}, с интервалом >= {DEAD_DAYS} дней): train {n_tr}, holdout {n_ho}\n")

    both_idx = tr.index.intersection(ho.index)
    print("=== 1. КАК АКТИВ ВЕДЁТ СЕБЯ, КОГДА BTC ПАДАЕТ ===")
    print(f"{'актив':>10} | {'бета':>5} {'корр.':>5} {'корр. в дни -2%+':>16} {'дн.':>4} | "
          f"{'бета':>5} {'корр.':>5} {'корр. в дни -2%+':>16} {'дн.':>4}")
    print(f"{'':>10} | {'train':^34} | {'holdout':^34}")
    order = ho.reindex(both_idx)["down_corr"].sort_values().index
    for s in order:
        a, b = tr.loc[s], ho.loc[s]
        tag = "  <-- ЗОЛОТО" if s == "XAUTUSDT" else ("  (стратегия)" if s in STRATEGY_COINS else "")
        print(f"{s:>10} | {a['beta']:>5.2f} {a['corr']:>5.2f} {a['down_corr']:>16.2f} {int(a['n_down']):>4d} | "
              f"{b['beta']:>5.2f} {b['corr']:>5.2f} {b['down_corr']:>16.2f} {int(b['n_down']):>4d}{tag}")

    print("\n=== 2. ЧТО ДЕЛАЕТ АКТИВ В СЛЕДУЮЩИЕ 7 ДНЕЙ ПОСЛЕ НАЧАЛА ПРОСАДКИ BTC (разница с BTC, и чистая за 0,36%) ===")
    print(f"{'актив':>10} | {'train: собыйт.':>14} {'превыш.':>8} {'чистая':>8} | {'holdout: собыйт.':>16} {'превыш.':>8} {'чистая':>8} | знак")
    for s in both_idx:
        a, b = tr.loc[s], ho.loc[s]
        same = "совпал" if (a["net7"] > 0) == (b["net7"] > 0) else "разный"
        both_pos = " <-- плюс на обоих" if a["net7"] > 0 and b["net7"] > 0 else ""
        print(f"{s:>10} | {int(a['n_ev']):>14d} {a['excess7']:>+8.1%} {a['net7']:>+8.1%} | "
              f"{int(b['n_ev']):>16d} {b['excess7']:>+8.1%} {b['net7']:>+8.1%} | {same}{both_pos}")

    print("\n=== 3. УСТОЙЧИВОСТЬ ОТБОРА (ранг активов по двум признакам: train против holdout) ===")
    for col, label in (("down_corr", "корреляция в дни падения BTC"), ("excess7", "превышение над BTC за 7 дней после начала просадки")):
        x, y = tr.loc[both_idx, col].rank(), ho.loc[both_idx, col].rank()
        print(f"  {label}: согласие train/holdout (Спирмен) {x.corr(y):+.2f}  по {len(both_idx)} активам")

    neg = [(s, ho.loc[s, 'down_corr']) for s in both_idx if ho.loc[s, 'down_corr'] < 0]
    print(f"\n  Активов с отрицательной корреляцией в дни падения BTC: train {int((tr['down_corr'] < 0).sum())}, "
          f"holdout {len(neg)} из {len(both_idx)}")
    if "XAUTUSDT" in tr.index and "XAUTUSDT" in ho.index:
        g_t, g_h = tr.loc["XAUTUSDT"], ho.loc["XAUTUSDT"]
        print(f"  Золото (XAUT): корреляция в дни падения BTC train {g_t['down_corr']:+.2f} / holdout {g_h['down_corr']:+.2f}; "
              f"средняя доходность в такие дни train {g_t['down_ret']:+.2%} / holdout {g_h['down_ret']:+.2%}")

    pd.concat({"train": tr, "holdout": ho}, axis=1).to_csv("counter_asset_summary.csv")
    print("\nЗаписано в counter_asset_summary.csv")


if __name__ == "__main__":
    main()
