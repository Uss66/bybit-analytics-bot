"""
Long-only Donchian trend ensemble with volatility sizing, on the 7 coins
(2026-10-15). The one idea from the literature search with published,
survivorship-free, net-of-fee evidence: Zarattini, Pagani, Barbon, "Catching
Crypto Trends" (SFI working paper, 2025). Their BTC result 2015-2025, before
costs: ensemble Sharpe 1.58, max drawdown 19% vs >80% buy-and-hold, beta 0.17.
This asks whether anything like it holds on OUR data, at OUR fee, and whether it
adds anything next to the honest FGI strategy.

Design, fixed before any result:
  * Daily bars. Day-D close = close of the 23:00 candle of D (candles are
    stamped with their open), i.e. the price at 00:00 UTC of D+1. The signal at
    the end of D trades at that price and earns D+1's return: no look-ahead.
  * Donchian channel of window n: Up = max close of the previous n days,
    Down = min. Long when the close breaks above Up; exit when the close falls
    below the channel midpoint (the paper's trailing stop).
  * Windows tested alone: 5, 10, 20, 30, 60, 90, 150, 250, 360 days.
    Ensembles: E_short = {5,10,20,30}; E_all = all nine. The position of an
    ensemble is the average of its windows' on/off states.
  * Sizing: each of the 7 coins is a sleeve with 1/7 of the capital. Weight =
    ensemble * min(1, 25% / 90-day annualised vol), then /7: unleveraged, cash
    when out. "No vol target" variant: ensemble / 7.
  * Rebalancing: trade a sleeve only when its on/off state changes or its
    weight has drifted more than 20% from target (the paper's rule); cost
    0.18% of traded value (the measured fee, above the paper's 10 bp).
  * Reported on windows of ONE full run (a 360-day window needs its history):
    train <= 2025-11-30, holdout >= 2025-12-01, calendar years, per coin.

Predeclared bar for "worth building on" (all of them):
  1. E_all or E_short: net Sharpe > 0.5 on train AND on holdout;
  2. positive in at least 4 of the 6 calendar years;
  3. at least 6 of the 9 single windows have positive Sharpe on both periods
     (a plateau, not a lucky window);
  4. a 50/50 blend with the honest FGI strategy beats the FGI strategy alone
     on ret/dd in both periods.

Usage:
    python scripts/trend_ensemble_study.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import SYMBOLS, load_panel
from momentum_study import simulate

FEE = 0.0018
WINDOWS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
E_SHORT = (5, 10, 20, 30)
VOL_TARGET, VOL_WIN, REBAL_BAND = 0.25, 90, 0.20
SPLIT = pd.Timestamp("2025-12-01", tz="UTC")
N = len(SYMBOLS)


def daily_close():
    conn = get_connection()
    cols = {}
    for s in SYMBOLS:
        d = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": s})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        h = d.set_index("ts")["close"]
        day = h[h.index.hour == 23]                      # close of the last candle of the day = price at 00:00 next day
        day.index = day.index.normalize()
        cols[s] = day
    conn.close()
    df = pd.DataFrame(cols)
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D")).ffill(limit=3)


def donchian_states(close, n):
    """Per day: 1 if long. Entry: close > max(previous n closes). Exit: close < midpoint of the channel."""
    c = close.values
    out = np.zeros_like(c)
    for j in range(c.shape[1]):
        inpos = False
        for t in range(n, c.shape[0]):
            win = c[t - n:t, j]
            if np.isnan(win).any() or np.isnan(c[t, j]):
                continue
            up, dn = win.max(), win.min()
            if not inpos and c[t, j] > up:
                inpos = True
            elif inpos and c[t, j] < (up + dn) / 2:
                inpos = False
            out[t, j] = 1.0 if inpos else 0.0
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def run(close, signal, vol_target=True):
    """Unit-based sleeve simulation. Returns daily equity (index = day-end), trades, per-coin gross pnl."""
    px = close.values
    sig = signal.values
    ret = close.pct_change()
    vol = (ret.rolling(VOL_WIN, min_periods=45).std() * np.sqrt(365)).values
    T = px.shape[0]
    cash, qty = 1.0, np.zeros(N)
    prev_state = np.zeros(N)
    equity = np.full(T, np.nan)
    gross = np.zeros(N)
    traded = 0.0
    last = np.full(N, np.nan)                             # last known price per coin
    for t in range(T):
        # A coin that has no price yet (BNB only starts in 2022-03, SOL in 2021-10, ...) or has a gap is
        # simply not tradable that day. The first version skipped the WHOLE day whenever any one of
        # the seven was missing, which meant no trading at all until March 2022.
        avail = ~np.isnan(px[t])
        p = np.where(avail, px[t], last)
        last = np.where(avail, px[t], last)
        val = np.where(np.isnan(p), 0.0, qty * np.nan_to_num(p))
        eq = cash + val.sum()
        equity[t] = eq
        if t + 1 < T:
            nxt = px[t + 1]
            ok = avail & ~np.isnan(nxt)
            gross += np.where(ok, val * (np.where(ok, nxt, 1.0) / np.where(ok, px[t], 1.0) - 1), 0.0)
        if t == 0:
            continue
        s = np.where(avail, sig[t], 0.0)
        v = np.where(np.isnan(vol[t]) | (vol[t] <= 0), np.inf, vol[t])
        sizing = np.minimum(1.0, VOL_TARGET / v) if vol_target else np.where(np.isnan(vol[t]), 0.0, 1.0)
        tgt = s * sizing / N                              # fraction of equity per sleeve
        cur = val / eq
        changed = (s > 0) != (prev_state > 0)
        drift = (cur > 0) & (np.abs(tgt - cur) > REBAL_BAND * np.maximum(cur, 1e-12))
        trade = avail & (changed | drift | ((s == 0) & (cur > 0)))
        for j in np.where(trade)[0]:
            target_val = tgt[j] * eq
            delta = target_val - val[j]
            cost = abs(delta) * FEE
            qty[j] += delta / p[j]
            cash -= delta + cost
            traded += abs(delta)
        prev_state = s.copy()
    return pd.Series(equity, index=close.index), traded, pd.Series(gross, index=close.columns)


def stats(eq, a, b):
    s = eq[(eq.index >= a) & (eq.index < b)].dropna()
    if len(s) < 60:
        return None
    r = s.pct_change().dropna()
    tot = s.iloc[-1] / s.iloc[0] - 1
    dd = ((s - s.cummax()) / s.cummax()).min()
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    return dict(tot=tot, cagr=cagr, vol=r.std() * np.sqrt(365), sharpe=sharpe, dd=dd,
                rdd=tot / abs(dd) if dd < 0 else np.nan)


def fmt(st):
    return "-".rjust(40) if st is None else (f"{st['tot']:>+7.0%} {st['cagr']:>+6.0%} {st['vol']:>5.0%} "
                                              f"{st['sharpe']:>6.2f} {st['dd']:>6.0%} {st['rdd']:>6.2f}")


def main():
    close = daily_close()
    t0, t1 = close.index[0], close.index[-1] + pd.Timedelta(days=1)
    print(f"Данные: {t0:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}, {N} монет, комиссия {FEE:.2%}, "
          f"порог ребалансировки {REBAL_BAND:.0%}, цель волатильности {VOL_TARGET:.0%}\n")

    states = {n: donchian_states(close, n) for n in WINDOWS}
    variants = {f"окно {n}д": states[n] for n in WINDOWS}
    variants["ансамбль короткий {5,10,20,30}"] = sum(states[n] for n in E_SHORT) / len(E_SHORT)
    variants["ансамбль все 9"] = sum(states.values()) / len(WINDOWS)

    hdr = f"{'':>31} | {'всего':>7} {'CAGR':>6} {'вол.':>5} {'Шарп':>6} {'просад':>6} {'ret/dd':>6}"
    results, eqs = {}, {}
    for per, a, b in (("TRAIN", t0, SPLIT), ("HOLDOUT", SPLIT, t1)):
        print(f"=== {per} ===\n{hdr}")
        for label, sg in variants.items():
            if label not in eqs:
                eqs[label] = run(close, sg)
            st = stats(eqs[label][0], a, b)
            results[(label, per)] = st
            print(f"{label:>31} | {fmt(st)}")
        if "ансамбль все 9" not in eqs or ("все 9 без цели", None) not in results:
            eqs["ансамбль все 9, без цели вол."] = eqs.get("ансамбль все 9, без цели вол.") or run(close, variants["ансамбль все 9"], vol_target=False)
        st = stats(eqs["ансамбль все 9, без цели вол."][0], a, b)
        results[("ансамбль все 9, без цели вол.", per)] = st
        print(f"{'ансамбль все 9, без цели вол.':>31} | {fmt(st)}")
        ew = close.pct_change().mean(axis=1).add(1).cumprod()
        btc = close["BTCUSDT"].pct_change().add(1).cumprod()
        print(f"{'удержание BTC':>31} | {fmt(stats(btc, a, b))}")
        print(f"{'удержание 7 монет поровну':>31} | {fmt(stats(ew, a, b))}\n")

    # ------------- calendar years, ensembles -------------
    print("=== ПО ГОДАМ: доходность / просадка ===")
    keys = ["ансамбль короткий {5,10,20,30}", "ансамбль все 9"]
    print(f"{'год':>6} " + " ".join(f"{k[:26]:>26}" for k in keys) + f" {'7 монет поровну':>20}")
    pos_years = {k: 0 for k in keys}
    ew = close.pct_change().mean(axis=1).add(1).cumprod()
    years = list(range(t0.year, t1.year + 1))
    for y in years:
        a = max(pd.Timestamp(f"{y}-01-01", tz="UTC"), t0)
        b = min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), t1)
        cells = []
        for k in keys:
            st = stats(eqs[k][0], a, b)
            cells.append("-" if st is None else f"{st['tot']:+.0%} / {st['dd']:.0%}")
            pos_years[k] += int(st is not None and st["tot"] > 0)
        st_ew = stats(ew, a, b)
        print(f"{y:>6} " + " ".join(f"{c:>26}" for c in cells) + f" {('-' if st_ew is None else f'{st_ew['tot']:+.0%} / {st_ew['dd']:.0%}'):>20}")

    # ------------- per coin -------------
    print("\n=== ПО МОНЕТАМ: доля валовой прибыли ансамбля «все 9» (до комиссий) ===")
    g = eqs["ансамбль все 9"][2]
    tot = g.sum()
    print("  " + ", ".join(f"{k[:3]} {v / tot:+.0%}" for k, v in g.items()) + f"   | монет в плюсе: {(g > 0).sum()} из {N}")
    for k in keys:
        print(f"  оборот «{k}»: {eqs[k][1]:.1f} капиталов за весь период")

    # ------------- blend with the honest FGI strategy -------------
    print("\n=== ЗАЩИТА ОТ ДУБЛЯ: смесь 50/50 с честной стратегией по индексу страха ===")
    conn = get_connection()
    panel = load_panel(conn)                               # honest score by default since 2026-10-15
    conn.close()
    _, _, eq_h = simulate(*panel, {}, fee=FEE)
    fg = eq_h[eq_h.index.hour == 23]
    fg.index = fg.index.normalize()
    fg = fg.reindex(close.index).ffill()
    fg = fg / fg.dropna().iloc[0]
    r_fg, r_tr = fg.pct_change(), eqs["ансамбль все 9"][0].pct_change()
    both = pd.concat([r_fg, r_tr], axis=1, keys=["fgi", "trend"]).dropna()
    print(f"  корреляция дневных доходностей: вся история {both.corr().iloc[0, 1]:+.2f}, "
          f"train {both[both.index < SPLIT].corr().iloc[0, 1]:+.2f}, holdout {both[both.index >= SPLIT].corr().iloc[0, 1]:+.2f}")
    blend = (1 + 0.5 * both["fgi"] + 0.5 * both["trend"]).cumprod()
    fg_c, tr_c = (1 + both["fgi"]).cumprod(), (1 + both["trend"]).cumprod()
    blend_ok = []
    for per, a, b in (("train", t0, SPLIT), ("holdout", SPLIT, t1)):
        sb, sf, stt = stats(blend, a, b), stats(fg_c, a, b), stats(tr_c, a, b)
        print(f"  {per:8s} только индекс страха: {fmt(sf)}\n  {'':8s} только тренд:         {fmt(stt)}\n  {'':8s} смесь 50/50:           {fmt(sb)}")
        blend_ok.append(sb is not None and sf is not None and sb["rdd"] > sf["rdd"])

    # ------------- verdict against the predeclared bar -------------
    print("\n=== ПРОВЕРКА ПО ЗАРАНЕЕ ЗАДАННОЙ ПЛАНКЕ ===")
    best = None
    for k in keys:
        ok1 = all(results[(k, p)] is not None and results[(k, p)]["sharpe"] > 0.5 for p in ("TRAIN", "HOLDOUT"))
        ok2 = pos_years[k] >= 4
        print(f"  1-2. «{k}»: Шарп>0,5 на train и holdout — {'да' if ok1 else 'НЕТ'}; плюсовых лет {pos_years[k]} из {len(years)} — {'да' if ok2 else 'НЕТ'}")
        best = best or (ok1 and ok2)
    plateau = sum(all(results[(f'окно {n}д', p)] is not None and results[(f'окно {n}д', p)]['sharpe'] > 0 for p in ('TRAIN', 'HOLDOUT')) for n in WINDOWS)
    print(f"  3. окон с положительным Шарпом на обоих периодах: {plateau} из {len(WINDOWS)} (нужно >= 6) — {'да' if plateau >= 6 else 'НЕТ'}")
    print(f"  4. смесь лучше одной стратегии по индексу страха по ret/dd: train {'да' if blend_ok[0] else 'НЕТ'}, holdout {'да' if blend_ok[1] else 'НЕТ'}")


if __name__ == "__main__":
    main()
