"""
Audit of the one number param_sensitivity_study.py could not explain:
with NO SMA trend filter the train return goes from +235% to +3446%.

That is 15x. A single parameter change of that size is far more often an
artifact than an edge, and it also contradicts the project's own earlier
diagnosis (entries below the 30-day SMA averaged -5.2% for SOL). So before
believing it, three checks:

  1. LAG  - delay the whole score series by 24h / 72h / 168h. A genuine,
            slowly varying signal (FGI band, 72h horizon) survives a day's
            delay with its shape intact. A result that leans on knowing the
            score a few hours early (look-ahead in how events are timestamped)
            collapses. Done for both the filtered and the unfiltered runs.
  2. WHO  - split the unfiltered run's trades by whether the entry was above
            or below SMA720 and show where the P&L comes from, and how much
            the biggest trades carry. A real effect is spread out; an
            artifact is a handful of absurd trades.
  3. BENCH - the equal-weight buy-and-hold of the same seven coins over the
            same windows, so +3446% can be put next to what the market
            itself did.

Usage:
    python scripts/sma_filter_audit.py
"""
import numpy as np
import pandas as pd

from db import get_connection
from rotation_study import load_panel, SYMBOLS
from momentum_study import simulate
from regime_study import window

FEE = 0.0018
TRAIN_END = pd.Timestamp("2025-11-30", tz="UTC") + pd.Timedelta(hours=23)
HOLD_START = pd.Timestamp("2025-12-01", tz="UTC")


def period(df, name):
    return df[df.index <= TRAIN_END] if name == "train" else df[df.index >= HOLD_START]


def main():
    conn = get_connection()
    close, score, sma = load_panel(conn)
    conn.close()
    nan_sma = pd.DataFrame(np.nan, index=sma.index, columns=sma.columns)

    print("=== 1. СДВИГ СИГНАЛА: доходность / просадка / ret-dd (train | holdout) ===")
    print(f"{'сдвиг score':>12} | {'С ФИЛЬТРОМ SMA720':^42} | {'БЕЗ ФИЛЬТРА':^42}")
    for lag in (0, 24, 72, 168):
        sc = score.shift(lag)
        cells = []
        for s_df in (sma, nan_sma):
            row = []
            for name in ("train", "holdout"):
                st, _, _ = simulate(period(close, name), period(sc, name), period(s_df, name), {}, fee=FEE)
                row.append(f"{st['total_ret']:>+8.0%} {st['max_dd']:>6.0%} {st['ratio']:>6.2f}")
            cells.append("  |  ".join(row))
        print(f"{lag:>10}ч | {cells[0]:^42} | {cells[1]:^42}")

    print("\n=== 2. ОТКУДА БЕРЁТСЯ ПРИБЫЛЬ БЕЗ ФИЛЬТРА (train) ===")
    tr_close, tr_score, tr_none, tr_sma = (period(close, "train"), period(score, "train"),
                                            period(nan_sma, "train"), period(sma, "train"))
    st, trades, eq = simulate(tr_close, tr_score, tr_none, {}, fee=FEE)
    t = trades[trades["exit_reason"] != "open_at_end"].copy()
    sma_at = lambda r: tr_sma.at[r["entry_ts"], r["symbol"]] if r["entry_ts"] in tr_sma.index else np.nan
    px_at = lambda r: tr_close.at[r["entry_ts"], r["symbol"]]
    t["below"] = [px_at(r) < sma_at(r) if not np.isnan(sma_at(r)) else False for _, r in t.iterrows()]
    print(f"  сделок {len(t)}, из них входов НИЖЕ SMA720: {int(t['below'].sum())} ({t['below'].mean():.0%})")
    for label, g in (("выше SMA720", t[~t["below"]]), ("ниже SMA720", t[t["below"]])):
        if len(g):
            print(f"  входы {label:11s}: {len(g):>4d} сделок, средняя {g['ret'].mean():>+6.1%}, "
                  f"медиана {g['ret'].median():>+6.1%}, доля плюсовых {(g['ret'] > 0).mean():.0%}, "
                  f"сумма P&L {g['pnl'].sum():>+10.0f}")
    top = t.sort_values("pnl", ascending=False)
    print(f"  10 лучших сделок несут {top['pnl'].head(10).sum() / t['pnl'].sum():.0%} суммарного P&L; "
          f"лучшая сделка {top['ret'].iloc[0]:+.0%} ({top['symbol'].iloc[0]}, {top['entry_ts'].iloc[0]:%Y-%m-%d})")
    print(f"  размеры: P&L лучшей {top['pnl'].iloc[0]:+,.0f} USDT при стартовых {tr_close.shape[0] and 1000}")

    print("\n=== 3. РЫНОК ЗА ТЕ ЖЕ ПЕРИОДЫ: равновесная покупка и удержание семи монет ===")
    for name in ("train", "holdout"):
        c = period(close, name).ffill()
        first = c.apply(lambda s: s.first_valid_index())
        # equal weight from each coin's first available hour inside the window
        rets = (c.iloc[-1] / c.apply(lambda s: s.dropna().iloc[0]) - 1)
        print(f"  {name:8s} по монетам: " + ", ".join(f"{k[:3]} {v:+.0%}" for k, v in rets.items())
              + f"  | среднее {rets.mean():+.0%}")


if __name__ == "__main__":
    main()
