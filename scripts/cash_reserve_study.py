"""
Should part of the cash be HELD BACK so dip-rebuy adds stay affordable?
(2026-09-30)

This closes a gap in my own earlier reasoning. On 2026-09-22 I argued for
keeping max_order at half the ceiling because the leftover "is the reserve
that funds the validated dip-rebuy add". That sounded right and was never
measured - and worse, the numbers the cap decision rested on
(slot_cap_study.py: cap=3 → +322% train / +38.2% holdout) come from an
engine that deploys `cash / free_slots`, i.e. keeps NO reserve at all. So
full deployment was what had been validated, and the reserve argument was
a plausible story on top of it.

The live question it decides: after the bot's next full exit it will have
~1367 USDT in cash and three empty slots. Deploy all of it (~456 per
position, nothing left for an add), or hold some back?

Everything else is the validated engine untouched - same rules, SMA720
filter, -8% stop on average cost, dip3_bounce2, cap 3, cooldown, fees,
hourly mark-to-market.

Usage:
    python scripts/cash_reserve_study.py --end-date 2025-11-30    # train
    python scripts/cash_reserve_study.py --start-date 2025-12-01  # holdout
"""
import argparse
import os

import pandas as pd

from db import get_connection
from rotation_study import load_panel
from slot_cap_study import simulate

RESERVES = [0.0, 0.25, 0.40]
CAP = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    close, score, sma = load_panel(conn, args.start_date, args.end_date)
    conn.close()
    period = f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}"
    print(f"Период: {period} ({len(close)} часов), кэп {CAP}\n")

    print(f"{'резерв':>10} {'сделок':>7} {'с докупкой':>11} {'доходность':>11} "
          f"{'max_dd':>8} {'ret/dd':>7} {'ср. размер входа':>17}")
    rows = []
    for reserve in RESERVES:
        stats, _, _ = simulate(close, score, sma, CAP, reserve_frac=reserve)
        rows.append(dict(period=period, reserve=reserve, **stats))
        print(f"{reserve:>9.0%} {stats['n_trades']:>7d} {stats['multi_tranche']:>11d} "
              f"{stats['total_ret']:>11.1%} {stats['max_dd']:>8.1%} {stats['ratio']:>7.2f} "
              f"{stats['avg_stake']:>17,.0f}")

    pd.DataFrame(rows).to_csv("cash_reserve_summary.csv", mode="a", index=False,
                              header=not os.path.exists("cash_reserve_summary.csv"))
    print("\nДописано в cash_reserve_summary.csv")


if __name__ == "__main__":
    main()
