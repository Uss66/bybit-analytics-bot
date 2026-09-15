"""
Step 1 of the LLM-confirmation-layer pilot (2026-09-14, see
project_llm_confirmation_layer memory): sample ~40 real historical trades
from the CURRENTLY ACTIVE strategy (fgi_greed/DXY/DVOL) in the train
period, and for each, package the "frozen" context (news events in the
preceding 48h, NO price/outcome data) that a blinded LLM judge will see.

Outcomes (ret) are saved SEPARATELY (llm_pilot_outcomes.csv) and never
shown to the blinded judge - joined back only after verdicts are collected,
to test whether judgment correlates with actual result without the judge
ever seeing it.

Usage:
    python scripts/llm_confirmation_pilot_sample.py
"""
import json
import random

import pandas as pd

from db import get_connection
from strategy import (
    load_events_with_returns, generate_signals, backtest, SPOT_TAKER_FEE,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
N_SAMPLE = 40
CONTEXT_WINDOW_HOURS = 48
RANDOM_SEED = 42


def main():
    conn = get_connection()
    events_raw = pd.read_sql(
        "SELECT id, ts, source, category, sentiment, title, content, raw "
        "FROM events WHERE category IS NOT NULL ORDER BY ts", conn)
    events_raw["ts"] = pd.to_datetime(events_raw["ts"], utc=True)
    events_for_signals = load_events_with_returns(conn)

    all_trades = []
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol=%(s)s ORDER BY ts",
                             conn, params={"s": symbol})
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        ohlcv = ohlcv[ohlcv["ts"] < TRAIN_END]
        signals = generate_signals(events_for_signals, symbol)
        trades_df, stats = backtest(symbol, ohlcv, signals, capital=10000, fee=SPOT_TAKER_FEE,
                                     stop_loss=-0.08, trend_filter_sma=720, cooldown_after_losses=3)
        if trades_df.empty:
            continue
        trades_df["symbol"] = symbol
        all_trades.append(trades_df[["symbol", "entry_ts", "entry_price", "exit_ts", "exit_price", "ret"]])
    conn.close()

    all_trades_df = pd.concat(all_trades, ignore_index=True)
    print(f"Total train-period trades across all 7 symbols: {len(all_trades_df)}")

    random.seed(RANDOM_SEED)
    sample_idx = random.sample(range(len(all_trades_df)), min(N_SAMPLE, len(all_trades_df)))
    sample_df = all_trades_df.iloc[sample_idx].reset_index(drop=True)
    sample_df["pilot_id"] = [f"T{i+1:03d}" for i in range(len(sample_df))]

    # coin-matching, same convention as event_study.py: whale-alert-derived
    # categories only count if raw.coin matches this trade's symbol
    coin_by_symbol = {s: s.replace("USDT", "") for s in SYMBOLS}
    COIN_SPECIFIC_CATEGORIES = {"exchange_inflow", "exchange_outflow", "wallet_to_wallet", "stablecoin_mint"}

    blinded_packets = []
    outcomes = []
    for row in sample_df.itertuples():
        window_start = row.entry_ts - pd.Timedelta(hours=CONTEXT_WINDOW_HOURS)
        coin = coin_by_symbol[row.symbol]
        mask = (events_raw["ts"] >= window_start) & (events_raw["ts"] < row.entry_ts)
        ctx = events_raw[mask].copy()

        def keep(r):
            if r["category"] in COIN_SPECIFIC_CATEGORIES:
                c = (r["raw"] or {}).get("coin") if isinstance(r["raw"], dict) else None
                return c == coin
            return True
        ctx = ctx[ctx.apply(keep, axis=1)]

        def clean(v, n):
            return "" if pd.isna(v) else str(v)[:n]

        news_items = [
            dict(ts=str(r.ts), source=r.source, category=r.category,
                 title=clean(r.title, 300), content=clean(r.content, 500))
            for r in ctx.itertuples()
        ]

        blinded_packets.append(dict(
            pilot_id=row.pilot_id,
            symbol=row.symbol,
            entry_ts=str(row.entry_ts),
            news_last_48h=news_items,
        ))
        outcomes.append(dict(pilot_id=row.pilot_id, symbol=row.symbol, entry_ts=str(row.entry_ts),
                              ret=row.ret, profitable=row.ret > 0))

    with open("llm_pilot_blinded_context.json", "w", encoding="utf-8") as f:
        json.dump(blinded_packets, f, indent=2, ensure_ascii=False)
    pd.DataFrame(outcomes).to_csv("llm_pilot_outcomes.csv", index=False)

    n_with_news = sum(1 for p in blinded_packets if p["news_last_48h"])
    print(f"Sampled {len(sample_df)} trades. {n_with_news} have >=1 news item in the preceding "
          f"{CONTEXT_WINDOW_HOURS}h window, {len(sample_df) - n_with_news} have none.")
    print("Wrote llm_pilot_blinded_context.json (NO outcome data) and llm_pilot_outcomes.csv (kept separate).")


if __name__ == "__main__":
    main()
