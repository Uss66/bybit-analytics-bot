"""
Hourly-scheduled paper-trading executor. Reuses strategy.py's exact
scoring logic (generate_signals + compute_score_series) so the live
decision rule is identical to what was backtested - no separate "live"
reimplementation to drift out of sync.

Two execution modes (--mode):
  "simulate" (default) - fills are simulated at the current public OHLCV
    close price, entirely locally, no exchange account or API key needed.
    Added 2026-09-13 after Bybit's testnet consistently returned error
    10024 ("regulatory restrictions") for API key creation with no
    resolution from support - rather than keep blocking on Bybit account
    access, this mode uses ONLY the public market-data API (no auth
    needed, and the whole project already depends on it for OHLCV) to get
    real-time paper trading running. Trade-off: no real order-book
    matching/slippage is modeled (acceptable at $100/trade on liquid
    pairs, and fees are still applied) - the goal is observing the
    strategy's real-time behavior, not exchange execution mechanics.
  "testnet" - real (fake-money) orders via Bybit's testnet API, using
    bybit_client.py. Kept for if/when Bybit account access is resolved.

State machine per symbol, mirroring strategy.py's backtest() exactly but
online (one hourly tick at a time, state persisted in `testnet_state`
between runs) instead of vectorized over a whole historical timeline:

  not in position:
    - skip if still in a cooldown window (after N consecutive losses)
    - skip if trend_filter_sma set and price is below its trailing SMA
    - enter (market buy) if score > 0

  in position:
    - exit (market sell) if price has dropped stop_loss% below entry
    - exit (market sell) if score is no longer > 0 (natural exit)
    - otherwise hold

Portfolio-wide exposure cap (added 2026-09-13, MAX_POSITIONS=3 by default):
since fgi_greed/DVOL are market-wide signals, several symbols often want to
enter in the very same hourly tick (the first live run opened 6 of 7 at
once). Exits/holds/skips run immediately per symbol; entry candidates are
collected across ALL symbols first, then arbitrated as one batch - if more
symbols want in than there's room for, the highest score wins the
contested slot(s), and the rest are logged as "skip_capacity" (not lost -
they get re-evaluated fresh next tick, same as any other skip). Validated
via strategy.py's simulate_portfolio() on the full extended history: best
risk-adjusted (return/drawdown) ratio of any cap on both train and
holdout, see README.md "Портфельный лимит экспозиции".

Every tick writes a row to `testnet_run_log` regardless of whether a trade
happened, so "why didn't it trade at time X" is always answerable later.

KNOWN LIMITATION (first iteration, watch closely rather than trust
blindly): if `wait_for_fill` times out waiting for an order to reach
Filled (order placed but exchange lag or a network hiccup on our side
means we never see the confirmation), the exception aborts BEFORE
`save_state`/`record_trade` run, so our local state won't reflect an order
that may have actually filled on the exchange. The next hourly tick would
then see the old state (e.g. still "flat") and could act on stale
information (e.g. try to buy again, doubling the position). Order IDs are
always printed/logged on the error path specifically so this is
reconcilable by hand from the testnet UI - check open orders/positions
there if a tick logs an "error" action. Not built out further because this
is testnet money and the first goal is observing real behavior, not
building a fully self-healing executor before we've seen it run once.

Usage (intended to be run hourly via Task Scheduler):
    python scripts/testnet_trader.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT
"""
import argparse
import math
import sys
from datetime import datetime, timezone

import pandas as pd

from db import get_connection
from strategy import (
    RULES, AVOID_RULES, SPOT_TAKER_FEE,
    load_events_with_returns, generate_signals, compute_score_series,
)
from telegram_notify import send_alert
from bybit_balance import get_full_balance

STOP_LOSS = -0.08
TREND_FILTER_SMA = 720
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_HOURS = 168
USDT_PER_TRADE = 100.0  # fixed paper-money position size per entry
MAX_POSITIONS = 3  # portfolio-wide cap on simultaneous open positions, added 2026-09-13 -
# validated via strategy.py's simulate_portfolio() on the full extended history: best
# risk-adjusted (return/drawdown) ratio of any cap on BOTH train (4.94x) and holdout
# (6.25x), see README.md "Портфельный лимит экспозиции". Ties at the same hourly tick
# (routine here - fgi_greed/DVOL are market-wide) are broken by highest score first.


def load_state(conn, symbol: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM testnet_state WHERE symbol = %s", (symbol,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO testnet_state (symbol) VALUES (%s) "
                "ON CONFLICT (symbol) DO NOTHING RETURNING *", (symbol,)
            )
            conn.commit()
            cur.execute("SELECT * FROM testnet_state WHERE symbol = %s", (symbol,))
            row = cur.fetchone()
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def save_state(conn, symbol: str, **fields):
    fields["updated_at"] = datetime.now(timezone.utc)
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE testnet_state SET {set_clause} WHERE symbol = %s", (*fields.values(), symbol))
    conn.commit()


def floor_to_precision(value: float, precision_str: str) -> float:
    """Round DOWN to the instrument's basePrecision (e.g. '0.000001') - never
    round up, since that could try to sell more than we actually hold."""
    decimals = len(precision_str.split(".")[1]) if "." in precision_str else 0
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def log_tick(conn, symbol, score, is_long_target, in_position, action, notes=""):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO testnet_run_log (symbol, score, is_long_target, in_position, action_taken, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (symbol, score, is_long_target, in_position, action, notes),
        )
    conn.commit()


def record_trade(conn, symbol, entry_ts, exit_ts, entry_price, exit_price, qty,
                  gross_ret, net_ret, exit_reason, entry_order_id, exit_order_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO testnet_trades
                (symbol, entry_ts, exit_ts, entry_price, exit_price, qty, gross_ret, net_ret,
                 exit_reason, entry_order_id, exit_order_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (symbol, entry_ts, exit_ts, entry_price, exit_price, qty, gross_ret, net_ret,
             exit_reason, entry_order_id, exit_order_id),
        )
    conn.commit()


def _balance_line() -> str:
    """Live real-mainnet balance for alert messages (read-only lookup, see
    bybit_balance.py - structurally incapable of placing an order
    regardless of the API key's own permissions). Shows BOTH wallets
    Bybit keeps separate, not just Unified - found 2026-09-14 that a user
    can have real funds sitting entirely in Funding (not tradeable without
    an internal transfer first) while Unified reads 0, which looks like
    "no money" if only one number is shown. Returns "" when
    BYBIT_API_KEY/SECRET aren't configured, so alerting degrades
    gracefully rather than ever blocking on this."""
    full = get_full_balance()
    if full is None:
        return ""
    line = f"\nБаланс для торговли (Unified): {full['unified_usdt']:,.2f} USDT"
    if full["funding_usdt"] > 0 or full["funding_btc"] > 0:
        line += (f"\n⚠️ В Funding-кошельке (НЕ доступно для торговли без перевода): "
                 f"{full['funding_usdt']:,.4f} USDT, {full['funding_btc']:.8f} BTC")
    return line


def _entry_alert_text(symbol: str, score: float, fill_price: float, fill_qty: float, mode: str) -> str:
    """Builds a clear, step-by-step Russian-language alert (2026-09-14, per
    user request "пусть бот дает более четкие инструкции по заключению
    сделки на русском языке") - concrete numbers a user can act on
    immediately, not just a terse English log line. Still only a
    NOTIFICATION - the user places any real order themselves, this project
    never executes real trades autonomously."""
    base_coin = symbol.replace("USDT", "")
    stop_price = fill_price * (1 + STOP_LOSS)
    mode_note = "бумажная сделка (симулятор)" if mode == "simulate" else f"реальный ордер размещён в режиме {mode}"
    return (
        f"\U0001F4C8 СИГНАЛ НА ВХОД — {symbol}\n\n"
        f"Цена: {fill_price:,.6g} USDT\n"
        f"Сила сигнала (score): {score:.1f}\n"
        f"Статус: {mode_note}{_balance_line()}\n\n"
        f"Что делать, если торгуешь реальными деньгами:\n"
        f"1. Открой Bybit → Spot → {symbol}\n"
        f"2. Рыночный ордер BUY на ~{USDT_PER_TRADE:.0f} USDT "
        f"(≈{fill_qty:.6g} {base_coin} по текущей цене)\n"
        f"3. Ориентировочный стоп-лосс: {stop_price:,.6g} USDT (−8% от входа) — "
        f"выставь сам, бот не делает это за тебя на реальном счёте\n\n"
        f"⚠️ Если уже держишь {symbol} по прошлому сигналу — не дублируй позицию."
    )


def _exit_alert_text(symbol: str, exit_price: float, exit_reason: str, net_ret: float, mode: str) -> str:
    reason_ru = {
        "stop_loss": "сработал стоп-лосс (−8% от входа)",
        "natural": "сигнал угас (score вернулся к нулю или ниже)",
    }.get(exit_reason, exit_reason)
    result_kind = "бумажный результат" if mode == "simulate" else "результат реального ордера"
    return (
        f"\U0001F4C9 СИГНАЛ НА ВЫХОД — {symbol}\n\n"
        f"Причина: {reason_ru}\n"
        f"Цена выхода: {exit_price:,.6g} USDT\n"
        f"{result_kind}: {net_ret:+.2%}{_balance_line()}\n\n"
        f"Что делать, если торгуешь реальными деньгами:\n"
        f"1. Открой Bybit → Spot → {symbol}\n"
        f"2. Рыночный ордер SELL на весь объём {symbol.replace('USDT', '')}, "
        f"купленный по этому сигналу"
    )


def enter_position(conn, client, mode: str, symbol: str, score: float, is_long_target: bool, price: float, dry_run: bool):
    """Executes an entry that has already cleared cooldown/trend-filter/
    capacity checks. Split out from process_symbol() so main() can collect
    ALL symbols' entry candidates first and apply the portfolio-wide
    MAX_POSITIONS cap (highest score wins a contested slot) before any
    order is actually placed."""
    if dry_run:
        print(f"[{symbol}] DRY RUN would BUY ~{USDT_PER_TRADE} USDT at ~{price}")
        log_tick(conn, symbol, score, is_long_target, False, "dry_run_would_enter")
        return

    if mode == "simulate":
        # fill at the current public close price - no exchange account
        # involved, so no fill-confirmation race to handle
        fill_price = price
        fill_qty = USDT_PER_TRADE / price
        order_id = f"SIM-{int(datetime.now(timezone.utc).timestamp())}"
    else:
        order = client.market_buy_quote(symbol, USDT_PER_TRADE)
        order_id = order.get("orderId")
        filled = client.wait_for_fill(symbol, order_id)
        fill_price = float(filled["avgPrice"])
        fill_qty = float(filled["cumExecQty"])

    save_state(conn, symbol, in_position=True, entry_ts=datetime.now(timezone.utc),
               entry_price=fill_price, entry_qty=fill_qty, entry_order_id=order_id)
    log_tick(conn, symbol, score, is_long_target, True, "enter", f"order {order_id} @ {fill_price}")
    print(f"[{symbol}] ENTERED at {fill_price}, qty {fill_qty}, order {order_id}")
    send_alert(_entry_alert_text(symbol, score, fill_price, fill_qty, mode))


def process_symbol(conn, client, mode: str, symbol: str, events: pd.DataFrame, dry_run: bool):
    """Handles exit/hold/skip decisions immediately (these don't need
    cross-symbol coordination). For a symbol that's flat and wants to
    enter, returns an entry-candidate dict instead of executing right away
    - main() collects these across all symbols and applies the portfolio-
    wide capacity cap before calling enter_position(). Returns None when
    there's nothing left for main() to decide on this tick."""
    ohlcv = pd.read_sql(
        "SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol}
    )
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    if len(ohlcv) < TREND_FILTER_SMA // 3:
        print(f"[{symbol}] not enough OHLCV history yet, skipping")
        return None

    signals = generate_signals(events, symbol)
    ohlcv["score"] = compute_score_series(signals, ohlcv)
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()

    latest = ohlcv.iloc[-1]
    score, price = float(latest["score"]), float(latest["close"])
    sma = float(latest["sma"]) if pd.notna(latest["sma"]) else None
    is_long_target = bool(score > 0)

    state = load_state(conn, symbol)
    in_position = state["in_position"]

    if not in_position:
        cooldown_until = state["cooldown_until_ts"]
        now = datetime.now(timezone.utc)
        if cooldown_until is not None and now < cooldown_until:
            log_tick(conn, symbol, score, is_long_target, False, "skip_cooldown",
                      f"cooldown until {cooldown_until}")
            print(f"[{symbol}] score={score:.1f} - in cooldown until {cooldown_until}, skipping")
            return None
        if not is_long_target:
            log_tick(conn, symbol, score, is_long_target, False, "hold_flat")
            print(f"[{symbol}] score={score:.1f} - flat, no entry signal")
            return None
        if TREND_FILTER_SMA and sma is not None and price < sma:
            log_tick(conn, symbol, score, is_long_target, False, "skip_trend_filter",
                      f"price {price} below SMA{TREND_FILTER_SMA} {sma:.4f}")
            print(f"[{symbol}] score={score:.1f} - signal active but price below trend SMA, skipping entry")
            return None

        # wants to ENTER - hand back to main() for portfolio-wide capacity arbitration
        return dict(symbol=symbol, score=score, is_long_target=is_long_target, price=price)

    # in position - check exit conditions
    entry_price = state["entry_price"]
    hit_stop = (price / entry_price - 1) <= STOP_LOSS
    exit_reason = "stop_loss" if hit_stop else ("natural" if not is_long_target else None)

    if exit_reason is None:
        log_tick(conn, symbol, score, is_long_target, True, "hold_position")
        print(f"[{symbol}] score={score:.1f} - holding position (entry {entry_price}, now {price})")
        return

    if dry_run:
        print(f"[{symbol}] DRY RUN would SELL (reason={exit_reason}) at ~{price}")
        log_tick(conn, symbol, score, is_long_target, True, f"dry_run_would_exit_{exit_reason}")
        return

    if mode == "simulate":
        qty = state["entry_qty"]
        exit_price = price
        order_id = f"SIM-{int(datetime.now(timezone.utc).timestamp())}"
    else:
        instrument = client.get_instrument_info(symbol)
        base_precision = instrument["lotSizeFilter"]["basePrecision"]
        base_coin = instrument["baseCoin"]
        # never trust the recorded entry_qty alone for the sell size - fees can be
        # deducted from the received base asset on the buy side, so the actual
        # held balance can be a hair less than what we recorded. Selling more
        # than we hold fails the order and leaves the position stuck open.
        actual_balance = client.get_coin_balance(base_coin)
        qty = floor_to_precision(min(state["entry_qty"], actual_balance), base_precision)
        if qty <= 0:
            raise RuntimeError(f"No sellable {base_coin} balance (recorded qty {state['entry_qty']}, "
                                f"actual balance {actual_balance}) - can't close position, needs manual check")
        qty_str = f"{qty:.{len(base_precision.split('.')[1]) if '.' in base_precision else 0}f}"

        order = client.market_sell_base(symbol, qty_str)
        order_id = order.get("orderId")
        filled = client.wait_for_fill(symbol, order_id)
        exit_price = float(filled["avgPrice"])

    gross_ret = (exit_price - entry_price) / entry_price
    net_ret = (exit_price * (1 - SPOT_TAKER_FEE)) / (entry_price * (1 + SPOT_TAKER_FEE)) - 1
    entry_ts = state["entry_ts"]
    exit_ts = datetime.now(timezone.utc)

    record_trade(conn, symbol, entry_ts, exit_ts, entry_price, exit_price, qty,
                 gross_ret, net_ret, exit_reason, state["entry_order_id"], order_id)

    consecutive_losses = state["consecutive_losses"]
    cooldown_until_ts = None
    if net_ret < 0:
        consecutive_losses += 1
        if consecutive_losses >= COOLDOWN_AFTER_LOSSES:
            cooldown_until_ts = exit_ts + pd.Timedelta(hours=COOLDOWN_HOURS)
            consecutive_losses = 0
    else:
        consecutive_losses = 0

    save_state(conn, symbol, in_position=False, entry_ts=None, entry_price=None, entry_qty=None,
               entry_order_id=None, consecutive_losses=consecutive_losses, cooldown_until_ts=cooldown_until_ts)
    log_tick(conn, symbol, score, is_long_target, False, f"exit_{exit_reason}",
             f"order {order_id} @ {exit_price}, net_ret {net_ret:.4f}")
    print(f"[{symbol}] EXITED ({exit_reason}) at {exit_price}, net_ret {net_ret:.2%}, order {order_id}")
    send_alert(_exit_alert_text(symbol, exit_price, exit_reason, net_ret, mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str,
                         default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT")
    parser.add_argument("--mode", choices=["simulate", "testnet"], default="simulate",
                         help="'simulate' fills locally at the public close price, no exchange account "
                              "needed (default). 'testnet' places real fake-money orders via Bybit's "
                              "testnet API (needs BYBIT_TESTNET_API_KEY/SECRET).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute signals and log what WOULD happen, but never record a trade.")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS,
                         help=f"Portfolio-wide cap on simultaneous open positions across --symbols "
                              f"(default {MAX_POSITIONS}, validated via strategy.py's simulate_portfolio() "
                              f"- see README.md). Pass 0 to disable.")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    max_positions = args.max_positions or None

    conn = get_connection()
    events = load_events_with_returns(conn)

    client = None
    if args.mode == "testnet" and not args.dry_run:
        from bybit_client import BybitTestnetClient
        try:
            client = BybitTestnetClient()
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    entry_candidates = []
    for symbol in symbols:
        try:
            result = process_symbol(conn, client, args.mode, symbol, events, args.dry_run)
            if result is not None:
                entry_candidates.append(result)
        except Exception as e:
            print(f"[{symbol}] ERROR: {e}", file=sys.stderr)
            log_tick(conn, symbol, None, None, None, "error", str(e))

    # Portfolio-wide capacity arbitration: exits above have already run and
    # freed any slots they occupied, so testnet_state now reflects this
    # tick's true open-position count. When more symbols want in than
    # there's room for (routine - fgi_greed/DVOL are market-wide and often
    # fire on several symbols the same hour), the highest score wins the
    # contested slot(s) - see strategy.py's simulate_portfolio() docstring
    # for why this tie-break was chosen over the original study's
    # incidental (and non-representative) processing-order tie-break.
    if entry_candidates:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM testnet_state WHERE symbol = ANY(%s) AND in_position",
                        (symbols,))
            currently_open = cur.fetchone()[0]
        remaining_slots = max(0, max_positions - currently_open) if max_positions is not None else len(entry_candidates)
        entry_candidates.sort(key=lambda c: c["score"], reverse=True)
        admitted, rejected = entry_candidates[:remaining_slots], entry_candidates[remaining_slots:]

        for c in admitted:
            enter_position(conn, client, args.mode, c["symbol"], c["score"], c["is_long_target"], c["price"], args.dry_run)
        for c in rejected:
            log_tick(conn, c["symbol"], c["score"], c["is_long_target"], False, "skip_capacity",
                      f"max_positions={max_positions}, currently_open={currently_open}")
            print(f"[{c['symbol']}] score={c['score']:.1f} - signal active but portfolio at capacity "
                  f"({currently_open}/{max_positions}), skipping entry")

    conn.close()


if __name__ == "__main__":
    main()
