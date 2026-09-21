"""
The layer that decides what to TELL the user, based on the real account
(2026-09-22). Replaces the paper book as the basis for advice.

Why it is separate from testnet_trader.py. That script simulates a fixed
100 USDT fill at the public close, which makes it a clean, uninterrupted
record of what the STRATEGY would have done - valuable, and worth
keeping exactly as it is. But it is a terrible description of what the
USER holds: it had BTC/BNB/LINK/XRP open while the exchange held only
BTC and BNB, and its BNB average (716.30) sat ~11% below the real one
(806.76), so its -8% stop pointed at a price that meant nothing. Advice
has to come from reality; the track record can stay paper. So the paper
tick keeps running (now with --no-alerts) and this module owns every
message the user actually acts on.

What it decides, per symbol, from the same scoring rules as the backtest:
  holding (real qty > dust):
    - SELL when price is -8% below the REAL average cost
    - SELL when the score stops being positive
    - ADD on a confirmed dip-rebuy bounce (dip3_bounce2, validated)
    - otherwise stay quiet
  flat:
    - BUY when the score is positive, price is above SMA720, the symbol
      is not in a post-loss cooldown, and a slot is free

Position size comes from the real free USDT balance divided by the free
slots (the rule chosen 2026-09-22), so the number in the message is one
the account can actually execute. When there is no free USDT the advice
says so plainly instead of naming a size that does not exist.

Advice is REPEATED-BUT-NOT-SPAMMED: the bot cannot act for the user, so
a signal can persist for many ticks while nothing happens. An alert goes
out when the recommended action CHANGES, or once every REMIND_HOURS
while it stands - not every 15 minutes.

EXECUTION (--execute, added 2026-09-22 at the user's request: "очень
много рисков, что я не смогу среагировать например во время сна"). With
the flag AND `trading_config.enabled` both true, the same decisions are
sent to the exchange instead of to the user. The flag alone is not
enough on purpose - a stray --execute in a workflow file cannot start
trading an account whose owner switched it off in the database.

Every buy is followed immediately by a STOP-LOSS ORDER ON THE EXCHANGE,
re-placed whenever a dip-rebuy add moves the average cost and cancelled
on exit. That resting order, not this 15-minute loop, is the real answer
to "what if it crashes while I am asleep": it lives on Bybit's matching
engine and fires in milliseconds even when this code, GitHub Actions,
Supabase and Telegram are all down at once.

Limits live in `trading_config` (see db/setup_trading.sql) and are
checked by trading_guard before every order: master switch, per-order
size, total capital the bot may deploy, and a 24h realized-loss limit
that switches the bot off by itself.

Usage:
    python scripts/live_advisor.py            # advice only (default)
    python scripts/live_advisor.py --dry-run  # decide and print, never send
    python scripts/live_advisor.py --execute  # place real orders, if enabled in the DB
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

import bybit_orders
import trading_guard
from bybit_balance import get_full_balance
from db import get_connection
from real_positions import positions as real_positions, fetch_spot_executions, upsert_executions
from strategy import load_events_with_returns, generate_signals, compute_score_series
from telegram_notify import send_alert
from testnet_trader import (
    MAX_PRICE_AGE_MINUTES, MAX_POSITIONS, STOP_LOSS, TREND_FILTER_SMA,
    COOLDOWN_AFTER_LOSSES, COOLDOWN_HOURS,
    DIP_REBUY_THRESHOLD, DIP_REBUY_REBOUND, DIP_REBUY_MAX_ADDS,
    price_feed_age,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
DUST = 1e-8
MIN_ORDER_USDT = 10.0     # below this Bybit spot will not take the order anyway
REMIND_HOURS = 6          # re-send a standing recommendation at most this often


def load_advisor_state(conn, symbol: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM advisor_state WHERE symbol = %s", (symbol,))
        row = cur.fetchone()
        if row is None:
            cur.execute("INSERT INTO advisor_state (symbol) VALUES (%s) ON CONFLICT DO NOTHING", (symbol,))
            conn.commit()
            cur.execute("SELECT * FROM advisor_state WHERE symbol = %s", (symbol,))
            row = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], row))


def save_advisor_state(conn, symbol: str, **fields):
    fields["updated_at"] = datetime.now(timezone.utc)
    clause = ", ".join(f"{k} = %s" for k in fields)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE advisor_state SET {clause} WHERE symbol = %s", (*fields.values(), symbol))
    conn.commit()


def log(conn, symbol, score, holding, action, notes=""):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO testnet_run_log (symbol, score, is_long_target, in_position, action_taken, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (symbol, score, bool(score and score > 0), holding, f"advice_{action}", notes),
        )
    conn.commit()


def market_frame(conn, symbol: str, events: pd.DataFrame):
    ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts",
                        conn, params={"s": symbol})
    if len(ohlcv) < TREND_FILTER_SMA // 3:
        return None
    ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
    ohlcv["score"] = compute_score_series(generate_signals(events, symbol), ohlcv)
    ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()
    last = ohlcv.iloc[-1]
    return dict(price=float(last["close"]), score=float(last["score"]),
                sma=float(last["sma"]) if pd.notna(last["sma"]) else None)


def settle_closed_position(conn, symbol: str, state: dict, pos: dict | None):
    """A position the user has sold (real qty back to zero) still has to
    feed the post-loss cooldown, exactly as a simulated exit would. The
    realized P&L comes from the ledger, so it reflects the price the user
    actually got - not an assumed exit at the signal's close."""
    realized_now = pos["realized_pnl"] if pos else 0.0
    delta = realized_now - (state["last_realized_pnl"] or 0.0)
    losses = state["consecutive_losses"] or 0
    cooldown_until = state["cooldown_until_ts"]
    if delta < 0:
        losses += 1
        if losses >= COOLDOWN_AFTER_LOSSES:
            cooldown_until = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
            losses = 0
    elif delta > 0:
        losses = 0
    save_advisor_state(conn, symbol, consecutive_losses=losses, cooldown_until_ts=cooldown_until,
                       last_realized_pnl=realized_now, running_low=None, tranche_count=0)
    return delta


def _fmt(x, digits=6):
    return f"{x:,.{digits}g}"


def buy_text(symbol, score, price, stake, free_usdt, funding_note):
    coin = symbol.replace("USDT", "")
    if stake and stake >= MIN_ORDER_USDT:
        size_line = (f"Сколько покупать: **{stake:,.2f} USDT** "
                     f"(≈{stake / price:,.6g} {coin} по текущей цене)\n"
                     f"Это свободные средства, делённые на свободные слоты.")
    else:
        size_line = (f"Свободных USDT сейчас {free_usdt:,.2f} — на вход не хватает.\n"
                     f"Сигнал есть; если хочешь в него войти, надо освободить средства.")
    return (
        f"\U0001F4C8 СИГНАЛ НА ВХОД — {symbol}\n\n"
        f"Цена: {_fmt(price)} USDT\n"
        f"Сила сигнала: {score:.1f}\n\n"
        f"{size_line}{funding_note}\n\n"
        f"Как покупать (спот, НЕ Convert — Convert берёт ~1.4% спреда):\n"
        f"1. Assets → Transfer: USDT из Funding в Unified Trading\n"
        f"2. Trade → Spot → пара {coin}/USDT\n"
        f"3. Вкладка Buy, тип Market, сумма в USDT\n"
        f"4. Стоп-лосс после покупки: {_fmt(price * (1 + STOP_LOSS))} USDT (−8% от цены входа)"
    )


def add_text(symbol, price, stake, pos, tranche_no):
    coin = symbol.replace("USDT", "")
    projected_avg = ((pos["invested"] + stake) / (pos["qty"] + stake / price)) if stake else pos["avg_cost"]
    size_line = (f"Сколько докупать: **{stake:,.2f} USDT** (≈{stake / price:,.6g} {coin})"
                 if stake and stake >= MIN_ORDER_USDT
                 else "Свободных USDT на докупку не хватает — сигнал фиксирую, действие за тобой.")
    return (
        f"\U0001F53C ДОКУПКА НА ОТСКОКЕ (транш {tranche_no}) — {symbol}\n\n"
        f"Цена просела от твоей средней и подтверждённо отскочила.\n"
        f"Цена сейчас: {_fmt(price)} USDT\n"
        f"Твоя средняя: {_fmt(pos['avg_cost'])} USDT, держишь {pos['qty']:,.8g} {coin}\n\n"
        f"{size_line}\n\n"
        f"После докупки средняя станет ≈{_fmt(projected_avg)} USDT, "
        f"стоп −8% сместится на ≈{_fmt(projected_avg * (1 + STOP_LOSS))} USDT — "
        f"докупка на просадке отодвигает стоп дальше от цены, а не поджимает его."
    )


def sell_text(symbol, price, reason, pos):
    coin = symbol.replace("USDT", "")
    pnl = (price - pos["avg_cost"]) * pos["qty"]
    reason_ru = {
        "stop_loss": f"цена упала на −8% от твоей реальной средней ({_fmt(pos['avg_cost'])} USDT)",
        "signal_gone": "сигнал угас — score вернулся к нулю или ниже",
    }[reason]
    return (
        f"\U0001F4C9 СИГНАЛ НА ВЫХОД — {symbol}\n\n"
        f"Причина: {reason_ru}\n"
        f"Цена: {_fmt(price)} USDT\n"
        f"Позиция: {pos['qty']:,.8g} {coin}, средняя {_fmt(pos['avg_cost'])} USDT\n"
        f"Результат при продаже сейчас: {pnl:+,.2f} USDT "
        f"({(price / pos['avg_cost'] - 1):+.2%})\n\n"
        f"Как продавать (спот):\n"
        f"1. Если монета в Funding — Assets → Transfer в Unified Trading\n"
        f"2. Trade → Spot → пара {coin}/USDT\n"
        f"3. Вкладка Sell, тип Market, слайдер 100%"
    )


def _refresh_ledger(conn, coin: str) -> dict | None:
    """Pull the just-placed fill into the ledger immediately instead of
    waiting for the next tick's sync, so the stop-loss that follows is
    computed from the REAL average cost rather than a projection."""
    try:
        upsert_executions(conn, fetch_spot_executions(conn, days=1))
    except Exception as e:
        print(f"[{coin}] could not refresh the ledger right after the fill: {e}")
    return real_positions(conn, coin).get(coin)


def cancel_stop_order(conn, symbol: str, state: dict) -> None:
    """A resting stop must never outlive the position it protects: left
    behind, it would one day sell coins the bot no longer manages."""
    link_id = state.get("stop_order_link_id")
    if not link_id:
        return
    try:
        bybit_orders.cancel_order(symbol, link_id=link_id, stop_order=True, conn=conn)
        print(f"[{symbol}] cancelled the resting stop order")
    except bybit_orders.OrderRejected as e:
        # Already gone (triggered or cancelled by hand) is a fine outcome.
        print(f"[{symbol}] stop order could not be cancelled ({e.ret_msg}) - assuming it is gone")
    save_advisor_state(conn, symbol, stop_order_link_id=None, stop_trigger_price=None)


def ensure_stop_order(conn, symbol: str, pos: dict, state: dict) -> str:
    """Place (or re-place) the exchange-side stop for the whole position.

    This is the protection that actually answers 'what if it drops while
    I am asleep': it sits on Bybit's matching engine and fires in
    milliseconds even if this code, GitHub Actions, Supabase and Telegram
    are all down. The bot's own 15-minute check stays as the second line.

    Re-placed after every add, because a dip-rebuy add lowers the average
    cost and therefore moves the -8% level down with it."""
    trigger = pos["avg_cost"] * (1 + STOP_LOSS)
    if state.get("stop_order_link_id") and state.get("stop_trigger_price"):
        if abs(state["stop_trigger_price"] - trigger) / trigger < 1e-6:
            return "стоп на бирже уже стоит"
        cancel_stop_order(conn, symbol, state)

    link_id = bybit_orders.make_link_id(symbol, "stop")
    request = dict(symbol=symbol, qty=pos["qty"], trigger=trigger)
    row = trading_guard.record_order(conn, symbol, "stop_order", link_id, request,
                                     reason=f"-8% of real avg {pos['avg_cost']:.6g}")
    try:
        result = bybit_orders.place_stop_loss(symbol, pos["qty"], trigger, link_id, conn=conn)
    except bybit_orders.OrderRejected as e:
        trading_guard.finish_order(conn, row, "rejected", response=e.body, pnl_usdt=None)
        return (f"⚠️ стоп-ордер на бирже НЕ принят ({e.ret_msg}) — "
                f"позиция защищена только проверкой бота раз в 15 минут")
    trading_guard.finish_order(conn, row, "accepted", order_id=result.get("orderId"), response=result)
    save_advisor_state(conn, symbol, stop_order_link_id=link_id, stop_trigger_price=trigger)
    return f"стоп-ордер выставлен на бирже: {_fmt(trigger)} USDT"


def execute_buy(conn, symbol: str, stake: float, price: float, intent: str, pos: dict | None) -> str:
    """Buy, then immediately protect. Returns the text to send."""
    coin = symbol.replace("USDT", "")
    link_id = bybit_orders.make_link_id(symbol, intent)
    row = trading_guard.record_order(conn, symbol, intent, link_id,
                                     dict(symbol=symbol, stake_usdt=stake, price=price),
                                     reason=f"score-driven {intent}")
    try:
        result = bybit_orders.market_buy(symbol, stake, link_id, conn=conn)
    except bybit_orders.OrderRejected as e:
        trading_guard.finish_order(conn, row, "rejected", response=e.body)
        return f"⚠️ ПОКУПКА {symbol} НЕ ПРОШЛА\n\nBybit отклонил ордер: {e.ret_msg}"
    trading_guard.finish_order(conn, row, "accepted", order_id=result.get("orderId"), response=result)
    trading_guard.note_bot_buy(conn, symbol, stake)

    new_pos = _refresh_ledger(conn, coin) or pos
    state = load_advisor_state(conn, symbol)
    stop_note = ensure_stop_order(conn, symbol, new_pos, state) if new_pos else "позиция ещё не видна в реестре"
    head = "КУПЛЕНО" if intent == "buy" else "ДОКУПЛЕНО НА ОТСКОКЕ"
    avg_line = (f"Средняя цена позиции: {_fmt(new_pos['avg_cost'])} USDT\n" if new_pos else "")
    return (
        f"\U00002705 {head} — {symbol}\n\n"
        f"Потрачено: {stake:,.2f} USDT по цене ≈{_fmt(price)}\n"
        f"{avg_line}"
        f"{stop_note}\n\n"
        f"Ордер: {result.get('orderId')}"
    )


def execute_sell(conn, symbol: str, pos: dict, price: float, reason: str) -> str:
    coin = symbol.replace("USDT", "")
    state = load_advisor_state(conn, symbol)
    cancel_stop_order(conn, symbol, state)

    intent = "sell_stop" if reason == "stop_loss" else "sell_signal"
    link_id = bybit_orders.make_link_id(symbol, intent)
    pnl = (price - pos["avg_cost"]) * pos["qty"]
    row = trading_guard.record_order(conn, symbol, intent, link_id,
                                     dict(symbol=symbol, qty=pos["qty"], price=price), reason=reason)
    try:
        result = bybit_orders.market_sell(symbol, pos["qty"], link_id, conn=conn)
    except bybit_orders.OrderRejected as e:
        trading_guard.finish_order(conn, row, "rejected", response=e.body)
        return (f"⚠️ ПРОДАЖА {symbol} НЕ ПРОШЛА\n\nBybit отклонил ордер: {e.ret_msg}\n"
                f"Позиция всё ещё открыта — посмотри вручную.")
    trading_guard.finish_order(conn, row, "accepted", order_id=result.get("orderId"),
                               response=result, pnl_usdt=pnl)
    trading_guard.note_bot_exit(conn, symbol)
    _refresh_ledger(conn, coin)

    reason_ru = {"stop_loss": "стоп-лосс: −8% от реальной средней",
                 "signal_gone": "сигнал угас"}[reason]
    return (
        f"\U00002705 ПРОДАНО — {symbol}\n\n"
        f"Причина: {reason_ru}\n"
        f"Продано {pos['qty']:,.8g} {coin} по ≈{_fmt(price)} USDT\n"
        f"Результат: {pnl:+,.2f} USDT ({(price / pos['avg_cost'] - 1):+.2%})\n\n"
        f"Ордер: {result.get('orderId')}"
    )


def should_alert(state: dict, action: str) -> bool:
    """Quiet while nothing changes, but not silent forever: the user may
    simply not have seen the first message."""
    if action == state["last_action"]:
        last = state["last_alert_ts"]
        if last is not None and datetime.now(timezone.utc) - last < timedelta(hours=REMIND_HOURS):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    parser.add_argument("--dry-run", action="store_true", help="Decide and print, never send or save")
    parser.add_argument("--execute", action="store_true",
                        help="Place real orders instead of only advising. Still subject to every "
                             "guard in trading_config - with enabled=false this flag does nothing.")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    conn = get_connection()
    age, latest_ts = price_feed_age(conn, symbols)
    if age is None or age > MAX_PRICE_AGE_MINUTES:
        print(f"ABORT: stale price feed ({age} min old) - testnet_trader.py owns that alert", file=sys.stderr)
        conn.close()
        sys.exit(1)

    events = load_events_with_returns(conn)
    book = real_positions(conn)
    balance = get_full_balance()
    free_usdt = balance["unified_usdt"] if balance else 0.0
    funding_note = ""
    if balance and balance["funding_usdt"] > MIN_ORDER_USDT:
        funding_note = (f"\n⚠️ Ещё {balance['funding_usdt']:,.2f} USDT лежит в Funding — "
                        f"для спота их надо перевести в Unified.")

    held = {s for s in symbols if (book.get(s.replace("USDT", "")) or {}).get("qty", 0) > DUST}
    free_slots = max(0, (args.max_positions or len(symbols)) - len(held))
    print(f"Real book: {sorted(held) or 'empty'} | free USDT {free_usdt:,.2f} | free slots {free_slots}")

    # Autonomy is off unless BOTH the flag and the database say so. The
    # flag alone cannot trade - that way a stray --execute in a workflow
    # or a shell history cannot start trading an account whose owner
    # switched it off.
    config = trading_guard.load_config(conn)
    executing = False
    if args.execute and not args.dry_run:
        try:
            trading_guard.check_can_trade(conn, config)
            executing = True
        except trading_guard.TradingBlocked as e:
            print(f"Execution disabled: {e}")
            if "daily loss limit" in str(e):
                send_alert(f"\U0001F6D1 АВТОТОРГОВЛЯ ОСТАНОВЛЕНА\n\n{e}")
    print(f"Mode: {'EXECUTING REAL ORDERS' if executing else 'advice only'}"
          + (f" | cap {config['max_capital_usdt']:,.0f} USDT, "
             f"deployed {trading_guard.bot_deployed_usdt(conn):,.2f}" if executing else ""))

    candidates = []
    for symbol in symbols:
        coin = symbol.replace("USDT", "")
        market = market_frame(conn, symbol, events)
        if market is None:
            continue
        price, score, sma = market["price"], market["score"], market["sma"]
        state = load_advisor_state(conn, symbol)
        pos = book.get(coin)
        holding = symbol in held

        if not holding:
            if state["last_realized_pnl"] is not None and pos and \
                    pos["realized_pnl"] != state["last_realized_pnl"] and not args.dry_run:
                delta = settle_closed_position(conn, symbol, state, pos)
                print(f"[{symbol}] position closed on the exchange, realized {delta:+,.2f} USDT")
                state = load_advisor_state(conn, symbol)

            cd = state["cooldown_until_ts"]
            if cd is not None and datetime.now(timezone.utc) < cd:
                log(conn, symbol, score, False, "skip_cooldown", f"until {cd}")
                continue
            if not score > 0:
                log(conn, symbol, score, False, "flat")
                continue
            if sma is not None and price < sma:
                log(conn, symbol, score, False, "skip_trend_filter", f"price {price} < SMA {sma:.6g}")
                continue
            candidates.append(dict(symbol=symbol, score=score, price=price, state=state))
            continue

        # ---- holding: stop, exit, or dip-rebuy add ----
        # While executing, the exchange-side stop is kept in sync on every
        # tick - a position whose stop went missing (cancelled by hand, or
        # an add that moved the average) is a position with no protection
        # between ticks, which is the whole thing this is meant to prevent.
        if executing and not state.get("stop_order_link_id"):
            print(f"[{symbol}] no resting stop - placing one: {ensure_stop_order(conn, symbol, pos, state)}")
            state = load_advisor_state(conn, symbol)

        if price / pos["avg_cost"] - 1 <= STOP_LOSS:
            action = "sell_stop"
            text = (execute_sell(conn, symbol, pos, price, "stop_loss") if executing
                    else sell_text(symbol, price, "stop_loss", pos))
        elif not score > 0:
            action = "sell_signal"
            text = (execute_sell(conn, symbol, pos, price, "signal_gone") if executing
                    else sell_text(symbol, price, "signal_gone", pos))
        else:
            running_low = min(state["running_low"] or price, price)
            tranches = state["tranche_count"] or 1
            dip = running_low / pos["avg_cost"] - 1
            bounce = price / running_low - 1
            add_now = (tranches - 1 < DIP_REBUY_MAX_ADDS
                       and dip <= DIP_REBUY_THRESHOLD and bounce >= DIP_REBUY_REBOUND)
            if not args.dry_run:
                save_advisor_state(conn, symbol, running_low=running_low)
            if add_now:
                stake = free_usdt / max(free_slots, 1) if free_usdt >= MIN_ORDER_USDT else 0.0
                action = "add"
                if executing and stake >= MIN_ORDER_USDT:
                    try:
                        stake = trading_guard.check_order(conn, config, stake)
                        text = execute_buy(conn, symbol, stake, price, "add", pos)
                    except trading_guard.TradingBlocked as e:
                        action = "add_blocked"
                        text = (f"\U000026A0 ДОКУПКА {symbol} пропущена\n\n{e}\n\n"
                                f"Сигнал на докупку был, лимиты не дали его исполнить.")
                else:
                    text = add_text(symbol, price, stake, pos, tranches + 1)
                if not args.dry_run:
                    save_advisor_state(conn, symbol, tranche_count=tranches + 1, running_low=price)
            else:
                log(conn, symbol, score, True, "hold",
                    f"avg {pos['avg_cost']:.6g}, stop {pos['stop_price']:.6g}, price {price:.6g}")
                print(f"[{symbol}] holding {pos['qty']:.8g} {coin}, avg {pos['avg_cost']:,.6g}, "
                      f"stop {pos['stop_price']:,.6g}, now {price:,.6g}")
                continue

        if should_alert(state, action):
            print(f"[{symbol}] ADVICE {action}")
            if not args.dry_run:
                send_alert(text)
                save_advisor_state(conn, symbol, last_action=action,
                                   last_alert_ts=datetime.now(timezone.utc))
            else:
                print(text, "\n")
        else:
            print(f"[{symbol}] advice {action} unchanged, staying quiet")
        log(conn, symbol, score, True, action)

    # ---- entries compete for the free slots, highest score first ----
    candidates.sort(key=lambda c: -c["score"])
    for i, c in enumerate(candidates):
        symbol, state = c["symbol"], c["state"]
        if i >= free_slots:
            log(conn, symbol, c["score"], False, "skip_capacity",
                f"max_positions={args.max_positions}, held={len(held)}")
            print(f"[{symbol}] score={c['score']:.1f} - no free slot ({len(held)}/{args.max_positions})")
            continue
        stake = free_usdt / free_slots if free_usdt >= MIN_ORDER_USDT else 0.0
        action = "buy" if stake else "buy_no_funds"
        text = buy_text(symbol, c["score"], c["price"], stake, free_usdt, funding_note)
        if executing and stake >= MIN_ORDER_USDT:
            try:
                # The guard can TRIM the stake (e.g. the cap leaves room for
                # 80 of the 120 the sizing rule wanted) - a smaller trade is
                # a fine outcome, silently skipping the signal is not.
                stake = trading_guard.check_order(conn, config, stake)
                text = execute_buy(conn, symbol, stake, c["price"], "buy", None)
                action = "buy_executed"
            except trading_guard.TradingBlocked as e:
                action = "buy_blocked"
                text = (f"\U000026A0 ВХОД {symbol} пропущен\n\n{e}\n\n"
                        f"Сигнал был (score {c['score']:.1f}), лимиты не дали его исполнить.")
        if should_alert(state, action):
            print(f"[{symbol}] ADVICE {action} (stake {stake:,.2f})")
            if not args.dry_run:
                send_alert(text)
                save_advisor_state(conn, symbol, last_action=action,
                                   last_alert_ts=datetime.now(timezone.utc))
            else:
                print(text, "\n")
        else:
            print(f"[{symbol}] advice {action} unchanged, staying quiet")
        log(conn, symbol, c["score"], False, action, f"stake {stake:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
