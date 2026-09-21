"""
On-demand "куда вложить свободные деньги" advisor (2026-09-18), triggered
by a Telegram message instead of the hourly push alerts. User asked for a
PULL-based query on top of the existing PUSH-based entry/exit alerts (see
project_telegram_alerts memory) - "when I have spare money, let me ask the
bot which coin to put it in, and whether it's even worth investing in
crypto right now."

Reuses strategy.py's exact live scoring (generate_signals +
compute_score_series) - same function testnet_trader.py calls every hour -
so the answer is never a separate, driftable reimplementation of the real
signal.

Deliberately reports the STRATEGY'S signal state, never personal
financial advice ("вложить" framed as "what the validated rules say right
now", not "you should buy"). Per user's explicit choice (2026-09-18): the
portfolio-wide MAX_POSITIONS cap is honored here too - if the cap is
already full, that's reported plainly (no new entry recommended by the
same risk-managed logic), even if some symbol's own score looks strong.

IMPORTANT DISTINCTION found 2026-09-18 (user was confused why the advisor
said the cap was full when they only actually hold BTC): the cap check
here counts REAL Bybit positions (get_coin_balance per symbol > dust),
NOT `testnet_state`'s paper/simulated positions. The paper state tracks
what the fully-followed validated strategy WOULD be doing (used by the
hourly push alerts - see project_telegram_alerts) - it's a different
question from "given what I actually hold, where should NEW spare real
money go", which is what THIS on-demand feature answers. The two can
disagree (paper had 4 "open", real was BTC-only) and that's expected,
not a bug - they're deliberately answering different questions.

Triggered by a Telegram webhook (2026-09-21, replaces the original ~30min
GitHub Actions cron poll - see project_telegram_advisor_webhook memory /
README). GitHub's `schedule:` trigger turned out to be unreliable by a
large, inconsistent margin (confirmed: an hourly cron actually firing every
3-6h) - not fixable by tuning the interval. Fix: a Supabase Edge Function
(supabase/functions/telegram-webhook) receives Telegram's webhook call the
instant a message arrives, does a fast trigger-phrase pre-filter, and fires
this workflow via `workflow_dispatch` (which GitHub does NOT delay/drop the
way it does `schedule:`), passing the single Telegram update as the
`update_json` input. This script now processes exactly that one update -
no more getUpdates()/cursor polling, because registering a webhook with
Telegram disables getUpdates for the bot entirely (409 Conflict if called).

Usage:
    python scripts/telegram_command_handler.py
    (expects TELEGRAM_UPDATE_JSON in the environment - set by
    telegram_commands.yml from the workflow_dispatch input)
"""
import json
import os

import pandas as pd

from db import get_connection
from strategy import load_events_with_returns, generate_signals, compute_score_series
from telegram_notify import send_alert, CHAT_ID
from testnet_trader import MAX_POSITIONS, TREND_FILTER_SMA, NO_REAL_BALANCE_EPS
from bybit_balance import get_coin_balance
import trading_guard

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]

# Recognized triggers - a formal bot command plus the natural-language
# phrasing the user actually used when asking for this feature. Keep this
# list short and specific rather than matching on loose keywords like
# "крипта" alone, so casual chat in the group never accidentally fires it.
TRIGGERS = ["/invest", "куда вложить", "стоит ли вкладывать", "стоит ли инвестировать"]

# Autonomous-trading controls (2026-09-22). Deliberately generous on the
# STOP side and strict on the resume side: a false positive on "стоп"
# costs one missed signal, a false positive on "включить" costs real
# money. /stop is checked before every other trigger.
STOP_TRIGGERS = ["/stop", "/trade off", "стоп", "останови", "выключи торговлю"]
RESUME_TRIGGERS = ["/trade on", "/resume"]
STATUS_TRIGGERS = ["/status", "/trade status"]


def compute_signal_snapshot(conn) -> pd.DataFrame:
    """One row per symbol: current score/price/trend-filter status, using
    the exact same scoring + entry-eligibility path as testnet_trader.py's
    live tick (score>0 AND price above its trailing SMA720) - never a
    separate reimplementation. A symbol can have score>0 and still be
    `trend_ok=False` (price below its own 30d trend) - testnet_trader.py
    would log that as skip_trend_filter and never actually enter, so the
    advisor must reflect the SAME two-part condition, not score alone."""
    events = load_events_with_returns(conn)
    rows = []
    for symbol in SYMBOLS:
        ohlcv = pd.read_sql(
            "SELECT ts, close FROM ohlcv WHERE symbol = %(s)s ORDER BY ts", conn, params={"s": symbol}
        )
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if len(ohlcv) < TREND_FILTER_SMA // 3:
            continue
        signals = generate_signals(events, symbol)
        ohlcv["score"] = compute_score_series(signals, ohlcv)
        ohlcv["sma"] = ohlcv["close"].rolling(TREND_FILTER_SMA, min_periods=TREND_FILTER_SMA // 3).mean()
        latest = ohlcv.iloc[-1]
        sma = latest["sma"]
        trend_ok = not (pd.notna(sma) and latest["close"] < sma)
        rows.append(dict(symbol=symbol, score=float(latest["score"]), price=float(latest["close"]),
                          trend_ok=trend_ok))
    return pd.DataFrame(rows)


def count_real_positions() -> tuple[int, bool]:
    """How many of the 7 symbols the user REALLY holds right now (Unified
    + Funding, dust-thresholded) - NOT testnet_state's paper positions,
    see module docstring for why that distinction matters here. Second
    return value is False if any lookup failed (creds missing / API
    hiccup), meaning the count is a floor, not necessarily exact -
    callers should say so rather than presenting it as certain."""
    count = 0
    reliable = True
    for symbol in SYMBOLS:
        bal = get_coin_balance(symbol.replace("USDT", ""))
        if bal is None:
            reliable = False
            continue
        if bal >= NO_REAL_BALANCE_EPS:
            count += 1
    return count, reliable


def build_reply(conn) -> str:
    open_count, balance_reliable = count_real_positions()
    snapshot = compute_signal_snapshot(conn)
    # entry-eligible = score>0 AND above its own trend filter, matching
    # testnet_trader.py's actual entry condition exactly (a positive score
    # alone isn't enough live - see compute_signal_snapshot's docstring)
    active = snapshot[(snapshot["score"] > 0) & snapshot["trend_ok"]].sort_values("score", ascending=False)
    blocked_by_trend = snapshot[(snapshot["score"] > 0) & ~snapshot["trend_ok"]]

    # One-line bottom-line verdict FIRST (2026-09-18, per user feedback -
    # the detail-first version made them re-derive the answer themselves
    # from the signal list + cap warning instead of just reading it).
    # Same three-way logic as the detail below, just stated up front.
    if open_count >= MAX_POSITIONS:
        verdict = (
            f"\U0001F534 ВЕРДИКТ: сейчас НЕ СТОИТ открывать новую позицию ни в одной монете — "
            f"у вас уже реально открыто {open_count} из {MAX_POSITIONS} допустимых по риск-профилю стратегии."
        )
    elif active.empty:
        verdict = "\U0001F7E1 ВЕРДИКТ: сейчас нет доступного сигнала входа ни по одной из 7 монет — вкладывать некуда."
    else:
        top = active.iloc[0]
        verdict = (
            f"\U0001F7E2 ВЕРДИКТ: сейчас лучше всего подходит {top.symbol} "
            f"(score={top.score:.1f}, цена {top.price:,.6g} USDT)."
        )

    lines = ["\U0001F4CA ОТВЕТ НА ЗАПРОС\n", verdict, ""]

    lines.append("Подробности:")
    if active.empty:
        lines.append(
            "Сейчас НИ ПО ОДНОЙ из 7 монет нет реально доступного сигнала на вход "
            "(score ≤ 0, либо цена ниже своего трендового фильтра). По собственной "
            "логике стратегии сейчас не время открывать новую позицию ни в одной из "
            "отслеживаемых монет."
        )
    else:
        lines.append("Активный и доступный сигнал на вход (score > 0, выше трендового фильтра), по убыванию силы:")
        for r in active.itertuples():
            lines.append(f"  • {r.symbol}: score={r.score:.1f}, цена {r.price:,.6g} USDT")

    if not blocked_by_trend.empty:
        names = ", ".join(blocked_by_trend["symbol"])
        lines.append(
            f"\n(Есть сигнал, но заблокирован трендовым фильтром — цена ниже SMA720: {names}. "
            f"Стратегия не входит в подтверждённый нисходящий тренд даже при положительном score.)"
        )

    lines.append(f"\nРеально открыто (по вашему балансу на Bybit): {open_count} из {MAX_POSITIONS} допустимых.")
    if not balance_reliable:
        lines.append(
            "⚠️ Не удалось проверить баланс по части монет (ключ/API недоступны) — "
            "число открытых позиций ниже может быть занижено."
        )

    if open_count >= MAX_POSITIONS:
        lines.append(
            "⚠️ Лимит одновременных позиций уже заполнен (или превышен) по вашим реальным остаткам. "
            "По валидированному риск-профилю (кэп=3 — лучшее соотношение доходность/просадка "
            "и на train, и на holdout) новую позицию открывать сейчас не стоит, даже если у "
            "какой-то монеты сильный score — свободных мест нет."
        )
    elif not active.empty:
        free_slots = MAX_POSITIONS - open_count
        top = active.iloc[0]
        lines.append(
            f"Свободных мест: {free_slots}. Если ориентироваться на текущую силу сигнала "
            f"(так же, как это делает сам бот при конкуренции за место), впереди {top.symbol} "
            f"(score={top.score:.1f})."
        )

    lines.append(
        "\nℹ️ Это срез текущего состояния валидированной стратегии, а не "
        "персональная инвестиционная рекомендация — решение и реальная сделка (если делаете) "
        "всегда за вами."
    )
    return "\n".join(lines)


def trading_status_text(conn) -> str:
    """What the bot is allowed to do right now, and what it has done -
    the answer to "is it still running and what did it get up to while I
    was asleep", without opening a dashboard."""
    config = trading_guard.load_config(conn)
    deployed = trading_guard.bot_deployed_usdt(conn)
    pnl24 = trading_guard.realized_pnl_24h(conn)
    orders = trading_guard.recent_orders(conn, hours=24)

    head = "\U00002705 Автоторговля ВКЛЮЧЕНА" if config["enabled"] else "\U0001F6D1 Автоторговля ВЫКЛЮЧЕНА"
    if not config["enabled"] and config.get("paused_reason"):
        head += f"\nПричина: {config['paused_reason']}"

    lines = [
        head, "",
        f"Задействовано ботом: {deployed:,.2f} из {config['max_capital_usdt']:,.0f} USDT",
        f"Максимум на ордер: {config['max_order_usdt']:,.0f} USDT",
        f"Реализовано за 24ч: {pnl24:+,.2f} USDT "
        f"(лимит убытка {config['max_daily_loss_usdt']:,.0f})",
        "",
    ]
    if orders:
        lines.append(f"Ордеров за 24ч: {len(orders)}")
        for o in orders[:8]:
            pnl = f"  {o['pnl_usdt']:+,.2f} USDT" if o["pnl_usdt"] is not None else ""
            lines.append(f"  {o['ts']:%d.%m %H:%M} {o['symbol']} {o['intent']} — {o['status']}{pnl}")
    else:
        lines.append("Ордеров за последние 24 часа не было.")
    lines += ["", "Команды: /stop — остановить, /trade on — включить, /status — этот отчёт"]
    return "\n".join(lines)


def main():
    raw = os.environ.get("TELEGRAM_UPDATE_JSON", "").strip()
    if not raw:
        print("[telegram_command_handler] no update_json input, nothing to do")
        return

    update = json.loads(raw)
    message = update.get("message") or update.get("channel_post")
    if not message:
        print("[telegram_command_handler] update has no message, ignoring")
        return
    if str(message.get("chat", {}).get("id")) != str(CHAT_ID):
        print("[telegram_command_handler] update from a different chat, ignoring")
        return
    if message.get("from", {}).get("is_bot"):
        print("[telegram_command_handler] update from a bot, ignoring")
        return

    text = (message.get("text") or "").strip().lower()

    # The kill switch comes first, and is matched before anything else:
    # when the user types "стоп" they need the bot to stop, not to think
    # about what else the message might have meant. It writes to
    # trading_config, which every tick reads before placing anything, so
    # the next tick (<=15 min) trades nothing - no deploy, no git push.
    if any(t in text for t in STOP_TRIGGERS):
        conn = get_connection()
        trading_guard.pause(conn, f"stopped from Telegram by {message.get('from', {}).get('username', 'user')}")
        conn.close()
        send_alert("\U0001F6D1 АВТОТОРГОВЛЯ ОСТАНОВЛЕНА\n\n"
                   "Новых ордеров бот больше не отправляет. Уже открытые позиции остаются как есть, "
                   "и выставленные на бирже стоп-ордера продолжают их защищать.\n\n"
                   "Включить обратно: /trade on")
        print("[telegram_command_handler] autotrading paused from Telegram")
        return

    if any(t in text for t in RESUME_TRIGGERS):
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("UPDATE trading_config SET enabled = true, paused_reason = NULL, "
                        "paused_at = NULL, updated_at = now() WHERE id = 1")
        conn.commit()
        config = trading_guard.load_config(conn)
        conn.close()
        send_alert(f"\U00002705 АВТОТОРГОВЛЯ ВКЛЮЧЕНА\n\n"
                   f"Потолок капитала: {config['max_capital_usdt']:,.0f} USDT\n"
                   f"Максимум на один ордер: {config['max_order_usdt']:,.0f} USDT\n"
                   f"Дневной лимит убытка: {config['max_daily_loss_usdt']:,.0f} USDT\n\n"
                   f"Остановить в любой момент: /stop")
        print("[telegram_command_handler] autotrading enabled from Telegram")
        return

    if any(t in text for t in STATUS_TRIGGERS):
        conn = get_connection()
        send_alert(trading_status_text(conn))
        conn.close()
        print("[telegram_command_handler] replied with trading status")
        return

    if not any(trigger in text for trigger in TRIGGERS):
        print("[telegram_command_handler] update did not match any trigger")
        return

    conn = get_connection()
    reply = build_reply(conn)
    conn.close()
    send_alert(reply)
    print("[telegram_command_handler] replied to an /invest-style query")


if __name__ == "__main__":
    main()
