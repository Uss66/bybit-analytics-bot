"""
"Что бот делал, пока меня не было" - один отчёт по автономной торговле
(2026-09-22, написан в день запуска специально для разбора через пару
дней).

Собирает в одном месте то, что иначе пришлось бы выковыривать из четырёх
источников: журнал ордеров, реестр реальных сделок, живые стоп-ордера на
бирже и лог тиков. Отдельно сверяет две вещи, которые важнее всех
остальных цифр:

  * есть ли ОТКРЫТАЯ ПОЗИЦИЯ БЕЗ СТОПА на бирже - единственное
    состояние, в котором ночное падение действительно опасно;
  * расходятся ли реальные сделки с тем, что в те же часы делал бумажный
    бот (testnet_trader.py). Бумажный трек - это стратегия в чистом
    виде; расхождение означает, что реальность вмешалась (проскальзывание,
    нехватка средств, лимиты, отклонённый ордер), и именно там надо
    искать разницу между "стратегия работает" и "бот работает".

Usage:
    python scripts/trading_review.py           # с момента запуска автономии
    python scripts/trading_review.py --days 7
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bybit_orders
import trading_guard
from bybit_balance import get_full_balance
from db import get_connection
from real_positions import positions as real_positions, latest_prices, reconcile

# Момент включения автономной торговли - точка отсчёта по умолчанию.
AUTONOMY_START = datetime(2026, 9, 21, 21, 56, tzinfo=timezone.utc)

# Снимок на момент запуска автономии — чтобы отчёт показывал ИЗМЕНЕНИЕ,
# а не только абсолютные числа. Лежит в локальном файле вне git
# (scripts/.baseline.json): репозиторий публичный, а состав и размер
# реального счёта публиковать незачем. Без файла отчёт просто не
# показывает строки сравнения — всё остальное работает как обычно.
BASELINE_FILE = Path(__file__).with_name(".baseline.json")


def load_baseline() -> dict:
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


BASELINE = load_baseline()
BASELINE_EQUITY_USDT = BASELINE.get("equity_usdt")
BASELINE_POSITIONS = BASELINE.get("positions", {})


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=None,
                        help="Окно отчёта в днях (по умолчанию - с момента запуска автономии)")
    args = parser.parse_args()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)) if args.days else AUTONOMY_START
    hours = (datetime.now(timezone.utc) - since).total_seconds() / 3600

    conn = get_connection()

    section(f"РЕЖИМ  (окно: {since:%d.%m %H:%M} UTC .. сейчас, {hours:.1f} ч)")
    config = trading_guard.load_config(conn)
    print(f"  автоторговля: {'ВКЛЮЧЕНА' if config['enabled'] else 'ВЫКЛЮЧЕНА'}"
          + (f"  причина остановки: {config['paused_reason']}" if config.get("paused_reason") else ""))
    print(f"  потолок {config['max_capital_usdt']:,.0f} USDT | "
          f"макс. ордер {config['max_order_usdt']:,.0f} | "
          f"дневной лимит убытка {config['max_daily_loss_usdt']:,.0f}")
    print(f"  задействовано ботом: {trading_guard.bot_deployed_usdt(conn):,.2f} USDT")

    section("ОРДЕРА БОТА")
    with conn.cursor() as cur:
        cur.execute("SELECT ts, symbol, intent, status, order_id, pnl_usdt, reason FROM bot_orders "
                    "WHERE ts >= %s ORDER BY ts", (since,))
        orders = cur.fetchall()
    if not orders:
        print("  ни одного ордера за период")
    for ts, symbol, intent, status, order_id, pnl, reason in orders:
        pnl_s = f"  P&L {pnl:+,.2f}" if pnl is not None else ""
        flag = "" if status == "accepted" else f"  <<< {status.upper()}"
        print(f"  {ts:%d.%m %H:%M} {symbol:9s} {intent:12s} {status:9s}{pnl_s}{flag}")
        if status != "accepted" and reason:
            print(f"            причина: {reason}")
    rejected = [o for o in orders if o[3] != "accepted"]
    if rejected:
        print(f"\n  ВНИМАНИЕ: {len(rejected)} ордер(ов) не прошли - разобраться, почему")

    section("ПОЗИЦИИ И ЗАЩИТА")
    book = {c: p for c, p in real_positions(conn).items() if p["qty"] > 0}
    prices = latest_prices(conn, book)
    try:
        resting = {o["symbol"]: o for o in bybit_orders.open_orders(stop_orders=True, conn=conn)}
    except Exception as e:
        resting = {}
        print(f"  не удалось прочитать стоп-ордера с биржи: {e}")

    unprotected = []
    for coin, pos in sorted(book.items()):
        symbol = f"{coin}USDT"
        price = prices.get(coin)
        stop_order = resting.get(symbol)
        base = BASELINE_POSITIONS.get(coin)
        pnl = (price - pos["avg_cost"]) * pos["qty"] if price else None
        to_stop = (price / pos["stop_price"] - 1) if price and pos["stop_price"] else None
        print(f"  {coin:4s} {pos['qty']:>12.8f}  средняя {pos['avg_cost']:>12,.2f}  "
              f"цена {price:>12,.2f}  P&L {pnl:>+9,.2f}" if price else f"  {coin}: нет цены")
        if to_stop is not None:
            print(f"        стоп {pos['stop_price']:,.2f} (до него {to_stop:+.1%})", end="")
        if stop_order:
            covered = float(stop_order.get("qty") or 0) / pos["qty"] if pos["qty"] else 0
            print(f"  | на бирже стоит ордер {stop_order.get('triggerPrice')} "
                  f"({covered:.0%} позиции)")
        else:
            unprotected.append(coin)
            print("  | СТОП-ОРДЕРА НА БИРЖЕ НЕТ")
        if base and abs(base["avg_cost"] - pos["avg_cost"]) > 0.01:
            print(f"        средняя изменилась с {base['avg_cost']:,.2f} - была докупка или новая сделка")

    if unprotected:
        print(f"\n  ВАЖНО: без биржевого стопа: {', '.join(unprotected)}. "
              f"Между тиками эти позиции ничем не защищены.")

    drift = reconcile(conn, book)
    print("\n  реестр и биржа сходятся" if not drift else f"\n  РАСХОЖДЕНИЕ реестра с биржей: {drift}")

    section("ДЕНЬГИ")
    realized_total = sum(p["realized_pnl"] for p in real_positions(conn).values())
    unrealized = sum((prices.get(c, 0) - p["avg_cost"]) * p["qty"] for c, p in book.items() if prices.get(c))
    balance = get_full_balance()
    equity = None
    if balance:
        equity = balance["unified_usdt"] + sum((prices.get(c, 0) * p["qty"]) for c, p in book.items())
        print(f"  свободно USDT: {balance['unified_usdt']:,.2f}"
              + (f"  (+ {balance['funding_usdt']:,.2f} в Funding, споту недоступны)"
                 if balance["funding_usdt"] > 1 else ""))
        print(f"  эквити (свободные + позиции по рынку): {equity:,.2f} USDT")
        if BASELINE_EQUITY_USDT:
            print(f"  на момент запуска автономии было: {BASELINE_EQUITY_USDT:,.2f} USDT  "
                  f"-> {equity - BASELINE_EQUITY_USDT:+,.2f} ({equity / BASELINE_EQUITY_USDT - 1:+.2%})")
    print(f"  реализовано всего за историю реестра: {realized_total:+,.2f} USDT")
    print(f"  нереализовано сейчас: {unrealized:+,.2f} USDT")
    print(f"  реализовано ботом за 24ч: {trading_guard.realized_pnl_24h(conn):+,.2f} USDT")

    section("ЗДОРОВЬЕ ЦИКЛА")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), min(ts), max(ts) FROM testnet_run_log "
                    "WHERE ts >= %s AND action_taken LIKE 'advice%%'", (since,))
        n, first, last = cur.fetchone()
        cur.execute("SELECT action_taken, count(*) FROM testnet_run_log WHERE ts >= %s "
                    "AND action_taken LIKE 'advice%%' GROUP BY 1 ORDER BY 2 DESC", (since,))
        by_action = cur.fetchall()
        cur.execute("SELECT count(*) FROM testnet_run_log WHERE ts >= %s "
                    "AND action_taken IN ('skip_stale_price', 'error')", (since,))
        problems = cur.fetchone()[0]
    print(f"  решений советника: {n}, с {first:%d.%m %H:%M} по {last:%d.%m %H:%M}" if n else "  тиков не было")
    if n:
        expected = hours * 4  # тик раз в 15 минут
        print(f"  ожидалось ~{expected * 7:.0f} записей (7 монет x 4 тика/час) - "
              f"{'похоже на норму' if n > expected * 7 * 0.7 else 'МЕНЬШЕ ОЖИДАЕМОГО, проверить прогоны CI'}")
    for action, count in by_action:
        print(f"    {action:28s} {count}")
    if problems:
        print(f"  ПРОБЛЕМНЫХ тиков (устаревшие данные / ошибки): {problems}")

    section("СТРАТЕГИЯ ПРОТИВ ИСПОЛНЕНИЯ")
    print("  Бумажный бот (testnet_trader.py) торгует ту же логику без реальных денег.")
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, entry_ts, exit_ts, net_ret, exit_reason FROM testnet_trades "
                    "WHERE exit_ts >= %s ORDER BY exit_ts", (since,))
        paper = cur.fetchall()
    if not paper:
        print("  бумажных сделок за период не закрывалось - сравнивать пока нечего")
    for symbol, entry_ts, exit_ts, net_ret, reason in paper:
        print(f"    {exit_ts:%d.%m %H:%M} {symbol:9s} {net_ret:+.2%}  ({reason})")
    print("\n  Расхождение бумажных сделок с ордерами выше - это цена реальности: "
          "проскальзывание, лимиты, нехватка свободных средств.")

    conn.close()


if __name__ == "__main__":
    main()
