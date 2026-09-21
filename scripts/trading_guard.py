"""
The things that must be true before the bot is allowed to send an order
(2026-09-22), and the audit trail of every order it sent.

These are separate from the strategy on purpose. The strategy answers
"is this a good trade"; this answers "is the bot allowed to trade at all,
right now, for this much". A bug in the first one costs a bad trade. A
missing check here costs the account.

The limits live in the `trading_config` table rather than in this file so
that stopping the bot never requires a deploy or a git push: flip
`enabled` in the database (or send /stop in Telegram) and the very next
tick places nothing. A bot you can only stop by editing code is a bot you
cannot stop at 3am.

Guards, in the order they are checked:
  1. enabled            - the master switch, default OFF
  2. max_order_usdt     - no single order larger than this
  3. max_capital_usdt   - total deployed by the BOT across all its
                          positions. Deliberately counts only what the bot
                          itself bought: the user's own holdings (BTC/BNB
                          bought by hand) are not the bot's to risk, and
                          counting them would have frozen it out entirely
                          at any sane ceiling.
  4. max_daily_loss_usdt - realized losses over the last 24h. On breach
                          the bot disables ITSELF and says why; it does
                          not keep trading while losing.
"""
from datetime import datetime, timedelta, timezone

DEFAULTS = dict(enabled=False, max_capital_usdt=300.0, max_order_usdt=150.0,
                max_daily_loss_usdt=30.0)


class TradingBlocked(Exception):
    """Raised when a guard refuses an order. The message is user-facing."""


def load_config(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM trading_config WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            return dict(DEFAULTS, paused_reason="trading_config row missing")
        return dict(zip([d[0] for d in cur.description], row))


def pause(conn, reason: str) -> None:
    """The bot switching itself off. Used by the daily-loss guard and by
    any order failure we cannot explain - stopping on confusion is always
    cheaper than continuing on it."""
    with conn.cursor() as cur:
        cur.execute("UPDATE trading_config SET enabled = false, paused_reason = %s, "
                    "paused_at = now(), updated_at = now() WHERE id = 1", (reason,))
    conn.commit()


def bot_deployed_usdt(conn) -> float:
    """What the bot currently has in the market, by its own bookkeeping."""
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(sum(bot_invested_usdt), 0) FROM advisor_state")
        return float(cur.fetchone()[0])


def realized_pnl_24h(conn) -> float:
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(sum(pnl_usdt), 0) FROM bot_orders "
                    "WHERE status = 'accepted' AND pnl_usdt IS NOT NULL AND ts > now() - interval '24 hours'")
        return float(cur.fetchone()[0])


def check_can_trade(conn, config: dict) -> None:
    """Global checks, run once per tick before any order is considered."""
    if not config.get("enabled"):
        reason = config.get("paused_reason")
        raise TradingBlocked(f"autotrading is off{f' ({reason})' if reason else ''}")

    loss = realized_pnl_24h(conn)
    limit = float(config["max_daily_loss_usdt"])
    if loss <= -abs(limit):
        pause(conn, f"daily loss limit hit: {loss:+,.2f} USDT in 24h (limit {limit:,.2f})")
        raise TradingBlocked(f"daily loss limit hit ({loss:+,.2f} USDT in 24h) - "
                             f"autotrading disabled itself, re-enable by hand after looking at why")


def check_order(conn, config: dict, stake_usdt: float) -> float:
    """Returns the stake the bot is actually allowed to spend, which may
    be smaller than the one requested. Raises when it cannot trade at all.

    Trimming rather than refusing matters: "the cap leaves room for 80 of
    the 120 USDT you wanted" is a perfectly good trade, while refusing
    outright would silently skip signals whenever the ceiling is close."""
    max_order = float(config["max_order_usdt"])
    max_capital = float(config["max_capital_usdt"])
    deployed = bot_deployed_usdt(conn)

    room = min(stake_usdt, max_order, max_capital - deployed)
    if room <= 0:
        raise TradingBlocked(
            f"no room under the limits: bot has {deployed:,.2f} of {max_capital:,.2f} USDT deployed"
        )
    return room


def record_order(conn, symbol: str, intent: str, link_id: str, request: dict, reason: str = "") -> int:
    """Written BEFORE the request leaves, so an order whose response is
    lost still leaves a trace to reconcile against. Returns the row id."""
    from psycopg2.extras import Json
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bot_orders (symbol, intent, order_link_id, request, status, reason) "
            "VALUES (%s, %s, %s, %s, 'sent', %s) RETURNING id",
            (symbol, intent, link_id, Json(request), reason),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def finish_order(conn, row_id: int, status: str, order_id: str | None = None,
                 response: dict | None = None, pnl_usdt: float | None = None) -> None:
    from psycopg2.extras import Json
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE bot_orders SET status = %s, order_id = %s, response = %s, pnl_usdt = %s WHERE id = %s",
            (status, order_id, Json(response) if response is not None else None, pnl_usdt, row_id),
        )
    conn.commit()


def note_bot_buy(conn, symbol: str, usdt: float) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE advisor_state SET bot_invested_usdt = coalesce(bot_invested_usdt, 0) + %s, "
                    "updated_at = now() WHERE symbol = %s", (usdt, symbol))
    conn.commit()


def note_bot_exit(conn, symbol: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE advisor_state SET bot_invested_usdt = 0, updated_at = now() "
                    "WHERE symbol = %s", (symbol,))
    conn.commit()


def recent_orders(conn, hours: int = 24) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT ts, symbol, intent, status, order_id, pnl_usdt, reason FROM bot_orders "
                    "WHERE ts > now() - %s::interval ORDER BY ts DESC", (f"{hours} hours",))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
