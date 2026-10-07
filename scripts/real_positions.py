"""
The bot's real-money ledger (2026-09-22): what the user ACTUALLY bought,
at what price, and therefore where the -8% stop-loss really sits.

Until now the live bot only knew a paper position - a fixed
USDT_PER_TRADE=100 fill simulated at the public close - so its
`entry_price` (the basis the stop is measured from) had no connection to
the real account. Checking the real account showed how far apart the two
had drifted: the bot's book said BTC/BNB/LINK/XRP, the exchange said
BTC and BNB only, and for BNB the paper average (716.30) sat ~11% below
the real one (806.76), i.e. the bot would have watched a real -8% pass
by without reacting.

TWO sources, because the user's fills come from two different places:

  * `/v5/asset/exchange/order-record` - Bybit **Convert**. This is where
    every historical purchase actually happened, which is why
    /v5/execution/list came back empty for months. Convert has no
    explicit fee: its cost is a ~1.4%-per-leg spread baked into the
    quoted rate (measured, see README "Convert против спота"), about 14x
    the 0.1% spot taker fee the whole backtest assumes. The price stored
    here is the EFFECTIVE one (USDT moved / coin received), so the spread
    is already inside the recorded cost basis - exactly as it should be.
  * `/v5/execution/list` - real spot fills. Empty today; it fills up as
    soon as trading moves to spot orders, and this module then picks
    them up with no further changes.

Accounting is weighted-average, matching how the live stop works on a
multi-tranche position (testnet_trader.py): a BUY adds qty and cost, a
SELL removes cost AT THE AVERAGE (not at the sale price), so selling
part of a position never moves the remaining average - and the stop
level stays where it was. Deposits and withdrawals of USDT do not touch
any of this; they change how much you can buy next, not what you paid
for what you already hold.

All Bybit calls go through the Vault-signed Postgres function (pg_net,
fixed eu-west-1) - the geo-block workaround that also makes this work
from GitHub Actions. No API credentials exist in this process.

Usage:
    python scripts/real_positions.py              # sync, then report
    python scripts/real_positions.py --report     # report from the DB only
    python scripts/real_positions.py --days 365   # widen the spot backfill
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

from psycopg2.extras import Json, execute_values

from bybit_balance import get_coin_balance
from db import get_connection
from pg_net_proxy import fetch_json

SIGNED_SQL_FN = "bybit_wallet_balance_start"
QUOTE = "USDT"
STOP_LOSS = -0.08  # keep in sync with testnet_trader.STOP_LOSS

# A remainder worth less than this is NOT a position, because Bybit will
# not accept a spot order below minOrderAmt = 5 USDT (checked 2026-10-07:
# same 5 for all seven pairs). So such a remainder can neither be sold
# nor protected by a stop - calling it a position only makes the bot
# attempt both every tick and get rejected every time.
#
# This has to be a VALUE, not a quantity. A coin-unit threshold cannot be
# right for seven coins at once - 1e-8 BTC and 1e-8 DOGE differ by five
# orders of magnitude in money - and that is exactly how the ETH stop of
# 2026-10-07 left 0.00000264 ETH (0.0068 USDT) behind and had the bot
# retrying a dust sale for hours.
#
# Measured against the remaining COST BASIS, which the ledger knows
# without a price lookup. The one blind spot: a position that cost 6 USDT
# and then halved is worth under 5 and would still be called a position.
# That error direction is the safe one - it keeps reporting a holding
# that really is there instead of silently zeroing one.
DUST_USDT = 5.0

# Bybit caps a single spot execution query at a 7-day window.
SPOT_WINDOW_DAYS = 7
DEFAULT_BACKFILL_DAYS = 180


def _signed_get(path: str, params: dict, conn=None) -> dict:
    body = fetch_json(SIGNED_SQL_FN, path, params, conn=conn)
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit error on {path}: {body.get('retCode')} {body.get('retMsg')}")
    return body.get("result") or {}


def fetch_convert_records(conn) -> list[dict]:
    """Bybit Convert history, normalized into the same shape as a spot fill.

    A Convert is always coin-for-coin, so only the legs against USDT can
    be priced in this ledger - a hypothetical BTC->BNB conversion would
    need two half-trades and is skipped with a warning rather than
    silently mispriced."""
    rows, cursor = [], ""
    while True:
        params = {"limit": "50"}
        if cursor:
            params["cursor"] = cursor
        result = _signed_get("/v5/asset/exchange/order-record", params, conn=conn)
        batch = result.get("orderBody") or []
        for r in batch:
            from_coin, to_coin = r["fromCoin"], r["toCoin"]
            from_amt, to_amt = float(r["fromAmount"]), float(r["toAmount"])
            if from_coin == QUOTE:
                coin, side, qty, quote_qty = to_coin, "BUY", to_amt, from_amt
            elif to_coin == QUOTE:
                coin, side, qty, quote_qty = from_coin, "SELL", from_amt, to_amt
            else:
                print(f"[real_positions] skipping {from_coin}->{to_coin} convert "
                      f"(no {QUOTE} leg, cannot be priced in this ledger)")
                continue
            rows.append(dict(
                source="convert", exec_id=str(r["exchangeTxId"]),
                ts=datetime.fromtimestamp(int(r["createdTime"]) / 1000, tz=timezone.utc),
                coin=coin, side=side, qty=qty, quote_qty=quote_qty,
                price=quote_qty / qty, fee_quote=None, raw=r,
            ))
        cursor = result.get("nextPageCursor") or ""
        if not cursor or not batch:
            break
    return rows


def fetch_spot_executions(conn, days: int) -> list[dict]:
    """Real spot fills, walked backwards in 7-day windows (Bybit's limit
    per query). Fees are explicit here, unlike Convert: Bybit charges a
    spot BUY's fee in the coin received and a SELL's in USDT, so the two
    cases have to be folded into the cost basis differently."""
    rows = []
    now = datetime.now(timezone.utc)
    for offset in range(0, days, SPOT_WINDOW_DAYS):
        end = now - timedelta(days=offset)
        start = end - timedelta(days=SPOT_WINDOW_DAYS)
        cursor = ""
        while True:
            params = {"category": "spot", "limit": "100",
                      "startTime": str(int(start.timestamp() * 1000)),
                      "endTime": str(int(end.timestamp() * 1000))}
            if cursor:
                params["cursor"] = cursor
            result = _signed_get("/v5/execution/list", params, conn=conn)
            batch = result.get("list") or []
            for r in batch:
                symbol = r["symbol"]
                if not symbol.endswith(QUOTE):
                    continue
                coin = symbol[: -len(QUOTE)]
                side = "BUY" if r["side"].upper() == "BUY" else "SELL"
                qty, price = float(r["execQty"]), float(r["execPrice"])
                quote_qty = float(r.get("execValue") or qty * price)
                fee, fee_coin = float(r.get("execFee") or 0), (r.get("feeCurrency") or "").upper()
                fee_quote = None
                if fee:
                    if fee_coin == coin:
                        qty -= fee            # fee taken out of the coin received
                        fee_quote = fee * price
                    else:
                        fee_quote = fee       # fee taken in USDT
                        quote_qty += fee if side == "BUY" else -fee
                rows.append(dict(
                    source="spot", exec_id=str(r["execId"]),
                    ts=datetime.fromtimestamp(int(r["execTime"]) / 1000, tz=timezone.utc),
                    coin=coin, side=side, qty=qty, quote_qty=quote_qty,
                    price=quote_qty / qty if qty else price, fee_quote=fee_quote, raw=r,
                ))
            cursor = result.get("nextPageCursor") or ""
            if not cursor or not batch:
                break
    return rows


def upsert_executions(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO real_executions
                (source, exec_id, ts, coin, side, qty, quote_qty, price, fee_quote, raw)
            VALUES %s
            ON CONFLICT (source, exec_id) DO UPDATE SET
                ts = EXCLUDED.ts, qty = EXCLUDED.qty, quote_qty = EXCLUDED.quote_qty,
                price = EXCLUDED.price, fee_quote = EXCLUDED.fee_quote, raw = EXCLUDED.raw
            """,
            [(r["source"], r["exec_id"], r["ts"], r["coin"], r["side"], r["qty"],
              r["quote_qty"], r["price"], r["fee_quote"], Json(r["raw"])) for r in rows],
        )
    conn.commit()
    return len(rows)


def positions(conn, coin: str | None = None) -> dict[str, dict]:
    """Replays the stored ledger in chronological order and returns the
    current weighted-average position per coin, including the price the
    -8% stop really belongs at. Realized P&L is tracked alongside, since
    a sale that leaves the average untouched still banks a result."""
    sql = "SELECT ts, coin, side, qty, quote_qty FROM real_executions"
    params = ()
    if coin:
        sql += " WHERE coin = %s"
        params = (coin,)
    sql += " ORDER BY ts"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    book: dict[str, dict] = {}
    for ts, c, side, qty, quote_qty in rows:
        pos = book.setdefault(c, dict(coin=c, qty=0.0, invested=0.0, realized_pnl=0.0,
                                      first_ts=ts, last_ts=ts, n_buys=0, n_sells=0))
        pos["last_ts"] = ts
        if side == "BUY":
            pos["qty"] += qty
            pos["invested"] += quote_qty
            pos["n_buys"] += 1
        else:
            avg = pos["invested"] / pos["qty"] if pos["qty"] else 0.0
            sold = min(qty, pos["qty"])
            pos["realized_pnl"] += quote_qty - avg * sold
            pos["invested"] -= avg * sold
            pos["qty"] -= sold
            pos["n_sells"] += 1

    for pos in book.values():
        # Dust left by rounding is not a position - treat it as closed.
        # Every exit leaves some: the sell is floored to the lot step, so
        # the sub-step remainder can never be sold. See DUST_USDT.
        if pos["qty"] <= 0 or pos["invested"] < DUST_USDT:
            # Kept, not discarded: the coins are still on the exchange, so
            # reconcile() has to expect them or it reports a phantom
            # withdrawal.
            pos["dust_qty"] = pos["qty"]
            pos["qty"] = 0.0
            pos["invested"] = 0.0
            pos["avg_cost"] = None
            pos["stop_price"] = None
        else:
            pos["dust_qty"] = 0.0
            pos["avg_cost"] = pos["invested"] / pos["qty"]
            pos["stop_price"] = pos["avg_cost"] * (1 + STOP_LOSS)
    return book


def tradeable(qty: float | None, price: float | None) -> bool:
    """Is this balance a position, or an unsellable remainder?

    The question only has a meaningful answer in money, which is why this
    takes a price: every exit leaves a sub-lot-step remainder behind, and
    `qty > 1e-8` calls 0.0000026 ETH a position just as readily as it
    would 0.0000026 BTC. See DUST_USDT.

    An unknown price returns True on purpose: a balance that cannot be
    valued must never be silently treated as nothing."""
    if not qty or qty <= 0:
        return False
    if price is None:
        return True
    return qty * price >= DUST_USDT


def last_exit(conn, coin: str) -> dict | None:
    """Who actually closed the position, read from the fills themselves.

    Deliberately NOT read from the bot's own advisor_state: on 2026-10-07
    the ETH exchange stop fired, and the failed dust-sale that followed
    cleared the stop bookkeeping before anything had reported the exit -
    so the bot's state no longer knew that its own stop was the cause.
    The exchange, by contrast, stamps every fill with the orderLinkId and
    stopOrderType that produced it, and those cannot be clobbered.

    Groups by orderId because one market order fills in many pieces (the
    ETH stop came back as six)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT max(ts) AS ts, sum(qty) AS qty, sum(quote_qty) AS quote,
                   max(raw->>'orderLinkId')   AS link_id,
                   max(raw->>'stopOrderType') AS stop_type,
                   max(raw->>'orderId')       AS order_id
            FROM real_executions
            WHERE coin = %s AND side = 'SELL'
            GROUP BY raw->>'orderId'
            ORDER BY max(ts) DESC
            LIMIT 1
        """, (coin,))
        row = cur.fetchone()
    if not row or not row[1]:
        return None
    ts, qty, quote, link_id, stop_type, order_id = row
    return dict(ts=ts, qty=qty, quote=quote, price=quote / qty,
                link_id=link_id, stop_type=stop_type, order_id=order_id,
                by_bot_stop=bool(stop_type) and str(link_id or "").startswith("BOT-"))


def latest_prices(conn, coins) -> dict[str, float]:
    out = {}
    with conn.cursor() as cur:
        for c in coins:
            cur.execute("SELECT close FROM ohlcv WHERE symbol = %s ORDER BY ts DESC LIMIT 1",
                        (f"{c}{QUOTE}",))
            row = cur.fetchone()
            if row:
                out[c] = row[0]
    return out


def reconcile(conn, book: dict[str, dict]) -> dict[str, dict]:
    """Compares the ledger against what the exchange says is actually
    there (Unified + Funding). Any gap means coins moved without a
    recorded trade: a deposit, a withdrawal, a transfer to another
    account, a card spend - or a fill this ledger has not seen.

    This matters more than it sounds. The ledger's average cost is only
    a valid stop basis while the quantity it describes is the quantity
    you hold; found 2026-09-22 that <кол-во скрыто> BTC had left the account
    with no matching record on any read-only endpoint the key can reach
    (it has Wallet:AccountTransfer but no Withdraw scope, so an on-chain
    withdrawal is invisible to it - and comes back as an empty list
    rather than an error, which is exactly how it stayed unnoticed).

    Returns {coin: {ledger_qty, real_qty, drift, drift_pct}} for coins
    where the two disagree by more than dust. `real_qty` None means the
    balance call failed - reported as unknown, never silently as zero."""
    out = {}
    for coin, pos in book.items():
        real = get_coin_balance(coin)
        ledger = pos["qty"]
        if real is None:
            out[coin] = dict(ledger_qty=ledger, real_qty=None, drift=None, drift_pct=None)
            continue
        # The unsellable remainder of a closed position is still physically
        # on the exchange, so the balance legitimately exceeds the ledger
        # quantity by exactly that much. Expecting it keeps a normal exit
        # from looking like coins leaving without a record.
        expected = ledger + pos.get("dust_qty", 0.0)
        drift = real - expected
        if abs(drift) <= max(1e-8, expected * 1e-6):
            continue
        out[coin] = dict(ledger_qty=ledger, real_qty=real, drift=drift,
                          drift_pct=drift / expected if expected else None)
    return out


def report(conn, check_balances: bool = True) -> None:
    book = positions(conn)
    open_book = {c: p for c, p in book.items() if p["qty"] > 0}
    if not book:
        print("No real executions recorded yet - run without --report to sync.")
        return
    prices = latest_prices(conn, book)

    print(f"{'coin':6s} {'qty':>14s} {'invested':>11s} {'avg cost':>12s} {'stop -8%':>12s} "
          f"{'price now':>12s} {'unrealized':>11s} {'realized':>10s}")
    for c, p in sorted(open_book.items()):
        now = prices.get(c)
        unreal = (now - p["avg_cost"]) * p["qty"] if now else None
        print(f"{c:6s} {p['qty']:>14.8f} {p['invested']:>11,.2f} {p['avg_cost']:>12,.2f} "
              f"{p['stop_price']:>12,.2f} {(f'{now:,.2f}' if now else '-'):>12s} "
              f"{(f'{unreal:+,.2f}' if unreal is not None else '-'):>11s} "
              f"{p['realized_pnl']:>+10,.2f}")
    closed = {c: p for c, p in book.items() if p["qty"] == 0}
    for c, p in sorted(closed.items()):
        print(f"{c:6s} {'(closed)':>14s} {'':>11s} {'':>12s} {'':>12s} {'':>12s} {'':>11s} "
              f"{p['realized_pnl']:>+10,.2f}")

    if not check_balances:
        return
    drift = reconcile(conn, open_book)
    if not drift:
        print("\nLedger matches the exchange balances.")
        return
    print("\nLEDGER vs EXCHANGE - these coins moved without a recorded trade:")
    for coin, d in sorted(drift.items()):
        if d["real_qty"] is None:
            print(f"  {coin}: balance unknown (API call failed) - not treated as zero")
            continue
        print(f"  {coin}: ledger {d['ledger_qty']:.8f}, exchange {d['real_qty']:.8f}, "
              f"drift {d['drift']:+.8f} ({d['drift_pct']:+.1%})")
    print("  A deposit, withdrawal, transfer out or an unseen fill. Until it is explained,\n"
          "  the average cost above still prices what you PAID, but not what you HOLD.")


def alert_block(conn, coin: str, check_balance: bool = True) -> str:
    """Russian block for a Telegram alert: what this coin's REAL position
    is and where its stop actually sits. Returns "" when the ledger knows
    nothing about the coin, so an alert never claims a position the user
    does not have.

    The whole point of the ledger being in the alert: the bot's own
    `entry_price` is a simulated 100 USDT fill, so the stop it quotes is
    fiction on a real account. This puts the honest number next to it."""
    try:
        pos = positions(conn, coin).get(coin)
    except Exception as e:  # an alert must never be lost to a ledger problem
        print(f"[real_positions] alert_block failed for {coin}: {e}")
        return ""
    if not pos or not pos["qty"]:
        return ""

    block = (f"\n\nТвоя РЕАЛЬНАЯ позиция по {coin}: {pos['qty']:.8g} "
             f"(вложено {pos['invested']:,.2f} USDT)\n"
             f"Реальная средняя цена: {pos['avg_cost']:,.6g} USDT → "
             f"стоп −8% на {pos['stop_price']:,.6g} USDT")
    if not check_balance:
        return block
    drift = reconcile(conn, {coin: pos}).get(coin)
    if drift and drift["real_qty"] is not None:
        block += (f"\n⚠️ На бирже фактически {drift['real_qty']:.8g} {coin} "
                  f"({drift['drift']:+.8g} к реестру) — часть монет двигалась "
                  f"без записанной сделки, средняя цена может быть неточной")
    return block


def true_up(conn, coin: str, price: float | None = None) -> dict | None:
    """Records the drift between ledger and exchange as an explicit
    adjustment row, so the ledger's quantity matches what is actually
    held. Returns the recorded row, or None when there was nothing to fix.

    Needed because not every coin movement is visible to a read-only key:
    the user converted BTC out through a path that appears on NO endpoint
    this key can read (checked every page of Convert history, all five
    convert account types, withdrawals in 30-day windows over six months,
    internal transfers and Earn - all empty), and the key has no Withdraw
    scope to see it directly.

    Crucially this does NOT distort the stop: weighted-average accounting
    removes cost AT THE AVERAGE, so the remaining position keeps exactly
    the average cost - and therefore the stop price - it had before. Only
    realized P&L is affected, and it is approximate by construction
    (priced at the current market unless a price is given), which is why
    the row is stored with source='adjustment' rather than pretending to
    be a trade."""
    pos = positions(conn, coin).get(coin)
    if not pos or not pos["qty"]:
        print(f"[real_positions] no ledger position in {coin} to true up")
        return None
    real = get_coin_balance(coin)
    if real is None:
        print(f"[real_positions] balance for {coin} unknown - refusing to true up on a guess")
        return None

    drift = real - pos["qty"]
    if abs(drift) <= max(1e-8, pos["qty"] * 1e-6):
        print(f"[real_positions] {coin} already matches the exchange")
        return None
    if price is None:
        price = latest_prices(conn, [coin]).get(coin)
        if price is None:
            print(f"[real_positions] no price for {coin} - pass --price explicitly")
            return None

    side = "SELL" if drift < 0 else "BUY"
    qty = abs(drift)
    now = datetime.now(timezone.utc)
    row = dict(
        source="adjustment", exec_id=f"trueup-{coin}-{int(now.timestamp())}", ts=now,
        coin=coin, side=side, qty=qty, quote_qty=qty * price, price=price, fee_quote=None,
        raw=dict(reason="ledger/exchange drift with no readable record on any endpoint",
                 ledger_qty=pos["qty"], exchange_qty=real, priced_at=price),
    )
    upsert_executions(conn, [row])
    print(f"[real_positions] {coin}: recorded {side} {qty:.8f} at {price:,.2f} "
          f"(ledger {pos['qty']:.8f} -> exchange {real:.8f})")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Skip the API sync, report what's stored")
    parser.add_argument("--true-up", type=str, metavar="COIN",
                        help="Record the ledger-vs-exchange drift for COIN as an explicit adjustment "
                             "(use when coins moved through a path the API cannot show). The remaining "
                             "position keeps its average cost, so the stop level does not move.")
    parser.add_argument("--price", type=float, default=None,
                        help="Price to value a --true-up adjustment at (default: latest close)")
    parser.add_argument("--days", type=int, default=DEFAULT_BACKFILL_DAYS,
                        help=f"How far back to walk spot executions (default {DEFAULT_BACKFILL_DAYS})")
    parser.add_argument("--no-reconcile", action="store_true",
                        help="Skip the ledger-vs-exchange balance check")
    args = parser.parse_args()

    conn = get_connection()
    if args.true_up:
        true_up(conn, args.true_up.upper(), args.price)
        report(conn, check_balances=not args.no_reconcile)
        conn.close()
        return
    if not args.report:
        convert_rows = fetch_convert_records(conn)
        spot_rows = fetch_spot_executions(conn, args.days)
        n = upsert_executions(conn, convert_rows + spot_rows)
        print(f"Synced {len(convert_rows)} convert + {len(spot_rows)} spot executions ({n} rows upserted)\n")
    report(conn, check_balances=not args.no_reconcile)
    conn.close()


if __name__ == "__main__":
    main()
