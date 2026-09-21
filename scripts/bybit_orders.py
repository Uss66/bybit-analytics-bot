"""
The only module in this project that can move money (2026-09-22).

Everything else reads. This places spot orders, through the same
pg_net-from-eu-west-1 transport as every other Bybit call (GitHub Actions
runners are geo-blocked), signed by SEPARATE Vault secrets - a spot-trade
key with no withdrawal permission, bound to the Supabase egress IP. The
read-only key the rest of the project uses stays unable to trade.

Nothing here decides anything. It takes an explicit instruction, sends it,
and reports exactly what the exchange said. The decisions, the limits and
the kill switch live in live_advisor.py / trading_config.

Idempotency is not optional here. The paper bot had a known hole - an
order could be placed while the confirmation was lost, leaving local state
out of sync - which was acceptable with fake money and is not acceptable
now. Every order carries an `orderLinkId` derived from (symbol, intent,
tick minute): Bybit rejects a duplicate outright, so a retry after a lost
response cannot double-buy. The caller records the id BEFORE sending.

Stop-loss orders are placed ON THE EXCHANGE right after a buy. That is
the part that actually answers "what if it crashes while I'm asleep": a
resting stop executes in milliseconds on Bybit's side even if this code,
GitHub Actions, Supabase and Telegram are all down at once. The bot's own
15-minute stop check remains as a second line, never the first.
"""
import re
from datetime import datetime, timezone

import bybit_public
from pg_net_proxy import fetch_json

TRADE_SQL_FN = "bybit_signed_post_start"
READ_SQL_FN = "bybit_wallet_balance_start"
CATEGORY = "spot"

_instrument_cache: dict[str, dict] = {}


class OrderRejected(RuntimeError):
    """Bybit accepted the request but refused the order (retCode != 0)."""

    def __init__(self, ret_code, ret_msg, body=None):
        self.ret_code, self.ret_msg, self.body = ret_code, ret_msg, body
        super().__init__(f"Bybit rejected the order: {ret_code} {ret_msg}")


def make_link_id(symbol: str, intent: str, when: datetime | None = None) -> str:
    """Idempotency key. Same symbol+intent inside the same minute produces
    the same id, so a retry of a request whose response was lost is
    refused by Bybit instead of doubling the position. Bybit allows up to
    36 chars of [A-Za-z0-9_-]."""
    when = when or datetime.now(timezone.utc)
    raw = f"BOT-{symbol}-{intent}-{when:%Y%m%d%H%M}"
    return re.sub(r"[^A-Za-z0-9_-]", "", raw)[:36]


def _post(path: str, body: dict, conn=None) -> dict:
    result = fetch_json(TRADE_SQL_FN, path, body, conn=conn)
    if result.get("retCode") != 0:
        raise OrderRejected(result.get("retCode"), result.get("retMsg"), result)
    return result.get("result") or {}


def _signed_get(path: str, params: dict, conn=None) -> dict:
    body = fetch_json(READ_SQL_FN, path, params, conn=conn)
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit error on {path}: {body.get('retCode')} {body.get('retMsg')}")
    return body.get("result") or {}


def instrument(symbol: str, conn=None) -> dict:
    """basePrecision / minOrderQty / minOrderAmt for a spot pair. Cached
    per process: these change about never, and every order needs them to
    round correctly (an over-precise qty is rejected outright)."""
    if symbol not in _instrument_cache:
        result = bybit_public.get("/v5/market/instruments-info",
                                  {"category": CATEGORY, "symbol": symbol}, conn=conn)
        info = (result.get("list") or [{}])[0]
        f = info.get("lotSizeFilter") or {}
        _instrument_cache[symbol] = dict(
            base_precision=f.get("basePrecision") or "0.000001",
            quote_precision=f.get("quotePrecision") or "0.01",
            min_order_qty=float(f.get("minOrderQty") or 0),
            min_order_amt=float(f.get("minOrderAmt") or 0),
            tick_size=float((info.get("priceFilter") or {}).get("tickSize") or 0),
        )
    return _instrument_cache[symbol]


def _floor_str(value: float, precision: str) -> str:
    """Round DOWN to the instrument's step. Never up: rounding up can ask
    to sell more than is held, or to spend more than is free."""
    decimals = len(precision.split(".")[1]) if "." in precision else 0
    factor = 10 ** decimals
    floored = int(value * factor) / factor
    return f"{floored:.{decimals}f}"


def market_buy(symbol: str, quote_usdt: float, link_id: str, conn=None) -> dict:
    """Spend `quote_usdt` USDT on `symbol` at market. Size is given in the
    QUOTE coin (marketUnit=quoteCoin) because that is how the strategy
    thinks - "put 120 USDT into ETH", not "buy 0.0431 ETH"."""
    spec = instrument(symbol, conn=conn)
    if spec["min_order_amt"] and quote_usdt < spec["min_order_amt"]:
        raise OrderRejected(-1, f"{quote_usdt:.2f} USDT is below the {symbol} minimum "
                                f"of {spec['min_order_amt']:.2f}")
    body = dict(category=CATEGORY, symbol=symbol, side="Buy", orderType="Market",
                qty=_floor_str(quote_usdt, spec["quote_precision"]),
                marketUnit="quoteCoin", orderLinkId=link_id)
    return _post("/v5/order/create", body, conn=conn)


def market_sell(symbol: str, base_qty: float, link_id: str, conn=None) -> dict:
    """Sell `base_qty` of the base coin at market."""
    spec = instrument(symbol, conn=conn)
    qty = _floor_str(base_qty, spec["base_precision"])
    if float(qty) <= 0 or (spec["min_order_qty"] and float(qty) < spec["min_order_qty"]):
        raise OrderRejected(-1, f"{base_qty} {symbol} is below the minimum order qty "
                                f"{spec['min_order_qty']}")
    body = dict(category=CATEGORY, symbol=symbol, side="Sell", orderType="Market",
                qty=qty, marketUnit="baseCoin", orderLinkId=link_id)
    return _post("/v5/order/create", body, conn=conn)


def place_stop_loss(symbol: str, base_qty: float, trigger_price: float, link_id: str, conn=None) -> dict:
    """A resting stop-market sell on the exchange - the protection that
    survives everything in this stack being down. `orderFilter=StopOrder`
    is what makes a spot order conditional on Bybit v5."""
    spec = instrument(symbol, conn=conn)
    qty = _floor_str(base_qty, spec["base_precision"])
    if float(qty) <= 0:
        raise OrderRejected(-1, f"nothing to protect: qty rounds to 0 ({base_qty})")
    trigger = _floor_str(trigger_price, str(spec["tick_size"]) if spec["tick_size"] else "0.01")
    body = dict(category=CATEGORY, symbol=symbol, side="Sell", orderType="Market",
                qty=qty, marketUnit="baseCoin", triggerPrice=trigger,
                orderFilter="StopOrder", orderLinkId=link_id)
    return _post("/v5/order/create", body, conn=conn)


def cancel_order(symbol: str, order_id: str | None = None, link_id: str | None = None,
                 stop_order: bool = False, conn=None) -> dict:
    body = dict(category=CATEGORY, symbol=symbol)
    if order_id:
        body["orderId"] = order_id
    elif link_id:
        body["orderLinkId"] = link_id
    else:
        raise ValueError("cancel_order needs order_id or link_id")
    if stop_order:
        body["orderFilter"] = "StopOrder"
    return _post("/v5/order/cancel", body, conn=conn)


def open_orders(symbol: str | None = None, stop_orders: bool = False, conn=None) -> list[dict]:
    """Live orders, read with the READ-ONLY key - reading never needs the
    trading credentials."""
    params = {"category": CATEGORY}
    if symbol:
        params["symbol"] = symbol
    if stop_orders:
        params["orderFilter"] = "StopOrder"
    return (_signed_get("/v5/order/realtime", params, conn=conn).get("list") or [])


def order_status(symbol: str, link_id: str, conn=None) -> dict | None:
    """What actually happened to an order we sent - by our own link id, so
    it works even when the response that carried Bybit's orderId was lost.
    Checks live orders first, then history."""
    for order in open_orders(symbol, conn=conn):
        if order.get("orderLinkId") == link_id:
            return order
    result = _signed_get("/v5/order/history",
                         {"category": CATEGORY, "symbol": symbol, "limit": "50"}, conn=conn)
    for order in result.get("list") or []:
        if order.get("orderLinkId") == link_id:
            return order
    return None
