"""
Proves the order path works BEFORE any order depends on it (2026-09-22).

Three things have to be true for the bot to be able to trade, and each
fails differently:
  1. the trading secrets exist in Vault (the SQL function raises if not),
  2. the POST signature Bybit computes matches ours - a body serialized
     even slightly differently comes back as retCode 10004 "error sign",
  3. the key's permissions and IP binding allow the call - retCode 10003 /
     10005 / a CloudFront 403 rather than a signature error.

The probe is a `cancel-all` on a symbol that has NO open orders, verified
first through the read-only key. That makes it a genuine no-op: it can
only report success or fail, never move money. An order-shaped probe (a
deliberately invalid order) was rejected as an idea - "invalid" is one
typo away from "executed".

Run it after db/setup_trading.sql, and again any time orders start
failing:
    python scripts/verify_trading_setup.py
"""
import sys

import bybit_orders
import trading_guard
from db import get_connection
from pg_net_proxy import fetch_json

PROBE_SYMBOL = "BTCUSDT"


def main():
    conn = get_connection()
    ok = True

    config = trading_guard.load_config(conn)
    print(f"[config] enabled={config['enabled']}  cap={config['max_capital_usdt']:,.0f} USDT  "
          f"max_order={config['max_order_usdt']:,.0f}  max_daily_loss={config['max_daily_loss_usdt']:,.0f}")
    if config.get("paused_reason"):
        print(f"[config] paused: {config['paused_reason']}")

    print("\n[1/3] open spot orders, via the READ-ONLY key ...")
    try:
        resting = bybit_orders.open_orders(conn=conn)
        print(f"      {len(resting)} open order(s) on the account")
        for o in resting:
            print(f"        {o.get('symbol')} {o.get('side')} {o.get('qty')} "
                  f"{o.get('orderType')} filter={o.get('orderFilter')}")
    except Exception as e:
        print(f"      FAILED: {e}")
        conn.close()
        sys.exit(1)

    on_probe_symbol = [o for o in resting if o.get("symbol") == PROBE_SYMBOL]
    if on_probe_symbol:
        print(f"      {PROBE_SYMBOL} has open orders - skipping the cancel-all probe so nothing of "
              f"yours gets cancelled. Re-run when it is clear, or the first real order will be the test.")
        conn.close()
        sys.exit(0)

    print(f"\n[2/3] signature + permissions, via a no-op cancel-all on {PROBE_SYMBOL} ...")
    body = fetch_json("bybit_signed_post_start", "/v5/order/cancel-all",
                      {"category": "spot", "symbol": PROBE_SYMBOL}, conn=conn)
    code, msg = body.get("retCode"), body.get("retMsg")
    if code == 0:
        print(f"      OK - Bybit accepted a signed, authorized POST (nothing was cancelled)")
    elif code == 10004:
        ok = False
        print(f"      SIGNATURE MISMATCH (10004): {msg}\n"
              f"      The body we hash differs from the body pg_net sends. Check that the SQL function "
              f"hashes p_body::text and passes the same jsonb.")
    elif code in (10003, 10005, 33004):
        ok = False
        print(f"      PERMISSION/IP PROBLEM ({code}): {msg}\n"
              f"      The key needs SPOT -> Trade, and its IP binding must include the Supabase egress "
              f"IP. Check the current one with: SELECT net.http_get('https://api.ipify.org?format=json');")
    else:
        print(f"      Bybit answered retCode={code} ({msg}) - not a signature error, so the path works; "
              f"read the message above on its own merits.")

    print("\n[3/3] instrument metadata (public, needed to round every order) ...")
    try:
        spec = bybit_orders.instrument(PROBE_SYMBOL, conn=conn)
        print(f"      {PROBE_SYMBOL}: basePrecision={spec['base_precision']} "
              f"minOrderQty={spec['min_order_qty']} minOrderAmt={spec['min_order_amt']}")
    except Exception as e:
        ok = False
        print(f"      FAILED: {e}")

    conn.close()
    print("\n" + ("READY - the bot can place orders the moment trading_config.enabled is true."
                  if ok else "NOT READY - fix the errors above before enabling."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
