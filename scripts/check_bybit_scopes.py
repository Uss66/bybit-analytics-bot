"""
One-off diagnostic (2026-09-21): which READ-ONLY Bybit v5 endpoints does
the real-account API key actually have permission for?

Why this exists: the live bot currently only knows a PAPER position (a
fixed USDT_PER_TRADE=100 simulated fill), so its -8% stop is measured
from a simulated average cost, not from what the user actually bought.
Wiring the bot to the REAL cost basis / balance means reading the account's
real spot executions and its deposit/withdrawal records - but a key
created with only "wallet read" checked cannot see those. This script
answers "what can we build on" before any of it gets built.

Deliberately GET-only, exactly like bybit_balance.py: there is no order-
placement code path anywhere in this file, so it cannot move money even if
the key happens to carry trade permission.

Run it YOURSELF from your own machine (not CI):
    ! python scripts/check_bybit_scopes.py

- from home the direct api.bybit.com call works; GitHub Actions runners
  get a CloudFront geo-block (403), see project_bybit_geoblock_proxy.
- prints only endpoint names, permission verdicts and row counts - never
  the key, the secret, or account figures.
"""
import hashlib
import hmac
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.bybit.com"
RECV_WINDOW = "20000"

# path -> (params, what wiring it would unlock)
PROBES = {
    "/v5/user/query-api": ({}, "the key's own declared permission set"),
    "/v5/account/wallet-balance": ({"accountType": "UNIFIED", "coin": "USDT"}, "tradeable USDT (already used)"),
    "/v5/asset/transfer/query-account-coins-balance": ({"accountType": "FUND"}, "funding wallet (already used)"),
    "/v5/execution/list": ({"category": "spot", "limit": "50"}, "REAL fills -> real average cost basis for the stop"),
    "/v5/order/history": ({"category": "spot", "limit": "20"}, "fallback if execution/list is not permitted"),
    "/v5/account/transaction-log": ({"accountType": "UNIFIED", "limit": "20"}, "unified-account money movements"),
    "/v5/asset/deposit/query-record": ({"limit": "20"}, "DEPOSITS -> equity curve that ignores top-ups"),
    "/v5/asset/withdraw/query-record": ({"limit": "20"}, "WITHDRAWALS -> same, in the other direction"),
    "/v5/asset/transfer/query-inter-transfer-list": ({"limit": "20"}, "Funding <-> Unified internal moves"),
}


def signed_get(key: str, secret: str, path: str, params: dict):
    ts = str(int(time.time() * 1000))
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sign = hmac.new(secret.encode(), (ts + key + RECV_WINDOW + query).encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN": sign,
    }
    resp = requests.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=20)
    return resp.status_code, resp.json()


def main():
    key = (os.getenv("BYBIT_API_KEY") or "").strip()
    secret = (os.getenv("BYBIT_API_SECRET") or "").strip()
    if not key or not secret:
        raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET missing from .env")

    print(f"Probing {len(PROBES)} read-only endpoints as key ...{key[-4:]}\n")
    for path, (params, unlocks) in PROBES.items():
        try:
            status, body = signed_get(key, secret, path, params)
        except requests.RequestException as e:
            print(f"  NETWORK  {path}\n           {e}")
            continue

        ret_code, ret_msg = body.get("retCode"), body.get("retMsg")
        result = body.get("result") or {}
        rows = result.get("list") if isinstance(result, dict) else None
        n = len(rows) if isinstance(rows, list) else None

        if status == 200 and ret_code == 0:
            verdict = f"OK       (rows: {n if n is not None else 'n/a'})"
        else:
            verdict = f"DENIED   (http {status}, retCode {ret_code}: {ret_msg})"
        print(f"  {verdict}\n           {path}  ->  {unlocks}")

        # /v5/user/query-api describes the key itself - the most useful single answer here
        if path == "/v5/user/query-api" and ret_code == 0:
            print(f"           readOnly={result.get('readOnly')} permissions={result.get('permissions')}")
    print("\nPaste this whole output back into the chat - no secrets are in it.")


if __name__ == "__main__":
    main()
