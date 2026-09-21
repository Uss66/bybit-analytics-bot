"""One-off diagnostic (2026-09-21): the real-balance check (bybit_balance.py)
has been returning 403 Forbidden for every coin, but ONLY when run from
GitHub Actions - identical code/credentials work fine from the user's home
IP. `resp.raise_for_status()` in bybit_balance.py's `_get()` only ever
logged the status line, never the response BODY, so there's no way to tell
apart an IP/WAF block from a credentials/signature problem from the logs
seen so far. This script makes the exact same signed request bybit_balance
makes, but prints/sends the full response (status, headers, body) so the
real cause can be read directly instead of guessed at. Delete once the 403
is resolved - this is not meant to be permanent.

Usage (run via GitHub Actions workflow_dispatch, NOT scheduled):
    python scripts/diagnose_bybit_403.py
"""
import hashlib
import hmac
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from telegram_notify import send_alert

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.bybit.com"
RECV_WINDOW = "20000"
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")


def main():
    if not API_KEY or not API_SECRET:
        send_alert("[diag] BYBIT_API_KEY/SECRET not set in this environment")
        return

    params = {"accountType": "UNIFIED", "coin": "BTC"}
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    timestamp = str(int(time.time() * 1000))
    raw = f"{timestamp}{API_KEY}{RECV_WINDOW}{query}"
    sign = hmac.new(API_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN": sign,
    }
    resp = requests.get(f"{BASE_URL}/v5/account/wallet-balance", params=params, headers=headers, timeout=20)

    outbound_ip = "unknown"
    try:
        outbound_ip = requests.get("https://api.ipify.org", timeout=10).text
    except Exception as e:
        outbound_ip = f"lookup failed: {e}"

    body = resp.text[:1500]
    msg = (
        f"[diag] Bybit wallet-balance call from this runner\n"
        f"outbound IP: {outbound_ip}\n"
        f"status: {resp.status_code}\n"
        f"server header: {resp.headers.get('server')}\n"
        f"cf-ray: {resp.headers.get('cf-ray')}\n"
        f"body: {body}"
    )
    print(msg)
    send_alert(msg)


if __name__ == "__main__":
    main()
