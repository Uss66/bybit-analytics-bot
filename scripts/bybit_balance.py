"""
Read-ONLY Bybit MAINNET wallet balance client (2026-09-14). Deliberately a
separate, minimal file from bybit_client.py - that client is testnet/demo
-only BY DESIGN (see its own docstring) specifically so it has no mainnet
code path capable of placing a real order. This file adds a mainnet
connection, but ONLY for reading the account balance (GET
/v5/account/wallet-balance) - there is no order-placement method anywhere
in this module, so even if the supplied API key happens to have trade
permission, this code has no way to use it. That structural limit matters
more than relying on the user to remember "use a read-only key."

Why this exists: the live alerting (see telegram_notify.py /
testnet_trader.py) previously only ever said "signal fired for $100" with
no connection to the user's actual capital. This lets an alert say "signal
fired - you have $X available" so the suggestion is concrete, without any
new ability to move money.

Setup (do this yourself - Claude does not have or want trade permission
on your real account):
  1. Log into bybit.com (your REAL account, not testnet.bybit.com).
  2. API Management -> Create New Key.
  3. Permissions: check ONLY "Read-Only" / wallet read. Do NOT check
     "Trade" or "Withdraw" for this key - there is no reason this key ever
     needs those, and leaving them off means even a leaked key can't move
     your funds.
  4. Set BYBIT_API_KEY / BYBIT_API_SECRET in .env (project root) - a
     DIFFERENT pair of env vars from BYBIT_TESTNET_API_KEY/SECRET used by
     bybit_client.py, so the two are never accidentally cross-wired.

If these env vars are missing, get_usdt_balance() returns None (not an
error) so the alerting code can gracefully omit the balance line rather
than crash the hourly tick.

Usage:
    from bybit_balance import get_usdt_balance
    bal = get_usdt_balance()  # float or None
"""
import hashlib
import hmac
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://api.bybit.com"  # real mainnet - READ ENDPOINTS ONLY are ever called from this file
RECV_WINDOW = "20000"

API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")


def _sign(payload: str, timestamp: str) -> str:
    raw = f"{timestamp}{API_KEY}{RECV_WINDOW}{payload}"
    return hmac.new(API_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()


def _get(path: str, params: dict) -> dict:
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    timestamp = str(int(time.time() * 1000))
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "X-BAPI-SIGN": _sign(query, timestamp),
    }
    resp = requests.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error on GET {path}: {data}")
    return data["result"]


def get_usdt_balance() -> float | None:
    """Returns the real mainnet USDT balance actually available for SPOT
    TRADING (Unified Trading Account) - this is NOT the same as "all money
    on the account". Bybit keeps a separate Funding wallet that spot orders
    cannot draw from directly; see get_full_balance() for the complete
    picture and why this matters (found 2026-09-14: user's real funds -
    0.1956 USDT + ~0.00996 BTC - sat entirely in Funding, Unified showed
    0, which looked like a bug until checked directly). Returns None if
    BYBIT_API_KEY/SECRET aren't configured or the call fails - callers
    must treat None as 'unknown, don't show a figure', never as 0."""
    if not API_KEY or not API_SECRET:
        return None
    try:
        result = _get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        for account in result.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == "USDT":
                    return float(c.get("walletBalance") or 0)
        return 0.0
    except (requests.exceptions.RequestException, RuntimeError) as e:
        print(f"[bybit_balance] balance check failed (non-fatal): {e}")
        return None


def get_full_balance() -> dict | None:
    """Returns a full picture across BOTH wallets Bybit keeps separate:
    {'unified_usdt': float, 'funding_usdt': float, 'funding_btc': float} -
    or None if credentials are missing/the call fails. `unified_usdt` is
    what spot trading can actually use right now; `funding_*` is money
    that needs an in-account transfer (Bybit app: Assets -> Transfer,
    Funding -> Unified Trading Account - a normal internal move, not a
    withdrawal) before it's tradeable. Surfacing both, not just one
    number, is the point - a user with funds parked in Funding but
    reading only a Unified balance of 0 would wrongly conclude they have
    nothing, exactly what happened here before this was added."""
    if not API_KEY or not API_SECRET:
        return None
    try:
        unified = _get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        unified_usdt = 0.0
        for account in unified.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == "USDT":
                    unified_usdt = float(c.get("walletBalance") or 0)

        funding = _get("/v5/asset/transfer/query-account-coins-balance", {"accountType": "FUND"})
        funding_usdt = funding_btc = 0.0
        for c in funding.get("balance", []):
            if c.get("coin") == "USDT":
                funding_usdt = float(c.get("walletBalance") or 0)
            elif c.get("coin") == "BTC":
                funding_btc = float(c.get("walletBalance") or 0)

        return dict(unified_usdt=unified_usdt, funding_usdt=funding_usdt, funding_btc=funding_btc)
    except (requests.exceptions.RequestException, RuntimeError) as e:
        print(f"[bybit_balance] full balance check failed (non-fatal): {e}")
        return None


def get_coin_balance(coin: str) -> float | None:
    """Real mainnet balance of a given coin (e.g. 'BTC'), summed across
    Unified + Funding wallets - added 2026-09-15 so exit alerts can check
    "does the user actually hold this coin at all" before suggesting a
    sell, since the bot's own testnet_state (paper positions) can diverge
    from what the user actually bought on past entry alerts (they may not
    have acted on every signal, or already sold manually outside the
    bot). Returns None (not 0) if credentials are missing or the call
    fails - callers must treat None as 'unknown, don't suppress the
    alert', never as 'confirmed zero', so a transient API hiccup never
    silently hides a real actionable signal."""
    if not API_KEY or not API_SECRET:
        return None
    try:
        total = 0.0
        unified = _get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": coin})
        for account in unified.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == coin:
                    total += float(c.get("walletBalance") or 0)

        funding = _get("/v5/asset/transfer/query-account-coins-balance", {"accountType": "FUND"})
        for c in funding.get("balance", []):
            if c.get("coin") == coin:
                total += float(c.get("walletBalance") or 0)

        return total
    except (requests.exceptions.RequestException, RuntimeError) as e:
        print(f"[bybit_balance] coin balance check failed for {coin} (non-fatal): {e}")
        return None


if __name__ == "__main__":
    full = get_full_balance()
    if full is None:
        print("BYBIT_API_KEY/BYBIT_API_SECRET not set, or fetch failed")
    else:
        print(f"Unified (tradeable): {full['unified_usdt']} USDT")
        print(f"Funding (needs transfer to trade): {full['funding_usdt']} USDT, {full['funding_btc']} BTC")
