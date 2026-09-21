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
  4. The key/secret live in Supabase Vault (`bybit_api_key`/
     `bybit_api_secret`), read by the `bybit_wallet_balance()` Postgres
     function - NOT in this process's own .env. This process just needs
     its normal POSTGRES_* connection (already required by every other
     script) - see the 2026-09-21 note below for why.

If the DB call fails, get_usdt_balance() returns None (not an error) so
the alerting code can gracefully omit the balance line rather than crash
the hourly tick.

Usage:
    from bybit_balance import get_usdt_balance
    bal = get_usdt_balance()  # float or None
"""
import json
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from db import get_connection

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 2026-09-21: calls no longer go straight to Bybit from here - GitHub
# Actions runners got a confirmed 403 from Bybit's CloudFront ("configured
# to block access from your country"), a geo-block unrelated to
# credentials/signature and not fixable by an IP whitelist. First attempt
# fixed it via a Supabase Edge Function proxy (Supabase's DB infra tested
# as NOT blocked) - but Edge Functions turned out to execute via anycast
# routing close to the CALLER, so a GitHub-Actions-triggered call still
# got routed through a blocked US region even though the exact same
# function succeeded when called from elsewhere. Postgres itself (unlike
# Edge Functions) runs in one FIXED region (this project's AWS eu-west-1)
# regardless of caller, confirmed not blocked - so the actual signed Bybit
# call now happens inside a Postgres function (bybit_wallet_balance(),
# using pgcrypto's hmac() + pg_net) that this module invokes over the
# same DB connection every other script already uses. See
# project_bybit_geoblock_proxy memory for the full story of both attempts.


def _get(path: str, params: dict) -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT bybit_wallet_balance(%s, %s)", (path, json.dumps(params)))
            result = cur.fetchone()[0]
    finally:
        conn.close()
    if result.get("status") != 200:
        raise RuntimeError(f"Bybit API error on GET {path} (via pg_net): {result}")
    body = result["body"]
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error on GET {path}: {body}")
    return body["result"]


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
    try:
        result = _get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        for account in result.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == "USDT":
                    return float(c.get("walletBalance") or 0)
        return 0.0
    except (psycopg2.Error, RuntimeError) as e:
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
    except (psycopg2.Error, RuntimeError) as e:
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
    except (psycopg2.Error, RuntimeError) as e:
        print(f"[bybit_balance] coin balance check failed for {coin} (non-fatal): {e}")
        return None


if __name__ == "__main__":
    full = get_full_balance()
    if full is None:
        print("bybit_wallet_balance() call failed - check the DB connection and Vault secrets")
    else:
        print(f"Unified (tradeable): {full['unified_usdt']} USDT")
        print(f"Funding (needs transfer to trade): {full['funding_usdt']} USDT, {full['funding_btc']} BTC")
