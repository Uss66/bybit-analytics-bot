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
  4. The key/secret now live as Supabase Edge Function secrets
     (BYBIT_API_KEY/BYBIT_API_SECRET on the `bybit-proxy` function), NOT
     in this process's own .env - see the 2026-09-21 note below for why.
     This process instead needs BYBIT_PROXY_URL (the function's invoke
     URL) and BYBIT_PROXY_SECRET (a shared secret checked by the proxy)
     in its own .env / GitHub Secrets.

If these env vars are missing, get_usdt_balance() returns None (not an
error) so the alerting code can gracefully omit the balance line rather
than crash the hourly tick.

Usage:
    from bybit_balance import get_usdt_balance
    bal = get_usdt_balance()  # float or None
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 2026-09-21: calls no longer go straight to Bybit from here - GitHub
# Actions runners got a confirmed 403 from Bybit's CloudFront ("configured
# to block access from your country"), a geo-block that has nothing to do
# with credentials/signature and can't be fixed by an IP whitelist (GH
# Actions runner IPs aren't stable anyway). Supabase's own infra (AWS
# eu-west-1) was confirmed NOT blocked, so the actual signed Bybit call now
# happens in a Supabase Edge Function (supabase/functions/bybit-proxy) -
# this module just forwards path+params to it and unwraps the response.
# The real Bybit API key/secret now live ONLY as Supabase Edge Function
# secrets, not in this process's environment - see
# project_bybit_geoblock_proxy memory for the full story.
PROXY_URL = os.environ.get("BYBIT_PROXY_URL")
PROXY_SECRET = os.environ.get("BYBIT_PROXY_SECRET")
_warned_missing_creds = False


def _warn_missing_creds():
    """One-time diagnostic print when the proxy isn't configured - mirrors
    the 2026-09-18 finding that a silent None with zero log trace makes a
    missing-secret problem look like a Bybit API problem. Silent None is
    still the right return value (callers must treat it as 'unknown', not
    '0'), but silent should not mean invisible in the log too."""
    global _warned_missing_creds
    if not _warned_missing_creds:
        print("[bybit_balance] BYBIT_PROXY_URL/BYBIT_PROXY_SECRET not set - balance checks disabled "
              "(check the workflow's env: block, or .env locally)")
        _warned_missing_creds = True


def _get(path: str, params: dict) -> dict:
    resp = requests.post(
        PROXY_URL,
        json={"path": path, "params": params},
        headers={"X-Proxy-Secret": PROXY_SECRET},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error on GET {path} (via proxy): {data}")
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
    if not PROXY_URL or not PROXY_SECRET:
        _warn_missing_creds()
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
    if not PROXY_URL or not PROXY_SECRET:
        _warn_missing_creds()
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
    if not PROXY_URL or not PROXY_SECRET:
        _warn_missing_creds()
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
        print("BYBIT_PROXY_URL/BYBIT_PROXY_SECRET not set, or fetch failed")
    else:
        print(f"Unified (tradeable): {full['unified_usdt']} USDT")
        print(f"Funding (needs transfer to trade): {full['funding_usdt']} USDT, {full['funding_btc']} BTC")
