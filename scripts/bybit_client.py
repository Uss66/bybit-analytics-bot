"""
Minimal Bybit v5 authenticated REST client for paper trading only - no
mainnet/real-money order-placement code path exists here on purpose.

Bybit has TWO separate paper-money systems, easy to conflate (see
https://bybit-exchange.github.io/docs/v5/demo) - this client supports
either via BYBIT_PAPER_MODE, defaulting to the one that keeps full
separation from any real account:

  "testnet" (default) - api-testnet.bybit.com, a fully separate account you
    sign up for at testnet.bybit.com, unrelated to any mainnet account.
    Generate the API key from the site's normal API Management page - do
    NOT toggle any "Demo Trading" switch on the testnet site itself first,
    Bybit's own docs call a key generated that way "meaningless". Funds:
    Assets -> Assets Overview -> Request Test Coins (covers USDT).

  "demo" - api-demo.bybit.com, "Demo Trading" mode toggled ON INSIDE YOUR
    REAL MAINNET ACCOUNT (log into bybit.com, switch to Demo Trading, then
    generate keys from there - NOT from testnet.bybit.com; Bybit's own docs
    say creating a demo key from the testnet site is "meaningless"). Funds
    are simulated (50k USDT/USDC, 1 BTC, 1 ETH auto-granted, top-uppable via
    /v5/account/demo-apply-money) but the account itself is your real
    identity, just with trading simulated - a different privacy/separation
    tradeoff than "testnet" even though no real money moves either way.

Reads credentials from environment variables (never hardcode, never log
them): BYBIT_TESTNET_API_KEY, BYBIT_TESTNET_API_SECRET (used for both
modes - the env var name doesn't change, only which key you generate and
which BASE_URL it's sent to). Also loads a `.env` file from the project
root if present (see load_dotenv() below) - so a shell restart isn't
needed after creating/editing it, unlike OS-level environment variables.
The `.env` file itself is never read or written by Claude - the user
creates/edits it directly with their own editor.

Usage as a library:
    from bybit_client import BybitTestnetClient
    client = BybitTestnetClient()
    client.get_instrument_info("BTCUSDT")
    client.market_buy_quote("BTCUSDT", usdt_amount=100)
    client.market_sell_base("BTCUSDT", qty=0.001)
"""
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# same .env db.py already loads DB credentials from - just adding the
# Bybit keys to that one file keeps a single place to manage local secrets.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MODE = os.environ.get("BYBIT_PAPER_MODE", "testnet").lower()
BASE_URL = "https://api-demo.bybit.com" if _MODE == "demo" else "https://api-testnet.bybit.com"
# Bybit rejects requests whose signed timestamp is off from its server
# clock by more than this window (retCode 10002). Bumped from the 5000ms
# default - this machine's local clock measured ~9.5s of drift and syncing
# it needs admin rights (not something to do on the user's behalf without
# them running it themselves). 20s stays well inside Bybit's max (60s) while
# giving real slack; the underlying clock drift is still worth the user
# fixing via Windows Settings > Time & Language > Sync now, this is a
# workaround, not a real fix for the drift itself.
RECV_WINDOW = "20000"


class BybitTestnetClient:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self.api_key = api_key or os.environ.get("BYBIT_TESTNET_API_KEY")
        self.api_secret = api_secret or os.environ.get("BYBIT_TESTNET_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "Missing BYBIT_TESTNET_API_KEY / BYBIT_TESTNET_API_SECRET environment variables. "
                "Generate testnet API keys at https://testnet.bybit.com and set them as env vars "
                "(never pass them as literals in code or commit them anywhere)."
            )
        self.session = requests.Session()

    def _sign(self, payload: str, timestamp: str) -> str:
        raw = f"{timestamp}{self.api_key}{RECV_WINDOW}{payload}"
        return hmac.new(self.api_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _headers(self, payload: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": self._sign(payload, timestamp),
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict) -> dict:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        resp = self.session.get(f"{BASE_URL}{path}", params=params, headers=self._headers(query), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error on GET {path}: {data}")
        return data["result"]

    def _post(self, path: str, body: dict) -> dict:
        payload = json.dumps(body)
        resp = self.session.post(f"{BASE_URL}{path}", data=payload, headers=self._headers(payload), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error on POST {path}: {data}")
        return data["result"]

    # --- public-ish (still needs no auth really, but kept here for convenience) ---

    def get_instrument_info(self, symbol: str) -> dict:
        result = self._get("/v5/market/instruments-info", {"category": "spot", "symbol": symbol})
        items = result.get("list", [])
        if not items:
            raise RuntimeError(f"No instrument info for {symbol}")
        return items[0]

    # --- account ---

    def get_wallet_balance(self, coin: str = "USDT") -> float:
        result = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": coin})
        for account in result.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == coin:
                    return float(c.get("walletBalance") or 0)
        return 0.0

    def get_coin_balance(self, coin: str) -> float:
        result = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": coin})
        for account in result.get("list", []):
            for c in account.get("coin", []):
                if c.get("coin") == coin:
                    return float(c.get("walletBalance") or 0)
        return 0.0

    # --- orders ---

    def market_buy_quote(self, symbol: str, usdt_amount: float) -> dict:
        """Market buy spending a fixed USDT amount (marketUnit=quoteCoin)."""
        body = {
            "category": "spot",
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": str(usdt_amount),
            "marketUnit": "quoteCoin",
        }
        return self._post("/v5/order/create", body)

    def market_sell_base(self, symbol: str, qty: str) -> dict:
        """Market sell a fixed base-asset quantity (marketUnit=baseCoin, the default)."""
        body = {
            "category": "spot",
            "symbol": symbol,
            "side": "Sell",
            "orderType": "Market",
            "qty": qty,
        }
        return self._post("/v5/order/create", body)

    def get_order(self, symbol: str, order_id: str) -> dict:
        result = self._get("/v5/order/realtime", {"category": "spot", "symbol": symbol, "orderId": order_id})
        items = result.get("list", [])
        return items[0] if items else {}

    def wait_for_fill(self, symbol: str, order_id: str, attempts: int = 8, delay_s: float = 1.0) -> dict:
        """Poll order/realtime until the market order shows as Filled and
        return its aggregated avgPrice/cumExecQty. Market orders usually fill
        within a second, but this must never silently fall back to an
        estimate - a wrong fill qty on a BUY leads to trying to sell more
        than we actually hold later, which fails and leaves the position
        stuck open. Raises if the order never reaches Filled in time."""
        for _ in range(attempts):
            order = self.get_order(symbol, order_id)
            if order.get("orderStatus") == "Filled" and float(order.get("cumExecQty") or 0) > 0:
                return order
            time.sleep(delay_s)
        raise RuntimeError(f"Order {order_id} for {symbol} did not reach Filled status after "
                            f"{attempts * delay_s:.0f}s (last status: {order.get('orderStatus')!r})")
