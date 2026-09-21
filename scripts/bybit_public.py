"""
Transport for Bybit's PUBLIC market endpoints (2026-09-22).

Tries a direct HTTPS call first (works from a normal machine, and is the
fastest path), and falls back to routing the same GET through Postgres'
pg_net (see pg_net_proxy.py) when the direct call is refused. That
fallback is what makes the live pipeline work from GitHub Actions at all:
Bybit's CloudFront geo-blocks the runners' US IPs with a 403, and this was
silently killing the OHLCV refresh - the bot kept evaluating its stop-loss
against a six-day-old price (found 2026-09-21).

The transport decision is made ONCE per process and then reused, so a
blocked environment pays for exactly one refused request instead of one
per symbol. Override with BYBIT_TRANSPORT=direct|proxy|auto (default
auto) - `proxy` is useful to test the CI path from a machine that is not
itself blocked.

Retries are deliberately NOT applied to a 403: a geo-block is not a
transient error, and the old code's 5x exponential backoff turned every
blocked call into a ~2-minute stall (7 symbols of that came close to the
workflow's own 10-minute timeout).
"""
import os

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pg_net_proxy import fetch_json

BASE_URL = "https://api.bybit.com"
PUBLIC_SQL_FN = "bybit_public_get_start"

SESSION = requests.Session()

_transport = None  # "direct" | "proxy", decided lazily on the first call


class BybitBlocked(RuntimeError):
    """Bybit refused the request for reasons a retry cannot fix (geo-block)."""


class TransientError(RuntimeError):
    """Network hiccup / 5xx / rate limit - worth retrying."""


def _direct_get(path: str, params: dict) -> dict:
    try:
        resp = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=20)
    except requests.RequestException as e:
        raise TransientError(str(e)) from e
    if resp.status_code in (403, 451):
        raise BybitBlocked(f"HTTP {resp.status_code} from {path} - geo-block, not retryable")
    if resp.status_code >= 500 or resp.status_code == 429:
        raise TransientError(f"HTTP {resp.status_code} from {path}")
    resp.raise_for_status()
    return resp.json()


def _proxy_get(path: str, params: dict, conn=None) -> dict:
    # every value must be a string - pg_net builds the query string from jsonb
    return fetch_json(PUBLIC_SQL_FN, path, {k: str(v) for k, v in params.items()}, conn=conn)


@retry(retry=retry_if_exception_type(TransientError), stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=1, max=15), reraise=True)
def get(path: str, params: dict, conn=None) -> dict:
    """GET a public Bybit v5 endpoint and return its parsed body, whichever
    transport this environment can actually use. Raises RuntimeError if the
    API itself reports a non-zero retCode."""
    global _transport

    forced = (os.getenv("BYBIT_TRANSPORT") or "auto").lower()
    if _transport is None:
        _transport = forced if forced in ("direct", "proxy") else "direct"

    if _transport == "direct":
        try:
            body = _direct_get(path, params)
        except BybitBlocked as e:
            if forced == "direct":
                raise
            print(f"[bybit_public] direct call blocked ({e}) - switching this process to the pg_net proxy")
            _transport = "proxy"
            body = _proxy_get(path, params, conn=conn)
    else:
        body = _proxy_get(path, params, conn=conn)

    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error on GET {path}: {body.get('retCode')} {body.get('retMsg')}")
    return body["result"]


def transport_in_use() -> str:
    """Which transport this process settled on - for logging, so a run's
    output says whether it went direct or had to tunnel through Postgres."""
    return _transport or "undecided"
