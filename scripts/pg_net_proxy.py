"""
Shared "call an external HTTP API from inside Postgres" helper (2026-09-22).

Why this exists: GitHub Actions runners get a CloudFront geo-block (403)
from api.bybit.com - confirmed for the signed balance endpoints in
September 2026, and (found 2026-09-21) it hits the PUBLIC market endpoints
just as hard: the `ohlcv` table had not gained a single candle since the
Supabase migration on 2026-09-15, because fetch_bybit_ohlcv.py's direct
call fails on every CI run. Supabase Postgres runs in one FIXED region
(this project's AWS eu-west-1, confirmed not blocked) regardless of who
calls it, so routing the HTTP request through pg_net inside the database
is the one transport that is known to work from CI. An Edge Function proxy
was tried first and did NOT work - Edge Functions route via anycast close
to the CALLER, so a GitHub-Actions-triggered call still landed on a
blocked region (see project_bybit_geoblock_proxy memory).

The two-phase enqueue/poll shape below is not an optimization, it is
required: pg_net's background worker only sees a queued request once the
enqueuing transaction COMMITS, so a function that enqueues and then waits
for its own result before returning can never see it complete. Phase 1
(SELECT <fn>(...)) is committed immediately; phase 2 polls
net._http_response in separate transactions.
"""
import json
import time

from db import get_connection

_POLL_INTERVAL_S = 0.3
_DEFAULT_TIMEOUT_S = 25


def fetch_json(sql_function: str, path: str, params: dict, timeout_s: float = _DEFAULT_TIMEOUT_S,
               conn=None) -> dict:
    """Runs `SELECT <sql_function>(path, params)` to enqueue a GET through
    pg_net, waits for the response row, and returns the parsed JSON body.

    `sql_function` is the SQL-side wrapper that builds the actual URL:
    `bybit_wallet_balance_start` (signs the request with the Vault-held
    API key) or `bybit_public_get_start` (unauthenticated market data).
    Pass an existing `conn` to reuse a connection; otherwise one is opened
    and closed here. Raises RuntimeError on timeout or a non-200 status -
    callers decide whether that is fatal."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {sql_function}(%s, %s)", (path, json.dumps(params)))
            request_id = cur.fetchone()[0]
        conn.commit()  # pg_net's worker only picks the request up after this commit

        deadline = time.monotonic() + timeout_s
        status = body_text = None
        while time.monotonic() < deadline:
            with conn.cursor() as cur:
                cur.execute("SELECT status_code, content FROM net._http_response WHERE id = %s", (request_id,))
                row = cur.fetchone()
            conn.commit()  # each poll is its own transaction - see module docstring
            if row and row[0] is not None:
                status, body_text = row
                break
            time.sleep(_POLL_INTERVAL_S)
    finally:
        if own_conn:
            conn.close()

    if status is None:
        raise RuntimeError(f"timeout waiting for pg_net response (fn={sql_function}, path={path})")
    if status != 200:
        raise RuntimeError(f"HTTP {status} via pg_net (fn={sql_function}, path={path}): {str(body_text)[:300]}")
    return json.loads(body_text)
