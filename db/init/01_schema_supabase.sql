-- Supabase-compatible schema (2026-09-15) - subset of db/init/01_schema.sql
-- for the GitHub-Actions-hosted live bot (see project_oracle_cloud_hosting /
-- README "GitHub Actions + Supabase" for why: Oracle Cloud Always Free
-- instance creation kept hitting regional capacity shortages, so the bot
-- moved to ephemeral GitHub Actions runs + a managed Postgres instead of a
-- persistent VM).
--
-- Deliberately NOT the timescaledb extension / create_hypertable() calls -
-- Supabase doesn't offer the timescaledb extension, but nothing in this
-- codebase actually depends on hypertable-specific SQL (no time_bucket(),
-- no continuous aggregates - verified by grep before this migration).
-- Hypertables are a transparent performance/partitioning layer over
-- standard SQL; plain tables work identically for every query these
-- scripts run.
--
-- Deliberately a SUBSET of tables - only what refresh_live_data.py and
-- testnet_trader.py actually read/write for the live signal (see
-- refresh_live_data.py's own docstring for the exact rule->table mapping).
-- The full historical/research dataset (5yr backtests, Elliott Wave study
-- tables, whale-alert tables, etc.) stays on the local Postgres only -
-- this Supabase instance is live-state-only, not a backtest replica.

CREATE TABLE IF NOT EXISTS ohlcv (
    symbol   TEXT NOT NULL,
    ts       TIMESTAMPTZ NOT NULL,
    open     DOUBLE PRECISION NOT NULL,
    high     DOUBLE PRECISION NOT NULL,
    low      DOUBLE PRECISION NOT NULL,
    close    DOUBLE PRECISION NOT NULL,
    volume   DOUBLE PRECISION,
    turnover DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts ON ohlcv (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS events (
    id         BIGSERIAL,
    source     TEXT NOT NULL,
    source_id  TEXT,
    ts         TIMESTAMPTZ NOT NULL,
    author     TEXT,
    title      TEXT,
    content    TEXT,
    url        TEXT,
    category   TEXT,
    sentiment  DOUBLE PRECISION,
    raw        JSONB,
    PRIMARY KEY (id, ts)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_events_source_sourceid_ts ON events (source, source_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_category ON events (category);

CREATE TABLE IF NOT EXISTS traditional_markets (
    ticker TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    close  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ticker, ts)
);

CREATE TABLE IF NOT EXISTS fear_greed_index (
    ts             TIMESTAMPTZ NOT NULL PRIMARY KEY,
    value          INT NOT NULL,
    classification TEXT
);

CREATE TABLE IF NOT EXISTS deribit_dvol (
    symbol TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    open   DOUBLE PRECISION,
    high   DOUBLE PRECISION,
    low    DOUBLE PRECISION,
    close  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Live paper-trading state - MUST be seeded from the local DB's current
-- values (open positions etc.), never created empty, or the bot loses
-- track of positions already open when the home-PC setup is retired.
--
-- entry_price/entry_qty now mean "average cost basis / total quantity
-- ACROSS ALL TRANCHES" (2026-09-18, dip-rebuy add-on - see
-- project_dip_rebuy_findings memory) once tranche_count > 1, not just the
-- original entry - the stop-loss and every other check already computed
-- off entry_price, so this reuses the existing column semantics rather
-- than introducing a parallel "avg_cost" field.
CREATE TABLE IF NOT EXISTS testnet_state (
    symbol             TEXT PRIMARY KEY,
    in_position        BOOLEAN NOT NULL DEFAULT FALSE,
    entry_ts           TIMESTAMPTZ,
    entry_price        DOUBLE PRECISION,
    entry_qty          DOUBLE PRECISION,
    entry_order_id     TEXT,
    consecutive_losses INT NOT NULL DEFAULT 0,
    cooldown_until_ts  TIMESTAMPTZ,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    tranche_count      INT NOT NULL DEFAULT 0,
    running_low        DOUBLE PRECISION
);
-- Idempotent add for a table that may already exist from before this
-- feature (ALTER instead of relying on CREATE TABLE, which no-ops once
-- the table is already there).
ALTER TABLE testnet_state ADD COLUMN IF NOT EXISTS tranche_count INT NOT NULL DEFAULT 0;
ALTER TABLE testnet_state ADD COLUMN IF NOT EXISTS running_low DOUBLE PRECISION;

-- One row per tranche (the original entry = tranche_num 0, each dip-rebuy
-- add increments it) - an audit trail so "why did the average cost
-- change" is always answerable later, matching this project's existing
-- practice (see testnet_run_log's own docstring).
CREATE TABLE IF NOT EXISTS testnet_position_tranches (
    id            BIGSERIAL PRIMARY KEY,
    symbol        TEXT NOT NULL,
    tranche_num   INT NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    qty           DOUBLE PRECISION NOT NULL,
    order_id      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_testnet_position_tranches_symbol ON testnet_position_tranches (symbol, created_at);

CREATE TABLE IF NOT EXISTS testnet_trades (
    id             BIGSERIAL PRIMARY KEY,
    symbol         TEXT NOT NULL,
    entry_ts       TIMESTAMPTZ NOT NULL,
    exit_ts        TIMESTAMPTZ NOT NULL,
    entry_price    DOUBLE PRECISION NOT NULL,
    exit_price     DOUBLE PRECISION NOT NULL,
    qty            DOUBLE PRECISION NOT NULL,
    gross_ret      DOUBLE PRECISION NOT NULL,
    net_ret        DOUBLE PRECISION NOT NULL,
    exit_reason    TEXT NOT NULL,
    entry_order_id TEXT,
    exit_order_id  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS testnet_run_log (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol          TEXT NOT NULL,
    score           DOUBLE PRECISION,
    is_long_target  BOOLEAN,
    in_position     BOOLEAN,
    action_taken    TEXT NOT NULL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_testnet_run_log_ts ON testnet_run_log (ts DESC);

-- On-demand "куда вложить" Telegram advisor (2026-09-18) - single-row
-- cursor remembering the last Telegram update_id already handled, so the
-- polling workflow (scripts/telegram_command_handler.py, every ~30min)
-- never re-answers an old message. See project_telegram_command_advisor
-- memory for the full design.
CREATE TABLE IF NOT EXISTS telegram_command_cursor (
    id              INT PRIMARY KEY DEFAULT 1,
    last_update_id  BIGINT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO telegram_command_cursor (id, last_update_id)
VALUES (1, 0) ON CONFLICT (id) DO NOTHING;

-- Bybit real-balance proxy (2026-09-21) - Supabase-only, not part of the
-- local Docker schema. GitHub Actions runners got a confirmed geo-block
-- (403) from Bybit's CloudFront when calling Bybit directly; a first fix
-- via a Supabase Edge Function proxy still got geo-blocked (Edge
-- Functions route via anycast close to the CALLER, so a GitHub-Actions
-- call still landed on a blocked region). Postgres itself runs in one
-- FIXED region regardless of caller, confirmed not blocked - so the
-- actual signed Bybit call happens here, via pg_net + pgcrypto's hmac().
-- scripts/bybit_balance.py calls this function over the normal DB
-- connection instead of hitting Bybit directly. See
-- project_bybit_geoblock_proxy memory for the full story.
--
-- IMPORTANT (found 2026-09-21, after a first version that tried to poll
-- net._http_response for the result INSIDE this same function/
-- transaction always timed out, even at 30s): pg_net's background worker
-- only sees a queued request once the transaction that enqueued it
-- COMMITS. A function that enqueues, then polls for the result before
-- returning (i.e. before its own transaction can commit), can never see
-- its own request complete - a self-deadlock, not a slowness problem.
-- Fix: this function ONLY enqueues and returns the request_id
-- immediately; the polling happens in Python (bybit_balance.py), as
-- separate, individually-committed queries against net._http_response -
-- exactly mirroring how the earlier ad-hoc two-step SQL Editor tests
-- (one query to enqueue, a later separate query to check the result)
-- worked cleanly.
--
-- Requires (run once, in the SQL Editor, not applied automatically by
-- this file - it references Vault secrets that must exist first):
--   SELECT vault.create_secret('<real key>', 'bybit_api_key', '...');
--   SELECT vault.create_secret('<real secret>', 'bybit_api_secret', '...');
CREATE OR REPLACE FUNCTION bybit_wallet_balance_start(p_path text, p_params jsonb)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_api_key      text;
    v_api_secret   text;
    v_recv_window  text := '20000';
    v_timestamp    text;
    v_query        text;
    v_sign         text;
    v_url          text;
    v_request_id   bigint;
BEGIN
    SELECT decrypted_secret INTO v_api_key FROM vault.decrypted_secrets WHERE name = 'bybit_api_key';
    SELECT decrypted_secret INTO v_api_secret FROM vault.decrypted_secrets WHERE name = 'bybit_api_secret';
    v_timestamp := (extract(epoch FROM clock_timestamp()) * 1000)::bigint::text;

    SELECT string_agg(key || '=' || value, '&' ORDER BY key)
        INTO v_query
    FROM jsonb_each_text(p_params);

    v_sign := encode(
        hmac(v_timestamp || v_api_key || v_recv_window || coalesce(v_query, ''), v_api_secret, 'sha256'),
        'hex'
    );
    v_url := 'https://api.bybit.com' || p_path
        || CASE WHEN v_query IS NOT NULL AND v_query != '' THEN '?' || v_query ELSE '' END;

    SELECT net.http_get(
        url := v_url,
        headers := jsonb_build_object(
            'X-BAPI-API-KEY', v_api_key,
            'X-BAPI-TIMESTAMP', v_timestamp,
            'X-BAPI-RECV-WINDOW', v_recv_window,
            'X-BAPI-SIGN', v_sign
        )
    ) INTO v_request_id;

    RETURN v_request_id;
END;
$$;

REVOKE ALL ON FUNCTION bybit_wallet_balance_start(text, jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION bybit_wallet_balance_start(text, jsonb) TO postgres;
