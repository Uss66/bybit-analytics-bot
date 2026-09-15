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
CREATE TABLE IF NOT EXISTS testnet_state (
    symbol             TEXT PRIMARY KEY,
    in_position        BOOLEAN NOT NULL DEFAULT FALSE,
    entry_ts           TIMESTAMPTZ,
    entry_price        DOUBLE PRECISION,
    entry_qty          DOUBLE PRECISION,
    entry_order_id     TEXT,
    consecutive_losses INT NOT NULL DEFAULT 0,
    cooldown_until_ts  TIMESTAMPTZ,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
