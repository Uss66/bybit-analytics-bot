CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Historical OHLCV candles (spot market)
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
SELECT create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts ON ohlcv (symbol, ts DESC);

-- Intraday OHLCV (interval < 1h, e.g. 15-minute bars) - kept SEPARATE from
-- `ohlcv` (2026-09-14) because that table's PK is (symbol, ts) only: fetching
-- a shorter interval for the same symbol into it would silently overwrite
-- the hourly candles the live testnet_trader.py bot depends on (a 15-min
-- bar's close at an hour boundary differs from that hour's true close).
-- Added for the Elliott Wave high-frequency re-test (see
-- project_elliott_wave_analysis memory) - not used by any active strategy.
CREATE TABLE IF NOT EXISTS ohlcv_intraday (
    symbol       TEXT NOT NULL,
    interval_min INT NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION,
    turnover     DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval_min, ts)
);
SELECT create_hypertable('ohlcv_intraday', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_intraday_symbol_ts ON ohlcv_intraday (symbol, interval_min, ts DESC);

-- News / social / event items (GDELT, RSS, later Twitter/Truth Social)
CREATE TABLE IF NOT EXISTS events (
    id         BIGSERIAL,
    source     TEXT NOT NULL,        -- 'gdelt', 'rss:cointelegraph', 'rss:coindesk', 'twitter', ...
    source_id  TEXT,                 -- external id / URL, used for dedup
    ts         TIMESTAMPTZ NOT NULL,
    author     TEXT,                 -- person/handle for social posts
    title      TEXT,
    content    TEXT,
    url        TEXT,
    category   TEXT,                 -- filled in by classification step
    sentiment  DOUBLE PRECISION,     -- filled in by classification step
    raw        JSONB,                -- original payload for reprocessing
    PRIMARY KEY (id, ts)
);
SELECT create_hypertable('events', 'ts', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS uq_events_source_sourceid_ts ON events (source, source_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_category ON events (category);

-- Tracks which time-slices of a backfill job have already been fetched
-- successfully, so a restarted/interrupted backfill resumes instead of
-- starting over from scratch.
CREATE TABLE IF NOT EXISTS fetch_log (
    source     TEXT NOT NULL,        -- 'gdelt', etc.
    period_key TEXT NOT NULL,        -- e.g. day as 'YYYY-MM-DD'
    status     TEXT NOT NULL,        -- 'success' or 'failed'
    item_count INT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, period_key)
);

-- Resume cursor for the Telegram channel scraper (t.me/s/<channel> paginates
-- backwards via ?before=<message_id>; this remembers where a channel's
-- backfill left off so a rerun continues deeper into history instead of
-- restarting from the latest message).
CREATE TABLE IF NOT EXISTS telegram_cursor (
    channel       TEXT PRIMARY KEY,
    before_cursor BIGINT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Perpetual futures funding rate and open interest (category=linear on
-- Bybit) - market-positioning signals, distinct from the spot ohlcv table.
CREATE TABLE IF NOT EXISTS funding_rate (
    symbol       TEXT NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    funding_rate DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('funding_rate', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol        TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    open_interest DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('open_interest', 'ts', if_not_exists => TRUE);

-- Traditional-market daily closes (S&P500, Nasdaq, gold, DXY, VIX, 10Y
-- yield) via Yahoo Finance's free public chart API - for correlating crypto
-- moves against broader macro risk-on/risk-off regimes.
CREATE TABLE IF NOT EXISTS traditional_markets (
    ticker TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    close  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ticker, ts)
);
SELECT create_hypertable('traditional_markets', 'ts', if_not_exists => TRUE);

-- Crypto Fear & Greed Index (alternative.me, free, daily, no key needed).
CREATE TABLE IF NOT EXISTS fear_greed_index (
    ts             TIMESTAMPTZ NOT NULL PRIMARY KEY,
    value          INT NOT NULL,
    classification TEXT
);
SELECT create_hypertable('fear_greed_index', 'ts', if_not_exists => TRUE);

-- Per-event forward returns computed by scripts/event_study.py, so
-- aggregate correlation stats can be recomputed cheaply without re-joining
-- events against ohlcv every time.
CREATE TABLE IF NOT EXISTS event_returns (
    event_id      BIGINT NOT NULL,
    event_ts      TIMESTAMPTZ NOT NULL,
    symbol        TEXT NOT NULL,
    horizon_hours INT NOT NULL,
    fwd_return    DOUBLE PRECISION,
    PRIMARY KEY (event_id, event_ts, symbol, horizon_hours)
);

-- Polymarket prediction-market implied probabilities (free public Gamma API
-- for market discovery + CLOB API for hourly price history, no key needed).
-- series_key groups a chain of successive short-lived markets (e.g. one per
-- FOMC meeting) into a single logical time series, since Polymarket doesn't
-- keep one market open for 2 years - each meeting gets its own market.
CREATE TABLE IF NOT EXISTS polymarket_odds (
    series_key   TEXT NOT NULL,        -- e.g. 'fed_rate_cut_25bps'
    ts           TIMESTAMPTZ NOT NULL,
    probability  DOUBLE PRECISION NOT NULL,  -- 0-1, implied by outcome price
    market_slug  TEXT NOT NULL,        -- which underlying market this point came from
    question     TEXT,
    PRIMARY KEY (series_key, ts, market_slug)
);
SELECT create_hypertable('polymarket_odds', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_polymarket_odds_series_ts ON polymarket_odds (series_key, ts DESC);

-- Deribit DVOL - crypto's equivalent of the VIX, options-implied
-- volatility index, via CryptoDataDownload's free daily CSV mirror (no key,
-- history back to 2021-03-24). A genuinely different signal type from the
-- spot-based Fear & Greed Index (options market pricing, not spot sentiment).
CREATE TABLE IF NOT EXISTS deribit_dvol (
    symbol TEXT NOT NULL,
    ts     TIMESTAMPTZ NOT NULL,
    open   DOUBLE PRECISION,
    high   DOUBLE PRECISION,
    low    DOUBLE PRECISION,
    close  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('deribit_dvol', 'ts', if_not_exists => TRUE);

-- Aggregate USD-pegged stablecoin market cap (DefiLlama, free, no key,
-- daily, history back to 2017) - proxy for new liquidity entering crypto.
CREATE TABLE IF NOT EXISTS defillama_stablecoins (
    ts        TIMESTAMPTZ NOT NULL PRIMARY KEY,
    total_usd DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('defillama_stablecoins', 'ts', if_not_exists => TRUE);

-- Testnet paper-trading execution state (2026-08-28) - tracks the live
-- position state machine per symbol so the hourly-scheduled trader script
-- (scripts/testnet_trader.py) can resume correctly across separate runs.
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

-- Every hourly tick gets a row here, even when no trade happens - the audit
-- trail for "why did/didn't the bot act at time X".
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

-- Elliott Wave analysis (2026-09-12): ZigZag pivots detected on daily-
-- resampled closes, plus rule-based 5/3-wave sequence candidates built on
-- top of them, feeding an LLM-driven pattern analysis + strategy step.
CREATE TABLE IF NOT EXISTS elliott_pivots (
    symbol        TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    pivot_type    TEXT NOT NULL,   -- 'high' or 'low'
    threshold_pct DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION,  -- volume on the pivot candle, backfilled from ohlcv
    confirmed_ts  TIMESTAMPTZ,       -- when the threshold-reversal was actually observed
                                     -- (added 2026-09-12 to fix a repainting/look-ahead
                                     -- bug - any signal using a pivot must enter no
                                     -- earlier than confirmed_ts, never ts itself)
    PRIMARY KEY (symbol, ts, threshold_pct)
);
SELECT create_hypertable('elliott_pivots', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_elliott_pivots_symbol_ts ON elliott_pivots (symbol, ts);

-- volume_* columns added 2026-09-12 per user request: volume behavior at
-- wave completion (climax or fading volume into a wave's end) is a classic
-- Elliott Wave confirmation/divergence signal, so it's captured alongside
-- each wave's price structure rather than treated as a separate feature.
CREATE TABLE IF NOT EXISTS elliott_waves (
    id                    BIGSERIAL PRIMARY KEY,
    symbol                TEXT NOT NULL,
    threshold_pct         DOUBLE PRECISION NOT NULL,
    wave_label            TEXT NOT NULL,   -- '1','2','3','4','5','A','B','C'
    sequence_id           BIGINT NOT NULL, -- groups one full 5-wave or 3-wave count together
    degree                TEXT,
    start_ts              TIMESTAMPTZ NOT NULL,
    end_ts                TIMESTAMPTZ NOT NULL,
    start_price           DOUBLE PRECISION NOT NULL,
    end_price             DOUBLE PRECISION NOT NULL,
    direction             TEXT NOT NULL,   -- 'up' or 'down'
    rule_valid            BOOLEAN NOT NULL DEFAULT TRUE,
    avg_volume            DOUBLE PRECISION,  -- mean daily volume during the wave
    end_volume            DOUBLE PRECISION,  -- volume on the completing (pivot) candle
    volume_vs_prior_wave  DOUBLE PRECISION,  -- avg_volume / previous wave's avg_volume in the same sequence
    volume_trend          TEXT,              -- 'rising' / 'falling' / 'flat' across the wave's candles
    notes                 TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_elliott_waves_symbol ON elliott_waves (symbol, start_ts);
CREATE INDEX IF NOT EXISTS idx_elliott_waves_sequence ON elliott_waves (sequence_id);

-- Curated list of key persons to track across social sources (future phase)
CREATE TABLE IF NOT EXISTS tracked_persons (
    id                 SERIAL PRIMARY KEY,
    name               TEXT NOT NULL,
    role               TEXT,
    twitter_handle     TEXT,
    truth_social_handle TEXT,
    notes              TEXT
);
