-- ============================================================
--  AUTONOMOUS TRADING SETUP (2026-09-22)
--  Run this ONCE in the Supabase SQL Editor, by hand.
--
--  This is the only code in the project able to move money, and it is
--  deliberately not applied by any script: creating it is the user's
--  decision, made once, with eyes open. Everything Claude runs
--  automatically stays read-only.
--
--  Order of operations:
--    1. Create the Bybit key first (bybit.com -> API Management ->
--       Create New Key -> System-generated):
--         - Read-Write
--         - "Only IPs with permissions granted": <IP скрыт>
--           (the Supabase Postgres egress IP - every order leaves from
--           there, not from your PC and not from GitHub's runners)
--         - Permissions: SPOT -> Trade  ONLY
--         - Withdrawal: LEAVE UNCHECKED. With it unchecked, the worst a
--           leaked key can do is trade badly; it can never move coins off
--           the account.
--    2. Paste the key/secret into step A below and run the file.
--       The secret is shown by Bybit exactly once - it goes straight from
--       that screen into Vault. Do not paste it into chat, a file, or .env.
-- ============================================================


-- ---------- A. the trading credentials, in Vault only ----------
-- Replace the two placeholders, run, then clear your editor buffer.

SELECT vault.create_secret('PASTE_API_KEY_HERE',    'bybit_trade_api_key',    'Spot trade only, no withdrawal, IP-bound');
SELECT vault.create_secret('PASTE_API_SECRET_HERE', 'bybit_trade_api_secret', 'Spot trade only, no withdrawal, IP-bound');


-- ---------- B. the signed POST path ----------
-- Same pg_net-from-eu-west-1 transport as every other Bybit call in this
-- project (GitHub Actions runners are geo-blocked by Bybit's CloudFront).
--
-- It signs with the SEPARATE trading secrets above, never the read-only
-- pair the rest of the project uses: the read path keeps working with a
-- key that cannot trade, even if this one is compromised.
--
-- Bybit signs a POST over the EXACT request body, so the body is hashed
-- as `p_body::text` and handed to pg_net as the same jsonb. If those ever
-- diverge Bybit answers retCode 10004 ("error sign"), which is
-- distinguishable from a permissions problem (10005) and from a rejected
-- order (a non-zero retCode with a human-readable message).

CREATE OR REPLACE FUNCTION bybit_signed_post_start(p_path text, p_body jsonb)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_api_key     text;
    v_api_secret  text;
    v_recv_window text := '20000';
    v_timestamp   text;
    v_body_text   text;
    v_sign        text;
    v_request_id  bigint;
BEGIN
    SELECT decrypted_secret INTO v_api_key FROM vault.decrypted_secrets WHERE name = 'bybit_trade_api_key';
    SELECT decrypted_secret INTO v_api_secret FROM vault.decrypted_secrets WHERE name = 'bybit_trade_api_secret';
    IF v_api_key IS NULL OR v_api_secret IS NULL THEN
        RAISE EXCEPTION 'trading secrets missing from Vault (bybit_trade_api_key / bybit_trade_api_secret)';
    END IF;

    v_timestamp := (extract(epoch FROM clock_timestamp()) * 1000)::bigint::text;
    v_body_text := p_body::text;
    v_sign := encode(hmac(v_timestamp || v_api_key || v_recv_window || v_body_text, v_api_secret, 'sha256'), 'hex');

    SELECT net.http_post(
        url := 'https://api.bybit.com' || p_path,
        body := p_body,
        headers := jsonb_build_object(
            'X-BAPI-API-KEY', v_api_key,
            'X-BAPI-TIMESTAMP', v_timestamp,
            'X-BAPI-RECV-WINDOW', v_recv_window,
            'X-BAPI-SIGN', v_sign,
            'Content-Type', 'application/json'
        ),
        timeout_milliseconds := 20000
    ) INTO v_request_id;

    RETURN v_request_id;
END;
$$;

REVOKE ALL ON FUNCTION bybit_signed_post_start(text, jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION bybit_signed_post_start(text, jsonb) TO postgres;


-- ---------- C. the limits the bot cannot exceed ----------
-- Single-row config read at the start of every tick. Kept in the DATABASE
-- rather than in code so that stopping the bot never requires a deploy:
-- flip `enabled` here (or send /stop in Telegram) and the very next tick
-- places no orders.
--
-- max_capital_usdt is the hard ceiling on what the bot may have deployed
-- at any moment across all positions it opened. It is checked against the
-- real ledger before every buy, so it cannot be out-drifted by a failed
-- state write. Start small, watch it behave on live money, raise later by
-- changing one number.

CREATE TABLE IF NOT EXISTS trading_config (
    id                   INT PRIMARY KEY DEFAULT 1,
    enabled              BOOLEAN NOT NULL DEFAULT false,
    max_capital_usdt     DOUBLE PRECISION NOT NULL DEFAULT 300,
    max_order_usdt       DOUBLE PRECISION NOT NULL DEFAULT 150,
    max_daily_loss_usdt  DOUBLE PRECISION NOT NULL DEFAULT 30,
    paused_reason        TEXT,
    paused_at            TIMESTAMPTZ,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT trading_config_single_row CHECK (id = 1)
);

INSERT INTO trading_config (id, enabled, max_capital_usdt, max_order_usdt, max_daily_loss_usdt)
VALUES (1, false, 300, 150, 30)
ON CONFLICT (id) DO NOTHING;


-- ---------- D. the audit trail ----------
-- Every order the bot sends, with the idempotency key it used, what Bybit
-- answered and why the bot thought it should trade. Written BEFORE the
-- request goes out, so an order that vanishes mid-flight still leaves a
-- record to reconcile against - the failure mode that made the paper bot
-- unreliable is not acceptable once the money is real.

CREATE TABLE IF NOT EXISTS bot_orders (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol         TEXT NOT NULL,
    intent         TEXT NOT NULL,          -- buy | add | sell_signal | sell_stop | stop_order
    order_link_id  TEXT NOT NULL UNIQUE,   -- idempotency key: Bybit rejects a duplicate
    request        JSONB NOT NULL,
    status         TEXT NOT NULL DEFAULT 'sent',  -- sent | accepted | rejected | failed
    order_id       TEXT,
    response       JSONB,
    reason         TEXT
);
ALTER TABLE bot_orders ADD COLUMN IF NOT EXISTS pnl_usdt DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_bot_orders_ts ON bot_orders (ts DESC);

-- How much of the cap the bot itself has deployed, per symbol. Counted
-- separately from the ledger on purpose: the ledger holds the user's own
-- hand-bought BTC/BNB too, and charging those against the bot's ceiling
-- would freeze it out of trading entirely at any sane limit.
ALTER TABLE advisor_state ADD COLUMN IF NOT EXISTS bot_invested_usdt DOUBLE PRECISION NOT NULL DEFAULT 0;

-- The resting stop-loss order currently protecting this position on the
-- EXCHANGE. Remembered so it can be cancelled and re-placed when a
-- dip-rebuy add moves the average cost, and cancelled on exit so a stale
-- stop cannot later sell coins the bot no longer manages.
ALTER TABLE advisor_state ADD COLUMN IF NOT EXISTS stop_order_link_id TEXT;
ALTER TABLE advisor_state ADD COLUMN IF NOT EXISTS stop_trigger_price DOUBLE PRECISION;

-- The size of the entry that opened this position, so every dip-rebuy
-- tranche matches it - the sizing dip_rebuy_study.py actually validated.
ALTER TABLE advisor_state ADD COLUMN IF NOT EXISTS entry_stake_usdt DOUBLE PRECISION;


-- ---------- E. verify ----------
-- After running everything above, this should return one row with
-- enabled=false. The bot stays OFF until you flip it deliberately.
SELECT enabled, max_capital_usdt, max_order_usdt, max_daily_loss_usdt FROM trading_config;
