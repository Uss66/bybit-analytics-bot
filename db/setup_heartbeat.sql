-- ============================================================
--  HEARTBEAT: тревога, когда бот замолчал (2026-09-30)
--  Выполнить ОДИН РАЗ в Supabase SQL Editor.
--
--  Зачем. 2026-09-30 бот встал на 8 часов (кончились минуты GitHub
--  Actions), и никто об этом не узнал, пока пользователь не задал
--  посторонний вопрос. Причина дыры структурная: тревога о протухших
--  данных отправляется ИЗНУТРИ тика, поэтому «тик не запустился вовсе» —
--  единственный отказ, о котором система не способна сообщить.
--
--  Поэтому heartbeat живёт в Postgres и ходит в Telegram напрямую через
--  pg_net. Он не зависит ни от GitHub Actions, ни от Python-кода, ни от
--  Edge Functions — то есть работает именно в том сценарии, ради
--  которого написан.
--
--  Что он НЕ делает: не торгует, не трогает позиции и не выключает бота.
--  Только смотрит на время последней записи в testnet_run_log и пишет в
--  Telegram.
-- ============================================================


-- ---------- A. токен бота в Vault ----------
-- Тот же токен, что в GitHub Secrets (TELEGRAM_BOT_TOKEN) и в локальном
-- .env. Вставить значения и выполнить; после этого очистить буфер
-- редактора. В файлы репозитория токен не попадает.

SELECT vault.create_secret('PASTE_TELEGRAM_BOT_TOKEN', 'telegram_bot_token', 'Bot token for heartbeat alerts from Postgres');
SELECT vault.create_secret('PASTE_TELEGRAM_CHAT_ID',   'telegram_chat_id',   'Chat id for heartbeat alerts from Postgres');


-- ---------- B. состояние ----------
-- `alerting` нужен, чтобы отличать «молчит впервые» от «молчит уже
-- давно» и чтобы прислать отдельное сообщение о восстановлении: тревога
-- без парной отбойной телеграммы заставляет проверять руками.

CREATE TABLE IF NOT EXISTS heartbeat_state (
    id             INT PRIMARY KEY DEFAULT 1,
    alerting       BOOLEAN NOT NULL DEFAULT false,
    last_alert_ts  TIMESTAMPTZ,
    last_check_ts  TIMESTAMPTZ,
    CONSTRAINT heartbeat_single_row CHECK (id = 1)
);
INSERT INTO heartbeat_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;


-- ---------- C. сама проверка ----------
-- p_stale_minutes = 120: такт 15 минут, поэтому два часа тишины — это
-- восемь пропущенных тиков подряд. Одиночный сбой прогона (сеть, лимит
-- API, случайная ошибка) такого не даёт, а настоящая остановка даёт
-- сразу. Порог намеренно не ниже: ложная тревога обесценивает канал
-- быстрее, чем пропущенная.
-- p_remind_hours = 6: пока молчание продолжается, напоминать редко.

CREATE OR REPLACE FUNCTION heartbeat_check(p_stale_minutes int DEFAULT 120,
                                           p_remind_hours int DEFAULT 6)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_last      timestamptz;
    v_age_min   numeric;
    v_token     text;
    v_chat      text;
    v_alerting  boolean;
    v_last_alert timestamptz;
    v_msg       text;
BEGIN
    SELECT max(ts) INTO v_last FROM testnet_run_log;
    SELECT alerting, last_alert_ts INTO v_alerting, v_last_alert FROM heartbeat_state WHERE id = 1;
    UPDATE heartbeat_state SET last_check_ts = now() WHERE id = 1;

    IF v_last IS NULL THEN
        RETURN 'нет записей в testnet_run_log — проверять нечего';
    END IF;
    v_age_min := extract(epoch FROM (now() - v_last)) / 60;

    SELECT decrypted_secret INTO v_token FROM vault.decrypted_secrets WHERE name = 'telegram_bot_token';
    SELECT decrypted_secret INTO v_chat  FROM vault.decrypted_secrets WHERE name = 'telegram_chat_id';
    IF v_token IS NULL OR v_chat IS NULL THEN
        RETURN 'секреты telegram_bot_token / telegram_chat_id отсутствуют в Vault';
    END IF;

    -- ---- бот жив ----
    IF v_age_min <= p_stale_minutes THEN
        IF v_alerting THEN
            v_msg := E'✅ БОТ СНОВА РАБОТАЕТ\n\n'
                  || 'Тики возобновились, последний ' || to_char(v_last, 'DD.MM HH24:MI') || ' UTC.' || E'\n'
                  || 'Проверь /status — не пропустил ли он сигнал за время молчания.';
            PERFORM net.http_post(
                url := 'https://api.telegram.org/bot' || v_token || '/sendMessage',
                body := jsonb_build_object('chat_id', v_chat, 'text', v_msg),
                headers := jsonb_build_object('Content-Type', 'application/json'));
            UPDATE heartbeat_state SET alerting = false, last_alert_ts = now() WHERE id = 1;
            RETURN 'восстановился, отправлено уведомление';
        END IF;
        RETURN 'ok, последний тик ' || round(v_age_min) || ' мин назад';
    END IF;

    -- ---- бот молчит ----
    IF v_alerting AND v_last_alert > now() - (p_remind_hours || ' hours')::interval THEN
        RETURN 'молчит ' || round(v_age_min) || ' мин, о тревоге уже сообщено';
    END IF;

    v_msg := E'\U0001F6A8 БОТ НЕ ТИКАЕТ\n\n'
          || 'Последнее решение: ' || to_char(v_last, 'DD.MM HH24:MI') || ' UTC ('
          || round(v_age_min / 60, 1) || ' ч назад).' || E'\n\n'
          || 'Открытые позиции защищены биржевыми стоп-ордерами — они сработают '
          || 'независимо от бота. Но новые входы, докупки и выходы по сигналу сейчас НЕ происходят.'
          || E'\n\n' || 'Что проверить в первую очередь: не кончились ли минуты GitHub Actions '
          || '(подпись — прогоны падают за несколько секунд, не начав ни одного шага). '
          || 'Затем — логи последнего прогона.';
    PERFORM net.http_post(
        url := 'https://api.telegram.org/bot' || v_token || '/sendMessage',
        body := jsonb_build_object('chat_id', v_chat, 'text', v_msg),
        headers := jsonb_build_object('Content-Type', 'application/json'));
    UPDATE heartbeat_state SET alerting = true, last_alert_ts = now() WHERE id = 1;
    RETURN 'отправлена тревога: молчит ' || round(v_age_min) || ' мин';
END;
$$;

REVOKE ALL ON FUNCTION heartbeat_check(int, int) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION heartbeat_check(int, int) TO postgres;


-- ---------- D. расписание ----------
-- Раз в 30 минут: чаще незачем при пороге в 2 часа, а нагрузка нулевая.

SELECT cron.unschedule('heartbeat') WHERE EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'heartbeat');
SELECT cron.schedule('heartbeat', '*/30 * * * *', $cron$ SELECT heartbeat_check() $cron$);


-- ---------- E. проверка ----------
-- Первый вызов руками: если бот сейчас молчит — придёт тревога, и это
-- и есть боевая проверка канала.
SELECT heartbeat_check();
