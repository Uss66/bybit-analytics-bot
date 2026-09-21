// Telegram webhook receiver (2026-09-21) - replaces polling telegram_commands.yml
// on GitHub's unreliable `schedule:` trigger (confirmed degrading to 3-6x slower
// than configured - see project_github_actions_cron_reliability memory/README).
// Telegram calls THIS function the instant a message arrives. This function does
// a fast trigger-phrase pre-filter only (not the authoritative check - that still
// lives in telegram_command_handler.py/strategy.py, never duplicated here) and
// fires workflow_dispatch for telegram_commands.yml via GitHub's REST API, which
// GitHub runs near-instantly (unlike `schedule:`). The Python script still does
// all real work (signal scoring, balance checks, the actual reply); this
// function's only job is "wake it up now instead of waiting for cron/fallback".
//
// Requires these Edge Function secrets (Dashboard -> Edge Functions -> Secrets):
//   TELEGRAM_WEBHOOK_SECRET - random string, also passed to Telegram's
//     setWebhook as secret_token, so only real Telegram calls are accepted.
//   GITHUB_ACTIONS_PAT - same fine-grained PAT (Actions: read/write on this
//     repo only) stored in Supabase Vault for the pg_cron hourly dispatch.
//   TELEGRAM_CHAT_ID - the one chat this bot serves (matches .env's value).

const TELEGRAM_SECRET_TOKEN = Deno.env.get("TELEGRAM_WEBHOOK_SECRET")!;
const GITHUB_PAT = Deno.env.get("GITHUB_ACTIONS_PAT")!;
const CHAT_ID = Deno.env.get("TELEGRAM_CHAT_ID")!;
const GITHUB_REPO = "Uss66/bybit-analytics-bot";
const WORKFLOW_FILE = "telegram_commands.yml";

// Mirrors telegram_command_handler.py's TRIGGERS + STOP/RESUME/STATUS_TRIGGERS
// - a fast pre-filter only. Keep the two lists in sync: this one decides
// whether the message is seen AT ALL. A
// false positive here just costs one harmless extra Actions run (the Python
// script re-checks properly and no-ops); a false negative just means this
// message waits for the low-frequency fallback schedule instead of firing
// instantly.
const TRIGGERS = [
  "/invest", "куда вложить", "стоит ли вкладывать", "стоит ли инвестировать",
  // Autonomous-trading controls (2026-09-22). These MUST be mirrored here:
  // a phrase missing from this pre-filter never reaches the Python handler
  // at all, so the command silently does nothing - which is exactly how
  // /status came back empty the first time it was tried. A stop command
  // that fails silently is the worst possible failure in this system.
  "/stop", "/trade off", "стоп", "останови", "выключи торговлю",
  "/trade on", "/resume",
  "/status", "/trade status",
];

Deno.serve(async (req) => {
  const secretHeader = req.headers.get("X-Telegram-Bot-Api-Secret-Token");
  if (secretHeader !== TELEGRAM_SECRET_TOKEN) {
    return new Response("forbidden", { status: 403 });
  }

  let update: Record<string, any>;
  try {
    update = await req.json();
  } catch {
    return new Response("ok", { status: 200 });
  }

  const message = update.message ?? update.channel_post;
  const text = String(message?.text ?? "").toLowerCase();
  const chatId = String(message?.chat?.id ?? "");
  const isBot = message?.from?.is_bot ?? false;

  // Any slash-command counts, not just the known list: a command this
  // pre-filter has not heard of is better handled by the Python script
  // (which no-ops on anything it does not recognise) than dropped here.
  // The cost of a false positive is one short Actions run; the cost of a
  // false negative is a command that silently does nothing.
  const isCommand = text.startsWith("/");
  const matched = chatId === CHAT_ID && !isBot &&
    (isCommand || TRIGGERS.some((t) => text.includes(t)));

  if (matched) {
    // Fire-and-forget: don't block Telegram's webhook response on GitHub's
    // API completing - Telegram expects a fast 200 or it may retry/disable
    // the webhook.
    fetch(
      `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GITHUB_PAT}`,
          Accept: "application/vnd.github+json",
          "Content-Type": "application/json",
          "X-GitHub-Api-Version": "2022-11-28",
        },
        body: JSON.stringify({
          ref: "main",
          inputs: { update_json: JSON.stringify(update) },
        }),
      },
    ).catch((e) => console.error("dispatch failed", e));
  }

  return new Response("ok", { status: 200 });
});
