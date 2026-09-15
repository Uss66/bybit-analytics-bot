"""
Minimal Telegram alert sender for testnet_trader.py (2026-09-14). Added so
the user finds out the MOMENT the mechanical strategy would enter/exit a
position, instead of discovering it later in logs - addresses "the moment
might be missed" without crossing the hard rule against this project
executing real trades autonomously: this only NOTIFIES, the user still
places any real order themselves.

Requires two env vars (same .env-loading convention as db.py/bybit_client.py):
  TELEGRAM_BOT_TOKEN  - from @BotFather
  TELEGRAM_CHAT_ID    - the user's chat id (see README section on setup)

If either is missing, send_alert() silently no-ops (prints a warning once)
rather than crashing the hourly tick - alerting is a nice-to-have on top of
the trading logic, never a reason to break the main pipeline.
"""
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_warned = False


def send_alert(message: str) -> bool:
    global _warned
    if not BOT_TOKEN or not CHAT_ID:
        if not _warned:
            print("[telegram_notify] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set - alerts disabled")
            _warned = True
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[telegram_notify] send failed: {resp.status_code} {resp.text}")
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(f"[telegram_notify] send failed: {e}")
        return False


if __name__ == "__main__":
    ok = send_alert("ByBit_Analitics: telegram_notify.py test message - if you see this, alerts are wired up correctly.")
    print("Sent OK" if ok else "Failed to send - check TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
