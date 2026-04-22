"""Parse Telegram getUpdates JSON from stdin.

Prints two integers separated by a space:
  <fire>  -- 1 if a fresh /sup from our chat (within last 10min) was found
  <max_id> -- highest update_id observed (-1 if none), used to ack the buffer

Reads:
  TELEGRAM_CHAT_ID  -- only listen to this chat
  NOW               -- current epoch seconds (avoid re-firing on stale msgs)
"""
import json
import os
import sys

WINDOW_SEC = 600  # 10 min


def main() -> int:
    data = json.load(sys.stdin)
    chat_id = str(os.environ["TELEGRAM_CHAT_ID"])
    now = int(os.environ["NOW"])
    fire = False
    max_id = -1
    for u in data.get("result", []):
        uid = u.get("update_id")
        if uid is not None and uid > max_id:
            max_id = uid
        msg = u.get("message") or {}
        text = (msg.get("text") or "").strip().lower()
        chat = str(((msg.get("chat") or {}).get("id") or ""))
        date = int(msg.get("date") or 0)
        if chat != chat_id:
            continue
        if not (
            text == "/sup"
            or text.startswith("/sup ")
            or text.startswith("/sup@")
        ):
            continue
        if now - date > WINDOW_SEC:
            continue
        fire = True
    print(f"{int(fire)} {max_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
