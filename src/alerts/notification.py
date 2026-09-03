"""Multi-channel notification dispatcher (Console, WebSocket, Webhook, Telegram)."""

import os
from pathlib import Path
from typing import Callable, List, Optional
import requests
from src.alerts.event_logger import AlertEvent
from src.utils.config import get_env, get_project_root


class NotificationDispatcher:
    """Dispatches emergency alerts across configured external channels."""

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.telegram_token = telegram_token or get_env("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or get_env("TELEGRAM_CHAT_ID")
        self.webhook_url = webhook_url or get_env("WEBHOOK_URL")
        self._ws_subscribers: List[Callable[[AlertEvent], None]] = []

    def subscribe_websocket(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register a WebSocket broadcast callback."""
        self._ws_subscribers.append(callback)

    def dispatch(self, event: AlertEvent) -> None:
        """Dispatch event across all active notification channels."""
        # 1. Local Console Logging
        self._log_to_console(event)

        # 2. WebSocket notification
        for subscriber in self._ws_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"[Warning] Failed to push alert to WebSocket subscriber: {e}")

        # 3. HTTP Webhook
        if self.webhook_url:
            self._send_webhook(event)

        # 4. Telegram Bot
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(event)

    def _log_to_console(self, event: AlertEvent) -> None:
        print("\n" + "=" * 70)
        print("🚨 EMERGENCY ALERT TRIGGERED 🚨")
        print(f"Event ID:   {event.event_id}")
        print(f"Person ID:  {event.person_id}")
        print(f"Timestamp:  {event.timestamp}")
        print(f"Risk Score: {event.risk_score:.2f}")
        print(f"Message:    {event.message}")
        print(f"Evidence:   {event.evidence}")
        if event.snapshot_path:
            print(f"Snapshot:   {event.snapshot_path}")
        print("=" * 70 + "\n")

    def _send_webhook(self, event: AlertEvent) -> None:
        try:
            import json
            from dataclasses import asdict
            payload = asdict(event)
            requests.post(self.webhook_url, json=payload, timeout=3.0)
        except Exception as e:
            print(f"[Warning] Webhook alert dispatch failed: {e}")

    def _send_telegram(self, event: AlertEvent) -> None:
        try:
            msg_text = (
                f"🚨 *POSSIBLE MEDICAL EMERGENCY DETECTED*\n\n"
                f"• *Person ID:* `{event.person_id}`\n"
                f"• *Risk Score:* `{event.risk_score:.2f}`\n"
                f"• *Primary Action:* `{event.evidence.get('primary_action', 'unknown')}`\n"
                f"• *Notice:* {event.message}\n"
            )

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(
                url,
                data={"chat_id": self.telegram_chat_id, "text": msg_text, "parse_mode": "Markdown"},
                timeout=5.0,
            )

            # Upload snapshot image if available
            if event.snapshot_path:
                abs_path = get_project_root() / event.snapshot_path
                if abs_path.exists():
                    photo_url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                    with open(abs_path, "rb") as photo_file:
                        requests.post(
                            photo_url,
                            data={"chat_id": self.telegram_chat_id, "caption": f"Snapshot Event: {event.event_id}"},
                            files={"photo": photo_file},
                            timeout=8.0,
                        )
        except Exception as e:
            print(f"[Warning] Telegram alert dispatch failed: {e}")
