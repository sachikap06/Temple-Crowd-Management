"""
notification_manager.py
=======================
Modular notification system with provider abstraction for the
Smart Temple Crowd Management Project.

Features:
  - Abstract NotificationProvider interface (swap in email/SMS/webhook later)
  - MockNotificationProvider for testing (clearly marked as NOT real delivery)
  - Configurable per-area per-status cooldowns to prevent notification spam
  - Escalation detection (NORMAL → MODERATE → CONGESTED)
  - Recovery notifications when crowd returns to safe levels
  - Persistent alert log (JSON)
  - Test mode flag
  - No real credentials, phone numbers, or API keys

Design:
  Notification logic is fully separated from crowd detection and dashboard code.
  The provider is configurable so a real service can be integrated later.
"""

import os
import json
import time
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

ALERT_LOG_FILE = os.path.join("output", "alert_log.json")


# ═══════════════════════════════════════════════════════════════════════════════
# RECIPIENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Recipient:
    """A notification recipient."""
    name: str
    role: str        # "administrator", "security", "crowd_control"
    contact: str     # placeholder — email/phone/webhook URL
    enabled: bool = True


DEFAULT_RECIPIENTS = [
    Recipient("Temple Administrator", "administrator", "admin@temple.local"),
    Recipient("Security Head", "security", "security@temple.local"),
    Recipient("Crowd Control Officer", "crowd_control", "crowdcontrol@temple.local"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlertRecord:
    """A single alert log entry."""
    timestamp: str
    epoch: float
    area_name: str
    crowd_count: int
    crowd_status: str          # NORMAL, MODERATE, CONGESTED
    previous_status: str
    reason: str
    recipients_notified: List[str]
    delivery_results: List[dict]
    is_test_mode: bool = False
    is_recovery: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION PROVIDER (Abstract Interface)
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationProvider(ABC):
    """
    Abstract base class for notification delivery.

    Implementations may send email, SMS, push notifications, or
    webhook calls. For now only MockNotificationProvider is provided.
    """

    @abstractmethod
    def send(self, recipient: Recipient, subject: str, message: str) -> dict:
        """
        Deliver a notification. Must return a dict containing at least:
          success (bool), provider (str), detail (str)
        """

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""


class MockNotificationProvider(NotificationProvider):
    """
    Mock provider for testing alert rules without sending real messages.
    Every delivery result is clearly marked [MOCK - NOT A REAL DELIVERY].
    """

    def __init__(self):
        self.sent_log: List[dict] = []

    def provider_name(self) -> str:
        return "MockProvider"

    def send(self, recipient: Recipient, subject: str, message: str) -> dict:
        result = {
            "success": True,
            "provider": "MockProvider",
            "detail": (
                f"[MOCK - NOT A REAL DELIVERY] "
                f"Notification logged for {recipient.name} ({recipient.role})"
            ),
            "recipient_name": recipient.name,
            "recipient_role": recipient.role,
            "recipient_contact": recipient.contact,
            "subject": subject,
            "message_preview": message[:200],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.sent_log.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationManager:
    """
    Manages crowd alert notifications with:

    Rules:
      NORMAL    → No routine alerts. Optional recovery notification
                  if the area was previously MODERATE or CONGESTED.
      MODERATE  → Notify the configured temple administrator.
      CONGESTED → Notify administrator AND all authorities
                  (security, crowd-control personnel).

    Features:
      - Per-area per-status cooldowns (configurable seconds)
      - Escalation detection (status worsens → immediate notification)
      - De-escalation / recovery notifications
      - Persistent alert log (JSON, last 500 entries)
      - Test mode flag for simulated runs
      - Provider failure handling (errors caught, logged, not raised)
    """

    def __init__(
        self,
        provider: Optional[NotificationProvider] = None,
        cooldown_seconds: int = 120,
        test_mode: bool = True,
        recipients: Optional[List[Recipient]] = None,
        alert_log_file: Optional[str] = None,
    ):
        self.provider = provider or MockNotificationProvider()
        self.cooldown_seconds = cooldown_seconds
        self.test_mode = test_mode
        self.recipients = recipients if recipients is not None else list(DEFAULT_RECIPIENTS)
        self.alert_log_file = alert_log_file or ALERT_LOG_FILE

        # Cooldown tracker: (area_name, status) → epoch
        self._last_alert_time: Dict[tuple, float] = {}
        # Current status per area for escalation detection
        self._current_status: Dict[str, str] = {}
        # Persistent alert log
        self.alert_log: List[AlertRecord] = []
        self._load_alert_log()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_alert_log(self):
        if os.path.exists(self.alert_log_file):
            try:
                with open(self.alert_log_file, "r") as f:
                    data = json.load(f)
                self.alert_log = [AlertRecord(**entry) for entry in data]
            except Exception:
                self.alert_log = []

    def _save_alert_log(self):
        out_dir = os.path.dirname(self.alert_log_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        data = [asdict(r) for r in self.alert_log]
        if len(data) > 500:
            data = data[-500:]
        with open(self.alert_log_file, "w") as f:
            json.dump(data, f, indent=2)

    # ── Cooldown logic ─────────────────────────────────────────────────────

    def _is_on_cooldown(self, area_name: str, status: str) -> bool:
        key = (area_name, status)
        last = self._last_alert_time.get(key, 0.0)
        return (time.time() - last) < self.cooldown_seconds

    # ── Recipient selection ────────────────────────────────────────────────

    def _get_recipients_for_status(self, status: str) -> List[Recipient]:
        """
        NORMAL    → [] (no notifications)
        MODERATE  → [administrator]
        CONGESTED → [administrator, security, crowd_control]
        """
        if status == "NORMAL":
            return []
        if status == "MODERATE":
            return [r for r in self.recipients if r.enabled and r.role == "administrator"]
        if status == "CONGESTED":
            return [r for r in self.recipients if r.enabled]
        return []

    # ── Message building ───────────────────────────────────────────────────

    def _build_message(
        self, area_name: str, count: int, status: str,
        reason: str, density: Optional[float] = None,
    ) -> tuple:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"[Temple Crowd Alert] {status} - {area_name}"

        density_line = f"\nCrowd Density: {density:.3f} people/sqm" if density is not None else ""
        test_line = (
            "\n\n[TEST MODE - This is a simulated notification, NOT a real delivery]"
            if self.test_mode else ""
        )

        body = (
            f"Temple Crowd Management Alert\n"
            f"{'=' * 44}\n"
            f"Area         : {area_name}\n"
            f"Crowd Count  : {count} people\n"
            f"Classification: {status}{density_line}\n"
            f"Timestamp    : {ts}\n"
            f"Reason       : {reason}\n"
            f"{'=' * 44}{test_line}"
        )
        return subject, body

    # ── Core: process a crowd update ───────────────────────────────────────

    def process_crowd_update(
        self,
        area_name: str,
        crowd_count: int,
        crowd_status: str,
        density: Optional[float] = None,
    ) -> Optional[AlertRecord]:
        """
        Process a new crowd reading. Sends notifications based on:
          - Status level (NORMAL/MODERATE/CONGESTED)
          - Escalation detection
          - Cooldown enforcement
          - Recovery detection

        Returns an AlertRecord if notifications were dispatched, else None.
        """
        previous_status = self._current_status.get(area_name, "NORMAL")
        self._current_status[area_name] = crowd_status

        # ── NORMAL: check for recovery, otherwise no alert ────────────
        if crowd_status == "NORMAL":
            if previous_status in ("CONGESTED", "MODERATE"):
                return self._send_recovery(area_name, crowd_count, previous_status, density)
            return None

        # ── Cooldown check ────────────────────────────────────────────
        if self._is_on_cooldown(area_name, crowd_status):
            return None

        # ── Determine reason ──────────────────────────────────────────
        severity = {"NORMAL": 0, "MODERATE": 1, "CONGESTED": 2}
        is_escalation = severity.get(crowd_status, 0) > severity.get(previous_status, 0)

        if is_escalation:
            reason = f"Crowd escalated from {previous_status} to {crowd_status} in {area_name}"
        elif crowd_status == previous_status:
            reason = f"Ongoing {crowd_status} crowd level in {area_name}"
        else:
            reason = f"Crowd status is {crowd_status} in {area_name}"

        # ── Get recipients ────────────────────────────────────────────
        recipients = self._get_recipients_for_status(crowd_status)
        if not recipients:
            return None

        # ── Send notifications ────────────────────────────────────────
        subject, message = self._build_message(
            area_name, crowd_count, crowd_status, reason, density
        )
        delivery_results = []
        notified_names = []
        for recipient in recipients:
            try:
                result = self.provider.send(recipient, subject, message)
                delivery_results.append(result)
                if result.get("success"):
                    notified_names.append(recipient.name)
            except Exception as e:
                delivery_results.append({
                    "success": False,
                    "provider": self.provider.provider_name(),
                    "detail": f"Error sending to {recipient.name}: {e}",
                    "error": str(e),
                })

        # ── Update cooldown ───────────────────────────────────────────
        self._last_alert_time[(area_name, crowd_status)] = time.time()

        # ── Record ────────────────────────────────────────────────────
        record = AlertRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch=time.time(),
            area_name=area_name,
            crowd_count=crowd_count,
            crowd_status=crowd_status,
            previous_status=previous_status,
            reason=reason,
            recipients_notified=notified_names,
            delivery_results=delivery_results,
            is_test_mode=self.test_mode,
            is_recovery=False,
        )
        self.alert_log.append(record)
        self._save_alert_log()
        return record

    # ── Recovery notification ──────────────────────────────────────────────

    def _send_recovery(
        self, area_name: str, crowd_count: int,
        previous_status: str, density: Optional[float],
    ) -> Optional[AlertRecord]:
        """Notify when crowd returns from elevated to NORMAL."""
        if self._is_on_cooldown(area_name, "RECOVERY"):
            return None

        reason = f"Crowd recovered from {previous_status} to NORMAL in {area_name}"
        subject = f"[Temple Crowd Recovery] {area_name} - Back to Normal"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        density_line = f"\nCrowd Density: {density:.3f} people/sqm" if density is not None else ""
        test_line = "\n[TEST MODE - Simulated notification]" if self.test_mode else ""

        body = (
            f"Temple Crowd Recovery Notice\n"
            f"{'=' * 44}\n"
            f"Area            : {area_name}\n"
            f"Current Count   : {crowd_count} people\n"
            f"Previous Status : {previous_status}\n"
            f"Current Status  : NORMAL{density_line}\n"
            f"Timestamp       : {ts}\n"
            f"{'=' * 44}{test_line}"
        )

        # Recovery → notify administrators only
        recipients = [r for r in self.recipients if r.enabled and r.role == "administrator"]

        delivery_results = []
        notified_names = []
        for recipient in recipients:
            try:
                result = self.provider.send(recipient, subject, body)
                delivery_results.append(result)
                if result.get("success"):
                    notified_names.append(recipient.name)
            except Exception as e:
                delivery_results.append({
                    "success": False,
                    "provider": self.provider.provider_name(),
                    "detail": f"Error: {e}",
                })

        self._last_alert_time[(area_name, "RECOVERY")] = time.time()

        record = AlertRecord(
            timestamp=ts,
            epoch=time.time(),
            area_name=area_name,
            crowd_count=crowd_count,
            crowd_status="NORMAL",
            previous_status=previous_status,
            reason=reason,
            recipients_notified=notified_names,
            delivery_results=delivery_results,
            is_test_mode=self.test_mode,
            is_recovery=True,
        )
        self.alert_log.append(record)
        self._save_alert_log()
        return record

    # ── Utilities ──────────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 20) -> List[AlertRecord]:
        """Return the most recent alert records."""
        return self.alert_log[-limit:]

    def clear_cooldowns(self):
        """Clear all cooldowns (useful for testing)."""
        self._last_alert_time.clear()

    def get_alert_summary(self) -> dict:
        """Summary of alert activity."""
        by_status: Dict[str, int] = {}
        by_area: Dict[str, int] = {}
        for r in self.alert_log:
            by_status[r.crowd_status] = by_status.get(r.crowd_status, 0) + 1
            by_area[r.area_name] = by_area.get(r.area_name, 0) + 1
        return {
            "total_alerts": len(self.alert_log),
            "alerts_by_status": by_status,
            "alerts_by_area": by_area,
            "test_mode": self.test_mode,
            "provider": self.provider.provider_name(),
            "cooldown_seconds": self.cooldown_seconds,
        }
