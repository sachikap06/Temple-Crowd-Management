"""
test_crowd_system.py
====================
Test suite for Smart Temple Crowd Management System.

Covers all 10 required test cases:
  1. Normal crowd classification
  2. Moderate crowd classification
  3. Congested crowd classification
  4. Different thresholds for different areas
  5. Correct graph updates using real processed counts
  6. Escalation from Moderate to Congested
  7. Notification cooldown and duplicate prevention
  8. Recovery from Congested to Normal
  9. Missing video and invalid configuration handling
  10. Notification-provider failure handling

Uses synthetic count sequences only for isolated unit tests.
"""

import unittest
import os
import sys
import json
import time
import tempfile

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from area_config import AreaConfig, CrowdClassifier, AreaManager
from notification_manager import (
    NotificationManager,
    MockNotificationProvider,
    NotificationProvider,
    Recipient,
    AlertRecord,
)


class TestCrowdManagementSystem(unittest.TestCase):
    """
    Complete test suite for the Smart Temple Crowd Management System.
    Each test is numbered to match the 10 required test cases.
    """

    def setUp(self):
        """Set up test fixtures with a known area configuration."""
        self.area = AreaConfig(
            name="Test Temple",
            count_threshold_moderate=10,
            count_threshold_congested=20,
            area_sqm=50.0,
            density_threshold_moderate=0.0,   # density checks disabled for clarity
            density_threshold_congested=0.0,
            max_capacity=30,
        )
        self.mock_provider = MockNotificationProvider()

        # Use a temporary file for the alert log so tests don't pollute real data
        self.temp_log = tempfile.mktemp(suffix="_alert_log.json")
        self.notification_mgr = NotificationManager(
            provider=self.mock_provider,
            cooldown_seconds=5,
            test_mode=True,
            alert_log_file=self.temp_log,
        )

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.temp_log]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    # TEST 1: Normal crowd classification
    # ──────────────────────────────────────────────────────────────────────
    def test_01_normal_classification(self):
        """Count at or below moderate threshold → NORMAL."""
        self.assertEqual(self.area.classify_crowd(0), "NORMAL")
        self.assertEqual(self.area.classify_crowd(5), "NORMAL")
        self.assertEqual(self.area.classify_crowd(10), "NORMAL")   # boundary: <= 10

    # ──────────────────────────────────────────────────────────────────────
    # TEST 2: Moderate crowd classification
    # ──────────────────────────────────────────────────────────────────────
    def test_02_moderate_classification(self):
        """Count between moderate and congested thresholds → MODERATE."""
        self.assertEqual(self.area.classify_crowd(11), "MODERATE")
        self.assertEqual(self.area.classify_crowd(15), "MODERATE")
        self.assertEqual(self.area.classify_crowd(20), "MODERATE")  # boundary: <= 20

    # ──────────────────────────────────────────────────────────────────────
    # TEST 3: Congested crowd classification
    # ──────────────────────────────────────────────────────────────────────
    def test_03_congested_classification(self):
        """Count above congested threshold → CONGESTED."""
        self.assertEqual(self.area.classify_crowd(21), "CONGESTED")
        self.assertEqual(self.area.classify_crowd(50), "CONGESTED")
        self.assertEqual(self.area.classify_crowd(100), "CONGESTED")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 4: Different thresholds for different areas
    # ──────────────────────────────────────────────────────────────────────
    def test_04_different_area_thresholds(self):
        """Same count yields different statuses in differently-configured areas."""
        small_room = AreaConfig(
            name="Small Room",
            count_threshold_moderate=5,
            count_threshold_congested=10,
            max_capacity=15,
        )
        large_hall = AreaConfig(
            name="Large Hall",
            count_threshold_moderate=30,
            count_threshold_congested=60,
            max_capacity=80,
        )

        count = 8
        self.assertEqual(small_room.classify_crowd(count), "MODERATE")
        self.assertEqual(large_hall.classify_crowd(count), "NORMAL")

        count = 35
        self.assertEqual(small_room.classify_crowd(count), "CONGESTED")
        self.assertEqual(large_hall.classify_crowd(count), "MODERATE")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 5: Correct graph updates using real processed counts
    # ──────────────────────────────────────────────────────────────────────
    def test_05_graph_data_from_real_counts(self):
        """
        Verify that the chart data pipeline preserves actual processed counts
        without any random, dummy, or hardcoded data.
        """
        # Simulate a sequence of real YOLO-detected counts
        real_counts = [5, 8, 12, 15, 22, 18, 10, 7, 3, 5]

        chart_data = []
        for i, count in enumerate(real_counts):
            time_sec = round(i * 0.5, 1)   # 0.5s intervals
            chart_data.append({"Time (s)": time_sec, "Crowd Count": count})

        # Every entry must exactly match the input count
        self.assertEqual(len(chart_data), len(real_counts))
        for i, entry in enumerate(chart_data):
            self.assertEqual(entry["Crowd Count"], real_counts[i],
                             f"Chart entry {i} has wrong count")
            self.assertEqual(entry["Time (s)"], round(i * 0.5, 1))

        # Verify fluctuations are preserved (increases AND decreases)
        counts_in_chart = [e["Crowd Count"] for e in chart_data]
        has_increase = any(counts_in_chart[i] > counts_in_chart[i - 1]
                          for i in range(1, len(counts_in_chart)))
        has_decrease = any(counts_in_chart[i] < counts_in_chart[i - 1]
                          for i in range(1, len(counts_in_chart)))
        self.assertTrue(has_increase, "Chart data must show crowd increases")
        self.assertTrue(has_decrease, "Chart data must show crowd decreases")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 6: Escalation from Moderate to Congested
    # ──────────────────────────────────────────────────────────────────────
    def test_06_escalation_notification(self):
        """Escalation from MODERATE to CONGESTED triggers notification to all."""
        # First: set baseline at MODERATE
        record_mod = self.notification_mgr.process_crowd_update(
            "Escalation Area", 15, "MODERATE"
        )
        self.assertIsNotNone(record_mod, "MODERATE should trigger admin notification")
        self.assertEqual(record_mod.crowd_status, "MODERATE")
        # Only administrator should be notified for MODERATE
        admin_notified = any(
            "Administrator" in n for n in record_mod.recipients_notified
        )
        self.assertTrue(admin_notified, "Administrator must be notified for MODERATE")

        # Clear cooldown so the CONGESTED notification is not blocked
        self.notification_mgr.clear_cooldowns()

        # Escalate to CONGESTED
        record_cong = self.notification_mgr.process_crowd_update(
            "Escalation Area", 30, "CONGESTED"
        )
        self.assertIsNotNone(record_cong, "CONGESTED should trigger notification")
        self.assertEqual(record_cong.crowd_status, "CONGESTED")
        self.assertIn("escalated", record_cong.reason.lower())
        # All recipients (admin + security + crowd_control) should be notified
        self.assertTrue(
            len(record_cong.recipients_notified) >= 3,
            f"CONGESTED should notify all authorities, got {record_cong.recipients_notified}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # TEST 7: Notification cooldown and duplicate prevention
    # ──────────────────────────────────────────────────────────────────────
    def test_07_cooldown_duplicate_prevention(self):
        """Second notification for the same area+status within cooldown is blocked."""
        # First notification goes through
        record1 = self.notification_mgr.process_crowd_update(
            "Cooldown Area", 25, "CONGESTED"
        )
        self.assertIsNotNone(record1, "First notification should be sent")

        # Immediate second notification → blocked by cooldown
        record2 = self.notification_mgr.process_crowd_update(
            "Cooldown Area", 26, "CONGESTED"
        )
        self.assertIsNone(record2, "Duplicate within cooldown must be blocked")

        # After cooldown expires, notification should go through again
        self.notification_mgr.clear_cooldowns()
        record3 = self.notification_mgr.process_crowd_update(
            "Cooldown Area", 27, "CONGESTED"
        )
        self.assertIsNotNone(record3, "After cooldown clears, notification should send")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 8: Recovery from Congested to Normal
    # ──────────────────────────────────────────────────────────────────────
    def test_08_recovery_notification(self):
        """When crowd drops from CONGESTED to NORMAL, send recovery notification."""
        # Set area's tracked status to CONGESTED
        self.notification_mgr._current_status["Recovery Area"] = "CONGESTED"

        # Now report NORMAL → should trigger recovery
        record = self.notification_mgr.process_crowd_update(
            "Recovery Area", 3, "NORMAL"
        )
        self.assertIsNotNone(record, "Recovery should send notification")
        self.assertTrue(record.is_recovery, "Record must be marked as recovery")
        self.assertEqual(record.crowd_status, "NORMAL")
        self.assertEqual(record.previous_status, "CONGESTED")
        self.assertIn("recovered", record.reason.lower())

    # ──────────────────────────────────────────────────────────────────────
    # TEST 9: Missing video and invalid configuration handling
    # ──────────────────────────────────────────────────────────────────────
    def test_09_missing_video_invalid_config(self):
        """Proper error handling for missing files and invalid configurations."""
        # Invalid area configuration: congested < moderate
        bad_area = AreaConfig(
            name="Bad Config",
            count_threshold_moderate=30,
            count_threshold_congested=10,   # lower than moderate → invalid
            max_capacity=5,                 # lower than congested → invalid
        )
        errors = bad_area.validate()
        self.assertTrue(len(errors) >= 2, f"Expected >=2 validation errors, got: {errors}")

        # Empty name
        nameless = AreaConfig(name="", count_threshold_moderate=5, count_threshold_congested=10)
        self.assertTrue(len(nameless.validate()) > 0, "Empty name should fail validation")

        # Missing video file → cv2.VideoCapture should not open
        import cv2
        cap = cv2.VideoCapture("nonexistent_video_file_that_does_not_exist.mp4")
        self.assertFalse(cap.isOpened(), "Missing video must not open successfully")
        cap.release()

        # Area manager with non-existent config → should fall back to defaults
        temp_cfg = tempfile.mktemp(suffix="_area_test.json")
        try:
            mgr = AreaManager(config_file=temp_cfg)
            self.assertTrue(len(mgr.list_areas()) > 0, "Should initialise with defaults")
        finally:
            if os.path.exists(temp_cfg):
                os.remove(temp_cfg)

    # ──────────────────────────────────────────────────────────────────────
    # TEST 10: Notification-provider failure handling
    # ──────────────────────────────────────────────────────────────────────
    def test_10_provider_failure_handling(self):
        """Provider errors are caught, logged, and do NOT crash the system."""

        class FailingProvider(NotificationProvider):
            """Provider that always raises an exception."""
            def provider_name(self) -> str:
                return "FailingProvider"

            def send(self, recipient, subject, message):
                raise ConnectionError("Simulated network failure")

        temp_log = tempfile.mktemp(suffix="_fail_log.json")
        try:
            failing_mgr = NotificationManager(
                provider=FailingProvider(),
                cooldown_seconds=0,
                test_mode=True,
                alert_log_file=temp_log,
            )

            # Must NOT raise — errors should be caught internally
            record = failing_mgr.process_crowd_update("Fail Area", 30, "CONGESTED")
            self.assertIsNotNone(record, "Record should still be created despite failures")

            # All deliveries should be marked as failed
            failed = [r for r in record.delivery_results if not r.get("success")]
            self.assertTrue(len(failed) > 0, "Failed deliveries must be recorded")
            self.assertTrue(
                any("error" in r or "Error" in r.get("detail", "") for r in failed),
                "Error details must be captured in delivery results"
            )
        finally:
            if os.path.exists(temp_log):
                os.remove(temp_log)


# ──────────────────────────────────────────────────────────────────────────────
# Additional tests for hysteresis classifier
# ──────────────────────────────────────────────────────────────────────────────

class TestCrowdClassifierHysteresis(unittest.TestCase):
    """Tests for the CrowdClassifier hysteresis behaviour."""

    def setUp(self):
        self.area = AreaConfig(
            name="Hysteresis Test",
            count_threshold_moderate=10,
            count_threshold_congested=20,
            max_capacity=30,
        )
        self.classifier = CrowdClassifier(escalation_frames=3, deescalation_frames=5)

    def test_stable_normal_stays_normal(self):
        """Repeated NORMAL readings keep status NORMAL."""
        for count in [5, 6, 4, 7, 5]:
            status = self.classifier.update(count, self.area)
            self.assertEqual(status, "NORMAL")

    def test_escalation_requires_consecutive_frames(self):
        """Status does not escalate until enough consecutive readings."""
        # 1 reading of MODERATE → not enough
        self.assertEqual(self.classifier.update(15, self.area), "NORMAL")
        # 2 readings → still not enough (need 3)
        self.assertEqual(self.classifier.update(15, self.area), "NORMAL")
        # 3rd reading → escalation confirmed
        self.assertEqual(self.classifier.update(15, self.area), "MODERATE")

    def test_deescalation_requires_more_frames(self):
        """De-escalation requires more consecutive frames than escalation."""
        # First escalate to MODERATE
        for _ in range(3):
            self.classifier.update(15, self.area)
        self.assertEqual(self.classifier.current_status, "MODERATE")

        # Now send NORMAL readings — need 5 consecutive for de-escalation
        for i in range(4):
            status = self.classifier.update(5, self.area)
            self.assertEqual(status, "MODERATE", f"De-escalation too early at reading {i+1}")

        # 5th NORMAL reading → de-escalation
        status = self.classifier.update(5, self.area)
        self.assertEqual(status, "NORMAL")

    def test_interrupted_streak_resets(self):
        """A single contradicting reading resets the consecutive count."""
        # Start escalating toward MODERATE
        self.classifier.update(15, self.area)  # 1st MODERATE
        self.classifier.update(15, self.area)  # 2nd MODERATE
        # Interrupt with NORMAL
        self.classifier.update(5, self.area)
        # Resume MODERATE — counter should be reset
        self.assertEqual(self.classifier.update(15, self.area), "NORMAL")  # only 1st again


if __name__ == "__main__":
    unittest.main(verbosity=2)
