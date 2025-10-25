"""
Unit tests for PII salt 90-day rotation with dual-valid window.

Tests:
- test_salt_rotates_every_90d: Verify salt rotation triggers every 90 days
- test_dual_valid_24h: Verify both old and new salts valid for 24h during rotation
- test_salt_age_metric: Verify salt_age_days metric <90 continuously
- test_user_key_resolution_no_failures: Verify zero user_key resolution failures during rotation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from core.security import SaltManager, hash_pii


class TestSaltRotation:
    """Test suite for PII salt 90-day rotation (Q_002)"""

    @pytest.fixture
    def salt_manager(self, redis_client, secret_manager):
        """Fixture to provide SaltManager with test dependencies"""
        return SaltManager(
            redis=redis_client,
            secret_manager=secret_manager,
            rotation_days=90,
            dual_valid_hours=24
        )

    @pytest.fixture
    def mock_time(self):
        """Fixture to mock system time for rotation testing"""
        return MagicMock()

    def test_salt_rotates_every_90d(self, salt_manager, mock_time):
        """
        ACCEPTANCE: unit: test_salt_rotation_90d.py::test_salt_rotates_every_90d passes
        RISK_GATE: rotation_schedule=90d, dual_valid_period=24h
        """
        # Initial salt
        initial_salt = salt_manager.get_current_salt()
        initial_created = salt_manager.get_salt_age_days()

        # Fast-forward 89 days (should NOT rotate)
        with patch("core.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=89)
            salt_manager.check_rotation_trigger()
            salt_89d = salt_manager.get_current_salt()
            assert salt_89d == initial_salt, "Salt should not rotate before 90 days"

        # Fast-forward to 90 days (should rotate)
        with patch("core.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=90)
            rotation_result = salt_manager.check_rotation_trigger()
            assert rotation_result["rotated"] is True, "Salt should rotate at 90 days"

            new_salt = salt_manager.get_current_salt()
            assert new_salt != initial_salt, "New salt should be different from initial"

    def test_dual_valid_24h(self, salt_manager):
        """
        ACCEPTANCE: unit: test_salt_rotation_90d.py::test_dual_valid_24h passes
        DRY_RUN_PROBE: Advance system time +90d in test env; trigger rotation cron; 
        assert both old+new salts valid for 24h
        """
        # Get initial salt
        initial_salt = salt_manager.get_current_salt()

        # Trigger rotation
        with patch("core.security.datetime") as mock_dt:
            rotation_time = datetime.utcnow() + timedelta(days=90)
            mock_dt.utcnow.return_value = rotation_time
            salt_manager.rotate_salt()

        new_salt = salt_manager.get_current_salt()
        assert new_salt != initial_salt

        # During dual-valid window (within 24h), both salts should work
        test_email = "test@example.com"

        # Hash with old salt
        hash_old = hash_pii(test_email, salt=initial_salt)

        # Hash with new salt
        hash_new = hash_pii(test_email, salt=new_salt)

        # Both hashes should be resolvable
        assert salt_manager.is_hash_valid(hash_old, test_email), "Old salt hash should be valid during dual-valid window"
        assert salt_manager.is_hash_valid(hash_new, test_email), "New salt hash should be valid during dual-valid window"

        # After 24h, only new salt should be valid
        with patch("core.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = rotation_time + timedelta(hours=25)
            assert not salt_manager.is_hash_valid(hash_old, test_email), "Old salt hash should be invalid after 24h"
            assert salt_manager.is_hash_valid(hash_new, test_email), "New salt hash should remain valid"

    def test_salt_age_metric(self, salt_manager, metrics_client):
        """
        ACCEPTANCE: metric: salt_age_days <90 continuously
        """
        # Check initial salt age
        age_days = salt_manager.get_salt_age_days()
        assert age_days < 90, f"Salt age {age_days} should be <90 days"

        # Verify metric is reported
        metric_value = metrics_client.get_gauge("salt_age_days")
        assert metric_value < 90, f"Metric salt_age_days={metric_value} should be <90"

        # Simulate 90-day advancement (should trigger rotation)
        with patch("core.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=90)
            salt_manager.check_rotation_trigger()

            # After rotation, age should reset to 0
            age_after_rotation = salt_manager.get_salt_age_days()
            assert age_after_rotation < 1, f"Salt age {age_after_rotation} should reset to 0 after rotation"

    def test_user_key_resolution_no_failures(self, salt_manager, db_session):
        """
        ACCEPTANCE: metric: user_key_resolution_failures=0 during rotation window
        """
        # Create test user with email hash using old salt
        test_email = "user@example.com"
        old_salt = salt_manager.get_current_salt()
        old_hash = hash_pii(test_email, salt=old_salt)

        # Store in DB
        user = {"email_hash": old_hash, "user_key": "ukey_001"}
        db_session.execute("INSERT INTO dim_user (user_key, email_hash) VALUES (:user_key, :email_hash)", user)
        db_session.commit()

        # Trigger rotation
        with patch("core.security.datetime") as mock_dt:
            rotation_time = datetime.utcnow() + timedelta(days=90)
            mock_dt.utcnow.return_value = rotation_time
            salt_manager.rotate_salt()

        new_salt = salt_manager.get_current_salt()

        # During dual-valid window, resolve user_key with NEW hash (generated from new salt)
        new_hash = hash_pii(test_email, salt=new_salt)

        # Both old and new hashes should resolve to same user_key
        resolved_old = salt_manager.resolve_user_key(old_hash)
        resolved_new = salt_manager.resolve_user_key(new_hash)

        assert resolved_old == "ukey_001", "Old hash should resolve during dual-valid window"
        assert resolved_new == "ukey_001", "New hash should resolve during dual-valid window"

        # Check metric: user_key_resolution_failures should be 0
        failures = metrics_client.get_counter("user_key_resolution_failures", lookback_hours=1)
        assert failures == 0, f"Expected 0 resolution failures, got {failures}"

    def test_rotation_cron_job(self, salt_manager):
        """
        Test the scheduled cron job that checks rotation daily
        """
        # Mock cron execution
        with patch("core.security.datetime") as mock_dt:
            # Day 89: should not rotate
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=89)
            result_89 = salt_manager.check_rotation_trigger()
            assert result_89["rotated"] is False

            # Day 90: should rotate
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=90)
            result_90 = salt_manager.check_rotation_trigger()
            assert result_90["rotated"] is True
            assert result_90["old_salt_expires_at"] is not None

    def test_concurrent_resolution_during_rotation(self, salt_manager):
        """
        Concurrency test: Multiple threads resolving user_keys during rotation
        All should succeed without failures
        """
        import threading
        from queue import Queue

        test_email = "concurrent@example.com"
        old_salt = salt_manager.get_current_salt()
        old_hash = hash_pii(test_email, salt=old_salt)

        # Trigger rotation in background
        def rotate_in_background():
            with patch("core.security.datetime") as mock_dt:
                mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=90)
                salt_manager.rotate_salt()

        rotation_thread = threading.Thread(target=rotate_in_background)
        rotation_thread.start()

        # Multiple threads trying to resolve during rotation
        results = Queue()

        def resolve_user_key():
            try:
                user_key = salt_manager.resolve_user_key(old_hash)
                results.put(("success", user_key))
            except Exception as e:
                results.put(("error", str(e)))

        threads = []
        for _ in range(10):
            t = threading.Thread(target=resolve_user_key)
            threads.append(t)
            t.start()

        rotation_thread.join()
        for t in threads:
            t.join()

        # All resolutions should succeed
        successes = 0
        errors = 0
        while not results.empty():
            status, _ = results.get()
            if status == "success":
                successes += 1
            else:
                errors += 1

        assert successes == 10, f"Expected 10 successful resolutions, got {successes}"
        assert errors == 0, f"Expected 0 errors during rotation, got {errors}"

    def test_kill_switch_disables_rotation(self, salt_manager):
        """
        Test SALT_ROTATION_ENABLED kill switch
        """
        # Disable rotation
        salt_manager.rotation_enabled = False

        initial_salt = salt_manager.get_current_salt()

        # Try to rotate (should be blocked)
        with patch("core.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(days=90)
            result = salt_manager.check_rotation_trigger()

        assert result["rotated"] is False, "Rotation should be blocked when kill switch disabled"
        assert salt_manager.get_current_salt() == initial_salt, "Salt should not change when rotation disabled"
