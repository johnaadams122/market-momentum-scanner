import pytest
from momentum_scanner.daemon_lock import single_instance_lock, AlreadyRunningError


def test_acquires_and_releases(tmp_path):
    lp = tmp_path / "daemon.lock"
    with single_instance_lock(lp):
        assert lp.exists()
    with single_instance_lock(lp):
        pass


def test_second_instance_refuses_immediately(tmp_path):
    lp = tmp_path / "daemon.lock"
    with single_instance_lock(lp):
        with pytest.raises(AlreadyRunningError):
            with single_instance_lock(lp):
                pass


def test_creates_missing_parent(tmp_path):
    lp = tmp_path / "state" / "daemon.lock"
    with single_instance_lock(lp):
        assert lp.exists()
