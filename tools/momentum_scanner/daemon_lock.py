"""Single-instance guard for the movers discovery daemon. An OS-held msvcrt
byte-range lock: the OS releases it on handle-close or process death, so there is no stale lockfile
and no PID-liveness reclaim. A second daemon fails fast with AlreadyRunningError."""
import os
import msvcrt
from contextlib import contextmanager
from pathlib import Path


class AlreadyRunningError(Exception):
    pass


@contextmanager
def single_instance_lock(lock_path):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            raise AlreadyRunningError(f"another daemon holds {lock_path}")
        try:
            yield
        finally:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    finally:
        os.close(fd)
