from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest


def test_exclusive_file_lock_rejects_a_concurrent_thread_and_releases_with_handle(
    tmp_path: Path,
) -> None:
    from bone_cancer_ai.utils.locking import exclusive_file_lock

    lock_path = tmp_path / "run" / ".run.lock"
    acquired = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with exclusive_file_lock(lock_path):
                acquired.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release the held lock")
        except BaseException as error:
            errors.append(error)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(timeout=5)
    try:
        with pytest.raises(TimeoutError, match="already locked"):
            with exclusive_file_lock(lock_path):
                pytest.fail("concurrent lock acquisition unexpectedly succeeded")
    finally:
        release.set()
        holder.join(timeout=5)

    assert not holder.is_alive()
    assert errors == []
    assert lock_path.is_file()
    with exclusive_file_lock(lock_path):
        pass


def test_exclusive_file_lock_is_process_backed_and_crash_releases_it(tmp_path: Path) -> None:
    from bone_cancer_ai.utils.locking import exclusive_file_lock

    lock_path = tmp_path / "run" / ".run.lock"
    child_code = "\n".join(
        [
            "import sys, time",
            "from pathlib import Path",
            "from bone_cancer_ai.utils.locking import exclusive_file_lock",
            "lock = exclusive_file_lock(Path(sys.argv[1]))",
            "lock.__enter__()",
            "print('acquired', flush=True)",
            "time.sleep(30)",
        ]
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "acquired"
        with pytest.raises(TimeoutError, match="already locked"):
            with exclusive_file_lock(lock_path):
                pytest.fail("cross-process lock acquisition unexpectedly succeeded")
    finally:
        child.kill()
        child.wait(timeout=5)

    with exclusive_file_lock(lock_path):
        pass
