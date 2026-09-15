"""Cross-platform exclusive file locks whose ownership follows an open OS handle."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire one nonblocking thread/process lock until the context exits."""

    path = Path(lock_path).resolve()
    thread_lock = _thread_lock(path)
    if not thread_lock.acquire(blocking=False):
        raise TimeoutError(f"resource is already locked: {path}")
    acquired = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                _try_file_lock(stream)
                acquired = True
            except OSError as error:
                raise TimeoutError(f"resource is already locked: {path}") from error
            try:
                yield
            finally:
                if acquired:
                    _unlock_file(stream)
    finally:
        thread_lock.release()


def _thread_lock(lock_path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(lock_path, threading.Lock())


def _try_file_lock(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
