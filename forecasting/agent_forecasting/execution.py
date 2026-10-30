"""Small Linux/main-thread execution primitives shared by M1 and r9.

Directory rename is the publication commit point. A subsequent fsync failure
cannot undo visibility: callers receive ``durable=False`` and must not retry
the committed run. This is atomic visibility, not a promise against power loss.
"""
from __future__ import annotations

import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


class DeadlineExceeded(TimeoutError):
    pass


def call_with_deadline(callback, seconds):
    """Interrupt a blocking callback, preserving the earliest enclosing alarm."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise DeadlineExceeded("no remaining execution time")
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    prior_deadline = started + previous_timer[0] if previous_timer[0] else None
    deadline = started + seconds
    if prior_deadline is not None:
        deadline = min(deadline, prior_deadline)
    prior_fired = False

    def expired(signum, frame):
        nonlocal prior_fired
        if prior_deadline is not None and time.monotonic() >= prior_deadline:
            prior_fired = True
            if callable(previous_handler):
                previous_handler(signum, frame)
        raise DeadlineExceeded("execution deadline exceeded")

    signal.signal(signal.SIGALRM, expired)  # explicitly requires main thread
    signal.setitimer(signal.ITIMER_REAL, max(deadline - time.monotonic(), 1e-6))
    try:
        result = callback()
        if time.monotonic() >= deadline:
            expired(signal.SIGALRM, None)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if prior_deadline is not None and not prior_fired:
            signal.setitimer(signal.ITIMER_REAL,
                            max(prior_deadline - time.monotonic(), 1e-6), previous_timer[1])
        elif prior_fired and previous_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[1], previous_timer[1])


@dataclass(frozen=True)
class DirectoryCommit:
    path: Path
    durable: bool


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory(staging, final, *, deadline=None, clock=time.monotonic):
    """Publish closed staged files/COMMIT together with one directory rename."""
    staging, final = Path(staging), Path(final)
    if not (staging / "COMMIT.json").is_file():
        raise ValueError("COMMIT.json must already exist in staging")
    if final.exists():
        raise FileExistsError(final)
    final.parent.mkdir(parents=True, exist_ok=True)

    def sync_staging():
        for path in staging.rglob("*"):
            if path.is_file():
                _sync(path)
        _sync(staging)

    if deadline is None:
        sync_staging()
    else:
        call_with_deadline(sync_staging, deadline - clock())
        if clock() >= deadline:
            raise DeadlineExceeded("publication deadline exceeded before commit")
    try:
        if deadline is None:
            os.rename(staging, final)
        else:
            call_with_deadline(lambda: os.rename(staging, final), deadline - clock())
    except BaseException:
        if not staging.exists() and (final / "COMMIT.json").is_file():
            return DirectoryCommit(final, False)
        raise
    try:
        if deadline is None:
            _sync(final.parent)
        else:
            call_with_deadline(lambda: _sync(final.parent), deadline - clock())
    except BaseException:
        return DirectoryCommit(final, False)
    return DirectoryCommit(final, True)
