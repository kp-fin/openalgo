"""Singleton entrypoint for hosted Python strategy subprocesses.

The strategy host launches this file instead of the strategy script directly.
We take an exclusive instance lock (path in OPENALGO_STRATEGY_LOCK_FILE) and
only then run the strategy as ``__main__``. A second copy exits immediately
instead of placing duplicate orders.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_OPENALGO_ROOT = Path(__file__).resolve().parent.parent
if str(_OPENALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENALGO_ROOT))

from utils.strategy_process_guard import ExclusiveFileLock  # noqa: E402

# Same convention as sysexits.h EX_TEMPFAIL — host should adopt the live copy.
INSTANCE_LOCK_HELD = 75


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    if len(args) < 2:
        sys.stderr.write("usage: _run_strategy.py <strategy_script>\n")
        return 2

    script = args[1]
    lock_file = os.environ.get("OPENALGO_STRATEGY_LOCK_FILE", "").strip()
    if not lock_file:
        sys.stderr.write("OPENALGO_STRATEGY_LOCK_FILE is not set\n")
        return 2

    lock = ExclusiveFileLock(lock_file)
    try:
        lock.acquire(blocking=False)
    except OSError:
        sys.stderr.write(
            f"Strategy instance lock held ({lock_file}); not starting a second copy.\n"
        )
        return INSTANCE_LOCK_HELD

    try:
        from utils.hosted_strategy_tag import install_hosted_strategy_tag

        install_hosted_strategy_tag()
        sys.argv = [script, *args[2:]]
        runpy.run_path(script, run_name="__main__")
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
