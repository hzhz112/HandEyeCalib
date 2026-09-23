"""Minimal test harness shared by the modules in this directory.

These tests run two ways. Under pytest they behave normally, with `skip()`
mapping onto `pytest.skip()`. As plain scripts -- `python -m tests.test_charuco`
-- `run_tests()` provides the reporting, because pytest is not a dependency of
this repository.
"""

from __future__ import annotations

import sys
import traceback


class Skip(Exception):
    """Raised when a test cannot run in this environment."""


def skip(reason: str) -> None:
    """Skip the current test, whichever way it is being run."""
    try:
        import pytest
    except ImportError:
        raise Skip(reason)
    pytest.skip(reason)


def run_tests(namespace: dict) -> int:
    """Run every test_* callable in `namespace`. Returns a process exit code."""
    tests = [
        obj
        for name, obj in sorted(namespace.items())
        if name.startswith("test_") and callable(obj)
    ]
    if not tests:
        print("No tests found.")
        return 1

    passed = skipped = failed = 0
    for test in tests:
        try:
            test()
        except Skip as reason:
            skipped += 1
            print(f"SKIP  {test.__name__}: {reason}")
        except Exception as error:  # noqa: BLE001 - a test runner reports everything
            failed += 1
            print(f"FAIL  {test.__name__}: {error!r}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {test.__name__}")

    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


def main(namespace: dict) -> None:
    sys.exit(run_tests(namespace))
