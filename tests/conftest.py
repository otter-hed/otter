"""Separate optional private-archive audits from public source tests."""
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "private_baseline: audit locally retained accepted NPZ archives")


def pytest_collection_modifyitems(config, items):
    root = Path(__file__).resolve().parents[1]
    # A partially missing/corrupted private archive set must still fail tests.
    # Only the intentionally archive-free public source export skips these audits.
    if not any((root / "benchmarks/baselines").glob("*/*.npz")):
        unavailable = pytest.mark.skip(reason="Optional private NPZ archive audit; public source excludes numerical archives")
        for item in items:
            if item.get_closest_marker("private_baseline"):
                item.add_marker(unavailable)
