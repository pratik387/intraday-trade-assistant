"""Tests for period boundary assignment.

Indian financial year: April 1 to March 31.
FY2022-23 = 2022-04-01 to 2023-03-31.
Period config = dict {period_name: (start_inclusive, end_inclusive)}.
"""
from datetime import date
import pytest

from tools.edge_discovery_legacy_gauntlet.periods import (
    assign_fy,
    get_discovery_subperiods,
    DiscoveryConfig,
)


def test_fy_boundary_march_31():
    assert assign_fy(date(2023, 3, 31)) == "FY2022-23"


def test_fy_boundary_april_1():
    assert assign_fy(date(2023, 4, 1)) == "FY2023-24"


def test_fy_mid_year():
    assert assign_fy(date(2023, 12, 15)) == "FY2023-24"


def test_discovery_config_lock():
    """Discovery period dates are locked and must not be mutated."""
    cfg = DiscoveryConfig(
        discovery_start=date(2023, 1, 1),
        discovery_end=date(2024, 12, 31),
        validation_start=date(2025, 1, 1),
        validation_end=date(2025, 9, 30),
        holdout_start=date(2025, 10, 1),
        holdout_end=date(2026, 3, 31),
    )
    # Dataclass must be frozen — mutation raises
    # FrozenInstanceError is a subclass of AttributeError (dataclasses stdlib).
    # Using AttributeError keeps the test precise — a broad Exception catch
    # would mask unrelated errors.
    with pytest.raises(AttributeError):
        cfg.discovery_end = date(2024, 6, 30)


def test_discovery_config_rejects_overlapping_validation():
    """Validation start must strictly follow discovery end — overlap is rejected."""
    with pytest.raises(ValueError, match="Discovery must end before Validation starts"):
        DiscoveryConfig(
            discovery_start=date(2023, 1, 1),
            discovery_end=date(2025, 6, 1),
            validation_start=date(2025, 1, 1),
            validation_end=date(2025, 9, 30),
            holdout_start=date(2025, 10, 1),
            holdout_end=date(2026, 3, 31),
        )


def test_discovery_config_rejects_overlapping_holdout():
    """Holdout start must strictly follow validation end."""
    with pytest.raises(ValueError, match="Validation must end before Holdout starts"):
        DiscoveryConfig(
            discovery_start=date(2023, 1, 1),
            discovery_end=date(2024, 12, 31),
            validation_start=date(2025, 1, 1),
            validation_end=date(2025, 12, 31),
            holdout_start=date(2025, 10, 1),
            holdout_end=date(2026, 3, 31),
        )


def test_discovery_subperiods_yields_two_halves():
    """Discovery splits into two equal halves for sub-period PF check."""
    cfg = DiscoveryConfig(
        discovery_start=date(2023, 1, 1),
        discovery_end=date(2024, 12, 31),
        validation_start=date(2025, 1, 1),
        validation_end=date(2025, 9, 30),
        holdout_start=date(2025, 10, 1),
        holdout_end=date(2026, 3, 31),
    )
    h1, h2 = get_discovery_subperiods(cfg)
    assert h1 == (date(2023, 1, 1), date(2024, 1, 1))
    assert h2 == (date(2024, 1, 2), date(2024, 12, 31))
