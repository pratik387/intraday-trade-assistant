"""Tests for tools.methodology.legacy_adapters.

For each registered adapter, we verify:
  - It loads the real legacy CSV without error
  - Output passes canonical schema validation
  - Row count preserved (no silent drops)
  - Sign-convention invariants hold on a sample of rows
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.methodology.legacy_adapters import (
    adapt_pre_results_t1_fade, get_adapter, ADAPTERS,
)
from tools.methodology.sanity_csv_schema import validate


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def test_get_adapter_returns_registered():
    fn = get_adapter("pre_results_t1_fade")
    assert callable(fn)


def test_get_adapter_raises_for_unknown_setup():
    with pytest.raises(KeyError, match="No legacy adapter"):
        get_adapter("does_not_exist")


# ---------------------------------------------------------------------------
# Adapter: pre_results_t1_fade
# ---------------------------------------------------------------------------

@pytest.fixture
def pre_results_t1_discovery_df() -> pd.DataFrame:
    p = _REPO_ROOT / "reports" / "sub9_sanity" / "_pre_results_t1_v2_trades_discovery.csv"
    if not p.exists():
        pytest.skip(f"Legacy CSV missing: {p}")
    return pd.read_csv(p)


def test_pre_results_t1_adapter_preserves_row_count(pre_results_t1_discovery_df):
    """Adapter must not silently drop rows."""
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)
    assert len(df_out) == len(df_in)


def test_pre_results_t1_adapter_output_passes_validator(pre_results_t1_discovery_df):
    """Critical: adapter output must validate cleanly (zero errors)."""
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)
    result = validate(df_out, setup_name="pre_results_t1_fade", layer="filtered_trades")
    assert result.is_valid, f"validation failed:\n{result.summary()}"


def test_pre_results_t1_adapter_sets_side_to_short(pre_results_t1_discovery_df):
    """This is a SHORT setup — every row must say SHORT."""
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    assert (df_out["side"] == "SHORT").all()


def test_pre_results_t1_adapter_symbol_has_nse_prefix(pre_results_t1_discovery_df):
    """Symbol must be NSE:XYZ format after adapter runs."""
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    assert df_out["symbol"].str.startswith("NSE:").all()


def test_pre_results_t1_adapter_pnl_pct_matches_prices(pre_results_t1_discovery_df):
    """For SHORT trades, pnl_pct should equal (entry-exit)/entry*100 within float tolerance.
    This is the sign-convention invariant — if it fails, the adapter has a bug
    that would cause walk-forward to silently invert verdicts.
    """
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    # Random sample of 100 rows for speed
    sample = df_out.sample(n=min(100, len(df_out)), random_state=20260520)
    computed = (sample["entry_price"] - sample["exit_price"]) / sample["entry_price"] * 100.0
    diff = (sample["pnl_pct"] - computed).abs()
    # Tolerance: 0.10pp (matches validator)
    assert (diff < 0.10).all(), (
        f"sign-convention check failed on {(diff >= 0.10).sum()} of {len(sample)} rows. "
        f"Max diff: {diff.max():.4f}pp"
    )


def test_pre_results_t1_adapter_exit_reason_canonical(pre_results_t1_discovery_df):
    """All exit_reason values must be in canonical set after adapter."""
    from tools.methodology.sanity_csv_schema import EXIT_REASONS
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    bad = ~df_out["exit_reason"].isin(EXIT_REASONS)
    assert not bad.any(), (
        f"non-canonical exit_reason values: "
        f"{sorted(df_out.loc[bad, 'exit_reason'].unique())}"
    )


def test_pre_results_t1_adapter_same_bar_sl_promotion(pre_results_t1_discovery_df):
    """Verify the 'stop' + same_bar=True → 'same_bar_sl' promotion happens."""
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)
    # Find rows that were 'stop' + same_bar_exit=True in input
    legacy_sb_sl_mask = (df_in["exit_reason"] == "stop") & (df_in["same_bar_exit"].astype(bool))
    if not legacy_sb_sl_mask.any():
        pytest.skip("No same-bar SL rows in discovery sample — adapter promotion path uncovered")
    # Those rows should have exit_reason=same_bar_sl in output
    assert (df_out.loc[legacy_sb_sl_mask, "exit_reason"] == "same_bar_sl").all()


def test_pre_results_t1_adapter_rupee_passthrough(pre_results_t1_discovery_df):
    """Rupee columns (realized_pnl_inr, fee_inr, net_pnl_inr) should be passed through."""
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    for col in ("realized_pnl_inr", "fee_inr", "net_pnl_inr"):
        assert col in df_out.columns, f"missing optional column: {col}"
    # Spot-check: rupee internal consistency
    sample = df_out.sample(n=min(50, len(df_out)), random_state=20260520)
    computed_net = sample["realized_pnl_inr"] - sample["fee_inr"]
    # Tolerance: 1 rupee or 1% of |realized|
    tol = np.maximum(1.0, sample["realized_pnl_inr"].abs() * 0.01)
    bad = (sample["net_pnl_inr"] - computed_net).abs() > tol
    assert not bad.any(), (
        f"{bad.sum()} rows where net_pnl_inr != realized_pnl_inr - fee_inr"
    )
