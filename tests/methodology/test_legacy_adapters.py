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


def test_pre_results_t1_adapter_pnl_pct_matches_prices_for_single_leg(pre_results_t1_discovery_df):
    """For SHORT single-leg trades (NOT t1_booked), pnl_pct should equal
    (entry-exit)/entry*100 within float tolerance. This is the sign-convention
    invariant.

    Multi-leg trades (t1_partial_booked=True) are excluded — their pnl_pct
    is blended and cannot be derived from single entry/exit.
    """
    df_out = adapt_pre_results_t1_fade(pre_results_t1_discovery_df)
    # Filter to single-leg rows only
    single_leg = df_out[~df_out["t1_partial_booked"]]
    sample = single_leg.sample(n=min(100, len(single_leg)), random_state=20260520)
    computed = (sample["entry_price"] - sample["exit_price"]) / sample["entry_price"] * 100.0
    diff = (sample["pnl_pct"] - computed).abs()
    assert (diff < 0.10).all(), (
        f"sign-convention check failed on {(diff >= 0.10).sum()} of {len(sample)} single-leg rows. "
        f"Max diff: {diff.max():.4f}pp"
    )


def test_pre_results_t1_adapter_blends_pnl_pct_for_t1_booked(pre_results_t1_discovery_df):
    """For t1_booked rows where entry==exit (breakeven hit), the legacy CSV
    has pnl_pct=0 (BUG), but the adapter recomputes pnl_pct from realized_pnl
    so it reflects the T1 partial profit.

    The CRITICAL regression test: post-adapter pnl_pct for these rows must
    be > 0 (T1 partial profit on half qty), not 0 (legacy bug).
    """
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)

    # Find rows that were buggy in legacy: t1_booked=True AND entry==exit AND stop
    legacy_bug_mask = (
        (df_in["t1_booked"].astype(bool))
        & (df_in["entry_price"] == df_in["exit_price"])
        & (df_in["exit_reason"] == "stop")
    )
    if not legacy_bug_mask.any():
        pytest.skip("No buggy-scratch rows in fixture (unexpected)")

    # In input: legacy pnl_pct was 0 for these
    assert (df_in.loc[legacy_bug_mask, "pnl_pct"] == 0.0).all()

    # In output: pnl_pct should be positive (T1 partial profit booked) for these
    out_pnl = df_out.loc[legacy_bug_mask, "pnl_pct"]
    assert (out_pnl > 0).all(), (
        f"adapter failed to blend pnl_pct for {(out_pnl <= 0).sum()} buggy-scratch rows"
    )
    # Magnitude check: for a 0.3% SL and 50% T1 partial, blended ~= 0.15%
    # Sanity: blended should be in [0.05%, 0.30%] range
    assert (out_pnl > 0.05).all() and (out_pnl < 0.30).all(), (
        f"blended pnl_pct out of expected range [0.05, 0.30]: "
        f"min={out_pnl.min():.4f}, max={out_pnl.max():.4f}"
    )


def test_pre_results_t1_adapter_emits_breakeven_stop_exit_reason(pre_results_t1_discovery_df):
    """For t1_booked + entry==exit + legacy='stop', adapter emits 'breakeven_stop'."""
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)

    bes_input_mask = (
        (df_in["t1_booked"].astype(bool))
        & (df_in["entry_price"] == df_in["exit_price"])
        & (df_in["exit_reason"] == "stop")
    )
    if not bes_input_mask.any():
        pytest.skip("No buggy-scratch rows in fixture")

    # All these rows in output should have exit_reason='breakeven_stop'
    assert (df_out.loc[bes_input_mask, "exit_reason"] == "breakeven_stop").all()
    # And t1_partial_booked=True
    assert df_out.loc[bes_input_mask, "t1_partial_booked"].all()


def test_pre_results_t1_adapter_sets_t1_partial_booked_correctly(pre_results_t1_discovery_df):
    """t1_partial_booked output column should mirror the legacy t1_booked column."""
    df_in = pre_results_t1_discovery_df
    df_out = adapt_pre_results_t1_fade(df_in)
    legacy = df_in["t1_booked"].astype(bool).values
    new = df_out["t1_partial_booked"].astype(bool).values
    assert (legacy == new).all()


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
