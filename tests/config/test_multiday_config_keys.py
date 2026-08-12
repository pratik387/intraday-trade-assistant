import json
from pathlib import Path

_CFG = Path(__file__).resolve().parents[2] / "config" / "configuration.json"
_MULTIDAY = ("mtf_capitulation_revert_long", "low52_capitulation_revert_long",
             "zscore_oversold_revert_long", "crash2d_revert_long")


def _load():
    return json.loads(_CFG.read_text(encoding="utf-8"))


def test_each_multiday_setup_has_composite_weight():
    """Equal-weight v1 for every ACTIVE setup; a disabled one may be zeroed.

    The point of this assertion is to stop weights being quietly tuned into an
    implicit alpha bet — composite ordering has no measured predictive power
    (consensus test 2026-08-12: permutation p=0.69), so unequal weights would be
    an unvalidated claim.

    A setup that is switched OFF is a different case: `crash2d_revert_long` was
    disabled 2026-08-12 (mean -0.885%, 95% CI [-1.758, -0.012], the only CI in
    the book excluding zero) and its weight set to 0.0 so it also stops pulling
    shared names in through consensus scoring. Zero-weight is therefore allowed
    ONLY while disabled; re-enabling must restore 1.0.
    """
    cfg = _load()
    for name in _MULTIDAY:
        block = cfg["setups"][name]
        assert isinstance(block["composite_weight"], (int, float))
        w = float(block["composite_weight"])
        enabled = bool(block.get("enabled")) or bool(block.get("paper_enabled"))
        if enabled:
            assert w == 1.0, f"{name} is active but not equal-weight (w={w})"
        else:
            assert w in (0.0, 1.0), f"{name} disabled with unexpected weight {w}"
        # cap_score_clip is family-level only (multi_day_portfolio); the
        # composite selector reads it there, not per-setup. No dead per-setup key.
        assert "cap_score_clip" not in block


def test_multi_day_portfolio_family_block_present():
    cfg = _load()
    fam = cfg["multi_day_portfolio"]
    assert int(fam["max_new_per_day"]) > 0
    assert int(fam["max_concurrent"]) > 0
    assert float(fam["cap_score_clip"]) > 0.0
    assert fam["tiebreaker"] == "tshock"
    assert isinstance(fam["selection_log_path"], str) and fam["selection_log_path"]
