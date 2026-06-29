import json
from pathlib import Path

_CFG = Path(__file__).resolve().parents[2] / "config" / "configuration.json"
_MULTIDAY = ("mtf_capitulation_revert_long", "low52_capitulation_revert_long",
             "zscore_oversold_revert_long", "crash2d_revert_long")


def _load():
    return json.loads(_CFG.read_text(encoding="utf-8"))


def test_each_multiday_setup_has_composite_weight_and_clip():
    cfg = _load()
    for name in _MULTIDAY:
        block = cfg["setups"][name]
        assert isinstance(block["composite_weight"], (int, float))
        assert float(block["composite_weight"]) == 1.0  # equal-weight v1
        assert float(block["cap_score_clip"]) > 0.0


def test_multi_day_portfolio_family_block_present():
    cfg = _load()
    fam = cfg["multi_day_portfolio"]
    assert int(fam["max_new_per_day"]) > 0
    assert int(fam["max_concurrent"]) > 0
    assert float(fam["cap_score_clip"]) > 0.0
    assert fam["tiebreaker"] == "tshock"
    assert isinstance(fam["selection_log_path"], str) and fam["selection_log_path"]
