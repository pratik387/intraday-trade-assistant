from services.multiday_composite_selector import MultiDayCompositeSelector


def _cand(sym, cap_score, tshock=2.0, close=100.0, trail_ret=-0.1):
    return {"symbol": sym, "cap_score": cap_score, "tshock": tshock,
            "close": close, "trail_ret": trail_ret, "adv_tier": 1, "rank_pct": 0.02}


def _sel(clip=3.0):
    return MultiDayCompositeSelector({"max_new_per_day": 10, "max_concurrent": 50,
                                      "cap_score_clip": clip, "tiebreaker": "tshock"})


def test_consensus_sum_outranks_single_setup():
    # ABC flagged by two setups (1.0 + 1.0 = 2.0) beats XYZ flagged once at 1.5.
    baskets = {
        "A2": [_cand("ABC", 1.0), _cand("XYZ", 1.5)],
        "C1": [_cand("ABC", 1.0)],
    }
    chosen = _sel().select(baskets, held_symbols=set(),
                           weights={"A2": 1.0, "C1": 1.0}, limit=10)
    assert [c["bare"] for c in chosen] == ["ABC", "XYZ"]
    abc = chosen[0]
    assert abc["composite"] == 2.0
    assert sorted(abc["contributors"]) == ["A2", "C1"]
    assert abc["per_setup_cap_score"] == {"A2": 1.0, "C1": 1.0}


def test_deep_single_setup_can_outrank_mild_consensus():
    baskets = {"A2": [_cand("DEEP", 3.0)], "C1": [_cand("MILD", 0.4)],
               "C4": [_cand("MILD", 0.4)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0, "C1": 1.0, "C4": 1.0}, limit=10)
    assert chosen[0]["bare"] == "DEEP"  # 3.0 > 0.8


def test_dedup_owner_is_max_weighted_contributor():
    baskets = {"A2": [_cand("ABC", 0.5)], "C1": [_cand("ABC", 2.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0, "C1": 1.0}, limit=10)
    assert len(chosen) == 1
    assert chosen[0]["owner"] == "C1"  # higher weighted cap_score


def test_held_symbols_excluded():
    baskets = {"A2": [_cand("ABC", 2.0), _cand("XYZ", 1.0)]}
    chosen = _sel().select(baskets, held_symbols={"ABC"}, weights={"A2": 1.0}, limit=10)
    assert [c["bare"] for c in chosen] == ["XYZ"]


def test_nse_prefix_normalized_for_held_and_output():
    baskets = {"A2": [_cand("NSE:ABC", 2.0)]}
    chosen = _sel().select(baskets, held_symbols={"ABC"}, weights={"A2": 1.0}, limit=10)
    assert chosen == []  # NSE:ABC dedupes against bare-held ABC


def test_limit_caps_after_ranking():
    baskets = {"A2": [_cand("A", 3.0), _cand("B", 2.0), _cand("C", 1.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 1.0}, limit=2)
    assert [c["bare"] for c in chosen] == ["A", "B"]


def test_cap_score_upper_clip_applied():
    baskets = {"A2": [_cand("ABC", 9.0)], "C1": [_cand("ABC", 9.0)]}
    chosen = _sel(clip=3.0).select(baskets, set(), {"A2": 1.0, "C1": 1.0}, limit=10)
    assert chosen[0]["composite"] == 6.0  # min(9,3)*1 + min(9,3)*1


def test_weights_scale_contribution():
    baskets = {"A2": [_cand("ABC", 1.0)], "C1": [_cand("XYZ", 1.0)]}
    chosen = _sel().select(baskets, set(), {"A2": 2.0, "C1": 1.0}, limit=10)
    assert chosen[0]["bare"] == "ABC" and chosen[0]["composite"] == 2.0
