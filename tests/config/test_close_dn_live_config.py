from config.filters_setup import load_filters

def test_close_dn_has_gtt_and_live_pilot_keys():
    cfg = load_filters()
    s = cfg["setups"]["close_dn_overnight_long"]
    assert float(s["catastrophe_stop_pct"]) == 5.0
    assert float(s["gtt_limit_buffer_pct"]) > 0.0
    ca = s["capital_allocation"]
    # Live-pilot caps documented for the 1-slot first run.
    assert int(ca["_live_pilot_max_concurrent_slots"]) == 1
    assert int(ca["_live_pilot_margin_per_slot_inr"]) == 10000
    # Full live caps: 2026-07-02 resize — 6 x Rs50k slots / Rs3L (corrected-fee
    # confidence card; the old 25 x Rs10k "capture-everything" failed the ship bar).
    assert int(ca["_live_margin_per_slot_inr"]) == 50000
    assert int(ca["_live_active_margin_inr"]) == 300000
    assert int(ca["_live_max_concurrent_slots"]) == 6
