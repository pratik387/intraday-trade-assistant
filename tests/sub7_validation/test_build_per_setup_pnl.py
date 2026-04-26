"""build_per_setup_pnl tests (sub7-T9)."""
import pandas as pd
from tools.sub7_validation.build_per_setup_pnl import calc_fee, build_net_per_setup


def test_calc_fee_long_trade_basic():
    fee = calc_fee(entry_price=100.0, exit_price=102.0, qty=100, side="BUY")
    assert 10.0 < fee < 12.0


def test_calc_fee_short_trade_basic():
    fee = calc_fee(entry_price=100.0, exit_price=98.0, qty=100, side="SELL")
    assert 10.0 < fee < 12.0


def test_calc_fee_zero_qty():
    assert calc_fee(100.0, 102.0, 0, "BUY") == 0.0


def test_build_net_per_setup_groups_by_setup():
    df = pd.DataFrame({
        'session_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'setup_type': ['mis_unwind_short', 'gap_fade_short', 'mis_unwind_short'],
        'entry_price': [100.0, 50.0, 200.0],
        'e1_price': [98.0, 52.0, 195.0],
        'qty': [100, 200, 50],
        'side': ['SELL', 'SELL', 'SELL'],
        'realized_pnl': [200.0, -400.0, 250.0],
        'executed': [True, True, True],
    })
    out = build_net_per_setup(df)
    assert set(out['setup_type'].unique()) == {'mis_unwind_short', 'gap_fade_short'}
    assert (out['fee'] > 0).all()
    assert 'net_pnl' in out.columns
    for _, row in out.iterrows():
        assert abs(row['net_pnl'] - (row['realized_pnl'] - row['fee'])) < 0.01
