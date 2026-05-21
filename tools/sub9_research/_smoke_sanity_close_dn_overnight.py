"""Smoke test for sanity_close_dn_overnight_long over a 2-month window."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.sub9_research.sanity_close_dn_overnight_long as mod

# Override windows to 2 months
mod.WINDOWS = {"discovery": (date(2024, 4, 1), date(2024, 5, 31))}
mod.MIN_TRADING_DAYS_COVERAGE = 0.50  # 2-month subset, looser coverage

print("=" * 80)
print("SMOKE TEST — close_dn_overnight_long (2 months, Apr-May 2024)")
print("=" * 80)

trades = mod.run_window("discovery")
print(f"\n[OK] {len(trades):,} trades generated")
if not trades.empty:
    print()
    print(trades["signed_vol_ratio_bin"].value_counts().to_string())
    print()
    print("cap_segment distribution:")
    print(trades["cap_segment"].value_counts().to_string())
    print()
    print("news_proximity:")
    print(trades["news_proximity"].value_counts().to_string())
    print()
    print("First 3 trade rows:")
    print(trades[["signal_date","symbol","entry_price","exit_price","net_pnl_inr","cap_segment","signed_vol_ratio","closing_30m_volume_z","news_proximity"]].head(3).to_string(index=False))
print("\nSMOKE PASSED" if not trades.empty else "\nSMOKE FAILED — no trades")
