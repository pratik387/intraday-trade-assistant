"""Execution cost model: bid-ask spread + slippage + market impact."""
from __future__ import annotations

from typing import Optional


def _adv_bucket(adv_shares: float) -> str:
    if adv_shares < 100_000:
        return "adv_lt_100k"
    if adv_shares < 500_000:
        return "adv_100k_500k"
    if adv_shares < 2_000_000:
        return "adv_500k_2m"
    return "adv_gt_2m"


class ExecutionCosts:
    """Per-cap x ADV spread + SL slippage + linear market impact (capped).

    Config keys (all required, no defaults):
      spread_by_cap_adv: {cap_segment: {adv_bucket: pct_per_side}}
      sl_slippage_bar_fraction: float
      sl_slippage_normal_pct: float
      market_impact_pct_per_pct_adv: float
      market_impact_cap_pct: float
    """

    def __init__(self, config_block: dict) -> None:
        required = (
            "spread_by_cap_adv",
            "sl_slippage_bar_fraction",
            "sl_slippage_normal_pct",
            "market_impact_pct_per_pct_adv",
            "market_impact_cap_pct",
        )
        for k in required:
            if k not in config_block:
                raise KeyError(f"cost_model config missing required key: {k}")
        self.cfg = config_block

    def spread_pct(self, cap_segment: str, adv_shares: float) -> float:
        bucket = _adv_bucket(adv_shares)
        try:
            return float(self.cfg["spread_by_cap_adv"][cap_segment][bucket])
        except KeyError:
            raise KeyError(
                f"No spread config for cap_segment={cap_segment} adv_bucket={bucket}"
            )

    def market_impact_pct(self, order_shares: float, adv_shares: float) -> float:
        if adv_shares <= 0:
            return 0.0
        size_pct = float(order_shares) / float(adv_shares)
        raw = size_pct * float(self.cfg["market_impact_pct_per_pct_adv"])
        return min(raw, float(self.cfg["market_impact_cap_pct"]))

    def sl_slippage_pct(self, bar_range_pct: Optional[float]) -> float:
        if bar_range_pct is not None and bar_range_pct > 0:
            return float(self.cfg["sl_slippage_bar_fraction"]) * float(bar_range_pct)
        return float(self.cfg["sl_slippage_normal_pct"])

    def apply_round_trip(
        self,
        gross_return_pct: float,
        cap_segment: str,
        adv_shares: float,
        order_shares: float,
        sl_hit: bool,
        sl_bar_range_pct: Optional[float],
    ) -> float:
        """Subtract entry-side spread + exit-side spread + impact (each side) + (if SL) slippage."""
        side_spread = self.spread_pct(cap_segment, adv_shares)
        side_impact = self.market_impact_pct(order_shares, adv_shares)
        total_drag = 2 * side_spread + 2 * side_impact
        if sl_hit:
            total_drag += self.sl_slippage_pct(sl_bar_range_pct)
        return gross_return_pct - total_drag
