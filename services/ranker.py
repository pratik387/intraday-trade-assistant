# services/ranker.py
from __future__ import annotations
from typing import List, Dict, Optional

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters

logger = get_agent_logger()


def _intraday_strength(iv: Dict, strategy_type: str = None) -> float:
    """
    Compute an intraday strength score from a compact feature dict.

    Expected keys in `iv` (all optional; missing treated as neutral):
      - volume_ratio: float
      - rsi: float
      - rsi_slope: float
      - adx: float
      - adx_slope: float
      - above_vwap: bool
      - dist_from_level_bpct: float   # signed % distance from key level (ORH/ORL/PDH/PDL/etc.)
      - squeeze_pctile: float         # 0..100 (lower = tighter)
      - acceptance_ok: bool
      - bias: "long" | "short"

    All tunables are read from entry_config.json (no in-code defaults).
    """
    try:
        cfg = load_filters()

        # ---- Get strategy-specific parameters if available ----
        strategy_profile = None
        if strategy_type and "strategy_ranking_profiles" in cfg:
            profiles = cfg["strategy_ranking_profiles"]
            for profile_name, profile in profiles.items():
                if strategy_type in profile.get("strategies", []):
                    strategy_profile = profile
                    logger.debug(f"ranker: using {profile_name} profile for strategy {strategy_type}")
                    break
            if not strategy_profile:
                logger.debug(f"ranker: no profile found for strategy {strategy_type}, using defaults")
        elif strategy_type and "strategy_ranking_profiles" not in cfg:
            logger.warning(f"ranker: strategy_ranking_profiles not found in config, using defaults for {strategy_type}")

        # ---- weights, caps, thresholds (strategy-specific or default) ----
        if strategy_profile:
            vol_div     = float(strategy_profile.get("rank_vol_ratio_divisor", cfg["rank_vol_ratio_divisor"]))
            vol_cap     = float(strategy_profile.get("rank_vol_cap", cfg["rank_vol_cap"]))
            adx_mid     = float(strategy_profile.get("rank_adx_mid", cfg["rank_adx_mid"]))
            adx_div     = float(strategy_profile.get("rank_adx_divisor", cfg["rank_adx_divisor"]))
            vwap_bonus  = float(strategy_profile.get("rank_vwap_bonus", cfg["rank_vwap_bonus"]))
            vwap_pen    = float(strategy_profile.get("rank_vwap_penalty", cfg["rank_vwap_penalty"]))
            dist_near_s = float(strategy_profile.get("rank_dist_near_score", cfg["rank_dist_near_score"]))
            acc_bonus   = float(strategy_profile.get("rank_acceptance_bonus", cfg["rank_acceptance_bonus"]))
        else:
            vol_div     = float(cfg["rank_vol_ratio_divisor"])
            vol_cap     = float(cfg["rank_vol_cap"])
            adx_mid     = float(cfg["rank_adx_mid"])
            adx_div     = float(cfg["rank_adx_divisor"])
            vwap_bonus  = float(cfg["rank_vwap_bonus"])
            vwap_pen    = float(cfg["rank_vwap_penalty"])
            dist_near_s = float(cfg["rank_dist_near_score"])
            acc_bonus   = float(cfg["rank_acceptance_bonus"])

        # Common parameters (not strategy-specific)
        rsi_mid     = float(cfg["rank_rsi_mid"])
        rsi_div     = float(cfg["rank_rsi_divisor"])
        rsi_floor   = float(cfg["rank_rsi_floor"])
        rsi_s_cap   = float(cfg["rank_rsi_slope_cap"])
        adx_floor   = float(cfg["rank_adx_floor"])
        adx_s_cap   = float(cfg["rank_adx_slope_cap"])

        dist_near   = float(cfg["rank_dist_near_bpct"])
        dist_ok     = float(cfg["rank_dist_ok_bpct"])
        dist_ok_s   = float(cfg["rank_dist_ok_score"])
        dist_far_s  = float(cfg["rank_dist_far_score"])

        sq_50       = float(cfg["rank_sq_score_le_50"])
        sq_70       = float(cfg["rank_sq_score_le_70"])
        sq_90       = float(cfg["rank_sq_score_ge_90"])

        # ---- inputs (safe casts) ----
        vol_ratio  = float(iv.get("volume_ratio", 0) or 0)
        rsi        = float(iv.get("rsi", 0) or 0)
        rsi_slope  = float(iv.get("rsi_slope", 0) or 0)
        adx        = float(iv.get("adx", 0) or 0)
        adx_slope  = float(iv.get("adx_slope", 0) or 0)
        above_vwap = bool(iv.get("above_vwap", False))
        dist_bpct  = float(iv.get("dist_from_level_bpct", 9.99) or 9.99)
        sq_pct     = iv.get("squeeze_pctile", None)
        acceptance_ok = bool(iv.get("acceptance_ok", False))
        bias = (iv.get("bias") or "long").lower()

        # ---- component scores ----
        s_vol  = min(vol_ratio / vol_div, vol_cap)
        s_rsi  = max((rsi - rsi_mid) / rsi_div, rsi_floor)
        s_rsis = max(min(rsi_slope, rsi_s_cap), 0.0)
        s_adx  = max((adx - adx_mid) / adx_div, adx_floor)
        s_adxs = max(min(adx_slope, adx_s_cap), 0.0)

        if bias == "short":
            s_vwap = vwap_bonus if not above_vwap else vwap_pen
        else:
            s_vwap = vwap_bonus if above_vwap else vwap_pen

        adist = abs(dist_bpct)
        if adist <= dist_near:
            s_dist = dist_near_s
        elif adist <= dist_ok:
            s_dist = dist_ok_s
        else:
            s_dist = dist_far_s

        s_sq = 0.0
        if isinstance(sq_pct, (int, float)):
            if sq_pct <= 50:
                s_sq = sq_50
            elif sq_pct <= 70:
                s_sq = sq_70
            elif sq_pct >= 90:
                s_sq = sq_90

        s_acc = acc_bonus if acceptance_ok else 0.0

        score = s_vol + s_rsi + s_rsis + s_adx + s_adxs + s_vwap + s_dist + s_sq + s_acc

        # Debug logging for strategy-aware scoring
        if strategy_type and strategy_profile:
            logger.debug(f"ranker: {strategy_type} score={score:.3f} (vol={s_vol:.2f}, vwap={s_vwap:.2f}, adx={s_adx:.2f}, acc={s_acc:.2f})")

        return float(score)

    except Exception as e:
        logger.exception(f"ranker: _intraday_strength error: {e}")
        return 0.0


def rank_candidates(rows: List[Dict], top_n: Optional[int] = None) -> List[Dict]:
    """
    Mutate `rows` with 'intraday_score' and 'rank_score', then return the top N.

    Each row is expected to contain:
      - 'intraday': Dict  (features used by _intraday_strength)
      - 'daily_score': float (optional)
      - 'symbol': str (optional; for logging)
    """
    try:
        cfg = load_filters()
        w_daily = float(cfg["rank_weight_daily"])
        w_intr  = float(cfg["rank_weight_intraday"])
        if top_n is None:
            top_n = int(cfg["rank_top_n"])

        for r in rows:
            iv = r.get("intraday", {}) or {}
            strategy_type = r.get("strategy_type") or r.get("signal_type")
            r["intraday_score"] = _intraday_strength(iv, strategy_type)
            r["rank_score"] = w_daily * float(r.get("daily_score", 0.0)) + w_intr * r["intraday_score"]

        rows.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
        out = rows[:top_n]

        if out:
            sh = ", ".join(f"{r.get('symbol', '?')}:{r['rank_score']:.2f}" for r in out[:10])
            logger.info(f"ranker: top{top_n} -> {sh}")
        else:
            logger.info("ranker: no candidates")

        return out

    except Exception as e:
        logger.exception(f"ranker: rank_candidates error: {e}")
        return []


def get_strategy_threshold(strategy_type: str) -> float:
    """
    Get the appropriate threshold for a given strategy type.
    """
    try:
        cfg = load_filters()

        # Check if strategy-specific profiles exist
        if "strategy_ranking_profiles" in cfg and strategy_type:
            profiles = cfg["strategy_ranking_profiles"]
            for profile_name, profile in profiles.items():
                if strategy_type in profile.get("strategies", []):
                    threshold = float(profile.get("base_threshold", cfg.get("execution_threshold", 2.0)))
                    logger.debug(f"ranker: threshold {threshold:.3f} for strategy {strategy_type} (profile: {profile_name})")
                    return threshold

        # Default threshold
        default_threshold = float(cfg.get("execution_threshold", 2.0))
        logger.debug(f"ranker: using default threshold {default_threshold:.3f} for strategy {strategy_type}")
        return default_threshold

    except Exception as e:
        logger.exception(f"ranker: get_strategy_threshold error: {e}")
        return 2.0
