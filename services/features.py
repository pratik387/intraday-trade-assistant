import numpy as np
import pandas as pd

def _wick_bpct(row):
    rng = float(row["high"] - row["low"])
    if rng <= 0: return 0.0, 0.0
    top = float(row["high"] - max(row["open"], row["close"]))
    bot = float(min(row["open"], row["close"]) - row["low"])
    return (top / rng) * 100.0, (bot / rng) * 100.0

def _last_return_z(df1m: pd.DataFrame, lookback: int = 30):
    if df1m is None or len(df1m) < lookback + 1: return 0.0
    close = df1m["close"].astype(float)
    ret = close.pct_change().iloc[-lookback:]
    mu, sd = float(ret.mean()), float(ret.std(ddof=1)) or 1e-9
    return float((ret.iloc[-1] - mu) / sd)

def _momentum_5m(df5m: pd.DataFrame, bars: int = 3):
    if df5m is None or len(df5m) < bars + 1: return 0.0
    c = df5m["close"].astype(float)
    return float((c.iloc[-1] - c.iloc[-1 - bars]) / c.iloc[-1 - bars])

def _volume_ratio(df5m: pd.DataFrame, med_window: int = 20):
    if df5m is None or len(df5m) == 0: return 1.0
    v = df5m["volume"].astype(float)
    med = float(v.tail(med_window).median() or 1.0)
    return float(v.iloc[-1] / med) if med > 0 else 1.0

def _volume_profile_patterns(df5m: pd.DataFrame):
    """
    Detect volume accumulation/distribution patterns for smart money flow detection.
    Returns: (accumulation_score, distribution_score, volume_trend_strength)
    """
    if df5m is None or len(df5m) < 5:
        return 0.0, 0.0, 0.0

    volume = df5m["volume"].astype(float)
    close = df5m["close"].astype(float)

    # Calculate volume trend over last 5 bars
    vol_recent = volume.tail(5)
    vol_slope = (vol_recent.iloc[-1] - vol_recent.iloc[0]) / max(vol_recent.iloc[0], 1.0)

    # Calculate price-volume relationship
    price_change = close.pct_change().tail(5)
    vol_change = volume.pct_change().tail(5).fillna(0)

    # Smart money patterns
    accumulation_score = 0.0
    distribution_score = 0.0

    # Pattern 1: Volume increasing while price consolidating (accumulation)
    if len(vol_recent) >= 3:
        vol_increasing = vol_recent.iloc[-1] > vol_recent.iloc[-2] > vol_recent.iloc[-3]
        price_consolidating = abs(price_change.tail(3).mean()) < 0.005  # <0.5% average move

        if vol_increasing and price_consolidating:
            accumulation_score = 0.8

    # Pattern 2: Volume decreasing on pullbacks (healthy accumulation)
    recent_pullbacks = price_change.tail(3) < -0.002  # >0.2% down moves
    if recent_pullbacks.any():
        pullback_volumes = vol_change[price_change < -0.002].tail(2)
        if len(pullback_volumes) > 0 and pullback_volumes.mean() < 0:  # Volume decreasing on pullbacks
            accumulation_score = max(accumulation_score, 0.6)

    # Pattern 3: High volume with price weakness (distribution)
    high_vol_bars = vol_change.tail(3) > 0.5  # >50% volume increase
    weak_price = price_change.tail(3) < 0.001  # <0.1% price gain despite volume

    if high_vol_bars.any() and weak_price.any():
        distribution_score = 0.7

    # Volume trend strength
    vol_trend_strength = min(abs(vol_slope), 2.0)  # Cap at 200% change

    return float(accumulation_score), float(distribution_score), float(vol_trend_strength)

def _gap_metrics(df5m: pd.DataFrame):
    # Needs prev day close and today open; if unavailable use first bar open/prev close proxy
    if df5m is None or len(df5m) < 2:
        return 0.0, 1.0
    open0 = float(df5m["open"].iloc[0])
    prev_close = float(df5m["close"].iloc[0])  # if this is same-bar close, adjust to prior session close if you store it
    gap_pct = abs((open0 - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
    # gap fill fraction: how much of |open-prev_close| has been retraced intraday
    intraday_high = float(df5m["high"].max())
    intraday_low  = float(df5m["low"].min())
    gap_size = abs(open0 - prev_close)
    if gap_size <= 1e-9:
        return 0.0, 1.0
    # If gap-down (open < prev_close) and we rallied: fill = (intraday_high - open)/gap_size
    # If gap-up   (open > prev_close) and we faded : fill = (open - intraday_low)/gap_size
    if open0 < prev_close:
        fill = max(0.0, intraday_high - open0) / gap_size
    else:
        fill = max(0.0, open0 - intraday_low) / gap_size
    return float(gap_pct), float(min(fill, 1.0))

def _orb_metrics(df5m: pd.DataFrame, orb_bars: int = 3):
    if df5m is None or len(df5m) < orb_bars + 1:
        return False, 1.0
    first = df5m.iloc[:orb_bars]
    or_low, or_high = float(first["low"].min()), float(first["high"].max())
    rest = df5m.iloc[orb_bars:]
    orb_high_made = bool((rest["high"] > or_high).any())
    curr_low = float(df5m["low"].iloc[-1])
    denom = max(or_high - or_low, 1e-9)
    pullback_frac = float((or_high - curr_low) / denom)
    return orb_high_made, pullback_frac

def _vwap(df: pd.DataFrame):
    # Basic rolling session VWAP; adapt if you maintain session-anchored VWAP elsewhere
    if df is None or len(df) == 0: return None
    pv = (df["close"] * df["volume"]).astype(float).cumsum()
    vv = df["volume"].astype(float).cumsum().replace(0.0, np.nan)
    return pv / vv

def compute_hcet_features(
    *,
    df1m_tail: pd.DataFrame,
    df5m_tail: pd.DataFrame,
    index_df5m: pd.DataFrame,
    sector_df5m: pd.DataFrame | None,
    structural_rr: float
):
    # news spike detection using volume analysis
    news_spike_flag = False
    if df1m_tail is not None and len(df1m_tail) > 0:
        try:
            vol = df1m_tail["volume"].astype(float)
            if len(vol) >= 5:
                recent_median = vol.tail(20).median()
                current_vol = vol.iloc[-1]
                news_spike_flag = (current_vol > recent_median * 3.0) if recent_median > 0 else False
        except Exception:
            news_spike_flag = False

    # shared
    sector_mom = _momentum_5m(sector_df5m or index_df5m, 3)
    index_mom  = _momentum_5m(index_df5m, 3)
    vol_ratio  = _volume_ratio(df5m_tail, 20)

    # volume profile patterns - Enhancement 2
    vol_accumulation, vol_distribution, vol_trend_strength = _volume_profile_patterns(df5m_tail)

    # wick %
    up_wick, lo_wick = (0.0, 0.0)
    if df5m_tail is not None and len(df5m_tail) > 0:
        up_wick, lo_wick = _wick_bpct(df5m_tail.iloc[-1])

    # gaps (use your day boundary prev_close if available)
    gap_pct, gap_fill_pct = _gap_metrics(df5m_tail)

    # ORB
    orb_high_made, pullback_frac = _orb_metrics(df5m_tail, 3)

    # VWAP reclaim bar (last bar crossed from below to above)
    vwap = _vwap(df5m_tail)
    vwap_reclaim = False
    if vwap is not None and len(vwap) >= 2:
        c = df5m_tail["close"].astype(float)
        v_prev, v_curr = float(vwap.iloc[-2]), float(vwap.iloc[-1])
        c_prev, c_curr = float(c.iloc[-2]), float(c.iloc[-1])
        vwap_reclaim = (c_prev < v_prev) and (c_curr > v_curr)

    # 1m return z
    ret_z = _last_return_z(df1m_tail, 30)

    return {
        "sector_momentum": sector_mom,
        "index_momentum": index_mom,
        "volume_ratio": vol_ratio,
        "structural_rr": float(structural_rr),
        "news_spike_flag": bool(news_spike_flag),

        "gap_pct": gap_pct,
        "gap_fill_pct": gap_fill_pct,
        "last_bar_upper_wick_bpct": up_wick,
        "last_bar_lower_wick_bpct": lo_wick,

        "orb_high_made": orb_high_made,
        "pullback_frac_of_or": pullback_frac,
        "vwap_reclaim_bar": vwap_reclaim,

        "ret_z": ret_z,
        # add rsi_divergence_2bar_bull if you compute swing points; else default False and keep HCET strict
        "rsi_divergence_2bar_bull": False,

        # volume profile patterns - Enhancement 2
        "volume_accumulation": vol_accumulation,
        "volume_distribution": vol_distribution,
        "volume_trend_strength": vol_trend_strength,
    }
