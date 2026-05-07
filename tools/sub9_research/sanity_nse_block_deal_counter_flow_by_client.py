"""Pre-coding sanity (round-5 patch) for nse_block_deal_counter_flow with
client-type bucketing.

Aggregate sanity (sanity_nse_block_deal_counter_flow.py) returned NET PF
0.729 / n=113 with all three diagnostic controls failing (no scaling with
block size, inverse scaling with gap-pct, T+0 control comparable to T+1).

The brief explicitly cited "institutional vs retail block-deal informational
asymmetry"; aggregating ALL client types ignores the heterogeneous
post-disclosure flow dynamics. This patched sanity classifies the
disclosed-side `client_name` into participant categories and re-runs the
T+1 sanity per (category x side) cell. Brief locks (PF>=1.10, n>=30 per
cell) are evaluated cell-by-cell; if ANY cell passes that gate AND has
Sharpe>0, the candidate becomes a NARROW SHIP CANDIDATE worth OOS
validation; otherwise the institutional-vs-retail asymmetry thesis is
falsified.

Usage:
    python tools/sub9_research/sanity_nse_block_deal_counter_flow_by_client.py

Output:
    reports/sub9_sanity/nse_block_deal_counter_flow_by_client_trades.csv
"""
from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params per brief 3.3 (mirror aggregate sanity) ----
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)

MIN_NET_VALUE_CR = 25.0          # >= 25 Cr per-row (and per-side aggregate) trigger floor
T0_CONTROL_EOD_HHMM = "15:10"
T0_CONTROL_HOLD_BARS = 1

T1_FIRST_BAR_HHMM = "09:15"
T1_ENTRY_BAR_HHMM = "09:25"
TIME_STOP_HHMM = "14:30"

HARD_STOP_PCT = 1.5
MIN_STOP_PCT = 1.0

T1_R_MULTIPLE = 1.0
T2_R_MULTIPLE = 2.0
T1_PARTIAL_FRACTION = 0.5
USE_BREAKEVEN_TRAIL_AFTER_T1 = True

RISK_PER_TRADE_RUPEES = 1000

# ---- Brief gates (per-cell) ----
GATE_PF = 1.10
GATE_N = 30
GATE_SHARPE = 0.0  # narrow-ship requires Sharpe>0


# =====================================================================
# Client-name classification
# =====================================================================
#
# Order matters: more-specific patterns first. Each category compiled to a
# regex; we test from top-to-bottom and assign on first match. Anything
# unmatched lands in OTHER.
#
# NOTE: substring patterns are case-insensitive, run on a normalized
# uppercase client_name with extra spacing collapsed.

# DII institutional names (Indian mutual funds, insurance, pension)
DII_PATTERNS = [
    r"\bMUTUAL\s*FUND\b",
    r"\bMF\b",
    r"\bINSURANCE\b",
    r"\bLIC\b",                       # LIC, LIC OF INDIA
    r"\bSBI\b",                       # SBI MF / SBI Life
    r"\bHDFC\s*LIFE\b",
    r"\bHDFC\s*MUTUAL\b",
    r"\bICICI\s*PRUDENTIAL\b",
    r"\bICICI\s*PRU\b",
    r"\bNIPPON\b",
    r"\bAXIS\s*MUTUAL\b",
    r"\bAXIS\s*MF\b",
    r"\bUTI\b",
    r"\bKOTAK\s*MAHINDRA\s*MUTUAL\b",
    r"\bKOTAK\s*MUTUAL\b",
    r"\bMIRAE\b",
    r"\bDSP\b",
    r"\bFRANKLIN\s*INDIA\b",
    r"\bADITYA\s*BIRLA\b",
    r"\bINVESCO\b",
    r"\bHSBC\s*MUTUAL\b",
    r"\bTATA\s*MUTUAL\b",
    r"\bTATA\s*AIA\b",
    r"\bCANARA\s*ROBECO\b",
    r"\bMOTILAL\s*OSWAL\s*MUTUAL\b",
    r"\bEDELWEISS\s*MUTUAL\b",
    r"\bIDFC\s*MUTUAL\b",
    r"\bPGIM\b",
    r"\bSUNDARAM\s*MUTUAL\b",
    r"\bPENSION\b",
    r"\bPROVIDENT\s*FUND\b",
    r"\bEPFO\b",
    r"\bNPS\b",
]

# FII / FPI offshore + foreign banks
FII_PATTERNS = [
    r"\bFII\b",
    r"\bFPI\b",
    r"\bFOREIGN\b",
    r"\bMORGAN\s*STANLEY\b",
    r"\bGOLDMAN\b",
    r"\bJ\.?P\.?\s*MORGAN\b",
    r"\bCITIGROUP\b",
    r"\bDEUTSCHE\b",
    r"\bSOCIETE\s*GENERALE\b",
    r"\bSOC\s*GEN\b",
    r"\bVANGUARD\b",
    r"\bBLACKROCK\b",
    r"\bBLACK\s*ROCK\b",
    r"\bGMO\b",
    r"\bWELLINGTON\b",
    r"\bT\s*ROWE\b",
    r"\bFIDELITY\b",
    r"\bABU\s*DHABI\b",
    r"\bGOVERNMENT\s*OF\s*SINGAPORE\b",
    r"\bGIC\b",
    r"\bTEMASEK\b",
    r"\bSCHRODER\b",
    r"\bFRANKLIN\s*TEMPLETON\b",
    r"\bBNP\s*PARIBAS\b",
    r"\bBOFA\b",
    r"\bBANK\s*OF\s*AMERICA\b",
    r"\bUBS\b",
    r"\bCREDIT\s*SUISSE\b",
    r"\bHSBC\b",                       # HSBC global excl. mutual matched above
    r"\bNOMURA\b",
    r"\bMACQUARIE\b",
    r"\bBARCLAYS\b",
    r"\bMERRILL\s*LYNCH\b",
    r"\bAIA\b",                        # AIA INTERNATIONAL etc.
    r"\bMARSHALL\s*WACE\b",
    r"\bMARSHALLWACE\b",
    r"\bJUPITER\b",
    r"\bGHISALLO\b",
    r"\bWHITE\s*IRIS\b",
    r"\bGREAT\s*TERRAIN\b",
    r"\bMASTER\s*FUND\b",
    r"\bODI\b",
    r"\bMAURITIUS\b",
    r"\bSINGAPORE\b",
    r"\bIRELAND\b",
    r"\bLUXEMBOURG\b",
    r"\bCAYMAN\b",
    r"\bOFFSHORE\b",
    r"\bOPPENHEIMER\b",
    r"\bNORGES\b",
    r"\bSTATE\s*STREET\b",
    r"\bNORTHERN\s*TRUST\b",
    r"\bMITSUBISHI\b",
    r"\bSUMITOMO\b",
    r"\bSWISS\b",
    r"\bTHE\s*JUPITER\b",
]

# Promoter / promoter-related entities
PROMOTER_PATTERNS = [
    r"\bPROMOTER\b",
    r"\bFAMILY\s*TRUST\b",
    r"\bINVESTMENTS\s*PVT\b",
    r"\bINVESTMENT\s*PVT\b",
    r"\bHOLDINGS\s*PVT\b",
    r"\bHOLDING\s*PVT\b",
    r"\bENTERPRISES\b",
    r"\bHUF\b",
]

# Broker proprietary / general securities firms
PROP_PATTERNS = [
    r"\bTRADING\b",
    r"\bSECURITIES\b",
    r"\bBROKING\b",
    r"\bCAPITAL\b",
    r"\bFINSERV\b",
    r"\bFINANCIAL\s*SERVICES\b",
    r"\bASSET\s*MANAG\b",
]


def _compile(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_RE_DII = _compile(DII_PATTERNS)
_RE_FII = _compile(FII_PATTERNS)
_RE_PROMOTER = _compile(PROMOTER_PATTERNS)
_RE_PROP = _compile(PROP_PATTERNS)


def classify_client(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "OTHER"
    s = str(name).upper()
    s = re.sub(r"[\s]+", " ", s).strip()
    # Order: DII (specific) -> Promoter -> FII (broad) -> Prop (broadest) -> OTHER.
    # Promoter checked before FII because "INVESTMENTS PVT LTD" can also
    # contain words like "CAPITAL" which would otherwise match prop. But
    # promoter takes priority over prop too.
    if _RE_DII.search(s):
        return "DII"
    if _RE_PROMOTER.search(s):
        return "PROMOTER"
    if _RE_FII.search(s):
        return "FII"
    if _RE_PROP.search(s):
        return "PROP"
    return "OTHER"


# =====================================================================
# Data loaders
# =====================================================================

def load_block_deals() -> pd.DataFrame:
    path = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= DISCOVERY_START) & (df["trade_date"] <= DISCOVERY_END)]
    return df.reset_index(drop=True)


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    return set(df["symbol"].astype(str).tolist())


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m_for_events(events: pd.DataFrame) -> pd.DataFrame:
    """Load 5m bars for each event's (T-0, T+1) months."""
    needed_months: set = set()
    for d in events["trade_date"].unique():
        needed_months.add((d.year, d.month))
        next_d = d + timedelta(days=7)
        needed_months.add((next_d.year, next_d.month))

    print(f"  loading {len(needed_months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    universe_syms = set(events["raw_symbol"].astype(str).unique())
    for yyyy, mm in sorted(needed_months):
        mdf = _load_5m_for_month(yyyy, mm)
        if mdf.empty:
            continue
        mdf = mdf[mdf["symbol"].isin(universe_syms)]
        if not mdf.empty:
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total event-relevant 5m bars: {len(big):,}")
    return big


# =====================================================================
# Aggregation (per-category, per-side)
# =====================================================================

def aggregate_to_signals_by_category(df: pd.DataFrame, fno_syms: set) -> pd.DataFrame:
    """Build per (trade_date, symbol, side, client_category) signals.

    Each block-deal row owns its disclosed-side client_name and is
    classified independently. We then aggregate notional WITHIN
    (date, symbol, side, category) and gate at >= MIN_NET_VALUE_CR.
    """
    print("\n-- Funnel (by client_category) --")
    print(f"  raw block-deal events           : {len(df):>6}")

    # Classify
    df = df.copy()
    df["client_category"] = df["client_name"].apply(classify_client)

    # Print classification distribution (raw)
    print("\n  classification distribution (raw, pre-filter):")
    for cat, n in df["client_category"].value_counts().items():
        cr = float(df.loc[df["client_category"] == cat, "trade_value_cr"].sum())
        print(f"    {cat:<10} n={n:>5}  total_cr={cr:>10.1f}")

    df_25 = df[df["trade_value_cr"] >= MIN_NET_VALUE_CR]
    print(f"\n  rows >=25cr per-row             : {len(df_25):>6}")
    print("  classification dist after >=25cr per-row:")
    for cat, n in df_25["client_category"].value_counts().items():
        print(f"    {cat:<10} n={n:>5}")

    df_25_fno = df[(df["trade_value_cr"] >= MIN_NET_VALUE_CR) & (df["symbol"].isin(fno_syms))]
    print(f"\n  rows >=25cr + F&O 200            : {len(df_25_fno):>6}")
    print("  classification dist after >=25cr + F&O 200:")
    for cat, n in df_25_fno["client_category"].value_counts().items():
        print(f"    {cat:<10} n={n:>5}")

    # Per-side per-category aggregation BEFORE >=25cr (capture full notional
    # split across multiple counter-parties with the same category).
    df_fno = df[df["symbol"].isin(fno_syms)].copy()
    side_agg = (
        df_fno
        .groupby(["trade_date", "symbol", "buy_or_sell", "client_category"], as_index=False)
        .agg(side_total_cr=("trade_value_cr", "sum"),
             n_lines=("trade_value_cr", "size"),
             max_line_cr=("trade_value_cr", "max"))
    )
    side_agg = side_agg[side_agg["side_total_cr"] >= MIN_NET_VALUE_CR].copy()
    print(f"\n  unique (date, symbol, side, cat) cells: {len(side_agg):>6}")

    # SHORT when block-deal disclosed BUY -> retail FOMO long -> we fade.
    # LONG  when block-deal disclosed SELL -> retail panic   -> we fade.
    side_agg["side"] = np.where(side_agg["buy_or_sell"] == "BUY", "SHORT", "LONG")
    side_agg["raw_symbol"] = side_agg["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    side_agg["nse_symbol"] = "NSE:" + side_agg["raw_symbol"]
    side_agg["cap_segment"] = side_agg["nse_symbol"].apply(get_cap_segment)
    side_agg["abs_net_cr"] = side_agg["side_total_cr"]

    print("\n  signal-cell distribution by (cat, side):")
    pivot = (
        side_agg.groupby(["client_category", "side"]).size().unstack(fill_value=0)
    )
    print(pivot.to_string())
    return side_agg.reset_index(drop=True)


# =====================================================================
# T+1 simulation (mirror aggregate sanity)
# =====================================================================

def _next_trading_day_with_data(sym_df: pd.DataFrame, t0: date) -> Optional[date]:
    future_days = sym_df[sym_df["d"] > t0]["d"].unique()
    if len(future_days) == 0:
        return None
    return min(future_days)


def simulate_t1(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    print("\n-- Simulating T+1 entries (per signal cell) --")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    funnel = {
        "signals_in": len(signals),
        "no_5m_data": 0,
        "no_t1_day": 0,
        "no_first_bar": 0,
        "no_t0_close": 0,
        "fail_gap_or_candle": 0,
        "no_entry_bar": 0,
        "fired_pre_latch": 0,
        "fired_post_latch": 0,
    }

    raw_trades: List[dict] = []
    for _, sig in signals.iterrows():
        raw_sym = sig["raw_symbol"]
        sym_df = days_per_sym.get(raw_sym)
        if sym_df is None or sym_df.empty:
            funnel["no_5m_data"] += 1
            continue

        t0 = sig["trade_date"]
        side = sig["side"]
        t1 = _next_trading_day_with_data(sym_df, t0)
        if t1 is None:
            funnel["no_t1_day"] += 1
            continue

        t0_df = sym_df[sym_df["d"] == t0]
        if t0_df.empty:
            funnel["no_t0_close"] += 1
            continue
        t0_close = float(t0_df.iloc[-1]["close"])

        t1_df = sym_df[sym_df["d"] == t1].sort_values("date").reset_index(drop=True)
        if t1_df.empty:
            funnel["no_t1_day"] += 1
            continue
        t1_df["hhmm"] = t1_df["date"].dt.strftime("%H:%M")

        first_bar_rows = t1_df[t1_df["hhmm"] == T1_FIRST_BAR_HHMM]
        if first_bar_rows.empty:
            funnel["no_first_bar"] += 1
            continue
        first_bar = first_bar_rows.iloc[0]
        first_open = float(first_bar["open"])
        first_close = float(first_bar["close"])

        gap_pct = (first_open / t0_close - 1.0) * 100.0
        is_green = first_close > first_open
        is_red = first_close < first_open

        if side == "SHORT":
            ok = (first_open > t0_close) and is_green
        else:
            ok = (first_open < t0_close) and is_red
        if not ok:
            funnel["fail_gap_or_candle"] += 1
            continue

        entry_rows = t1_df[t1_df["hhmm"] == T1_ENTRY_BAR_HHMM]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["open"])

        sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
        if side == "SHORT":
            hard_sl = entry_price * (1.0 + sl_pct_used / 100.0)
            stop_distance = hard_sl - entry_price
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            hard_sl = entry_price * (1.0 - sl_pct_used / 100.0)
            stop_distance = entry_price - hard_sl
            t1_target = entry_price + T1_R_MULTIPLE * stop_distance
            t2_target = entry_price + T2_R_MULTIPLE * stop_distance

        if stop_distance <= 0:
            continue

        entry_idx_arr = t1_df.index[t1_df["date"] == entry_ts].tolist()
        if not entry_idx_arr:
            funnel["no_entry_bar"] += 1
            continue
        entry_idx = entry_idx_arr[0]
        forward = t1_df.iloc[entry_idx + 1:].copy()
        forward = forward[forward["hhmm"] <= TIME_STOP_HHMM]
        if forward.empty:
            funnel["no_entry_bar"] += 1
            continue

        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = bar["hhmm"]

            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts
                    exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True
                    t1_exit_price = t1_target
                    t1_exit_ts = ts
                if hit_t1 and low <= t2_target:
                    exit_ts = ts
                    exit_price = t2_target
                    exit_reason = "t2"
                    break
            else:
                if low <= active_sl:
                    exit_ts = ts
                    exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True
                    t1_exit_price = t1_target
                    t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
                    exit_ts = ts
                    exit_price = t2_target
                    exit_reason = "t2"
                    break

            if hhmm >= TIME_STOP_HHMM:
                exit_ts = ts
                exit_price = close
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "time_stop"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = max(int(qty * T1_PARTIAL_FRACTION), 1)
            qty_t2 = max(qty - qty_t1, 0)
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            kite_side = "SELL" if side == "SHORT" else "BUY"
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, kite_side)
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, kite_side) if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            kite_side = "SELL" if side == "SHORT" else "BUY"
            fee = calc_fee(entry_price, exit_price, qty, kite_side)
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        raw_trades.append({
            "T0_signal_date": t0,
            "T1_entry_date": t1,
            "symbol": "NSE:" + raw_sym,
            "client_category": sig["client_category"],
            "cap_segment": sig["cap_segment"],
            "side": side,
            "side_total_cr": float(sig["side_total_cr"]),
            "abs_net_cr": float(sig["abs_net_cr"]),
            "t0_close": t0_close,
            "t1_first_open": first_open,
            "t1_first_close": first_close,
            "gap_pct": gap_pct,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "stop_distance": stop_distance,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "hit_t1": hit_t1,
            "t1_exit_price": t1_exit_price,
            "t1_exit_ts": t1_exit_ts,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    funnel["fired_pre_latch"] = len(raw_trades)

    # Latch: one fire per (symbol, T+1, side, client_category).
    if raw_trades:
        trades_df = pd.DataFrame(raw_trades)
        trades_df = (
            trades_df.sort_values(
                ["symbol", "T1_entry_date", "side", "client_category", "abs_net_cr"],
                ascending=[True, True, True, True, False],
            )
            .drop_duplicates(
                subset=["symbol", "T1_entry_date", "side", "client_category"], keep="first",
            )
            .reset_index(drop=True)
        )
    else:
        trades_df = pd.DataFrame()

    funnel["fired_post_latch"] = len(trades_df)
    return trades_df, funnel


# =====================================================================
# Reporting
# =====================================================================

def _pf_wr(grp: pd.DataFrame) -> tuple[float, float, int, float, float]:
    n = len(grp)
    if n == 0:
        return float("nan"), float("nan"), 0, 0.0, 0.0
    npnl = grp["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    net = float(npnl.sum())
    daily_col = "T1_entry_date" if "T1_entry_date" in grp.columns else "T0_signal_date"
    daily = grp.groupby(daily_col)["net_pnl"].sum()
    sharpe = round(float(daily.mean() / daily.std()), 3) if daily.std() > 0 else 0.0
    return pf, wr, n, net, sharpe


def report_aggregate(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[aggregate] no trades")
        return
    pf, wr, n, net, sharpe = _pf_wr(trades)
    print("\n=== AGGREGATE (all categories combined) ===")
    print(f"  n={n}  PF={pf}  WR={wr}%  netPnL=Rs.{int(net):,}  Sharpe={sharpe}")


def report_per_category(trades: pd.DataFrame) -> Dict[str, dict]:
    print("\n=== Per client_category (aggregated across sides) ===")
    cat_summary: Dict[str, dict] = {}
    for cat, grp in trades.groupby("client_category"):
        pf, wr, n, net, sharpe = _pf_wr(grp)
        cat_summary[cat] = {"pf": pf, "wr": wr, "n": n, "net": net, "sharpe": sharpe}
        print(f"  {cat:<10} n={n:>4}  PF={pf:>6}  WR={wr:>5}%  netPnL=Rs.{int(net):>10,}  Sharpe={sharpe}")
    return cat_summary


def report_per_category_side(trades: pd.DataFrame) -> List[dict]:
    print("\n=== Per (client_category x side) cells ===")
    print(f"  Brief gates: PF>={GATE_PF}  n>={GATE_N}  Sharpe>{GATE_SHARPE}")
    cells: List[dict] = []
    for (cat, sd), grp in trades.groupby(["client_category", "side"]):
        pf, wr, n, net, sharpe = _pf_wr(grp)
        passes = (
            isinstance(pf, (int, float)) and pf >= GATE_PF
            and n >= GATE_N
            and isinstance(sharpe, (int, float)) and sharpe > GATE_SHARPE
        )
        flag = " *** PASS ***" if passes else ""
        cells.append({
            "client_category": cat, "side": sd, "n": n, "pf": pf,
            "wr": wr, "net": net, "sharpe": sharpe, "passes": passes,
        })
        print(f"  {cat:<10} {sd:<6} n={n:>4}  PF={pf:>6}  WR={wr:>5}%  netPnL=Rs.{int(net):>10,}  Sharpe={sharpe}{flag}")
    return cells


def report_per_cat_side_breakdown(trades: pd.DataFrame) -> None:
    """Detailed per-cell sub-breakdowns: cap_segment, monthly, exit_reason."""
    print("\n=== Per (client_category x side) detail breakdowns ===")
    for (cat, sd), grp in trades.groupby(["client_category", "side"]):
        if len(grp) < 5:
            continue
        pf, wr, n, net, sharpe = _pf_wr(grp)
        print(f"\n  -- {cat} x {sd} (n={n}, PF={pf}, Sharpe={sharpe}) --")
        # cap_segment
        if "cap_segment" in grp.columns:
            for cap, sub in grp.groupby("cap_segment"):
                pf2, wr2, n2, net2, _ = _pf_wr(sub)
                print(f"     cap={cap:<12} n={n2:>3}  PF={pf2:>6}  netPnL=Rs.{int(net2):>9,}")
        # exit_reason
        if "exit_reason" in grp.columns:
            for rsn, sub in grp.groupby("exit_reason"):
                pf2, wr2, n2, net2, _ = _pf_wr(sub)
                print(f"     reason={rsn:<18} n={n2:>3}  PF={pf2:>6}")


def _print_funnel(funnel: dict) -> None:
    print("\n-- Trade-funnel --")
    print(f"  signals fed in              : {funnel['signals_in']}")
    print(f"  no 5m data for symbol       : {funnel['no_5m_data']}")
    print(f"  no T+1 trading day          : {funnel['no_t1_day']}")
    print(f"  no T+0 close bar            : {funnel['no_t0_close']}")
    print(f"  no 09:15 first bar          : {funnel['no_first_bar']}")
    print(f"  fail gap-or-candle gate     : {funnel['fail_gap_or_candle']}")
    print(f"  no 09:25 entry bar          : {funnel['no_entry_bar']}")
    print(f"  -> fired (pre-latch)        : {funnel['fired_pre_latch']}")
    print(f"  -> fired (post-latch, FINAL): {funnel['fired_post_latch']}")


def _final_verdict(cells: List[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"FINAL VERDICT  (per-cell gate: PF>={GATE_PF}, n>={GATE_N}, Sharpe>{GATE_SHARPE})")
    print("=" * 70)

    passing = [c for c in cells if c["passes"]]
    if passing:
        print(f"\n  >>> {len(passing)} (client_category x side) cell(s) PASSED brief gate.")
        for c in sorted(passing, key=lambda x: x["pf"], reverse=True):
            print(f"      {c['client_category']:<10} {c['side']:<6} "
                  f"n={c['n']:>4}  PF={c['pf']:>6}  WR={c['wr']:>5}%  Sharpe={c['sharpe']}")
        print("\n  Recommendation: NARROW SHIP CANDIDATE — promote passing cells to OOS validation.")
    else:
        print("\n  All (client_category x side) cells FAILED PF>=1.10 + n>=30 + Sharpe>0.")
        print("  Aggregate cohort already failed (NET PF 0.729). Per-category slicing")
        print("  exposes no client-type subpopulation with positive expectancy.")
        print("\n  >>> Brief's 'institutional vs retail block-deal informational asymmetry'")
        print("      thesis is FALSIFIED at the disclosed-side / T+1 fade timeframe.")
        print("      RETIRE candidate.")


# =====================================================================
# Driver
# =====================================================================

def main():
    print("=== nse_block_deal_counter_flow_by_client — pre-coding sanity (round-5) ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")

    print("\nLoading block-deal events ...")
    df = load_block_deals()
    print(f"  raw events in Discovery: {len(df)}")

    print("\nLoading F&O 200 universe ...")
    fno_syms = load_fno_universe()
    print(f"  fno symbols: {len(fno_syms)}")

    signals = aggregate_to_signals_by_category(df, fno_syms)
    if signals.empty:
        print("No signals after filter; aborting.")
        return

    print("\nLoading 5m feathers for event months ...")
    big5m = build_5m_for_events(signals)
    if big5m.empty:
        print("No 5m bars; aborting.")
        return

    trades, funnel = simulate_t1(signals, big5m)
    _print_funnel(funnel)

    if trades.empty:
        print("\nNo T+1 trades fired. Cannot run report.")
        return

    # ---- Output reports ----
    report_aggregate(trades)
    report_per_category(trades)
    cells = report_per_category_side(trades)
    report_per_cat_side_breakdown(trades)

    _final_verdict(cells)

    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nse_block_deal_counter_flow_by_client_trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"\nT+1 trade log: {out_csv}")


if __name__ == "__main__":
    main()
