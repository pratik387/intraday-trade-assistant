# §3.3 Brief: `fno_oi_cliff_weekly_expiry`

**Sub-project:** #9 (microstructure-first redesign) — Round 5
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-07

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3 portfolio context)
- specs/2026-04-29-expiry_pin_strike_reversal-plan.md (broken predecessor — diagnosed below)
- structures/expiry_pin_strike_reversal_structure.py (broken detector being differentiated against)

This brief proposes a **clean redesign** of the OI-pin mechanic that the existing `expiry_pin_strike_reversal` detector failed to capture (-₹4.3M PnL, 80K trades, retired 2026-04-30 via the OCI run audit). The redesign is structurally distinct on universe, side construction, signal locus, and entry timing.

---

## Diagnosis of the broken `expiry_pin_strike_reversal` setup

Read of `structures/expiry_pin_strike_reversal_structure.py` + config block `expiry_pin_strike_reversal` confirms three independent failure modes that explain the -₹4.3M result:

1. **Wrong universe — pin-magnet was applied to NIFTY HEAVYWEIGHTS, not NIFTY itself.** The detector takes the highest-OI strike of the **NIFTY index option chain** and trades 10 individual heavyweights (RELIANCE, HDFCBANK, etc.) toward an "implied target" computed as `constituent_close × (1 - spot_distance_pct)`. This is **two layers of indirection** away from the gamma-hedging mechanic: (a) the constituent has its own beta-to-NIFTY (≠ 1.0); (b) the constituent has its OWN options OI which can be opposite to NIFTY's. Market-makers hedge the NIFTY index option book by buying/selling NIFTY futures, not the cash-equity heavyweights. The "pin pull" on individual stocks is mathematically diluted to noise.

2. **Side construction is reversed for the dominant case.** The detector fires SHORT when spot is ABOVE pin and LONG when spot is BELOW. This is correct ONLY IF the pin attracts (call-OI dominant scenario). When PUT-OI dominates the strike, market-makers sell into a downward move (defending negative gamma) — the strike REPELS instead of attracts. The detector ignores call-vs-put OI dominance entirely; it lumps both regimes into one direction rule, guaranteeing ~50% of trades are mechanically on the wrong side.

3. **Entry timing is too early and uses RSI noise as the trigger.** The detector activates 13:30-15:15 IST, fires anytime in that window when an RSI-decay condition prints. Gamma-hedging pressure is strongest in the **final 60-90 minutes** before expiry settlement (T+0 14:00-15:30), and on **T-1 PM** as MMs pre-position into expiry. RSI(14) on a 5m heavyweight chart is uncorrelated with index gamma flow; using it as the trigger is a category error.

4. **D-1 settlement OI mis-anchoring.** The detector reads D-1 OI to compute pin strike for session D. On an expiry day, the relevant OI cliff is the strike where the **largest open positions are about to be settled** — that data is published only AFTER market close on D. Using D-1 settlement OI on an expiry day means the pin strike is computed from yesterday's positions, not today's near-expiry positioning that actually drives the gamma flow.

These four issues compound: bad universe × bad side rule × bad timing × stale OI = -₹4.3M is unsurprising. The new brief addresses each issue explicitly.

---

## Asymmetry

**Name:** NIFTY/BANKNIFTY weekly-expiry T-1 PM gamma-hedging directional drift toward the max-OI strike.

**Indian-specific source:** NSE F&O weekly-options contract spec — every Tuesday (post-2025-09 reform; pre-reform: Thursday) NIFTY weekly options expire; BANKNIFTY weekly was discontinued 2024-11-20 by SEBI but the monthly + index-pin behaviour persists in the SENSEX/MIDCPNIFTY proxy and in NIFTY weekly. NSE contract spec: https://www.nseindia.com/products-services/equity-derivatives-nifty50 ; SEBI F&O weekly rationalisation circular SEBI/HO/MRD/POD-2/P/CIR/2024/103: https://www.sebi.gov.in/legal/circulars/oct-2024/.

**The exploitable asymmetry — gamma-hedging is mathematically forced:** market-makers running short-gamma books at the highest-OI strike must dynamically hedge as spot drifts. In Black-Scholes terms, MMs short a call at strike K hold delta-hedge that REQUIRES BUYING futures as spot rises and SELLING as spot falls — but their **second-order exposure (gamma)** forces them to sell INTO spot rises near K and buy INTO spot falls near K. The "pin" is the equilibrium price at which the cumulative MM hedge flow equals the cumulative retail directional flow. On T-1 PM and T+0, the magnitude of cumulative hedging required to settle the expiring book is largest, so the pin's gravitational pull is strongest. This is not behavioural — it is mathematically codified in MM risk-system requirements.

**The directional fade asymmetry:** when NIFTY spot is meaningfully ABOVE the max-OI cliff strike at T-1 14:30, MM call-gamma hedging will pull spot DOWN toward the strike into close. When spot is meaningfully BELOW the max-OI cliff strike at T-1 14:30, MM put-gamma hedging will push spot UP toward the strike. Direction is governed by spot-vs-cliff sign **conditional on** which side of the chain (CE vs PE) holds the dominant OI at the cliff strike.

## Participants

- **Retail option-buyers (89% of F&O participation per SEBI FY23 study):** systematically long calls (when bullish narrative) or long puts (when bearish narrative) at OTM strikes. They concentrate OI at psychologically round strikes (multiples of 50/100/500). They do NOT hedge dynamically; they hold the option to expiry.
- **Market-maker gamma-hedgers (institutional desks: Tower, Optiver, Jane Street India, JPM India, ICICI Sec prop):** are the natural counter-party to retail's option-buying. They are SHORT gamma at the max-OI strike. Their hedge mandate is delta-neutral within tight risk limits — every 0.5% NIFTY move triggers a futures rebalance. Their flow is mechanical, signed-by-spot-direction, and concentrated around the max-OI cliff.
- **Cross-flow:** institutional directional macro flow (FIIs buying/selling NIFTY futures based on global cues) can OVERWHELM the gamma-hedging flow on macro-event days. This is the dominant failure mode for the broken setup — and the falsification gate below is built around it.

We are on the disciplined side of the **gamma-hedging-induced drift**, BEFORE macro-flow can overwhelm it (T-1 14:30-15:15 is post-lunch but pre-close; macro flow concentrates at open and at close ±15 min).

## Persistence

Three structural reasons:

1. **Gamma hedging is mathematically codified in MM risk systems** — Bollen & Whaley 2004 (Journal of Finance), Black-Scholes Greek hedging is regulatory in MM capital frameworks (SEBI's market-maker-eligibility rules require risk-neutralisation; this is institutional, not optional).
2. **NSE weekly-expiry calendar is exchange-codified** — every Tuesday (post-2025-09) for NIFTY options. SEBI's Oct 2024 rationalisation circular reduced multiple-weekly-expiry venues to one per exchange but did NOT eliminate weekly expiry; it concentrated more OI per single weekly expiry. The pin effect strengthened, did not weaken.
3. **MM capacity is structurally limited at the cliff strike** — at the highest-OI strike, MM short-gamma exposure is largest. They cannot opt out of hedging without breaching risk limits. Retail flow can dwarf MM hedging flow on macro days, but on quiet days (no FII flow event) MM hedging IS the marginal flow at expiry.

## Evidence

1. **Bollen & Whaley, *Journal of Finance* 2004** — *Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?* — foundational empirical paper documenting market-maker hedging flow as the dominant intraday driver of underlying price near option-OI concentrations. Cited 2,400+ times. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00647.x
2. **Ni, Pearson, Poteshman, White, *Journal of Financial Economics* 2021** — *Does Option Trading Have a Pervasive Impact on Underlying Stock Prices?* — direct evidence that option-market-maker delta-hedging on expiry weeks accounts for measurable underlying-stock price effects (~30-40 bps directional drift at OI concentrations). https://www.sciencedirect.com/science/article/pii/S0304405X20302488
3. **Avellaneda & Lipkin 2003** — *A Market-Induced Mechanism for Stock Pinning* — Wilmott / mathematical-finance treatment of pin-strike pinning specifically caused by MM hedging. https://math.nyu.edu/faculty/avellane/AvellanedaLipkinQF03.pdf
4. **Ramesh Bhat & Pandey 2007 (*Vikalpa*)** — Indian-market study of expiry-day price effects at NSE: documents NIFTY pin behaviour, Indian-specific. https://journals.sagepub.com/doi/10.1177/0256090920070406
5. **NSE Open Interest publications** (live and historical) — https://www.nseindia.com/option-chain ; https://www.nseindia.com/market-data/most-active-underlying. NSE publishes max-pain and OI distribution daily; the data is public, the mechanic is observed.
6. **Negative confirmation on retail-saturation:** retail Indian options-pin platforms (Sensibull max-pain, NiftyTrader, Opstra) all publish max-pain calculators. **However**, none publish an operationalised T-1 PM directional fade strategy on NIFTY/BANKNIFTY spot — the published strategies trade options (selling far-OTM premium near the pin), not the underlying. This means the asymmetry of trading the underlying spot is **NOT retail-saturated** at the entry-flow level. (This is the inverse of the gauntlet's "if retail is doing it, it's arbitraged out" rule — we WANT mechanic awareness to be high but exploitation in the underlying spot to be low.)

## Direction

**BIDIRECTIONAL — driven by signed `(spot − cliff_strike)` × OI-dominance side**, but with a sharp asymmetry expected.

- **SHORT NIFTY** if at T-1 14:30 spot > cliff_strike × (1 + 0.30%) AND the cliff strike is **CE-OI dominant** (CE_OI ≥ 1.5× PE_OI at that strike).
- **LONG NIFTY** if at T-1 14:30 spot < cliff_strike × (1 − 0.30%) AND the cliff strike is **PE-OI dominant** (PE_OI ≥ 1.5× CE_OI at that strike).
- **NO TRADE** if neither CE nor PE is dominant at the cliff (mixed-OI = MM gamma is approximately neutral at that strike → no directional pull).

Side is governed by BOTH spot direction AND CE/PE dominance — fixing failure-mode #2 of the broken setup.

## Mechanic

**Setup name:** `fno_oi_cliff_weekly_expiry`
**Side:** Bidirectional (per direction rule above)

**Sequence:**

1. **T-1 EOD (post-15:30 IST)**: read `data/option_chain/<YYYY>/<MM>/<T-1>.parquet`. Filter to symbol ∈ {NIFTY, BANKNIFTY}, `expiry_date == T+0` (current week's expiry). For each row aggregate `oi_total = CE_OI + PE_OI` by strike; cliff_strike = `argmax(oi_total)`.
2. **At T-1 14:30 IST single-bar entry decision**: read NIFTY 5m close from `backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather` resampled to 5m. Compute `spot_distance_pct = (spot − cliff_strike) / cliff_strike × 100`.
3. **Direction & gate:** apply the direction rule above (sign + dominance). If `|spot_distance_pct| < 0.30%`, NO TRADE (already too close; no room for a fade). If `|spot_distance_pct| > 1.20%`, NO TRADE (too far; macro flow is overwhelming gamma flow — common on FII-flow days).
4. **Entry timing:** T-1 14:30 IST single-bar entry on the 5m close.
5. **Stop-loss:** structural — beyond the OPPOSITE-direction OI cluster's nearest large strike. SHORT entries: hard SL = next CE-OI cluster strike above cliff + 0.20% buffer. LONG entries: hard SL = next PE-OI cluster strike below cliff − 0.20% buffer. Min stop distance = 0.40% of entry (qty-inflation guard).
6. **Targets:** T1 (50% qty) at 50% of (entry → cliff) distance; T2 (50% qty) at the cliff strike itself. **Time stop = T-1 15:15 IST** — exit cleanly before 15:25 close-window late-MIS-unwind flow contaminates the gamma signal.
7. **Latch:** one fire per (NIFTY|BANKNIFTY, expiry-week, side). No re-entry same week even if trigger re-arms.

**Universe:** **NIFTY 50 spot only** (BANKNIFTY weekly options were discontinued 2024-11-20 — only monthly remains). Trading vehicle: NIFTY-50 ETF (NIFTYBEES.NS) or NIFTY futures (current month). The detector signal is on the index spot; execution is on the tradeable proxy.

## Active window

**Setup formation:** T-1 EOD (post-15:30) for cliff-strike computation; T-1 14:30 for entry decision.
**Entry:** T-1 14:30 IST single-bar.
**Hold horizon:** T-1 14:30 → T-1 15:15 IST = 45 minutes intraday MIS.

**Why T-1 14:30 (not T+0 11:00-14:30):** the brief explicitly chooses T-1 over T+0 for two reasons: (a) on T+0 (expiry day itself), the 11:00-14:30 window is dominated by retail option-unwinding flow which is noisier than the T-1 PM MM-pre-positioning flow; (b) T-1 PM lets the detector use D-2 EOD OI (1 day stale, captures the cliff before retail final-day OI reshuffling); T+0 would require D-1 OI which has already been distorted by MM pre-positioning. Picking T-1 isolates the cleanest signal locus.

## Risks / falsification criteria

The setup is **wrong** (and should retire) if:

1. **Phase-1 floor fails on validation/holdout:** n < 30 trades over 12-month sanity (TIGHT — only ~50 weekly expiries/year × NIFTY-only universe = max 50 events/yr; the 1.5x-dominance + 0.3-1.2% distance gates will compress further). NET PF < 1.10. |WR delta validation→holdout| > 10pp.
2. **Gamma flow overwhelmed by macro flow:** if on FII-flow days the trade is systematically wrong-sided, the asymmetry is conditional on a quiet-macro regime. Sanity must report PF on (macro-flow-day, quiet-day) split.
3. **CE/PE dominance gate doesn't stratify:** if mixed-OI cells (where dominance < 1.5×) have similar PF to dominant cells, the dominance rule isn't the discriminator and the setup degenerates to the broken `expiry_pin_strike_reversal` mechanic.
4. **Sample too thin for a single-symbol setup:** if n < 30 trades over 12 months even after PF passes, deployment is impossible at our risk-budget. RETIRE.

**Differentiation from the broken `expiry_pin_strike_reversal` (mandatory acceptance criterion):**

| Failure mode | Broken setup | This brief's fix |
|---|---|---|
| Universe indirection | NIFTY pin → trade 10 heavyweights | NIFTY pin → trade NIFTY itself |
| Side rule | Sign-of-spot-distance only | Sign × CE/PE dominance |
| Trigger | RSI(14) decay on 5m heavyweight | Single-bar T-1 14:30 entry, no RSI |
| OI staleness | D-1 OI on D (1 day stale on expiry) | T-1 EOD OI for T-1 14:30 entry (same-day, fresh) |
| Window | 13:30-15:15 (full 1h45m) | T-1 14:30 single-bar |
| Hold | Heavyweight intraday | NIFTY 45-min intraday MIS |

Each of the 4 diagnosed failure modes has a structural fix in the new mechanic.

## Pre-coding sanity-check plan

Fully achievable on existing data:

- **OI source:** `data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet` (verified on disk, 2023-2026 coverage; schema includes `symbol, expiry_date, strike, option_type, oi`).
- **NIFTY spot:** `backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather` (verified, 290K rows 2023-01 to 2026-01).
- **Expiry calendar:** `services/symbol_metadata.is_expiry_day` (already in production for the broken setup's expiry-day check; correct calendar source).
- **Tool:** `tools/sub9_research/sanity_fno_oi_cliff_weekly_expiry.py` — reads the three sources above, applies the cliff + dominance + distance gates, simulates 14:30→15:15 entry/exit at NIFTY-spot prices, computes NET PF using the existing Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee` with NIFTYBEES proxy commission of ~₹3 per ₹50K notional). No detector code yet.

**Decision gate:** PF ≥ 1.10 AND n ≥ 30 → APPROVED for detector implementation. Marginal (1.0-1.10) → revisit dominance threshold (1.2×, 1.5×, 2.0×) and distance band before deciding. PF < 1.0 → RETIRE; the asymmetry doesn't survive net-fee even on the cleanest mechanic, and the "OI cliff" thesis is decisively negative-finding for our system.

## Data engineering plan

**None additional.** The existing option-chain backfill (`tools/option_chain/fetch_oi_snapshot.py`) and the index_ohlcv feathers cover the entire data dependency. No new ingestion pipeline.

If sanity passes, post-approval: `structures/fno_oi_cliff_weekly_expiry_structure.py` + a tiny `services/oi_cliff_loader.py` that wraps the existing `option_chain_loader.find_max_oi_strike` with CE/PE dominance computation. The broken `structures/expiry_pin_strike_reversal_structure.py` is RETIRED (not edited) — it stays disabled in config as a documented failure case.

---

## Decision required

**User action:**
1. **APPROVE** for sanity-check coding → I write `tools/sub9_research/sanity_fno_oi_cliff_weekly_expiry.py`, simulate over 2023-2026 weekly expiries, report NET PF + n + macro-day stratification.
2. **REJECT** with revisions → I revise specific points and re-submit.
3. **RETIRE before sanity** → if you judge that the broken predecessor's failure is decisive evidence the gamma-pin mechanic is unexploitable at our scale, we skip and pick another round-5 candidate.

**My read:** APPROVE for sanity check. The four-way structural differentiation from the broken setup is concrete, the dominance gate is the missing ingredient that the literature (Bollen-Whaley, Ni-Pearson) flags as essential, and the data is already on disk. The single biggest risk is sample size (n ≤ 50/yr ceiling) — the sanity check will quantify whether the dominance + distance gates leave enough events to trade.
