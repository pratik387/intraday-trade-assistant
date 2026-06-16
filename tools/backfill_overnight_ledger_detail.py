"""Backfill per-trade detail (symbol/entry/exit/fees/gross/qty) into the
close_dn_overnight_long decay-tripwire ledger for trades settled BEFORE the
detail-persist change shipped.

Why this is possible: the tripwire ledger stored net-only {net_pnl_inr, ts_iso}
rows, but each settled trade's full detail survived in that day's 16:00 OCI
snapshot of `overnight_slots.json` — the daily archive captures slots while they
are still `t1_settling` (settle at T+1 09:30, release at T+2 09:30, archive at
T+1 16:00). The most recent settle day (not yet archived) is recovered from the
LIVE `state/overnight_slots.json` directly.

Matching: per settle-date, a snapshot's t1_settling slots correspond 1:1 to the
ledger rows with that date; we key on (date, net_pnl) where net_pnl == the slot's
realized_pnl_inr (verified identical). Enrichment is ADDITIVE — net_pnl_inr and
ts_iso are preserved byte-for-byte, the rolling-PF gate is unaffected.

Safe to run mid-day: the only concurrent writer is the verify-exit cron at 09:30,
already past for the day. The script backs up the ledger and defaults to dry-run.

Usage (run from a NEUTRAL cwd so the repo's local `oci/` package doesn't shadow
the SDK):
    cd /tmp && <repo>/.venv/bin/python <repo>/tools/backfill_overnight_ledger_detail.py --dry-run
    cd /tmp && <repo>/.venv/bin/python <repo>/tools/backfill_overnight_ledger_detail.py --apply
"""
from __future__ import annotations

# Import the SDK BEFORE adding the repo root to sys.path, so the local `oci/`
# package does not shadow the `oci` SDK (mirrors oci/tools/upload_overnight_state.py).
try:
    import oci as oci_sdk
except Exception:  # pragma: no cover - SDK absent in unit-test env
    oci_sdk = None

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SETUP_NAME = "close_dn_overnight_long"
DEFAULT_BUCKET = "paper-trading-logs"
OCI_PREFIX = f"overnight/{SETUP_NAME}/"
EXIT_REASON = "t1_settle"  # same label the forward handler writes


# ─── Pure logic (unit-tested) ──────────────────────────────────────────────


def slot_to_detail(slot: Dict) -> Dict:
    """Derive the ledger detail fields from a settled (t1_settling) slot record.

    fees_inr matches the forward handler's convention: brokerage + interest.
    gross = net + total cost.
    """
    buy = float(slot["buy_fill_price"])
    sell = float(slot["sell_fill_price"])
    fees = float(slot.get("fees_inr") or 0.0)
    interest = float(slot.get("interest_inr") or 0.0)
    realized = float(slot["realized_pnl_inr"])
    total_cost = fees + interest
    qty: Optional[int] = None
    notional = slot.get("notional_inr")
    if notional and buy:
        qty = int(round(float(notional) / buy))
    return {
        "symbol": slot.get("symbol"),
        "entry_price": buy,
        "exit_price": sell,
        "exit_reason": EXIT_REASON,
        "qty": qty,
        "fees_inr": total_cost,
        "gross_pnl_inr": realized + total_cost,
    }


def _key(date: str, pnl: float) -> Tuple[str, float]:
    # realized_pnl_inr and net_pnl_inr are identical floats; round only to defuse
    # any repr drift across the JSON round-trip.
    return (date, round(float(pnl), 4))


def enrich_ledger_trades(
    trades: List[Dict], slots_by_date: Dict[str, List[Dict]]
) -> Tuple[List[Dict], List[Dict]]:
    """Enrich net-only ledger rows in place from settled slot records.

    Returns (enriched_trades, unmatched_trades). Rows that already carry a
    'symbol' are left untouched. net_pnl_inr / ts_iso are preserved exactly.
    """
    index: Dict[Tuple[str, float], Dict] = {}
    for date, slots in slots_by_date.items():
        for s in slots:
            if s.get("status") != "t1_settling":
                continue
            if s.get("realized_pnl_inr") is None:
                continue
            index[_key(date, s["realized_pnl_inr"])] = slot_to_detail(s)

    enriched: List[Dict] = []
    unmatched: List[Dict] = []
    for t in trades:
        out = dict(t)
        if "symbol" in t and t.get("symbol") is not None:
            enriched.append(out)  # already detailed
            continue
        ts = str(t.get("ts_iso", ""))
        detail = index.get(_key(ts[:10], t.get("net_pnl_inr", 0.0))) if len(ts) >= 10 else None
        if detail is None:
            unmatched.append(out)
            enriched.append(out)
            continue
        for k, v in detail.items():
            if v is not None:
                out[k] = v
        enriched.append(out)
    return enriched, unmatched


# ─── I/O wrappers ──────────────────────────────────────────────────────────


def _oci_client():
    if oci_sdk is None:
        raise RuntimeError("oci SDK not importable (run from a neutral cwd, not the repo root)")
    cfg = oci_sdk.config.from_file()
    client = oci_sdk.object_storage.ObjectStorageClient(cfg)
    ns = client.get_namespace().data
    return client, ns


def fetch_snapshot_slots(client, ns: str, bucket: str) -> Dict[str, List[Dict]]:
    """Return {date: [slots]} for every archived overnight_slots.json snapshot."""
    objs = oci_sdk.pagination.list_call_get_all_results(
        client.list_objects, ns, bucket, prefix=OCI_PREFIX
    ).data.objects
    out: Dict[str, List[Dict]] = {}
    for o in objs:
        if not o.name.endswith("overnight_slots.json"):
            continue
        date = o.name[len(OCI_PREFIX):].split("/")[0]
        raw = client.get_object(ns, bucket, o.name).data.content
        out[date] = json.loads(raw).get("slots", [])
    return out


def load_live_slots(state_path: Path) -> Dict[str, List[Dict]]:
    """Recover the most-recent (not-yet-archived) settle day from the live pool.

    Slots currently t1_settling settled this morning; key them under their
    expected_exit_date (== the settle date the ledger recorded)."""
    if not state_path.exists():
        return {}
    slots = json.loads(state_path.read_text(encoding="utf-8")).get("slots", [])
    out: Dict[str, List[Dict]] = {}
    for s in slots:
        if s.get("status") != "t1_settling":
            continue
        d = s.get("expected_exit_date")
        if d:
            out.setdefault(d, []).append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(_REPO_ROOT),
                    help="engine repo root (default: this tool's repo)")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--apply", action="store_true", help="write the enriched ledger (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.repo_root)
    ledger_path = root / "state" / f"decay_tripwire_{SETUP_NAME}.json"
    live_slots_path = root / "state" / "overnight_slots.json"

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    trades = ledger.get("trades", [])
    net_only = [t for t in trades if t.get("symbol") is None]
    print(f"ledger: {len(trades)} trades, {len(net_only)} net-only (backfill targets)")

    client, ns = _oci_client()
    slots_by_date = fetch_snapshot_slots(client, ns, args.bucket)
    print(f"OCI snapshots: {sorted(slots_by_date.keys())}")
    live = load_live_slots(live_slots_path)
    for d, s in live.items():
        slots_by_date.setdefault(d, [])
        # live takes precedence for its date (freshest)
        slots_by_date[d] = s
    print(f"live t1_settling dates: {sorted(live.keys())}")

    enriched, unmatched = enrich_ledger_trades(trades, slots_by_date)
    n_filled = sum(1 for a, b in zip(trades, enriched)
                   if a.get("symbol") is None and b.get("symbol") is not None)
    print(f"matched + enriched: {n_filled}/{len(net_only)}")
    if unmatched:
        print(f"UNMATCHED ({len(unmatched)}):")
        for u in unmatched:
            print(f"   {u.get('ts_iso')}  net={u.get('net_pnl_inr')}")

    # show a couple of sample enrichments
    for b in enriched:
        if b.get("symbol") is not None and "entry_price" in b:
            print(f"   e.g. {b['ts_iso'][:10]} {b['symbol']} "
                  f"entry={b['entry_price']} exit={b['exit_price']} "
                  f"fees={round(b['fees_inr'],2)} net={round(b['net_pnl_inr'],2)}")
            break

    if not args.apply:
        print("\nDRY-RUN — no changes written. Re-run with --apply to persist.")
        return 0

    backup = ledger_path.with_suffix(
        f".json.bak-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(ledger_path, backup)
    ledger["trades"] = enriched
    tmp = ledger_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp.replace(ledger_path)
    print(f"\nAPPLIED. backup -> {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
