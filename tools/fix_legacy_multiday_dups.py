"""Retro-tag legacy multiday duplicate ledger rows as attribution mirrors.

Before the composite selector went live (entries from 2026-06-30), the
per-setup-island code could buy the SAME symbol into 2-3 setups' books — real
double/triple positions (e.g. SALASAR -27,293 x2 settled 06-30; ALLCARGO x2
settled 06-24). Those rows predate the `attributed` flag, so pooled dashboard
views double-count them, distorting strategy-level PnL.

This tags EXACT duplicates — identical (symbol, settle-date, entry_price,
exit_price, qty) across the four multiday tripwire ledgers — keeping ONE row
untagged (owner = first setup alphabetically, deterministic) and marking the
rest `attributed: true` plus `_legacy_dup: true` for provenance. Rows that
differ in price/qty/date are left alone: they were genuinely distinct
positions and belong in the actual history.

Idempotent; backs up each ledger before mutation. Run on the multiday deploy:
    python tools/fix_legacy_multiday_dups.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUPS = [
    "crash2d_revert_long",
    "low52_capitulation_revert_long",
    "mtf_capitulation_revert_long",
    "zscore_oversold_revert_long",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    ledgers = {}
    for s in SETUPS:
        f = ROOT / "state" / f"decay_tripwire_{s}.json"
        if f.exists():
            ledgers[s] = (f, json.loads(f.read_text(encoding="utf-8")))

    # Group rows by exact-duplicate key across setups (alphabetical setup order
    # makes the owner choice deterministic and idempotent).
    groups = defaultdict(list)  # key -> [(setup, row_dict)]
    for s in sorted(ledgers):
        for t in ledgers[s][1].get("trades", []):
            if not isinstance(t, dict) or not t.get("symbol"):
                continue
            key = (t["symbol"], str(t.get("ts_iso"))[:10],
                   t.get("entry_price"), t.get("exit_price"), t.get("qty"))
            groups[key].append((s, t))

    changed = defaultdict(int)
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        # rows are in alphabetical setup order; first = owner, rest = mirrors.
        for s, t in rows[1:]:
            if t.get("attributed") is True:
                continue  # already tagged (idempotent re-run)
            print(f"tagging mirror: {s:<32} {key[0]:<16} {key[1]} net={t.get('net_pnl_inr'):+.0f}")
            if not args.dry_run:
                t["attributed"] = True
                t["_legacy_dup"] = True
            changed[s] += 1

    if args.dry_run:
        print("dry-run: no files written | would tag:", dict(changed) or "nothing")
        return 0
    for s, n in changed.items():
        f, data = ledgers[s]
        shutil.copy2(f, str(f) + ".bak-legacydup")
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {f.name}: {n} row(s) tagged (backup .bak-legacydup)")
    if not changed:
        print("nothing to tag — ledgers already clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
