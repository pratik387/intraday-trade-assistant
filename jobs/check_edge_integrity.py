"""Nightly edge-integrity monitors (spec: specs/2026-07-28-edge-integrity-monitors-design.md).

Three monitor classes, all config-driven from the `edge_integrity` block of
config/configuration.json (missing key = startup error, per CLAUDE.md):

  Monitor 1 — Rule-change watch (regulatory preconditions):
    Per-setup preconditions [{type:"rule", watch:"<key>"}] matched against the
    `affects` tokens of data/sebi_calendar/rule_changes.csv. A watched key with
    a row whose effective_date >= today - rule_watch_lookback_days triggers a
    pause. This is the monitor that would have caught the Oct-2025 MWPL
    circular on day one instead of detecting it through -Rs206k of PnL.

  Monitor 2 — Data-source health (scraped-feed preconditions):
    Per-setup preconditions [{type:"data_source", path, date_column,
    max_staleness_days, min_rows_per_week}]. Staleness (max(date_column) too
    old) or weekly-volume collapse (announcements_fr pattern: the feed died
    but the file kept existing) triggers a pause. min_rows_per_week=0
    disables the volume check (for quarterly-clustered feeds).

  Monitor 3 — Factor-cluster tripwire (portfolio class):
    edge_integrity.factor_clusters.<name>: combined daily net PnL summed
    across members (July-2026 factor-study methodology), z-scored over the
    ledger's own mean/sigma. z(last lookback_days) <= -drawdown_threshold_sigma
    pauses NEW entries for ALL members. Complements per-setup tripwires.

Pause mechanism: cb_state = "paused_precondition" written via
tools.methodology.setup_metadata.write_setup_block_atomic — the SAME machinery
as jobs/check_circuit_breakers.py, so honoring by the intraday orchestrator
(services/plan_orchestrator._setup_should_fire) and the overnight/multiday
crons comes from their existing cb_state reads. Un-pause is MANUAL.

dry_run=true (v1 default): logs WOULD_PAUSE lines and writes the nightly
report JSON to <report_dir>/, but never mutates config or any state file.

Usage:
    .venv/Scripts/python jobs/check_edge_integrity.py \
        [--config config/configuration.json] \
        [--trades-csv reports/run_latest/trade_report.csv] \
        [--today 2026-07-28]

Run via cron nightly after the EOD data refresh (same host as
jobs/check_circuit_breakers.py). Idempotent: already-paused/disabled setups
are never re-paused; re-runs on the same date overwrite the same report file.
All timestamps IST-naive.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.methodology.setup_metadata import write_setup_block_atomic
from utils.time_util import _now_naive_ist

# Same alert channel as the overnight verify-exit failsafe: the agent logger
# (config/logging_config) surfaces WARNING/CRITICAL lines in the cron log.
# Fall back to a stdlib logger when running outside the deployed environment.
try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except Exception:  # pragma: no cover - fallback for stripped test envs
    import logging
    logger = logging.getLogger(__name__)


PAUSED_STATE = "paused_precondition"
# cb_states that mean "already not firing" — never re-pause over these
# (idempotency + never clobber a circuit-breaker disable).
_INACTIVE_CB_STATES = ("disabled", PAUSED_STATE)


class ConfigError(ValueError):
    """Missing/invalid edge_integrity configuration (fail-fast, per CLAUDE.md)."""


@dataclass(frozen=True)
class Finding:
    monitor: str          # "rule_watch" | "data_health" | "factor_tripwire"
    setup_name: str
    action: str           # "pause" | "ok" | "skip"
    reason: str
    detail: Dict


# ---------------------------------------------------------------------------
# Config loading + startup validation
# ---------------------------------------------------------------------------

def _require(block: dict, key: str, where: str):
    if key not in block:
        raise ConfigError(f"edge_integrity: missing required key {key!r} in {where}")
    return block[key]


def load_edge_integrity_config(cfg: dict) -> dict:
    """Extract + validate the edge_integrity block. Raises ConfigError on any
    missing key (NO defaults in code)."""
    if "edge_integrity" not in cfg:
        raise ConfigError("config has no `edge_integrity` block")
    ei = cfg["edge_integrity"]
    for key in ("enabled", "dry_run", "rule_watch_lookback_days",
                "rule_changes_csv", "report_dir", "factor_clusters"):
        _require(ei, key, "edge_integrity")
    for cname, cluster in ei["factor_clusters"].items():
        if cname.startswith("_"):
            continue
        for key in ("members", "lookback_days", "drawdown_threshold_sigma",
                    "min_baseline_days"):
            _require(cluster, key, f"edge_integrity.factor_clusters.{cname}")
        if not cluster["members"]:
            raise ConfigError(
                f"edge_integrity.factor_clusters.{cname}: members is empty")
    return ei


def validate_preconditions(cfg: dict) -> List[str]:
    """Startup validation. Returns a list of error strings (empty = OK).

    - Every setup with regulatory_sensitivity == "rule_dependent" MUST declare
      at least one precondition of type "rule" (spec section 1, Monitor 1).
    - Every declared precondition must be well-formed (required keys present).
    """
    errors: List[str] = []
    for name, block in (cfg.get("setups") or {}).items():
        pres = block.get("preconditions") or []
        if block.get("regulatory_sensitivity") == "rule_dependent":
            if not any(p.get("type") == "rule" for p in pres):
                errors.append(
                    f"setup {name!r} has regulatory_sensitivity=rule_dependent "
                    f"but no preconditions[] entry of type 'rule'")
        for i, p in enumerate(pres):
            ptype = p.get("type")
            if ptype == "rule":
                if not p.get("watch"):
                    errors.append(f"setup {name!r} preconditions[{i}]: "
                                  f"type 'rule' requires a 'watch' key")
            elif ptype == "data_source":
                for key in ("path", "date_column", "max_staleness_days",
                            "min_rows_per_week"):
                    if key not in p:
                        errors.append(f"setup {name!r} preconditions[{i}]: "
                                      f"type 'data_source' missing {key!r}")
            else:
                errors.append(f"setup {name!r} preconditions[{i}]: "
                              f"unknown type {ptype!r}")
    return errors


# ---------------------------------------------------------------------------
# Monitor 1 — rule-change watch
# ---------------------------------------------------------------------------

def load_rule_changes(csv_path: Path) -> pd.DataFrame:
    """rule_changes.csv with parsed effective_date and affects token lists."""
    df = pd.read_csv(csv_path)
    for col in ("effective_date", "affects"):
        if col not in df.columns:
            raise ConfigError(f"{csv_path}: missing column {col!r}")
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
    df["_affects_tokens"] = (
        df["affects"].fillna("").astype(str)
        .apply(lambda s: [t.strip() for t in s.split(";") if t.strip()])
    )
    return df


def check_rule_watch(
    setup_name: str,
    preconditions: List[dict],
    rules_df: pd.DataFrame,
    today: date,
    lookback_days: int,
) -> List[Finding]:
    """Trigger when a watched rule key gains a row with
    effective_date >= today - lookback_days (spec Monitor 1). Rows with a
    FUTURE effective_date also trigger — the row is added when the circular is
    announced, which is exactly when the researcher must be alerted."""
    findings: List[Finding] = []
    cutoff = today - timedelta(days=int(lookback_days))
    for p in preconditions:
        if p.get("type") != "rule":
            continue
        watch = str(p["watch"])
        hits = rules_df[
            rules_df["_affects_tokens"].apply(lambda toks: watch in toks)
            & (rules_df["effective_date"] >= cutoff)
        ]
        if len(hits) > 0:
            cols = [c for c in ("effective_date", "category", "severity",
                                "description") if c in hits.columns]
            rows = hits[cols]
            findings.append(Finding(
                monitor="rule_watch", setup_name=setup_name, action="pause",
                reason=f"rule_watch:{watch}",
                detail={
                    "watch": watch,
                    "cutoff": cutoff.isoformat(),
                    "matching_rows": [
                        {k: (v.isoformat() if isinstance(v, date) else str(v))
                         for k, v in r.items()}
                        for r in rows.to_dict("records")
                    ],
                },
            ))
        else:
            findings.append(Finding(
                monitor="rule_watch", setup_name=setup_name, action="ok",
                reason=f"rule_watch:{watch}",
                detail={"watch": watch, "cutoff": cutoff.isoformat()},
            ))
    return findings


# ---------------------------------------------------------------------------
# Monitor 2 — data-source health
# ---------------------------------------------------------------------------

def _read_date_column(path: Path, date_column: str) -> pd.Series:
    """Read only the date column of a parquet/csv feed as datetimes (NaT dropped)."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=[date_column])
    else:
        df = pd.read_csv(path, usecols=[date_column])
    return pd.to_datetime(df[date_column], errors="coerce").dropna()


def check_data_source(
    setup_name: str,
    precondition: dict,
    today: date,
    repo_root: Path,
) -> Finding:
    """Staleness + weekly-volume-collapse check for one data_source precondition.

    Trigger conditions (any -> pause):
      - file missing or empty (feed never materialized / deleted)
      - staleness: (today - max(date_column)).days > max_staleness_days
      - volume collapse: rows with date in (today-7d, today] < min_rows_per_week
        (skipped when min_rows_per_week <= 0 — quarterly-clustered feeds)
    """
    rel = str(precondition["path"])
    path = Path(rel)
    if not path.is_absolute():
        path = repo_root / rel
    date_column = str(precondition["date_column"])
    max_staleness_days = int(precondition["max_staleness_days"])
    min_rows_per_week = int(precondition["min_rows_per_week"])

    if not path.exists():
        return Finding(
            monitor="data_health", setup_name=setup_name, action="pause",
            reason="data_source_missing",
            detail={"path": rel},
        )
    try:
        dates = _read_date_column(path, date_column)
    except Exception as e:
        return Finding(
            monitor="data_health", setup_name=setup_name, action="pause",
            reason="data_source_unreadable",
            detail={"path": rel, "error": str(e)},
        )
    if len(dates) == 0:
        return Finding(
            monitor="data_health", setup_name=setup_name, action="pause",
            reason="data_source_empty",
            detail={"path": rel, "date_column": date_column},
        )

    last_date = dates.max().date()
    staleness_days = (today - last_date).days
    detail = {
        "path": rel,
        "date_column": date_column,
        "last_date": last_date.isoformat(),
        "staleness_days": staleness_days,
        "max_staleness_days": max_staleness_days,
    }
    if staleness_days > max_staleness_days:
        return Finding(
            monitor="data_health", setup_name=setup_name, action="pause",
            reason="data_source_stale", detail=detail,
        )

    if min_rows_per_week > 0:
        week_start = pd.Timestamp(today) - pd.Timedelta(days=7)
        rows_last_week = int(((dates > week_start)
                              & (dates <= pd.Timestamp(today))).sum())
        detail.update({
            "rows_last_week": rows_last_week,
            "min_rows_per_week": min_rows_per_week,
        })
        if rows_last_week < min_rows_per_week:
            return Finding(
                monitor="data_health", setup_name=setup_name, action="pause",
                reason="data_source_volume_collapse", detail=detail,
            )

    return Finding(
        monitor="data_health", setup_name=setup_name, action="ok",
        reason="data_source_healthy", detail=detail,
    )


# ---------------------------------------------------------------------------
# Monitor 3 — factor-cluster tripwire
# ---------------------------------------------------------------------------

def load_member_daily_pnl(
    member: str,
    setup_cfg: dict,
    trades_csv: Optional[Path],
    repo_root: Path,
) -> pd.Series:
    """Daily net PnL series (index: date, values: INR) for one cluster member.

    Source precedence (no double-count):
      1. decay_tripwire.state_file ledger (paper + live multiday legs settle
         here). `attributed` mirror rows are EXCLUDED — the composite feeds
         one position's PnL to every contributor's tripwire, so a portfolio
         view summing members would count it len(members) times otherwise.
      2. trades_csv (setup_type == member, actual_pnl_after_charges grouped by
         signal_date) — the intraday members' path, same schema as
         jobs/check_circuit_breakers.py.
    Returns an empty series when neither source has rows for this member.
    """
    tw_cfg = (setup_cfg or {}).get("decay_tripwire")
    if tw_cfg is not None and tw_cfg.get("state_file"):
        state_file = Path(str(tw_cfg["state_file"]))
        if not state_file.is_absolute():
            state_file = repo_root / state_file
        if state_file.exists():
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            rows = []
            for t in data.get("trades", []):
                if t.get("attributed"):
                    continue  # mirror row owned by another setup
                rows.append({
                    "d": datetime.fromisoformat(str(t["ts_iso"])).date(),
                    "pnl": float(t["net_pnl_inr"]),
                })
            if rows:
                df = pd.DataFrame(rows)
                return df.groupby("d")["pnl"].sum().sort_index()
        return pd.Series(dtype=float)

    if trades_csv is not None and Path(trades_csv).exists():
        df = pd.read_csv(trades_csv)
        needed = {"signal_date", "setup_type", "actual_pnl_after_charges"}
        if needed.issubset(df.columns):
            sub = df[df["setup_type"] == member].copy()
            if len(sub) > 0:
                sub["d"] = pd.to_datetime(sub["signal_date"]).dt.date
                return (sub.groupby("d")["actual_pnl_after_charges"]
                        .sum().sort_index())
    return pd.Series(dtype=float)


def combine_cluster_daily_pnl(
    daily_by_member: Dict[str, pd.Series],
) -> pd.Series:
    """Combined daily net PnL = per-day sum across members (factor-study
    methodology: members contribute 0 on days they didn't trade)."""
    frames = {m: s for m, s in daily_by_member.items() if len(s) > 0}
    if not frames:
        return pd.Series(dtype=float)
    wide = pd.DataFrame(frames).fillna(0.0)
    return wide.sum(axis=1).sort_index()


def check_factor_cluster(
    cluster_name: str,
    cluster_cfg: dict,
    daily_by_member: Dict[str, pd.Series],
) -> tuple:
    """Evaluate the drawdown tripwire. Returns (findings, stats).

    Trigger: z = (sum(last lookback_days trading days) - lookback*mean)
                 / (sigma*sqrt(lookback))  <=  -drawdown_threshold_sigma
    mean/sigma computed over the FULL combined ledger (>= min_baseline_days
    distinct days required before the tripwire is armed).
    """
    lookback = int(cluster_cfg["lookback_days"])
    threshold_sigma = float(cluster_cfg["drawdown_threshold_sigma"])
    min_baseline = int(cluster_cfg["min_baseline_days"])
    members = [str(m) for m in cluster_cfg["members"]]

    combined = combine_cluster_daily_pnl(daily_by_member)
    stats: Dict = {
        "cluster": cluster_name,
        "members": members,
        "n_days": int(len(combined)),
        "lookback_days": lookback,
        "drawdown_threshold_sigma": threshold_sigma,
        "min_baseline_days": min_baseline,
    }
    if len(combined) < max(min_baseline, lookback):
        stats["status"] = "insufficient_data"
        findings = [Finding(
            monitor="factor_tripwire", setup_name=m, action="skip",
            reason=f"cluster:{cluster_name}:insufficient_data",
            detail={"n_days": int(len(combined)),
                    "required": max(min_baseline, lookback)},
        ) for m in members]
        return findings, stats

    mean = float(combined.mean())
    sigma = float(combined.std(ddof=1))
    recent_sum = float(combined.iloc[-lookback:].sum())
    if sigma <= 0:
        stats["status"] = "degenerate_sigma"
        return [Finding(
            monitor="factor_tripwire", setup_name=m, action="skip",
            reason=f"cluster:{cluster_name}:degenerate_sigma",
            detail={"sigma": sigma},
        ) for m in members], stats

    z = (recent_sum - lookback * mean) / (sigma * math.sqrt(lookback))
    breach = z <= -threshold_sigma
    stats.update({
        "status": "evaluated",
        "daily_mean_inr": mean,
        "daily_sigma_inr": sigma,
        "recent_sum_inr": recent_sum,
        "recent_window_start": combined.index[-lookback].isoformat(),
        "recent_window_end": combined.index[-1].isoformat(),
        "z": z,
        "breach": bool(breach),
    })
    action = "pause" if breach else "ok"
    reason = (f"cluster:{cluster_name}:drawdown_z={z:.2f}"
              f"<=-{threshold_sigma}" if breach
              else f"cluster:{cluster_name}:z={z:.2f}")
    findings = [Finding(
        monitor="factor_tripwire", setup_name=m, action=action, reason=reason,
        detail={"z": z, "recent_sum_inr": recent_sum,
                "daily_mean_inr": mean, "daily_sigma_inr": sigma},
    ) for m in members]
    return findings, stats


# ---------------------------------------------------------------------------
# Pause application + report
# ---------------------------------------------------------------------------

def apply_pauses(
    config_path: Path,
    cfg: dict,
    findings: List[Finding],
    today: date,
    dry_run: bool,
) -> List[dict]:
    """Flip cb_state -> paused_precondition for every 'pause' finding.

    Idempotent: setups already in an inactive cb_state are skipped. In
    dry_run, NOTHING is written — WOULD_PAUSE lines are logged instead.
    Returns the list of applied/deferred pause records for the report.
    """
    records: List[dict] = []
    seen: set = set()
    setups = cfg.get("setups") or {}
    for f in findings:
        if f.action != "pause" or f.setup_name in seen:
            continue
        seen.add(f.setup_name)
        block = setups.get(f.setup_name) or {}
        current = block.get("cb_state")
        if current in _INACTIVE_CB_STATES:
            logger.info("[edge_integrity] %s already cb_state=%s — skip re-pause "
                        "(trigger %s)", f.setup_name, current, f.reason)
            records.append({"setup": f.setup_name, "applied": False,
                            "reason": f.reason,
                            "skip_cause": f"already_{current}"})
            continue
        if dry_run:
            logger.warning("[edge_integrity] WOULD_PAUSE %s (%s / %s) — dry_run, "
                           "no state changed", f.setup_name, f.monitor, f.reason)
            records.append({"setup": f.setup_name, "applied": False,
                            "reason": f.reason, "skip_cause": "dry_run"})
            continue
        write_setup_block_atomic(config_path, f.setup_name, {
            "cb_state": PAUSED_STATE,
            "cb_disabled_at": today.isoformat(),
            "cb_disabled_reason": f"edge_integrity:{f.monitor}:{f.reason}",
        })
        logger.critical("[edge_integrity] PAUSED %s — cb_state=%s (%s / %s). "
                        "Manual un-pause after researcher review.",
                        f.setup_name, PAUSED_STATE, f.monitor, f.reason)
        records.append({"setup": f.setup_name, "applied": True,
                        "reason": f.reason})
    return records


def write_report(
    report_dir: Path,
    today: date,
    *,
    dry_run: bool,
    findings: List[Finding],
    cluster_stats: List[dict],
    pause_records: List[dict],
    validation_errors: List[str],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"edge_integrity_report_{today.isoformat()}.json"
    payload = {
        "generated_at": _now_naive_ist().isoformat(timespec="seconds"),
        "run_date": today.isoformat(),
        "dry_run": dry_run,
        "validation_errors": validation_errors,
        "findings": [asdict(f) for f in findings],
        "factor_clusters": cluster_stats,
        "pauses": pause_records,
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path, trades_csv: Optional[Path], today: date,
        repo_root: Optional[Path] = None) -> dict:
    """Run all monitors once. Returns a summary dict (for tests/cron logs)."""
    repo_root = repo_root or REPO_ROOT
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    ei = load_edge_integrity_config(cfg)
    if not bool(ei["enabled"]):
        logger.info("[edge_integrity] disabled in config — nothing to do")
        return {"enabled": False}
    dry_run = bool(ei["dry_run"])

    # Startup validation: fail fast, do NOT run monitors on an invalid config.
    validation_errors = validate_preconditions(cfg)
    if validation_errors:
        for e in validation_errors:
            logger.critical("[edge_integrity] CONFIG VALIDATION: %s", e)
        raise ConfigError("; ".join(validation_errors))

    findings: List[Finding] = []
    setups = cfg.get("setups") or {}

    # Monitor 1 + 2 — per-setup preconditions
    rules_csv = Path(str(ei["rule_changes_csv"]))
    if not rules_csv.is_absolute():
        rules_csv = repo_root / rules_csv
    rules_df = load_rule_changes(rules_csv)
    lookback_days = int(ei["rule_watch_lookback_days"])
    for name, block in setups.items():
        # Review fix 2026-07-28: killed/soft-disabled setups keep their
        # preconditions[] as documentation, but monitoring them would clobber
        # their recorded kill reason with paused_precondition and add alert
        # noise. Only active (enabled or paper_enabled) setups are monitored.
        if not (block.get("enabled") or block.get("paper_enabled")):
            continue
        pres = block.get("preconditions") or []
        if not pres:
            continue
        findings.extend(check_rule_watch(name, pres, rules_df, today,
                                         lookback_days))
        for p in pres:
            if p.get("type") == "data_source":
                findings.append(check_data_source(name, p, today, repo_root))

    # Monitor 3 — factor clusters
    cluster_stats: List[dict] = []
    for cname, cluster in ei["factor_clusters"].items():
        if cname.startswith("_"):
            continue
        daily_by_member = {
            m: load_member_daily_pnl(m, setups.get(m) or {}, trades_csv,
                                     repo_root)
            for m in cluster["members"]
        }
        # Review fix 2026-07-28: a member with an entirely empty daily series
        # silently contributes 0 PnL and degrades cluster coverage (e.g.
        # panic_crash_revert_long has no decay-tripwire state file and relies
        # on --trades-csv). Surface it as a loud finding every run.
        for m, series in daily_by_member.items():
            if len(series) == 0:
                logger.warning(
                    "[edge_integrity] cluster %s member %s has an EMPTY daily "
                    "PnL series — check decay-tripwire state file / "
                    "--trades-csv wiring", cname, m)
                findings.append(Finding(
                    monitor="factor_tripwire",
                    setup_name=m,
                    action="ok",
                    reason="member_series_empty",
                    detail={"cluster": cname,
                            "note": "member contributes nothing to the "
                                    "tripwire; fix data wiring"},
                ))
        cfindings, cstats = check_factor_cluster(cname, cluster,
                                                 daily_by_member)
        findings.extend(cfindings)
        cluster_stats.append(cstats)

    pause_records = apply_pauses(config_path, cfg, findings, today, dry_run)

    report_dir = Path(str(ei["report_dir"]))
    if not report_dir.is_absolute():
        report_dir = repo_root / report_dir
    report_path = write_report(report_dir, today, dry_run=dry_run,
                               findings=findings, cluster_stats=cluster_stats,
                               pause_records=pause_records,
                               validation_errors=[])

    n_pause = sum(1 for f in findings if f.action == "pause")
    summary = {
        "enabled": True,
        "dry_run": dry_run,
        "n_findings": len(findings),
        "n_pause_triggers": n_pause,
        "n_pauses_applied": sum(1 for r in pause_records if r.get("applied")),
        "report_path": str(report_path),
    }
    logger.info("[edge_integrity] done: %s", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "config" / "configuration.json")
    p.add_argument("--trades-csv", type=Path, required=True,
                   help="trade_report.csv for intraday cluster members "
                        "(same schema as jobs/check_circuit_breakers.py; "
                        "REQUIRED — review fix 2026-07-28 so intraday cluster "
                        "members are never silently dropped from Monitor 3)")
    p.add_argument("--today",
                   type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                   default=None,
                   help="override run date (IST-naive); default = today IST")
    args = p.parse_args(argv)

    if args.today is not None:
        today = args.today
    else:
        # IST-naive per CLAUDE.md — never the host wall-clock date.
        from utils.time_util import _now_naive_ist
        today = _now_naive_ist().date()

    summary = run(args.config, args.trades_csv, today)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
