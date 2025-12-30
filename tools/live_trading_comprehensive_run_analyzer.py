#!/usr/bin/env python
"""
Live Trading Comprehensive Run Analyzer

Processes a single live/paper trading session folder to generate:
1. analytics.jsonl - Trade exits with PnL (from TradingLogger)
2. diagnostics CSV - Trade details in CSV format
3. Comprehensive analysis report - JSON with performance metrics

Usage:
    python tools/live_trading_comprehensive_run_analyzer.py paper_20251229_073712
    python tools/live_trading_comprehensive_run_analyzer.py logs/paper_20251229_073712
    python tools/live_trading_comprehensive_run_analyzer.py  # Auto-discovers latest session

Output:
    - {session_dir}/analytics.jsonl
    - {session_dir}/diagnostics.csv
    - analysis/reports/misc/analysis_report_{session_id}.json
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ----- repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_session_dir(session_arg: str | None) -> Path | None:
    """
    Find the session directory from argument or auto-discover latest.

    Args:
        session_arg: Session folder name or path (e.g., "paper_20251229_073712" or full path)

    Returns:
        Path to session directory, or None if not found
    """
    if session_arg:
        # Try as-is first
        if Path(session_arg).exists():
            return Path(session_arg)

        # Try under logs/
        logs_path = ROOT / "logs" / session_arg
        if logs_path.exists():
            return logs_path

        # Try in current directory
        cwd_path = Path.cwd() / session_arg
        if cwd_path.exists():
            return cwd_path

        # Try as direct path from ROOT
        root_path = ROOT / session_arg
        if root_path.exists():
            return root_path

        print(f"[ERROR] Session directory not found: {session_arg}")
        print(f"        Tried: {session_arg}, logs/{session_arg}, {cwd_path}, {root_path}")
        return None

    # Auto-discover latest session
    print("[INFO] No session specified, auto-discovering latest...")

    # Look for paper_* or live_* directories in ROOT and logs/
    candidates = []

    # Check ROOT directory
    for item in ROOT.iterdir():
        if item.is_dir() and (item.name.startswith("paper_") or item.name.startswith("live_")):
            events_file = item / "events.jsonl"
            if events_file.exists():
                candidates.append(item)

    # Check logs directory
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        for item in logs_dir.iterdir():
            if item.is_dir() and (item.name.startswith("paper_") or item.name.startswith("live_")):
                events_file = item / "events.jsonl"
                if events_file.exists():
                    candidates.append(item)

    if not candidates:
        print("[ERROR] No session directories found with events.jsonl")
        return None

    # Sort by modification time (newest first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    print(f"[INFO] Found latest session: {latest.name}")
    return latest


def generate_analytics(session_dir: Path) -> bool:
    """
    Generate analytics.jsonl from events.jsonl using TradingLogger.

    Returns:
        True if successful, False otherwise
    """
    session_id = session_dir.name
    events_file = session_dir / "events.jsonl"

    if not events_file.exists():
        print(f"[SKIP] No events.jsonl found in {session_dir}")
        return False

    if events_file.stat().st_size == 0:
        print(f"[SKIP] events.jsonl is empty in {session_dir}")
        return False

    print(f"\n[1/3] Generating analytics from events...")
    print(f"      Session: {session_id}")
    print(f"      Events file: {events_file}")

    try:
        from services.logging.trading_logger import TradingLogger

        logger = TradingLogger(session_id, str(session_dir))
        logger.populate_analytics_from_events()

        analytics_file = session_dir / "analytics.jsonl"
        if analytics_file.exists():
            # Count trades
            trade_count = 0
            with open(analytics_file) as f:
                for line in f:
                    if line.strip():
                        ev = json.loads(line)
                        if ev.get("is_final_exit"):
                            trade_count += 1
            print(f"      Analytics generated: {trade_count} trades")
        else:
            print(f"      Analytics file created (check for content)")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate analytics: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_diagnostics_csv(session_dir: Path) -> bool:
    """
    Generate diagnostics CSV from events.jsonl.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n[2/3] Generating diagnostics CSV...")

    try:
        from diagnostics.diagnostics_report_builder import build_csv_from_events

        csv_path = build_csv_from_events(log_dir=str(session_dir))
        if csv_path and Path(csv_path).exists():
            size = Path(csv_path).stat().st_size
            print(f"      CSV written: {csv_path} ({size} bytes)")
            return True
        else:
            print(f"      CSV generation returned no path")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to generate diagnostics CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_analysis(session_dir: Path) -> bool:
    """
    Run comprehensive_run_analyzer.py for the session.

    For a single session, we use the session directory as the "run prefix" equivalent.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n[3/3] Running comprehensive analysis...")

    session_id = session_dir.name

    # The comprehensive analyzer expects a prefix that matches log directories
    # For live/paper sessions, we can pass the full session name as prefix
    # But we need to handle the case where the session is not in logs/

    # First, check if session is in logs/ - if not, we need to create a symlink or copy
    logs_dir = ROOT / "logs"
    logs_session = logs_dir / session_id

    # If session is not under logs/, we need to handle it
    if session_dir.parent != logs_dir:
        print(f"      Note: Session is not under logs/, analysis may need adjustment")
        # Try running analyzer anyway with the session name as prefix

    try:
        analyzer_path = ROOT / "comprehensive_run_analyzer.py"

        if not analyzer_path.exists():
            print(f"[ERROR] Analyzer not found: {analyzer_path}")
            return False

        print(f"      Analyzer: {analyzer_path}")
        print(f"      Prefix: {session_id}")

        # Run the analyzer with the session ID as prefix
        result = subprocess.run(
            [sys.executable, str(analyzer_path), session_id],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300
        )

        if result.returncode == 0:
            print(f"      Analysis completed successfully")

            # Find the generated report
            reports_dir = ROOT / "analysis" / "reports" / "misc"
            if reports_dir.exists():
                reports = list(reports_dir.glob(f"analysis_report_{session_id}*.json"))
                if reports:
                    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
                    print(f"      Report: {latest_report}")
                    return True

            # Check stdout for any output
            if result.stdout:
                # Look for report path in output
                for line in result.stdout.split('\n'):
                    if 'report' in line.lower() or 'saved' in line.lower():
                        print(f"      {line.strip()}")

            return True
        else:
            print(f"[ERROR] Analysis failed (exit code {result.returncode})")
            if result.stderr:
                print(f"      Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Analysis timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(session_dir: Path) -> None:
    """Print a summary of the session's trading results."""
    print(f"\n{'='*60}")
    print("TRADING SESSION SUMMARY")
    print(f"{'='*60}")

    session_id = session_dir.name
    analytics_file = session_dir / "analytics.jsonl"

    if not analytics_file.exists():
        print("No analytics data available")
        return

    # Parse analytics
    trades = []
    with open(analytics_file) as f:
        for line in f:
            if line.strip():
                ev = json.loads(line)
                if ev.get("is_final_exit"):
                    trades.append(ev)

    if not trades:
        print("No completed trades found")
        return

    # Calculate metrics
    total_pnl = sum(t.get("total_trade_pnl", 0) for t in trades)
    winners = [t for t in trades if t.get("total_trade_pnl", 0) > 0]
    losers = [t for t in trades if t.get("total_trade_pnl", 0) <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    avg_winner = sum(t.get("total_trade_pnl", 0) for t in winners) / len(winners) if winners else 0
    avg_loser = sum(t.get("total_trade_pnl", 0) for t in losers) / len(losers) if losers else 0

    # Get unique setups
    setups = {}
    for t in trades:
        setup = t.get("setup_type", "unknown")
        if setup not in setups:
            setups[setup] = {"count": 0, "pnl": 0}
        setups[setup]["count"] += 1
        setups[setup]["pnl"] += t.get("total_trade_pnl", 0)

    print(f"\nSession: {session_id}")
    print(f"\n{'-'*40}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*40}")
    print(f"  Total Trades:    {len(trades)}")
    print(f"  Winners:         {len(winners)}")
    print(f"  Losers:          {len(losers)}")
    print(f"  Win Rate:        {win_rate:.1f}%")
    print(f"  Total P&L:       Rs {total_pnl:,.2f}")
    print(f"  Avg Winner:      Rs {avg_winner:,.2f}")
    print(f"  Avg Loser:       Rs {avg_loser:,.2f}")

    print(f"\n{'-'*40}")
    print("BY SETUP TYPE")
    print(f"{'-'*40}")
    for setup, data in sorted(setups.items(), key=lambda x: x[1]["pnl"], reverse=True):
        status = "[OK]" if data["pnl"] > 0 else "[X]"
        print(f"  {setup:<30} {data['count']:>3} trades  Rs {data['pnl']:>10,.2f} {status}")

    print(f"\n{'='*60}")


def main():
    print("="*60)
    print("LIVE TRADING COMPREHENSIVE RUN ANALYZER")
    print("="*60)

    # Get session argument
    session_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Find session directory
    session_dir = find_session_dir(session_arg)
    if not session_dir:
        print("\nUsage:")
        print("  python tools/live_trading_comprehensive_run_analyzer.py paper_20251229_073712")
        print("  python tools/live_trading_comprehensive_run_analyzer.py  # Auto-discover latest")
        return 1

    print(f"\nProcessing: {session_dir}")

    # Step 1: Generate analytics
    analytics_ok = generate_analytics(session_dir)

    # Step 2: Generate diagnostics CSV
    csv_ok = generate_diagnostics_csv(session_dir)

    # Step 3: Run comprehensive analysis (only if analytics generated)
    analysis_ok = False
    if analytics_ok:
        analysis_ok = run_comprehensive_analysis(session_dir)
    else:
        print("\n[SKIP] Skipping comprehensive analysis (no analytics)")

    # Print summary
    print_summary(session_dir)

    # Final status
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Analytics:    {'[OK]' if analytics_ok else '[SKIP]'}")
    print(f"  CSV Report:   {'[OK]' if csv_ok else '[SKIP]'}")
    print(f"  Analysis:     {'[OK]' if analysis_ok else '[SKIP]'}")
    print(f"\nOutput directory: {session_dir}")

    return 0 if analytics_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
