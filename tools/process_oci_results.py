#!/usr/bin/env python
"""
Process OCI backtest results - Generate analytics and comprehensive analysis.

This script mimics what engine.py does after a parallel backtest run:
1. Moves session directories from extraction folder to logs/
2. Generates analytics.jsonl and performance.json for each session
3. Generates diagnostics CSV reports
4. Runs comprehensive analysis across all sessions

Usage:
    python tools/process_oci_results.py <extracted_folder_path>

Example:
    python tools/process_oci_results.py logs_temp_extract/20251105-035540_full/20251105-035540
"""
import sys
import shutil
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.logging.trading_logger import TradingLogger
from diagnostics.diagnostics_report_builder import build_csv_from_events

def move_sessions_to_logs(source_dir: Path, run_id: str) -> list:
    """Move all session directories from extraction to logs/ directory"""
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MOVING SESSIONS TO LOGS DIRECTORY")
    print(f"{'='*80}")
    print(f"Source: {source_dir}")
    print(f"Destination: {logs_dir}")
    print()

    moved_sessions = []

    # Get all date directories (session directories)
    session_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    print(f"Found {len(session_dirs)} session directories")
    print()

    for session_dir in session_dirs:
        date_str = session_dir.name  # e.g., "2023-12-01"

        # Create new session name with run_id prefix
        # Format: bt_YYYYMMDD_<run_id>
        date_formatted = date_str.replace('-', '')
        new_session_name = f"bt_{date_formatted}_{run_id}"

        dest_dir = logs_dir / new_session_name

        # Move directory
        if dest_dir.exists():
            print(f"  [SKIP] {new_session_name} already exists")
            moved_sessions.append(new_session_name)
        else:
            shutil.move(str(session_dir), str(dest_dir))
            print(f"  [MOVED] {date_str} -> {new_session_name}")
            moved_sessions.append(new_session_name)

    print()
    print(f"Moved {len(moved_sessions)} sessions to logs/")
    print()

    return moved_sessions

def generate_analytics_for_session(session_id: str) -> bool:
    """Generate analytics for a single session (copied from engine.py)"""
    log_dir = str(ROOT / "logs" / session_id)
    events_file = ROOT / "logs" / session_id / "events.jsonl"

    # Skip if no events
    if not events_file.exists() or events_file.stat().st_size == 0:
        print(f"  [SKIP] {session_id} - no events")
        return False

    try:
        print(f"  [PROCESSING] {session_id}")

        # Generate analytics.jsonl and performance.json
        logger = TradingLogger(session_id, log_dir)
        logger.populate_analytics_from_events()

        # Generate diagnostics CSV
        csv_path = build_csv_from_events(log_dir=log_dir)

        print(f"    -> Analytics and CSV generated")
        return True

    except Exception as e:
        print(f"    -> ERROR: {str(e)[:100]}")
        return False

def process_all_sessions(session_ids: list):
    """Process analytics for all sessions"""
    print(f"\n{'='*80}")
    print(f"GENERATING ANALYTICS FOR {len(session_ids)} SESSIONS")
    print(f"{'='*80}")
    print()

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for session_id in session_ids:
        result = generate_analytics_for_session(session_id)
        if result:
            processed_count += 1
        else:
            # Check if it was skipped or errored
            events_file = ROOT / "logs" / session_id / "events.jsonl"
            if not events_file.exists() or events_file.stat().st_size == 0:
                skipped_count += 1
            else:
                error_count += 1

    print()
    print(f"{'='*80}")
    print(f"ANALYTICS SUMMARY")
    print(f"{'='*80}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (no events): {skipped_count}")
    print(f"  Errors: {error_count}")
    print()

def run_comprehensive_analysis(run_id: str, session_ids: list):
    """Run comprehensive analysis across all sessions for this specific run"""
    print(f"\n{'='*80}")
    print(f"RUNNING COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    print()

    # Since comprehensive_run_analyzer doesn't support suffix filtering,
    # we'll run our own simplified analysis here
    import json
    import pandas as pd

    print(f"Analyzing {len(session_ids)} sessions from run {run_id}...")
    print()

    # Load all executed trades
    all_trades = []
    for session_id in session_ids:
        analytics_file = ROOT / "logs" / session_id / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        try:
            with open(analytics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            trade = json.loads(line)
                            if trade.get('stage') == 'EXIT' and 'pnl' in trade:
                                all_trades.append({
                                    'session': session_id,
                                    'symbol': trade.get('symbol', ''),
                                    'setup_type': trade.get('setup_type', ''),
                                    'regime': trade.get('regime', ''),
                                    'pnl': trade.get('pnl', 0),
                                    'exit_reason': trade.get('reason', '')
                                })
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue

    if not all_trades:
        print("[WARNING] No executed trades found")
        return

    df = pd.DataFrame(all_trades)

    # Calculate summary stats
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).sum() / len(df) * 100
    avg_pnl = df['pnl'].mean()

    print(f"{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trades: {len(df)}")
    print(f"Total PnL: Rs.{total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average PnL/Trade: Rs.{avg_pnl:.2f}")
    print()

    # Setup performance
    print("SETUP PERFORMANCE:")
    for setup in df['setup_type'].unique():
        setup_df = df[df['setup_type'] == setup]
        setup_pnl = setup_df['pnl'].sum()
        setup_wr = (setup_df['pnl'] > 0).sum() / len(setup_df) * 100
        print(f"  {setup}: Rs.{setup_pnl:.0f} ({setup_wr:.1f}% WR, {len(setup_df)} trades)")

    print()

    # Regime performance
    print("REGIME PERFORMANCE:")
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        regime_pnl = regime_df['pnl'].sum()
        regime_wr = (regime_df['pnl'] > 0).sum() / len(regime_df) * 100
        print(f"  {regime}: Rs.{regime_pnl:.0f} ({regime_wr:.1f}% WR, {len(regime_df)} trades)")

    print()
    print("[SUCCESS] Analysis completed")
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/process_oci_results.py <extracted_folder_path>")
        print()
        print("Example:")
        print("  python tools/process_oci_results.py logs_temp_extract/20251105-035540_full/20251105-035540")
        sys.exit(1)

    source_path = Path(sys.argv[1])

    if not source_path.exists():
        print(f"ERROR: Source path does not exist: {source_path}")
        sys.exit(1)

    if not source_path.is_dir():
        print(f"ERROR: Source path is not a directory: {source_path}")
        sys.exit(1)

    # Extract run_id from parent folder name (e.g., "20251105-035540")
    # The structure is: logs_temp_extract/20251105-035540_full/20251105-035540/
    run_id = source_path.name  # Get the innermost folder name
    if not run_id:
        # If source_path is the parent, try to get it differently
        parent_name = source_path.parent.name
        if parent_name.endswith('_full'):
            run_id = parent_name.replace('_full', '')
        else:
            run_id = parent_name

    print(f"\n{'='*80}")
    print(f"OCI BACKTEST RESULTS PROCESSOR")
    print(f"{'='*80}")
    print(f"Source: {source_path}")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}")

    # Step 1: Move sessions to logs/
    session_ids = move_sessions_to_logs(source_path, run_id)

    if not session_ids:
        print("ERROR: No sessions found to process")
        sys.exit(1)

    # Step 2: Generate analytics for all sessions
    process_all_sessions(session_ids)

    # Step 3: Run comprehensive analysis for ONLY this run's sessions
    run_comprehensive_analysis(run_id, session_ids)

    print()
    print(f"{'='*80}")
    print(f"[SUCCESS] OCI RESULTS PROCESSING COMPLETE")
    print(f"{'='*80}")
    print()
    print(f"Processed {len(session_ids)} sessions")
    print(f"Run ID: {run_id}")
    print()
    print("Next steps:")
    print("  1. Check logs/ directory for individual session reports")
    print("  2. Review comprehensive analysis output above")
    print("  3. Check for analytics.jsonl, performance.json, and diagnostics.csv in each session")
    print()

if __name__ == "__main__":
    main()
