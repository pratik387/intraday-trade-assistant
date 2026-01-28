#!/usr/bin/env python3
"""
Run Manager CLI
---------------
Manage trading runs and generate combined reports.

Usage:
    python tools/run_manager_cli.py start "Backtest 2025-09-14"
    python tools/run_manager_cli.py status
    python tools/run_manager_cli.py sessions
    python tools/run_manager_cli.py report
    python tools/run_manager_cli.py end
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.logging.run_manager import RunManager
from diagnostics.diagnostics_report_builder import build_combined_csv_from_current_run

def main():
    parser = argparse.ArgumentParser(description="Manage trading runs")
    parser.add_argument("command", choices=["start", "status", "sessions", "report", "end"],
                       help="Command to execute")
    parser.add_argument("description", nargs="?", default="",
                       help="Run description (for start command)")

    args = parser.parse_args()

    # Initialize run manager
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    run_manager = RunManager(logs_dir)

    if args.command == "start":
        description = args.description or "Manual run"
        run_id = run_manager.start_new_run(description)
        print(f"[+] Started new run: {run_id}")
        print(f"    Description: {description}")

    elif args.command == "status":
        run_info = run_manager.get_current_run_info()
        if run_info:
            print(f"[*] Active run: {run_info['run_id']}")
            print(f"    Description: {run_info.get('description', 'N/A')}")
            print(f"    Started: {run_info.get('start_time', 'N/A')}")
            print(f"    Sessions: {len(run_info.get('sessions', []))}")
        else:
            print("[-] No active run")

    elif args.command == "sessions":
        sessions = run_manager.get_current_run_sessions()
        if sessions:
            print(f"[*] Current run has {len(sessions)} sessions:")
            for i, session_id in enumerate(sessions, 1):
                print(f"  {i}. {session_id}")
        else:
            print("[-] No sessions in current run")

    elif args.command == "report":
        try:
            csv_path = build_combined_csv_from_current_run()
            print(f"[+] Combined report generated: {csv_path}")
        except Exception as e:
            print(f"[-] Failed to generate report: {e}")

    elif args.command == "end":
        manifest = run_manager.end_current_run()
        if manifest:
            print(f"[+] Completed run: {manifest['run_id']}")
            print(f"    Total sessions: {len(manifest.get('sessions', []))}")
            print(f"    Archived to: {manifest['run_id']}.json")
        else:
            print("[-] No active run to end")

if __name__ == "__main__":
    main()