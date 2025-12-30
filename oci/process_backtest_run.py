#!/usr/bin/env python3
"""
OCI Backtest Processing Script

This script automates the complete processing of a backtest run:
1. Extracts the backtest zip file (or uses existing directory)
2. Generates analytics files for all sessions (postprocessing)
3. Generates comprehensive analysis report
4. Injects run ID and time period metadata into the report

Usage:
    python oci/process_backtest_run.py <backtest_zip_or_directory>

Examples:
    python oci/process_backtest_run.py backtest_20251109-125133.zip
    python oci/process_backtest_run.py 20251228-075841_full

The run ID (e.g., 20251228-075841) and time period are automatically
extracted and added to the generated JSON report for reference.
"""

import sys
import os
import zipfile
import subprocess
from pathlib import Path
import shutil
import time
import json
import re

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def _extract_run_id(input_path: str) -> str:
    """
    Extract the run ID from input path.

    Examples:
        - 20251228-075841_full -> 20251228-075841
        - backtest_20251228-075841.zip -> 20251228-075841
        - E:\path\20251228-075841_extracted\... -> 20251228-075841
    """
    path = Path(input_path)
    name = path.stem if path.suffix else path.name

    # Pattern: YYYYMMDD-HHMMSS (timestamp format)
    pattern = r'(\d{8}-\d{6})'
    match = re.search(pattern, name)
    if match:
        return match.group(1)

    # Check parent directories
    for parent in path.parents:
        match = re.search(pattern, parent.name)
        if match:
            return match.group(1)

    return None


def _get_time_period(sessions_root: Path) -> dict:
    """
    Get time period from session folder dates.

    Returns:
        Dict with start_date, end_date, and total_days
    """
    session_dirs = [
        d for d in sessions_root.iterdir()
        if d.is_dir() and len(d.name) == 10
        and d.name[4] == '-' and d.name[7] == '-'
        and d.name.startswith('20')
    ]

    if not session_dirs:
        return None

    dates = sorted([d.name for d in session_dirs])
    return {
        'start_date': dates[0],
        'end_date': dates[-1],
        'total_sessions': len(dates)
    }


def _find_sessions_root(extract_dir: Path):
    """
    Find the sessions root directory containing YYYY-MM-DD session folders.
    Returns None if not found.
    """
    # Check if extract_dir itself contains session dirs (YYYY-MM-DD format)
    def has_session_dirs(directory):
        return any(
            d.is_dir() and len(d.name) == 10
            and d.name[4] == '-' and d.name[7] == '-'
            and d.name.startswith('20')
            for d in directory.iterdir() if d.is_dir()
        )

    if has_session_dirs(extract_dir):
        return extract_dir

    # Try format 1: *_full directory
    full_dirs = list(extract_dir.glob("*_full"))
    if full_dirs:
        full_dir = full_dirs[0]
        session_dirs = [d for d in full_dir.iterdir() if d.is_dir()]
        if session_dirs:
            sessions_root = session_dirs[0]
            if has_session_dirs(sessions_root):
                return sessions_root

    # Try format 2: timestamp directories
    timestamp_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
    for potential_root in timestamp_dirs:
        if has_session_dirs(potential_root):
            return potential_root
        # Check if this is a single session folder
        if (potential_root / 'events.jsonl').exists() or (potential_root / 'screening.jsonl').exists():
            return extract_dir

    return None


def extract_backtest_zip(input_path: str) -> str:
    """
    Extract backtest zip file to a directory, or use existing directory.

    Args:
        input_path: Path to the backtest zip file OR existing directory

    Returns:
        Path to the directory containing session folders
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"=" * 80)
    print(f"STEP 1: Processing Input")
    print(f"=" * 80)

    # Check if input is a directory (already extracted or _full directory)
    if input_path.is_dir():
        # Try to find sessions root within this directory
        sessions_root = _find_sessions_root(input_path)
        if sessions_root:
            print(f"Using existing directory: {sessions_root}")
            session_folders = [d for d in sessions_root.iterdir() if d.is_dir() and d.name.startswith('20')]
            print(f"OK: Found {len(session_folders)} session folder(s)")
            print()
            return str(sessions_root)

        # Check subdirectories (e.g., 20251228-075841_full/20251228-075841/)
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                sessions_root = _find_sessions_root(subdir)
                if sessions_root:
                    print(f"Using existing directory: {sessions_root}")
                    session_folders = [d for d in sessions_root.iterdir() if d.is_dir() and d.name.startswith('20')]
                    print(f"OK: Found {len(session_folders)} session folder(s)")
                    print()
                    return str(sessions_root)

        raise ValueError(f"Could not find session directories in {input_path}")

    # Input is a zip file - extract it
    zip_path = input_path
    extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"

    print(f"Extracting {zip_path.name}")

    # Check if already extracted with valid session directories
    if extract_dir.exists():
        # Try to find valid session directories
        sessions_root = _find_sessions_root(extract_dir)
        if sessions_root:
            print(f"Already extracted: {sessions_root}")
            return str(sessions_root)

    # Extract zip file
    print(f"Extracting to: {extract_dir}")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Handle three possible zip structures:
    # 1. backtest_TIMESTAMP_extracted/TIMESTAMP_full/TIMESTAMP/ (6-month backtest format)
    # 2. backtest_TIMESTAMP_extracted/TIMESTAMP/ (old single-day format)
    # 3. backtest_TIMESTAMP_extracted/YYYY-MM-DD/ (new single-day format from cleanup script)

    # Try format 1: *_full directory
    full_dirs = list(extract_dir.glob("*_full"))

    if full_dirs:
        # Format 1: TIMESTAMP_full/TIMESTAMP/
        full_dir = full_dirs[0]
        session_dirs = [d for d in full_dir.iterdir() if d.is_dir()]

        if not session_dirs:
            raise ValueError(f"Could not find session directory in {full_dir}")

        sessions_root = session_dirs[0]
    else:
        # Format 2 or 3: Look for directories starting with '20'
        timestamp_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name.startswith('20')]

        if not timestamp_dirs:
            raise ValueError(f"Could not find timestamp directory in {extract_dir}")

        # Check if this is a sessions root (contains subdirs) or a single session
        potential_root = timestamp_dirs[0]
        session_subdirs = [d for d in potential_root.iterdir() if d.is_dir() and d.name.startswith('20')]

        if session_subdirs:
            # Format 2: This is a sessions root containing multiple session folders
            sessions_root = potential_root
        else:
            # Format 3: This IS a session folder itself
            # Check if it has session files (events.jsonl, etc.)
            has_session_files = (
                (potential_root / 'events.jsonl').exists() or
                (potential_root / 'screening.jsonl').exists()
            )

            if has_session_files:
                # This is a single session - use parent as sessions_root
                sessions_root = extract_dir
            else:
                sessions_root = potential_root

    # Count session folders
    session_folders = [d for d in sessions_root.iterdir() if d.is_dir() and d.name.startswith('20')]

    print(f"OK: Extracted successfully")
    print(f"OK: Found {len(session_folders)} session folder(s)")
    print(f"OK: Sessions root: {sessions_root}")
    print()

    return str(sessions_root)

def generate_analytics(sessions_root: str) -> bool:
    """
    Run postprocessing to generate analytics.jsonl and performance.json for all sessions.

    Args:
        sessions_root: Path to the directory containing session folders

    Returns:
        True if successful, False otherwise
    """
    print(f"=" * 80)
    print(f"STEP 2: Generating Analytics Files")
    print(f"=" * 80)

    # Run postprocessing script (resolve absolute path from script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    postprocess_script = project_root / "tools" / "postprocess_extracted_backtest.py"

    cmd = [
        "python",
        str(postprocess_script),
        sessions_root
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print("ERROR: FAILED to generate analytics files")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False

    # Show last 20 lines of output
    output_lines = result.stdout.strip().split('\n')
    if len(output_lines) > 20:
        print("... (showing last 20 lines)")
        print('\n'.join(output_lines[-20:]))
    else:
        print(result.stdout)

    print()
    print(f"OK: Analytics generation completed in {elapsed:.1f}s")
    print()

    return True

def generate_comprehensive_report(sessions_root: str) -> str:
    """
    Generate comprehensive analysis report using ComprehensiveRunAnalyzer.

    Args:
        sessions_root: Path to the directory containing session folders

    Returns:
        Path to the generated report file
    """
    print(f"=" * 80)
    print(f"STEP 3: Generating Comprehensive Analysis Report")
    print(f"=" * 80)

    # Convert to absolute path to ensure ComprehensiveRunAnalyzer can find it
    sessions_root_abs = str(Path(sessions_root).resolve())

    # Get absolute paths for project directories
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cache_dir = project_root / "cache" / "ohlcv_archive"
    reports_dir = project_root / "analysis" / "reports" / "misc"

    # Create Python script to run the analyzer
    # Use raw string literal for Windows paths
    script = f"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(r'{str(project_root)}')
sys.path.insert(0, str(project_root))

from comprehensive_run_analyzer import ComprehensiveRunAnalyzer

# Create analyzer pointing to the sessions directory
# Pass absolute paths for cache and output directory
analyzer = ComprehensiveRunAnalyzer(
    run_prefix='20',  # Match all session folders (2023-12-01, 2024-01-05, etc.)
    logs_dir=r'{sessions_root_abs}',
    ohlcv_cache_dir=r'{str(cache_dir)}'
)

print('Running comprehensive analysis...')
report_file = analyzer.run_comprehensive_analysis()

if report_file:
    # Move report to analysis/reports/misc directory
    from pathlib import Path
    import shutil

    report_path = Path(report_file)
    misc_reports_dir = Path(r'{str(reports_dir)}')
    misc_reports_dir.mkdir(parents=True, exist_ok=True)

    final_report_path = misc_reports_dir / report_path.name
    shutil.move(str(report_path), str(final_report_path))

    print(f'SUCCESS! Report saved to: {{final_report_path}}')
    # Print the report file path so we can capture it
    print(f'REPORT_FILE:{{final_report_path}}')
else:
    print('FAILED: Analysis returned no report file')
    sys.exit(1)
"""

    # Run the script
    result = subprocess.run(
        ["python", "-c", script],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: FAILED to generate comprehensive report")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return None

    print(result.stdout)

    # Extract report file path from output
    report_file = None
    for line in result.stdout.split('\n'):
        if line.startswith('REPORT_FILE:'):
            report_file = line.replace('REPORT_FILE:', '')
            break

    if not report_file:
        print("ERROR: Could not find report file path in output")
        return None

    print()
    print(f"OK: Comprehensive report generated successfully")
    print()

    return report_file


def inject_report_metadata(report_file: str, run_id: str, time_period: dict) -> bool:
    """
    Inject run ID and time period metadata into the generated JSON report.

    Args:
        report_file: Path to the JSON report file
        run_id: The backtest run ID (e.g., '20251228-075841')
        time_period: Dict with start_date, end_date, total_sessions

    Returns:
        True if successful, False otherwise
    """
    try:
        report_path = Path(report_file)
        if not report_path.exists():
            print(f"WARNING: Report file not found: {report_file}")
            return False

        # Load the report
        with open(report_path, 'r') as f:
            report = json.load(f)

        # Inject metadata at the top level
        # Use OrderedDict-like insertion by creating new dict with desired order
        metadata = {
            'backtest_run_id': run_id,
            'time_period': time_period
        }

        # Insert at beginning of report
        updated_report = {**metadata, **report}

        # Save back
        with open(report_path, 'w') as f:
            json.dump(updated_report, f, indent=2, default=str)

        print(f"OK: Injected run metadata into report")
        print(f"    Run ID: {run_id}")
        if time_period:
            print(f"    Time Period: {time_period['start_date']} to {time_period['end_date']} ({time_period['total_sessions']} sessions)")

        return True

    except Exception as e:
        print(f"WARNING: Failed to inject metadata: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Error: Missing backtest input argument")
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <backtest_zip_or_directory>")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} backtest_20251109-125133.zip")
        print(f"  python {sys.argv[0]} 20251228-075841_full")
        print()
        print("This script:")
        print("  1. Extracts the backtest zip file (or uses existing directory)")
        print("  2. Generates analytics.jsonl for all sessions (with fees per trade)")
        print("  3. Generates comprehensive report via ComprehensiveRunAnalyzer")
        print("  4. Injects run ID and time period metadata into the report")
        print()
        print("For detailed reports with MIS leverage and tax calculations,")
        print("run generate_backtest_report.py after this script.")
        sys.exit(1)

    input_path = sys.argv[1]

    # Extract run ID from input path
    run_id = _extract_run_id(input_path)

    print()
    print("=" * 80)
    print("OCI BACKTEST PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Input: {input_path}")
    if run_id:
        print(f"Run ID: {run_id}")
    print("=" * 80)
    print()

    try:
        # Step 1: Extract zip (or use existing directory)
        sessions_root = extract_backtest_zip(input_path)

        # Step 2: Generate analytics files (includes fees per trade)
        if not generate_analytics(sessions_root):
            print()
            print("=" * 80)
            print("PIPELINE FAILED AT STEP 2: Analytics Generation")
            print("=" * 80)
            sys.exit(1)

        # Step 3: Generate comprehensive report
        report_file = generate_comprehensive_report(sessions_root)

        if not report_file:
            print()
            print("=" * 80)
            print("PIPELINE FAILED AT STEP 3: Report Generation")
            print("=" * 80)
            sys.exit(1)

        # Step 4: Inject run metadata into report
        time_period = _get_time_period(Path(sessions_root))
        if run_id or time_period:
            print("=" * 80)
            print("STEP 4: Injecting Run Metadata")
            print("=" * 80)
            inject_report_metadata(report_file, run_id, time_period)
            print()

        # Success!
        print()
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print(f"Extracted Directory: {sessions_root}")
        print(f"Comprehensive Report: {report_file}")
        if run_id:
            print(f"Backtest Run ID: {run_id}")
        if time_period:
            print(f"Time Period: {time_period['start_date']} to {time_period['end_date']}")
        print()
        print("Next step: Run generate_backtest_report.py for detailed analysis")
        print("  python tools/generate_backtest_report.py")
        print("=" * 80)
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print(f"PIPELINE FAILED WITH ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
