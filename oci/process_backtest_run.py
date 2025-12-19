#!/usr/bin/env python3
"""
OCI Backtest Processing Script

This script automates the complete processing of a backtest run:
1. Extracts the backtest zip file
2. Generates analytics files for all sessions (postprocessing)
3. Generates comprehensive analysis report

Usage:
    python oci/process_backtest_run.py <backtest_zip_file>

Example:
    python oci/process_backtest_run.py backtest_20251109-125133.zip
"""

import sys
import os
import zipfile
import subprocess
from pathlib import Path
import shutil
import time
import json

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from tools.calculate_net_pnl import calculate_net_pnl, find_mis_file
    NET_PNL_AVAILABLE = True
except ImportError:
    NET_PNL_AVAILABLE = False
    def find_mis_file():
        return None


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


def extract_backtest_zip(zip_path: str) -> str:
    """
    Extract backtest zip file to a directory.

    Args:
        zip_path: Path to the backtest zip file

    Returns:
        Path to the extracted directory containing session folders
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Extract to <filename>_extracted/ directory
    extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"

    print(f"=" * 80)
    print(f"STEP 1: Extracting {zip_path.name}")
    print(f"=" * 80)

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


def calculate_and_append_net_pnl(sessions_root: str, report_file: str, mis_file: str = None) -> bool:
    """
    Calculate net PnL with MIS leverage, fees, and taxation and append to report.

    Args:
        sessions_root: Path to the directory containing session folders
        report_file: Path to the JSON report file to update
        mis_file: Optional path to MIS margin file

    Returns:
        True if successful, False otherwise
    """
    print(f"=" * 80)
    print(f"STEP 4: Calculating Net PnL (MIS, Fees, Tax)")
    print(f"=" * 80)

    if not NET_PNL_AVAILABLE:
        print("WARNING: Net PnL calculation not available (missing calculate_net_pnl module)")
        return False

    # Find MIS file if not provided
    if not mis_file:
        mis_file = find_mis_file()
        if mis_file:
            print(f"Using MIS file: {mis_file}")
        else:
            print("No MIS file found. Using default 1x leverage (NRML).")

    try:
        # Calculate net PnL
        net_pnl_result = calculate_net_pnl(
            sessions_root,
            mis_file=mis_file,
            use_mis=True,
            verbose=True
        )

        # Load existing report
        with open(report_file, 'r') as f:
            report_data = json.load(f)

        # Add net PnL analysis to report
        report_data['net_pnl_analysis'] = {
            'gross_pnl_nrml': net_pnl_result['gross_pnl_nrml'],
            'mis_multiplier_avg': net_pnl_result['mis_multiplier_avg'],
            'gross_pnl_with_mis': net_pnl_result['gross_pnl_mis'],
            'total_fees': net_pnl_result['total_fees'],
            'profit_after_fees': net_pnl_result['profit_after_fees'],
            'tax_breakdown': net_pnl_result['tax'],
            'net_pnl_final': net_pnl_result['net_pnl'],
            'mis_file_used': mis_file or 'default_1x_leverage'
        }

        # Save updated report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print()
        print(f"OK: Net PnL analysis added to report")
        print(f"    Gross PnL (NRML):     Rs {net_pnl_result['gross_pnl_nrml']:>12,.0f}")
        print(f"    Avg MIS Multiplier:      {net_pnl_result['mis_multiplier_avg']:>10.2f}x")
        print(f"    Gross PnL (with MIS): Rs {net_pnl_result['gross_pnl_mis']:>12,.0f}")
        print(f"    Total Fees:           Rs {net_pnl_result['total_fees']:>12,.0f}")
        print(f"    Tax (30% + 4% cess):  Rs {net_pnl_result['tax']['total_tax']:>12,.0f}")
        print(f"    NET PNL FINAL:        Rs {net_pnl_result['net_pnl']:>12,.0f}")
        print()

        return True

    except Exception as e:
        print(f"ERROR calculating net PnL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Error: Missing backtest zip file argument")
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <backtest_zip_file> [mis_margin_file.xlsx | existing_report.json]")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} backtest_20251109-125133.zip")
        print(f"  python {sys.argv[0]} backtest_20251109-125133.zip zerodha_mis_margin.xlsx")
        print(f"  python {sys.argv[0]} backtest_20251109-125133.zip existing_report.json  # Skip steps 2-3, just add MIS data")
        sys.exit(1)

    zip_file = sys.argv[1]

    # Handle optional second argument - could be MIS file or existing report file
    mis_file = None
    existing_report = None
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if arg2.endswith('.json'):
            # User provided an existing report file to update
            existing_report = arg2
        elif arg2.endswith('.xlsx') or arg2.endswith('.xls'):
            # User provided MIS margins file
            mis_file = arg2
        else:
            # Assume it's MIS file
            mis_file = arg2

    print()
    print("=" * 80)
    print("OCI BACKTEST PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Backtest Zip: {zip_file}")
    print("=" * 80)
    print()

    try:
        # Step 1: Extract zip
        sessions_root = extract_backtest_zip(zip_file)

        # If user provided an existing report file, skip steps 2-3 and just add MIS data
        if existing_report:
            print()
            print("=" * 80)
            print("USING EXISTING REPORT FILE - Skipping Steps 2-3")
            print("=" * 80)
            print(f"Existing Report: {existing_report}")
            report_file = existing_report
        else:
            # Step 2: Generate analytics files
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

        # Step 4: Calculate and append net PnL (MIS, fees, tax)
        calculate_and_append_net_pnl(sessions_root, report_file, mis_file)

        # Success!
        print()
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print(f"Extracted Directory: {sessions_root}")
        print(f"Comprehensive Report: {report_file}")
        print()
        print("You can now analyze the report to understand backtest performance.")
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
