#!/usr/bin/env python3
"""
Comprehensive Backtest Report Generator - Orchestrator Script

Usage:
    python tools/generate_backtest_report.py <backtest_zip_or_directory> [--baseline <baseline_zip_or_directory>]

Example:
    python tools/generate_backtest_report.py backtest_20251107-123840.zip --baseline backtest_20251106-165927.zip
    python tools/generate_backtest_report.py backtest_20251107-083559_extracted/20251107-083559_full/20251107-083559 --baseline backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927

This script orchestrates:
0. ZIP extraction (if input is a ZIP file)
1. Postprocessing (analytics.jsonl generation)
2. CSV report generation (diagnostics_report_builder)
3. Comprehensive performance analysis
4. Baseline comparison (if provided)
5. Executive summary generation

Similar to engine.py's final reporting, but for extracted backtests.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import zipfile
import tempfile
import shutil

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.logging.trading_logger import TradingLogger
from diagnostics.diagnostics_report_builder import build_csv_from_events

# Import net PnL calculator
try:
    from tools.calculate_net_pnl import calculate_net_pnl, MISLeverageCalculator, ZerodhaFeeCalculator, TaxCalculator, find_mis_file
    NET_PNL_AVAILABLE = True
except ImportError:
    NET_PNL_AVAILABLE = False

    def find_mis_file():
        return None


def extract_zip_if_needed(zip_or_dir_path: str) -> Path:
    """
    Extract ZIP file if input is a ZIP, otherwise return directory path.
    Returns the path to the session directory (containing YYYY-MM-DD folders).
    Skips extraction if already extracted.
    """
    path = Path(zip_or_dir_path)

    if path.suffix == '.zip':
        # Check if already extracted
        extract_dir = path.parent / f"{path.stem}_extracted"

        if extract_dir.exists():
            # Check if it contains valid session directories
            session_parent = _find_session_directory(extract_dir)
            if session_parent:
                print(f"Already extracted: {session_parent}")
                return session_parent

        print(f"Extracting ZIP file: {path.name}")

        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the session directory
        session_parent = _find_session_directory(extract_dir)

        if not session_parent:
            raise ValueError(f"Could not find session directory in extracted ZIP: {path}")

        print(f"Extracted to: {session_parent}")
        return session_parent
    else:
        return path


def _find_session_directory(extract_dir: Path) -> Path:
    """
    Find the session directory containing YYYY-MM-DD folders.
    Drills down through nested structure like: extracted/{run_id}_full/{run_id}/YYYY-MM-DD/...
    """
    # First check if extract_dir itself contains session dirs
    has_session_dirs = any(
        d.is_dir() and len(d.name) == 10
        and d.name[4] == '-' and d.name[7] == '-'
        for d in extract_dir.iterdir() if d.is_dir()
    )
    if has_session_dirs:
        return extract_dir

    # Otherwise search recursively
    candidates = list(extract_dir.rglob('*'))
    for candidate in candidates:
        if candidate.is_dir():
            # Check if this directory contains YYYY-MM-DD folders
            has_session_dirs = any(
                d.is_dir() and len(d.name) == 10
                and d.name[4] == '-' and d.name[7] == '-'
                for d in candidate.iterdir() if d.is_dir()
            )
            if has_session_dirs:
                return candidate

    return None


class BacktestReportGenerator:
    def __init__(self, backtest_dir: str, baseline_dir: str = None):
        self.backtest_dir = Path(backtest_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.run_id = self.backtest_dir.name if len(self.backtest_dir.name) > 10 else self.backtest_dir.parent.parent.name

        if not self.backtest_dir.exists():
            raise ValueError(f"Backtest directory not found: {backtest_dir}")

        self.session_dirs = sorted([d for d in self.backtest_dir.iterdir()
                                     if d.is_dir() and len(d.name) == 10
                                     and d.name[4] == '-' and d.name[7] == '-'])

        if not self.session_dirs:
            raise ValueError(f"No session directories found in {backtest_dir}")

        print(f"=" * 80)
        print(f"Backtest Report Generator")
        print(f"=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Sessions: {len(self.session_dirs)}")
        print(f"Date range: {self.session_dirs[0].name} to {self.session_dirs[-1].name}")
        if self.baseline_dir:
            print(f"Baseline: {self.baseline_dir.name}")
        print(f"=" * 80)
        print()

    def step1_postprocess_analytics(self):
        """Step 1: Generate analytics.jsonl from events.jsonl for all sessions"""
        print("STEP 1: Postprocessing Analytics")
        print("-" * 80)

        # Use the existing postprocess_extracted_backtest.py script which properly handles analytics generation
        import subprocess

        script_path = Path(__file__).parent / "postprocess_extracted_backtest.py"
        result = subprocess.run(
            [sys.executable, str(script_path), str(self.backtest_dir)],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"ERROR: Postprocessing failed")
            print(result.stderr)
            return False

        print()
        return True

    def step2_generate_csv_reports(self):
        """Step 2: Generate CSV reports from events.jsonl"""
        print("STEP 2: Generating CSV Reports")
        print("-" * 80)

        for idx, session_dir in enumerate(self.session_dirs, 1):
            session_id = session_dir.name
            csv_file = session_dir / 'trades.csv'

            if csv_file.exists():
                print(f"  [{idx}/{len(self.session_dirs)}] {session_id}: trades.csv exists, skipping")
                continue

            try:
                build_csv_from_events(log_dir=str(session_dir))
                print(f"  [{idx}/{len(self.session_dirs)}] {session_id}: trades.csv generated")
            except Exception as e:
                print(f"  [{idx}/{len(self.session_dirs)}] {session_id}: ERROR - {e}")

        print(f"\nCSV generation complete: {len(self.session_dirs)} sessions")
        print()

    def step3_aggregate_results(self):
        """Step 3: Aggregate all analytics.jsonl into comprehensive summary"""
        print("STEP 3: Aggregating Results")
        print("-" * 80)

        all_trades = []
        for session_dir in self.session_dirs:
            analytics_file = session_dir / 'analytics.jsonl'
            if not analytics_file.exists():
                continue

            with open(analytics_file) as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('stage') == 'EXIT':
                            event['session'] = session_dir.name
                            all_trades.append(event)

        if not all_trades:
            print("ERROR: No trade data found!")
            return None

        df = pd.DataFrame(all_trades)

        # Calculate comprehensive metrics
        results = {
            'run_id': self.run_id,
            'sessions': len(self.session_dirs),
            'date_range': f"{self.session_dirs[0].name} to {self.session_dirs[-1].name}",
            'total_trades': len(df),
            'total_pnl': float(df['pnl'].sum()),
            'avg_pnl_per_trade': float(df['pnl'].mean()),
            'win_rate': float(len(df[df['pnl'] > 0]) / len(df)),
            'wins': int(len(df[df['pnl'] > 0])),
            'losses': int(len(df[df['pnl'] <= 0])),
            'avg_winner': float(df[df['pnl'] > 0]['pnl'].mean()) if len(df[df['pnl'] > 0]) > 0 else 0,
            'avg_loser': float(df[df['pnl'] <= 0]['pnl'].mean()) if len(df[df['pnl'] <= 0]) > 0 else 0,
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] <= 0]['pnl'].sum()) if df[df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf'),
        }

        # Exit reason breakdown
        exit_reasons = df['reason'].value_counts().to_dict()
        exit_reasons_pnl = {reason: float(df[df['reason'] == reason]['pnl'].sum())
                           for reason in exit_reasons.keys()}

        results['exit_reasons'] = exit_reasons
        results['exit_reasons_pnl'] = exit_reasons_pnl

        # Strategy breakdown
        if 'strategy' in df.columns:
            strategy_stats = {}
            for strategy in df['strategy'].unique():
                strat_df = df[df['strategy'] == strategy]
                strategy_stats[str(strategy)] = {
                    'trades': int(len(strat_df)),
                    'pnl': float(strat_df['pnl'].sum()),
                    'win_rate': float(len(strat_df[strat_df['pnl'] > 0]) / len(strat_df)) if len(strat_df) > 0 else 0
                }
            results['strategy_breakdown'] = strategy_stats

        print(f"Total Trades: {results['total_trades']}")
        print(f"Total P&L: Rs.{results['total_pnl']:.2f}")
        print(f"Avg P&L/Trade: Rs.{results['avg_pnl_per_trade']:.2f}")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print()

        return results

    def step4_compare_with_baseline(self, current_results):
        """Step 4: Compare with baseline if provided"""
        if not self.baseline_dir:
            return None

        print("STEP 4: Baseline Comparison")
        print("-" * 80)

        # Load baseline trades
        baseline_sessions = sorted([d for d in self.baseline_dir.iterdir()
                                    if d.is_dir() and len(d.name) == 10])

        baseline_trades = []
        for session_dir in baseline_sessions:
            analytics_file = session_dir / 'analytics.jsonl'
            if analytics_file.exists():
                with open(analytics_file) as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            if event.get('stage') == 'EXIT':
                                baseline_trades.append(event)

        if not baseline_trades:
            print("WARNING: No baseline trade data found")
            return None

        baseline_df = pd.DataFrame(baseline_trades)

        baseline_results = {
            'total_trades': len(baseline_df),
            'total_pnl': float(baseline_df['pnl'].sum()),
            'avg_pnl_per_trade': float(baseline_df['pnl'].mean()),
            'win_rate': float(len(baseline_df[baseline_df['pnl'] > 0]) / len(baseline_df)),
            'profit_factor': abs(baseline_df[baseline_df['pnl'] > 0]['pnl'].sum() / baseline_df[baseline_df['pnl'] <= 0]['pnl'].sum())
        }

        comparison = {
            'total_trades': {
                'baseline': baseline_results['total_trades'],
                'current': current_results['total_trades'],
                'change': current_results['total_trades'] - baseline_results['total_trades'],
                'change_pct': ((current_results['total_trades'] / baseline_results['total_trades']) - 1) * 100 if baseline_results['total_trades'] > 0 else 0
            },
            'total_pnl': {
                'baseline': baseline_results['total_pnl'],
                'current': current_results['total_pnl'],
                'change': current_results['total_pnl'] - baseline_results['total_pnl'],
                'change_pct': ((current_results['total_pnl'] / baseline_results['total_pnl']) - 1) * 100 if baseline_results['total_pnl'] > 0 else 0
            },
            'win_rate': {
                'baseline': baseline_results['win_rate'],
                'current': current_results['win_rate'],
                'change': current_results['win_rate'] - baseline_results['win_rate'],
                'change_pct': ((current_results['win_rate'] / baseline_results['win_rate']) - 1) * 100 if baseline_results['win_rate'] > 0 else 0
            },
            'profit_factor': {
                'baseline': baseline_results['profit_factor'],
                'current': current_results['profit_factor'],
                'change': current_results['profit_factor'] - baseline_results['profit_factor'],
                'change_pct': ((current_results['profit_factor'] / baseline_results['profit_factor']) - 1) * 100 if baseline_results['profit_factor'] > 0 else 0
            }
        }

        print(f"{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Change':<15}")
        print("-" * 80)
        for metric, values in comparison.items():
            metric_display = metric.replace('_', ' ').title()
            baseline_val = values['baseline']
            current_val = values['current']
            change = values['change']
            change_pct = values['change_pct']

            # Format based on metric type
            if 'pnl' in metric:
                print(f"{metric_display:<25} Rs.{baseline_val:<13.2f} Rs.{current_val:<13.2f} {change:+.2f} ({change_pct:+.1f}%)")
            elif 'rate' in metric:
                print(f"{metric_display:<25} {baseline_val*100:<13.1f}% {current_val*100:<13.1f}% {change*100:+.1f}pp ({change_pct:+.1f}%)")
            elif 'factor' in metric:
                print(f"{metric_display:<25} {baseline_val:<15.2f} {current_val:<15.2f} {change:+.2f} ({change_pct:+.1f}%)")
            else:
                print(f"{metric_display:<25} {baseline_val:<15} {current_val:<15} {change:+} ({change_pct:+.1f}%)")

        print()
        return comparison

    def step5_calculate_net_pnl(self, results, mis_file: str = None):
        """Step 5: Calculate net PnL with MIS leverage, fees, and taxation"""
        print("STEP 5: Calculating Net PnL (MIS, Fees, Tax)")
        print("-" * 80)

        if not NET_PNL_AVAILABLE:
            print("WARNING: Net PnL calculation not available (missing calculate_net_pnl module)")
            return None

        # Look for MIS margin file using the find_mis_file function
        if not mis_file:
            mis_file = find_mis_file()
            if mis_file:
                print(f"  Using MIS file: {mis_file}")

        try:
            net_pnl_result = calculate_net_pnl(
                str(self.backtest_dir),
                mis_file=mis_file,
                use_mis=True,
                verbose=False
            )

            results['net_pnl_analysis'] = {
                'gross_pnl_nrml': net_pnl_result['gross_pnl_nrml'],
                'mis_multiplier_avg': net_pnl_result['mis_multiplier_avg'],
                'gross_pnl_with_mis': net_pnl_result['gross_pnl_mis'],
                'total_fees': net_pnl_result['total_fees'],
                'profit_after_fees': net_pnl_result['profit_after_fees'],
                'tax_breakdown': net_pnl_result['tax'],
                'net_pnl_final': net_pnl_result['net_pnl']
            }

            print(f"  Gross PnL (NRML):     Rs {net_pnl_result['gross_pnl_nrml']:>12,.0f}")
            print(f"  Avg MIS Multiplier:      {net_pnl_result['mis_multiplier_avg']:>10.2f}x")
            print(f"  Gross PnL (with MIS): Rs {net_pnl_result['gross_pnl_mis']:>12,.0f}")
            print(f"  Total Fees:           Rs {net_pnl_result['total_fees']:>12,.0f}")
            print(f"  Tax (30% + 4% cess):  Rs {net_pnl_result['tax']['total_tax']:>12,.0f}")
            print(f"  NET PNL FINAL:        Rs {net_pnl_result['net_pnl']:>12,.0f}")
            print()

            return net_pnl_result

        except Exception as e:
            print(f"ERROR calculating net PnL: {e}")
            return None

    def step6_generate_executive_summary(self, results, comparison=None):
        """Step 6: Generate executive summary report"""
        print("STEP 6: Generating Executive Summary")
        print("-" * 80)

        report_file = self.backtest_dir / f"EXECUTIVE_SUMMARY_{self.run_id}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BACKTEST EXECUTIVE SUMMARY - {self.run_id}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sessions: {results['sessions']}\n")
            f.write(f"Date Range: {results['date_range']}\n")
            f.write("\n")

            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Total P&L: Rs.{results['total_pnl']:.2f}\n")
            f.write(f"Avg P&L/Trade: Rs.{results['avg_pnl_per_trade']:.2f}\n")
            f.write(f"Win Rate: {results['win_rate']*100:.1f}% ({results['wins']}/{results['total_trades']})\n")
            f.write(f"Avg Winner: Rs.{results['avg_winner']:.2f}\n")
            f.write(f"Avg Loser: Rs.{results['avg_loser']:.2f}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
            f.write("\n")

            # Net PnL Analysis (if available)
            if 'net_pnl_analysis' in results:
                net = results['net_pnl_analysis']
                f.write("NET PNL ANALYSIS (MIS + Fees + Tax)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Gross P&L (NRML):      Rs.{net['gross_pnl_nrml']:>12,.0f}\n")
                f.write(f"Avg MIS Multiplier:       {net['mis_multiplier_avg']:>10.2f}x\n")
                f.write(f"Gross P&L (with MIS):  Rs.{net['gross_pnl_with_mis']:>12,.0f}\n")
                f.write(f"Total Fees:            Rs.{net['total_fees']:>12,.0f}\n")
                f.write(f"Profit after Fees:     Rs.{net['profit_after_fees']:>12,.0f}\n")
                f.write(f"Tax (30% + 4% cess):   Rs.{net['tax_breakdown']['total_tax']:>12,.0f}\n")
                f.write(f"NET P&L FINAL:         Rs.{net['net_pnl_final']:>12,.0f}\n")
                f.write("\n")

            f.write("EXIT REASON BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: x[1], reverse=True):
                pnl = results['exit_reasons_pnl'][reason]
                pct = (count / results['total_trades']) * 100
                f.write(f"  {reason:<25} {count:>5} ({pct:>5.1f}%) -> Rs.{pnl:>10.2f}\n")
            f.write("\n")

            if 'strategy_breakdown' in results:
                f.write("STRATEGY BREAKDOWN\n")
                f.write("-" * 80 + "\n")
                for strategy, stats in sorted(results['strategy_breakdown'].items(),
                                             key=lambda x: x[1]['pnl'], reverse=True):
                    f.write(f"  {strategy:<30} Trades: {stats['trades']:>4}  "
                          f"Win Rate: {stats['win_rate']*100:>5.1f}%  "
                          f"P&L: Rs.{stats['pnl']:>10.2f}\n")
                f.write("\n")

            if comparison:
                f.write("BASELINE COMPARISON\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Change':<20}\n")
                f.write("-" * 80 + "\n")
                for metric, values in comparison.items():
                    metric_display = metric.replace('_', ' ').title()
                    if 'pnl' in metric:
                        f.write(f"{metric_display:<25} Rs.{values['baseline']:<13.2f} Rs.{values['current']:<13.2f} "
                              f"{values['change']:+.2f} ({values['change_pct']:+.1f}%)\n")
                    elif 'rate' in metric:
                        f.write(f"{metric_display:<25} {values['baseline']*100:<13.1f}% {values['current']*100:<13.1f}% "
                              f"{values['change']*100:+.1f}pp ({values['change_pct']:+.1f}%)\n")
                    else:
                        f.write(f"{metric_display:<25} {values['baseline']:<15.2f} {values['current']:<15.2f} "
                              f"{values['change']:+.2f} ({values['change_pct']:+.1f}%)\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"Executive summary saved to: {report_file}")
        print()

        return report_file

    def generate_full_report(self, mis_file: str = None):
        """Main orchestrator - run all steps"""
        try:
            # Step 1: Postprocess analytics (includes CSV generation via postprocess_extracted_backtest.py)
            if not self.step1_postprocess_analytics():
                print("ERROR: Failed to postprocess analytics")
                return 1

            # Step 2: Aggregate results
            results = self.step3_aggregate_results()
            if not results:
                print("ERROR: Failed to aggregate results")
                return 1

            # Step 3: Compare with baseline (if provided)
            comparison = self.step4_compare_with_baseline(results)

            # Step 4: Calculate net PnL with MIS, fees, and taxation
            self.step5_calculate_net_pnl(results, mis_file)

            # Step 5: Generate executive summary
            report_file = self.step6_generate_executive_summary(results, comparison)

            print("=" * 80)
            print("REPORT GENERATION COMPLETE")
            print("=" * 80)
            print(f"Run ID: {self.run_id}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Total P&L: Rs.{results['total_pnl']:.2f}")
            print(f"Executive Summary: {report_file}")
            print("=" * 80)

            return 0

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    # Find default MIS file
    default_mis = find_mis_file()

    parser = argparse.ArgumentParser(description='Generate comprehensive backtest report')
    parser.add_argument('backtest_path', help='Path to backtest ZIP file or extracted directory')
    parser.add_argument('--baseline', help='Path to baseline backtest ZIP file or directory for comparison', default=None)
    parser.add_argument('--mis-file', help='Path to Zerodha MIS margin Excel file for net PnL calculation',
                       default=default_mis)

    args = parser.parse_args()

    # Extract ZIPs if needed
    print("=" * 80)
    print("STEP 0: ZIP Extraction (if needed)")
    print("=" * 80)

    backtest_dir = extract_zip_if_needed(args.backtest_path)
    baseline_dir = extract_zip_if_needed(args.baseline) if args.baseline else None

    print()

    # Use provided file or auto-detected default
    mis_file = args.mis_file
    if mis_file and not Path(mis_file).exists():
        print(f"Warning: MIS file '{mis_file}' not found. Will search for alternatives.")
        mis_file = find_mis_file()

    generator = BacktestReportGenerator(str(backtest_dir), str(baseline_dir) if baseline_dir else None)
    return generator.generate_full_report(mis_file=mis_file)


if __name__ == "__main__":
    sys.exit(main())
