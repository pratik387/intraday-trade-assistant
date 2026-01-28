"""
Combine 6 half-yearly backtest reports into a comprehensive 3-year analysis.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPORTS = [
    ("analysis/reports/misc/analysis_report_20_20251221_122639.json", "2023-H1", "2023-01-02 to 2023-06-30"),
    ("analysis/reports/misc/analysis_report_20_20251221_122751.json", "2023-H2", "2023-07-03 to 2023-12-29"),
    ("analysis/reports/misc/analysis_report_20_20251221_122908.json", "2024-H1", "2024-01-01 to 2024-06-28"),
    ("analysis/reports/misc/analysis_report_20_20251221_123040.json", "2024-H2", "2024-07-01 to 2024-12-31"),
    ("analysis/reports/misc/analysis_report_20_20251221_123156.json", "2025-H1", "2025-01-02 to 2025-06-30"),
    ("analysis/reports/misc/analysis_report_20_20251221_123324.json", "2025-H2", "2025-07-01 to 2025-12-12"),
]

def load_reports():
    """Load all 6 reports."""
    reports = []
    for path, period, date_range in REPORTS:
        with open(path) as f:
            data = json.load(f)
        data["period"] = period
        data["date_range"] = date_range
        reports.append(data)
    return reports

def combine_performance_summary(reports):
    """Combine performance summaries."""
    total_trades = sum(r["performance_summary"]["total_trades"] for r in reports)
    total_pnl = sum(r["performance_summary"]["total_pnl"] for r in reports)
    total_sessions = sum(r["sessions_analyzed"] for r in reports)

    # Collect all winning/losing trades for win rate
    winners = sum(r["performance_summary"]["total_trades"] * r["performance_summary"]["win_rate"] / 100 for r in reports)

    return {
        "total_sessions": total_sessions,
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
        "avg_pnl_per_session": round(total_pnl / total_sessions, 2) if total_sessions > 0 else 0,
        "win_rate": round(winners / total_trades * 100, 2) if total_trades > 0 else 0,
        "best_trade": max(r["performance_summary"]["best_trade"] for r in reports),
        "worst_trade": min(r["performance_summary"]["worst_trade"] for r in reports),
        "by_period": [
            {
                "period": r["period"],
                "date_range": r["date_range"],
                "sessions": r["sessions_analyzed"],
                "trades": r["performance_summary"]["total_trades"],
                "pnl": round(r["performance_summary"]["total_pnl"], 2),
                "win_rate": round(r["performance_summary"]["win_rate"], 2),
                "avg_pnl": round(r["performance_summary"]["avg_pnl_per_trade"], 2),
            }
            for r in reports
        ]
    }

def combine_setup_analysis(reports):
    """Combine setup analysis across all periods."""
    combined = defaultdict(lambda: {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0,
        "by_period": {}
    })

    for r in reports:
        period = r["period"]
        for setup, stats in r.get("setup_analysis", {}).items():
            combined[setup]["total_trades"] += stats.get("total_trades", 0)
            combined[setup]["winning_trades"] += stats.get("winning_trades", 0)
            combined[setup]["total_pnl"] += stats.get("total_pnl", 0)
            combined[setup]["by_period"][period] = {
                "trades": stats.get("total_trades", 0),
                "pnl": round(stats.get("total_pnl", 0), 2),
                "win_rate": round(stats.get("win_rate", 0), 2)
            }

    # Calculate final metrics
    result = {}
    for setup, stats in combined.items():
        if stats["total_trades"] == 0:
            continue
        result[setup] = {
            "total_trades": stats["total_trades"],
            "win_rate": round(stats["winning_trades"] / stats["total_trades"] * 100, 2),
            "total_pnl": round(stats["total_pnl"], 2),
            "avg_pnl": round(stats["total_pnl"] / stats["total_trades"], 2),
            "by_period": stats["by_period"]
        }

    # Sort by total PnL
    return dict(sorted(result.items(), key=lambda x: x[1]["total_pnl"], reverse=True))

def combine_regime_analysis(reports):
    """Combine regime analysis across all periods."""
    combined = defaultdict(lambda: {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0,
        "by_period": {}
    })

    for r in reports:
        period = r["period"]
        for regime, stats in r.get("regime_analysis", {}).items():
            combined[regime]["total_trades"] += stats.get("total_trades", 0)
            combined[regime]["winning_trades"] += stats.get("winning_trades", 0)
            combined[regime]["total_pnl"] += stats.get("total_pnl", 0)
            combined[regime]["by_period"][period] = {
                "trades": stats.get("total_trades", 0),
                "pnl": round(stats.get("total_pnl", 0), 2),
                "win_rate": round(stats.get("win_rate", 0), 2)
            }

    result = {}
    for regime, stats in combined.items():
        if stats["total_trades"] == 0:
            continue
        result[regime] = {
            "total_trades": stats["total_trades"],
            "win_rate": round(stats["winning_trades"] / stats["total_trades"] * 100, 2),
            "total_pnl": round(stats["total_pnl"], 2),
            "avg_pnl": round(stats["total_pnl"] / stats["total_trades"], 2),
            "by_period": stats["by_period"]
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["total_pnl"], reverse=True))

def extract_net_pnl_analysis(reports):
    """Extract net PnL analysis from each report."""
    net_pnl_data = []
    for r in reports:
        if "net_pnl_analysis" in r:
            data = r["net_pnl_analysis"]
            net_pnl_data.append({
                "period": r["period"],
                "gross_pnl_nrml": data.get("gross_pnl_nrml", 0),
                "gross_pnl_with_mis": data.get("gross_pnl_with_mis", 0),
                "total_fees": data.get("total_fees", 0),
                "net_pnl_final": data.get("net_pnl_final", 0),
            })

    if not net_pnl_data:
        return None

    return {
        "by_period": net_pnl_data,
        "combined": {
            "gross_pnl_nrml": round(sum(d["gross_pnl_nrml"] for d in net_pnl_data), 2),
            "gross_pnl_with_mis": round(sum(d["gross_pnl_with_mis"] for d in net_pnl_data), 2),
            "total_fees": round(sum(d["total_fees"] for d in net_pnl_data), 2),
            "net_pnl_final": round(sum(d["net_pnl_final"] for d in net_pnl_data), 2),
        }
    }

def calculate_yearly_breakdown(reports):
    """Calculate yearly breakdown."""
    years = defaultdict(lambda: {"sessions": 0, "trades": 0, "pnl": 0, "periods": []})

    for r in reports:
        year = r["period"].split("-")[0]
        years[year]["sessions"] += r["sessions_analyzed"]
        years[year]["trades"] += r["performance_summary"]["total_trades"]
        years[year]["pnl"] += r["performance_summary"]["total_pnl"]
        years[year]["periods"].append(r["period"])

    result = {}
    for year, data in sorted(years.items()):
        result[year] = {
            "sessions": data["sessions"],
            "trades": data["trades"],
            "pnl": round(data["pnl"], 2),
            "avg_pnl_per_trade": round(data["pnl"] / data["trades"], 2) if data["trades"] > 0 else 0,
            "avg_pnl_per_session": round(data["pnl"] / data["sessions"], 2) if data["sessions"] > 0 else 0,
            "periods": data["periods"]
        }

    return result

def main():
    print("=" * 70)
    print("3-YEAR BACKTEST COMBINED ANALYSIS")
    print("=" * 70)

    reports = load_reports()

    # Performance Summary
    perf = combine_performance_summary(reports)
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Period: Jan 2023 - Dec 2025 (3 years)")
    print(f"Total Sessions: {perf['total_sessions']}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Total PnL: Rs {perf['total_pnl']:,.2f}")
    print(f"Avg PnL/Trade: Rs {perf['avg_pnl_per_trade']:,.2f}")
    print(f"Avg PnL/Session: Rs {perf['avg_pnl_per_session']:,.2f}")
    print(f"Win Rate: {perf['win_rate']:.1f}%")
    print(f"Best Trade: Rs {perf['best_trade']:,.2f}")
    print(f"Worst Trade: Rs {perf['worst_trade']:,.2f}")

    # Period breakdown
    print(f"\n{'='*70}")
    print("HALF-YEARLY BREAKDOWN")
    print("="*70)
    print(f"{'Period':<10} {'Date Range':<30} {'Sessions':>8} {'Trades':>8} {'PnL':>12} {'Win%':>8} {'Avg':>10}")
    print("-" * 90)
    for p in perf["by_period"]:
        print(f"{p['period']:<10} {p['date_range']:<30} {p['sessions']:>8} {p['trades']:>8} {p['pnl']:>12,.2f} {p['win_rate']:>7.1f}% {p['avg_pnl']:>10,.2f}")

    # Yearly breakdown
    yearly = calculate_yearly_breakdown(reports)
    print(f"\n{'='*70}")
    print("YEARLY BREAKDOWN")
    print("="*70)
    print(f"{'Year':<6} {'Sessions':>10} {'Trades':>10} {'PnL':>15} {'Avg/Trade':>12} {'Avg/Session':>12}")
    print("-" * 70)
    for year, data in yearly.items():
        print(f"{year:<6} {data['sessions']:>10} {data['trades']:>10} {data['pnl']:>15,.2f} {data['avg_pnl_per_trade']:>12,.2f} {data['avg_pnl_per_session']:>12,.2f}")

    # Setup Analysis
    setups = combine_setup_analysis(reports)
    print(f"\n{'='*70}")
    print("SETUP PERFORMANCE (sorted by total PnL)")
    print("="*70)
    print(f"{'Setup':<35} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
    print("-" * 80)
    for setup, stats in setups.items():
        print(f"{setup:<35} {stats['total_trades']:>8} {stats['win_rate']:>7.1f}% {stats['total_pnl']:>12,.2f} {stats['avg_pnl']:>10,.2f}")

    # Top 5 setups by PnL
    print(f"\n{'='*70}")
    print("TOP 5 SETUPS (by Total PnL)")
    print("="*70)
    for i, (setup, stats) in enumerate(list(setups.items())[:5], 1):
        print(f"\n{i}. {setup}")
        print(f"   Total: {stats['total_trades']} trades | Win Rate: {stats['win_rate']:.1f}% | PnL: Rs {stats['total_pnl']:,.2f}")
        print(f"   By Period: ", end="")
        parts = []
        for period, pdata in stats["by_period"].items():
            if pdata["trades"] > 0:
                parts.append(f"{period}: {pdata['trades']}t/Rs{pdata['pnl']:,.0f}")
        print(" | ".join(parts))

    # Worst performing setups
    worst_setups = dict(sorted(setups.items(), key=lambda x: x[1]["total_pnl"])[:3])
    print(f"\n{'='*70}")
    print("BOTTOM 3 SETUPS (lowest PnL)")
    print("="*70)
    for setup, stats in worst_setups.items():
        print(f"  {setup}: {stats['total_trades']} trades | Win: {stats['win_rate']:.1f}% | PnL: Rs {stats['total_pnl']:,.2f}")

    # Regime Analysis
    regimes = combine_regime_analysis(reports)
    print(f"\n{'='*70}")
    print("REGIME PERFORMANCE")
    print("="*70)
    print(f"{'Regime':<15} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Avg PnL':>10}")
    print("-" * 60)
    for regime, stats in regimes.items():
        print(f"{regime:<15} {stats['total_trades']:>8} {stats['win_rate']:>7.1f}% {stats['total_pnl']:>12,.2f} {stats['avg_pnl']:>10,.2f}")

    # Net PnL Analysis
    net_pnl = extract_net_pnl_analysis(reports)
    if net_pnl:
        print(f"\n{'='*70}")
        print("NET PnL ANALYSIS (after fees & taxes)")
        print("="*70)
        c = net_pnl["combined"]
        print(f"Gross PnL (NRML): Rs {c['gross_pnl_nrml']:,.2f}")
        print(f"Gross PnL (with MIS): Rs {c['gross_pnl_with_mis']:,.2f}")
        print(f"Total Fees & Taxes: Rs {c['total_fees']:,.2f}")
        print(f"Net PnL Final: Rs {c['net_pnl_final']:,.2f}")

    # Monthly average (approximate)
    months = perf["total_sessions"] / 21  # ~21 trading days per month
    print(f"\n{'='*70}")
    print("ANNUALIZED METRICS")
    print("="*70)
    print(f"Total Months: ~{months:.0f}")
    print(f"Avg Monthly PnL: Rs {perf['total_pnl'] / months:,.2f}")
    print(f"Avg Trades/Month: {perf['total_trades'] / months:.1f}")
    print(f"Avg Trades/Day: {perf['total_trades'] / perf['total_sessions']:.2f}")

    # Save combined report
    combined_report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "date_range": "2023-01-02 to 2025-12-12",
        "total_years": 3,
        "performance_summary": perf,
        "yearly_breakdown": yearly,
        "setup_analysis": setups,
        "regime_analysis": regimes,
        "net_pnl_analysis": net_pnl,
    }

    output_path = Path("analysis/reports/misc/combined_3year_analysis.json")
    with open(output_path, "w") as f:
        json.dump(combined_report, f, indent=2)
    print(f"\n[OK] Full report saved to: {output_path}")

if __name__ == "__main__":
    main()
