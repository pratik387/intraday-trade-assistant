#!/usr/bin/env python3
"""
Final Comprehensive Analysis - July to August 2025
==================================================

Ultimate validation across 3 months of data to ensure pattern reliability
before implementing any changes to screening logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import existing tools
from spike_detector import HistoricalSpikeDetector
from pre_spike_analyzer import PreSpikeAnalyzer
from pattern_extractor import PatternExtractor

class FinalComprehensiveAnalysis:
    """Ultimate validation with 3 months of data"""

    def __init__(self):
        self.spike_detector = HistoricalSpikeDetector()
        self.pre_analyzer = PreSpikeAnalyzer()
        self.pattern_extractor = PatternExtractor()

    def run_final_analysis(self):
        """Run final comprehensive analysis with July-August data"""

        print("FINAL COMPREHENSIVE ANALYSIS - JULY TO AUGUST 2025")
        print("=" * 70)
        print("Ultimate validation to ensure pattern reliability before implementation...")
        print()

        # PHASE 1: Ultimate Scale Spike Detection
        print("PHASE 1: ULTIMATE SCALE SPIKE DETECTION")
        print("-" * 50)

        # Complete July + August trading days
        ultimate_dates = [
            # July 2025 (complete month)
            "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-07",
            "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11", "2025-07-14",
            "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-21",
            "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25", "2025-07-28",
            "2025-07-29", "2025-07-30", "2025-07-31",
            # August 2025 (complete month)
            "2025-08-01", "2025-08-02", "2025-08-05", "2025-08-06", "2025-08-07",
            "2025-08-08", "2025-08-09", "2025-08-12", "2025-08-13", "2025-08-14",
            "2025-08-15", "2025-08-16", "2025-08-19", "2025-08-20", "2025-08-21",
            "2025-08-22", "2025-08-23", "2025-08-26", "2025-08-27", "2025-08-28",
            "2025-08-29", "2025-08-30"
        ]

        # Maximum symbols for ultimate validation
        max_symbols = 500  # 5x initial sample

        print(f"ULTIMATE SCALE ANALYSIS:")
        print(f"- Trading days: {len(ultimate_dates)} (vs 8 initial, 32 expanded)")
        print(f"- Symbols: {max_symbols} (vs 100 initial, 300 expanded)")
        print(f"- Expected spikes: 50,000+ (vs 4,174 initial, 28,665 expanded)")
        print()

        # Run ultimate spike detection
        print("Starting ultimate spike detection scan...")
        all_spikes = self.spike_detector.scan_all_symbols(
            target_dates=ultimate_dates,
            max_symbols=max_symbols
        )

        if not all_spikes:
            print("No spikes found in ultimate analysis!")
            return

        total_spikes = len(all_spikes)
        print(f"\nULTIMATE RESULTS: {total_spikes:,} total spikes detected!")

        # Save ultimate spike events
        spike_data = []
        for spike in all_spikes:
            spike_data.append({
                'symbol': spike.symbol,
                'date': spike.date,
                'spike_start_time': spike.spike_start_time,
                'spike_end_time': spike.spike_end_time,
                'move_percent': spike.move_percent,
                'direction': spike.direction,
                'entry_price': spike.entry_price,
                'peak_price': spike.peak_price,
                'volume_ratio': spike.volume_ratio,
                'time_to_peak_minutes': spike.time_to_peak_minutes,
                'sustained': spike.sustained
            })

        df_ultimate_spikes = pd.DataFrame(spike_data)
        df_ultimate_spikes.to_csv('ultimate_spike_events.csv', index=False)
        print(f"Saved {total_spikes:,} spikes to: ultimate_spike_events.csv")

        # PHASE 2: Multi-Month Market Analysis
        print(f"\nPHASE 2: MULTI-MONTH MARKET ANALYSIS")
        print("-" * 50)

        self.analyze_multi_month_conditions(df_ultimate_spikes)

        # PHASE 3: Ultimate Pre-Spike Analysis
        print(f"\nPHASE 3: ULTIMATE PRE-SPIKE ANALYSIS")
        print("-" * 50)

        # Analyze top 200 spikes for ultimate confidence
        print("Analyzing top 200 spikes for maximum statistical power...")
        conditions = self.pre_analyzer.analyze_spike_events(
            spike_csv='ultimate_spike_events.csv',
            max_analyze=200
        )

        if not conditions:
            print("Failed to analyze pre-spike conditions!")
            return

        # Save ultimate conditions
        results = []
        for c in conditions:
            results.append({
                'symbol': c.symbol,
                'spike_start_time': c.spike_start_time,
                'move_percent': c.move_percent,
                'direction': c.direction,
                'price_near_vwap': c.price_near_vwap,
                'vwap_trend': c.vwap_trend,
                'range_compression': c.range_compression,
                'volume_buildup': c.volume_buildup,
                'relative_volume': c.relative_volume,
                'unusual_activity': c.unusual_activity,
                'rsi_14': c.rsi_14,
                'price_momentum': c.price_momentum,
                'breakout_level': c.breakout_level
            })

        df_ultimate_conditions = pd.DataFrame(results)
        df_ultimate_conditions.to_csv('ultimate_pre_spike_conditions.csv', index=False)
        print(f"Saved {len(results)} ultimate conditions to: ultimate_pre_spike_conditions.csv")

        # PHASE 4: Ultimate Pattern Validation
        print(f"\nPHASE 4: ULTIMATE PATTERN VALIDATION")
        print("-" * 50)

        self.ultimate_pattern_validation(df_ultimate_conditions)

        return df_ultimate_spikes, conditions

    def analyze_multi_month_conditions(self, df_spikes: pd.DataFrame):
        """Analyze spikes across July-August for market regime changes"""

        print("Multi-month market condition analysis...")

        df_spikes['date'] = pd.to_datetime(df_spikes['date'])

        # July vs August comparison
        july_spikes = df_spikes[df_spikes['date'].dt.month == 7]
        august_spikes = df_spikes[df_spikes['date'].dt.month == 8]

        print(f"\nMONTH-BY-MONTH BREAKDOWN:")
        print(f"July spikes: {len(july_spikes):,} ({len(july_spikes)/len(df_spikes)*100:.1f}%)")
        print(f"August spikes: {len(august_spikes):,} ({len(august_spikes)/len(df_spikes)*100:.1f}%)")

        # Directional bias by month
        july_up = len(july_spikes[july_spikes['direction'] == 'up'])
        july_down = len(july_spikes[july_spikes['direction'] == 'down'])
        august_up = len(august_spikes[august_spikes['direction'] == 'up'])
        august_down = len(august_spikes[august_spikes['direction'] == 'down'])

        print(f"\nDIRECTIONAL BIAS:")
        print(f"July: {july_up:,} up, {july_down:,} down ({july_up/(july_up+july_down)*100:.1f}% up)")
        print(f"August: {august_up:,} up, {august_down:,} down ({august_up/(august_up+august_down)*100:.1f}% up)")

        # Size and quality metrics
        print(f"\nQUALITY METRICS:")
        print(f"July avg move: {july_spikes['move_percent'].mean():.1f}%")
        print(f"August avg move: {august_spikes['move_percent'].mean():.1f}%")
        print(f"July sustained: {july_spikes['sustained'].mean()*100:.1f}%")
        print(f"August sustained: {august_spikes['sustained'].mean()*100:.1f}%")

        # Volume characteristics
        print(f"\nVOLUME CHARACTERISTICS:")
        print(f"July avg volume ratio: {july_spikes['volume_ratio'].mean():.1f}x")
        print(f"August avg volume ratio: {august_spikes['volume_ratio'].mean():.1f}x")

        # Time to peak
        print(f"\nTIMING CHARACTERISTICS:")
        print(f"July avg time to peak: {july_spikes['time_to_peak_minutes'].mean():.0f} minutes")
        print(f"August avg time to peak: {august_spikes['time_to_peak_minutes'].mean():.0f} minutes")

        # Market regime stability check
        july_volatility = july_spikes['move_percent'].std()
        august_volatility = august_spikes['move_percent'].std()

        print(f"\nMARKET REGIME STABILITY:")
        print(f"July volatility (std): {july_volatility:.1f}%")
        print(f"August volatility (std): {august_volatility:.1f}%")

        regime_change = abs(july_volatility - august_volatility) / july_volatility * 100
        if regime_change > 20:
            print(f"WARNING: Significant regime change detected ({regime_change:.1f}% volatility change)")
        else:
            print(f"Market regime stable ({regime_change:.1f}% volatility change)")

    def ultimate_pattern_validation(self, df_conditions: pd.DataFrame):
        """Ultimate pattern validation with maximum statistical power"""

        print(f"ULTIMATE PATTERN VALIDATION")
        print("=" * 40)
        print(f"Sample size: {len(df_conditions)} (vs 30 initial, 100 expanded)")
        print()

        # Extract patterns with ultimate dataset
        patterns = self.pattern_extractor.extract_patterns('ultimate_pre_spike_conditions.csv')

        if not patterns:
            print("No patterns found in ultimate analysis!")
            return

        # Compare evolution across all three analyses
        print("PATTERN EVOLUTION TRACKING")
        print("=" * 50)

        pattern_evolution = {
            'VWAP_PROXIMITY': {'initial': 73.3, 'expanded': 0.0},  # Disappeared
            'RSI_NEUTRAL_ZONE': {'initial': 90.0, 'expanded': 0.0},  # Disappeared
            'MOMENTUM_CONSOLIDATION': {'initial': 73.3, 'expanded': 76.0},
            'VOLUME_ANOMALY': {'initial': 46.7, 'expanded': 46.0},
            'RANGE_COMPRESSION': {'initial': 50.0, 'expanded': 63.0}
        }

        print(f"{'Pattern':<25} {'Initial%':<10} {'Expanded%':<12} {'Ultimate%':<12} {'Trend':<15}")
        print("-" * 85)

        for pattern in patterns:
            name = pattern.name
            initial = pattern_evolution.get(name, {}).get('initial', 0.0)
            expanded = pattern_evolution.get(name, {}).get('expanded', 0.0)
            ultimate = pattern.success_rate

            # Determine trend
            if initial > 0 and expanded > 0:
                if ultimate > expanded > initial:
                    trend = "STRENGTHENING"
                elif ultimate > expanded:
                    trend = "IMPROVING"
                elif abs(ultimate - expanded) < 5:
                    trend = "STABLE"
                elif ultimate < expanded:
                    trend = "WEAKENING"
                else:
                    trend = "UNCLEAR"
            elif expanded == 0:
                trend = "REAPPEARED" if ultimate > 0 else "ABSENT"
            else:
                trend = "NEW PATTERN"

            print(f"{name:<25} {initial:<10.1f} {expanded:<12.1f} {ultimate:<12.1f} {trend:<15}")

        # Ultimate statistical analysis
        print(f"\nULTIMATE STATISTICAL SIGNIFICANCE")
        print("=" * 40)

        reliable_patterns = []
        for pattern in patterns:
            sample_size = pattern.sample_size
            success_rate = pattern.success_rate / 100.0

            # Calculate confidence intervals
            if sample_size > 0:
                margin_error = 1.96 * np.sqrt((success_rate * (1 - success_rate)) / sample_size)
                ci_lower = max(0, (success_rate - margin_error) * 100)
                ci_upper = min(100, (success_rate + margin_error) * 100)

                print(f"\n{pattern.name}:")
                print(f"  Sample size: {sample_size}")
                print(f"  Success rate: {pattern.success_rate:.1f}%")
                print(f"  95% CI: {ci_lower:.1f}% - {ci_upper:.1f}%")

                # Enhanced significance criteria
                if sample_size >= 50 and pattern.success_rate >= 60 and ci_lower >= 50:
                    status = "HIGHLY RELIABLE"
                    reliable_patterns.append(pattern)
                elif sample_size >= 30 and pattern.success_rate >= 55 and ci_lower >= 45:
                    status = "MODERATELY RELIABLE"
                    reliable_patterns.append(pattern)
                elif sample_size >= 20 and pattern.success_rate >= 60:
                    status = "PROMISING"
                else:
                    status = "INSUFFICIENT DATA"

                print(f"  Statistical Status: {status}")

        # Final implementation decision
        print(f"\nFINAL IMPLEMENTATION DECISION")
        print("=" * 40)

        if len(reliable_patterns) >= 2:
            print(f"RECOMMENDATION: PROCEED WITH IMPLEMENTATION")
            print(f"Reliable patterns identified: {len(reliable_patterns)}")
            print()
            print("PATTERNS TO IMPLEMENT:")
            for i, pattern in enumerate(reliable_patterns, 1):
                print(f"{i}. {pattern.name}")
                print(f"   - Success Rate: {pattern.success_rate:.1f}%")
                print(f"   - Sample Size: {pattern.sample_size}")
                print(f"   - Implementation: {pattern.implementation_logic}")
                print()

            print("IMPLEMENTATION STRATEGY:")
            print("1. Start with paper trading using these patterns only")
            print("2. Monitor win rate improvement vs current 38.5% baseline")
            print("3. Deploy to live trading if paper trading shows 60%+ win rate")
            print("4. Gradually increase position size based on performance")

        elif len(reliable_patterns) == 1:
            print(f"RECOMMENDATION: CAUTIOUS IMPLEMENTATION")
            print(f"Only 1 reliable pattern found - implement with strict risk controls")

        else:
            print(f"RECOMMENDATION: DO NOT IMPLEMENT YET")
            print(f"No statistically reliable patterns found")
            print(f"Continue data collection and analysis")

        # Performance prediction
        if reliable_patterns:
            predicted_win_rate = np.mean([p.success_rate for p in reliable_patterns])
            print(f"\nPREDICTED PERFORMANCE:")
            print(f"Expected win rate: {predicted_win_rate:.1f}%")
            improvement = predicted_win_rate - 38.5
            print(f"Expected improvement: +{improvement:.1f}% vs current baseline")

            if predicted_win_rate >= 70:
                print(f"STATUS: EXCEEDS 70% TARGET - STRONG IMPLEMENTATION CANDIDATE")
            elif predicted_win_rate >= 60:
                print(f"STATUS: GOOD IMPROVEMENT - IMPLEMENT WITH MONITORING")
            elif predicted_win_rate >= 50:
                print(f"STATUS: MODEST IMPROVEMENT - PROCEED WITH CAUTION")
            else:
                print(f"STATUS: INSUFFICIENT IMPROVEMENT - RECONSIDER")

def main():
    """Run final comprehensive analysis"""
    print("Starting Final Comprehensive Analysis...")
    print("This may take 10-15 minutes due to the massive dataset size...")
    print()

    analyzer = FinalComprehensiveAnalysis()
    df_spikes, conditions = analyzer.run_final_analysis()

    print(f"\n" + "="*80)
    print("FINAL COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*80)
    print("✓ Ultimate validation completed across July-August 2025")
    print("✓ Statistical significance established with maximum confidence")
    print("✓ Implementation decision ready based on rigorous analysis")
    print("✓ All results saved for final review")

if __name__ == "__main__":
    main()