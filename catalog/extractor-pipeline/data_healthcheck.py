#!/usr/bin/env python3
"""
Simple health check for aggregated data file.
Reports coverage and gaps against intended extraction scale.
"""

import os
import sys
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv


def find_aggregated_file(directory="data/aggregated"):
    """Find the most recent aggregated CSV file."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [f for f in os.listdir(directory) if f.endswith("_aggregated.csv")]

    if not files:
        raise FileNotFoundError(f"No aggregated CSV files found in {directory}")

    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def load_aggregated_data(file_path):
    """Load and parse aggregated data."""
    df = pd.read_csv(file_path)
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    df['interval_end'] = pd.to_datetime(df['interval_end'])
    df = df.sort_values('interval_start')
    return df


def find_gaps(actual_intervals, expected_intervals):
    """Find gaps in the time series."""
    missing = expected_intervals.difference(actual_intervals)

    if len(missing) == 0:
        return []

    gaps = []
    gap_start = missing[0]
    gap_end = missing[0]

    for i in range(1, len(missing)):
        if missing[i] == gap_end + timedelta(hours=1):
            gap_end = missing[i]
        else:
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'hours': int((gap_end - gap_start).total_seconds() / 3600) + 1
            })
            gap_start = missing[i]
            gap_end = missing[i]

    gaps.append({
        'start': gap_start,
        'end': gap_end,
        'hours': int((gap_end - gap_start).total_seconds() / 3600) + 1
    })

    return gaps


def main():
    load_dotenv('.env')

    interval_start_str = os.getenv("INTERVAL_START")
    interval_end_str = os.getenv("INTERVAL_END")

    if not interval_start_str or not interval_end_str:
        print("ERROR: INTERVAL_START and INTERVAL_END must be set in .env")
        sys.exit(1)

    expected_start = pd.to_datetime(interval_start_str, format='%Y-%m-%d-%H:%M')
    expected_end = pd.to_datetime(interval_end_str, format='%Y-%m-%d-%H:%M')

    print("=" * 70)
    print("AGGREGATED DATA HEALTH CHECK")
    print("=" * 70)

    aggregated_file = find_aggregated_file()
    print(f"\nFile: {aggregated_file}")

    df = load_aggregated_data(aggregated_file)

    actual_start = df['interval_start'].min()
    actual_end = df['interval_start'].max()

    print(f"\nIntended Scale:")
    print(f"  {expected_start} to {expected_end}")
    print(f"  Duration: {(expected_end - expected_start).days} days")

    print(f"\nActual Data:")
    print(f"  {actual_start} to {actual_end}")
    print(f"  Duration: {(actual_end - actual_start).days} days")
    print(f"  Total rows: {len(df):,}")

    expected_intervals = pd.date_range(start=expected_start, end=expected_end, freq='1h')
    actual_intervals = pd.DatetimeIndex(df['interval_start'])

    actual_in_range = actual_intervals[
        (actual_intervals >= expected_start) & (actual_intervals <= expected_end)
    ]

    coverage_pct = (len(actual_in_range) / len(expected_intervals)) * 100

    print(f"\nCoverage (within intended scale):")
    print(f"  Expected: {len(expected_intervals):,} intervals")
    print(f"  Present:  {len(actual_in_range):,} intervals")
    print(f"  Missing:  {len(expected_intervals) - len(actual_in_range):,} intervals")
    print(f"  Coverage: {coverage_pct:.2f}%")

    gaps = find_gaps(actual_in_range, expected_intervals)

    if gaps:
        print(f"\nGaps Found: {len(gaps)}")
        print("\n" + "-" * 70)
        print(f"{'#':<4} {'Start':<20} {'End':<20} {'Hours':<10} {'Days':<8}")
        print("-" * 70)

        for i, gap in enumerate(gaps, 1):
            days = gap['hours'] / 24
            print(f"{i:<4} {str(gap['start']):<20} {str(gap['end']):<20} "
                  f"{gap['hours']:<10} {days:<8.2f}")

        total_gap_hours = sum(g['hours'] for g in gaps)
        print("-" * 70)
        print(f"Total missing: {total_gap_hours:,} hours ({total_gap_hours/24:.1f} days)")
    else:
        print("\nNo gaps - perfect coverage!")

    print("\n" + "=" * 70)
    status = "EXCELLENT" if coverage_pct > 95 else "GOOD" if coverage_pct > 80 else "FAIR" if coverage_pct > 60 else "POOR"
    print(f"Coverage: {coverage_pct:.2f}% - {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
