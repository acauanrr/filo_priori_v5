"""
Cálculo de APFD por Build.

Este módulo implementa o cálculo correto de APFD seguindo as regras de negócio:
1. APFD é calculado POR BUILD, não globalmente
2. Apenas builds com pelo menos 1 failure são consideradas
3. Builds com apenas 1 TC têm APFD = 1.0 (regra de negócio)
4. Gera relatório no formato: method_name, build_id, test_scenario, count_tc, count_commits, apfd, time

Autor: Filo-Priori V5
Data: 2025-10-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time


def calculate_apfd_single_build(ranks: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate APFD for a single build.

    Args:
        ranks: Array of ranks (1-indexed, lower rank = higher priority)
        labels: Binary labels (1=failure, 0=pass)

    Returns:
        APFD score (0 to 1, higher is better)

    Formula:
        APFD = 1 - (sum of failure ranks) / (n_failures * n_tests) + 1 / (2 * n_tests)
    """
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)

    n_tests = len(labels_arr)
    fail_indices = np.where(labels_arr == 1.0)[0]
    n_failures = len(fail_indices)

    # Business rule: if no failures, APFD is undefined (skip this build)
    if n_failures == 0:
        return None

    # Business rule: if only 1 test case, APFD = 1.0
    if n_tests == 1:
        return 1.0

    # Get ranks of failures
    failure_ranks = ranks_arr[fail_indices]

    # Calculate APFD
    apfd = 1.0 - failure_ranks.sum() / (n_failures * n_tests) + 1.0 / (2.0 * n_tests)

    return float(np.clip(apfd, 0.0, 1.0))


def calculate_apfd_per_build(
    df: pd.DataFrame,
    method_name: str = "filo_priori_v5",
    test_scenario: str = "full"
) -> pd.DataFrame:
    """
    Calculate APFD per build for the entire test set.

    Args:
        df: DataFrame with columns:
            - Build_ID: Build identifier
            - TC_Key: Test case identifier (optional, for counting)
            - label_binary: True label (1=failure, 0=pass)
            - rank: Priority rank (1-indexed, lower is better)
            - (optional) commit: commit info
        method_name: Name of the prioritization method
        test_scenario: Type of test scenario

    Returns:
        DataFrame with columns:
            - method_name
            - build_id
            - test_scenario
            - count_tc: Number of test cases in build
            - count_commits: Number of unique commits (or 0 if not available)
            - apfd: APFD score for the build
            - time: Processing time (placeholder, set to 0)
    """
    results = []

    # Group by Build_ID
    grouped = df.groupby('Build_ID')

    for build_id, build_df in grouped:
        # Count test cases
        count_tc = len(build_df)

        # Count failures
        n_failures = (build_df['label_binary'] == 1.0).sum()

        # Business rule: Skip builds with no failures
        if n_failures == 0:
            continue

        # Count unique commits (if available)
        if 'commit' in build_df.columns:
            # Assuming commit is a list-like field, count unique commits
            count_commits = 0  # Placeholder - would need to parse commit field
        else:
            count_commits = 0

        # Get ranks and labels for this build
        ranks = build_df['rank'].values
        labels = build_df['label_binary'].values

        # Calculate APFD for this build
        apfd = calculate_apfd_single_build(ranks, labels)

        # Skip if APFD is None (shouldn't happen due to earlier check, but safe)
        if apfd is None:
            continue

        # Add to results
        results.append({
            'method_name': method_name,
            'build_id': build_id,
            'test_scenario': test_scenario,
            'count_tc': count_tc,
            'count_commits': count_commits,
            'apfd': apfd,
            'time': 0.0  # Placeholder - could be filled with actual time if tracked
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by build_id for consistency
    results_df = results_df.sort_values('build_id').reset_index(drop=True)

    return results_df


def generate_apfd_report(
    df: pd.DataFrame,
    method_name: str = "filo_priori_v5",
    test_scenario: str = "full",
    output_path: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate complete APFD report with summary statistics.

    Args:
        df: DataFrame with test results
        method_name: Name of the prioritization method
        test_scenario: Type of test scenario
        output_path: Optional path to save CSV report

    Returns:
        Tuple of (results_df, summary_stats)
        - results_df: Per-build APFD results
        - summary_stats: Dictionary with summary statistics
    """
    # Calculate APFD per build
    results_df = calculate_apfd_per_build(df, method_name, test_scenario)

    # Calculate summary statistics
    summary_stats = {
        'total_builds': len(results_df),
        'mean_apfd': results_df['apfd'].mean(),
        'median_apfd': results_df['apfd'].median(),
        'std_apfd': results_df['apfd'].std(),
        'min_apfd': results_df['apfd'].min(),
        'max_apfd': results_df['apfd'].max(),
        'total_test_cases': results_df['count_tc'].sum(),
        'mean_tc_per_build': results_df['count_tc'].mean(),
        'builds_apfd_1.0': (results_df['apfd'] == 1.0).sum(),
        'builds_apfd_gte_0.7': (results_df['apfd'] >= 0.7).sum(),
        'builds_apfd_lt_0.5': (results_df['apfd'] < 0.5).sum()
    }

    # Save to CSV if path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"APFD report saved to: {output_path}")

    return results_df, summary_stats


def print_apfd_summary(summary_stats: Dict):
    """Print formatted APFD summary statistics."""
    print("\n" + "="*70)
    print("APFD PER BUILD - SUMMARY STATISTICS")
    print("="*70)
    print(f"Total builds analyzed: {summary_stats['total_builds']}")
    print(f"Total test cases: {summary_stats['total_test_cases']}")
    print(f"Mean TCs per build: {summary_stats['mean_tc_per_build']:.1f}")
    print(f"\nAPFD Statistics:")
    print(f"  Mean:   {summary_stats['mean_apfd']:.4f}")
    print(f"  Median: {summary_stats['median_apfd']:.4f}")
    print(f"  Std:    {summary_stats['std_apfd']:.4f}")
    print(f"  Min:    {summary_stats['min_apfd']:.4f}")
    print(f"  Max:    {summary_stats['max_apfd']:.4f}")
    print(f"\nAPFD Distribution:")
    print(f"  Builds with APFD = 1.0:  {summary_stats['builds_apfd_1.0']} ({summary_stats['builds_apfd_1.0']/summary_stats['total_builds']*100:.1f}%)")
    print(f"  Builds with APFD ≥ 0.7:  {summary_stats['builds_apfd_gte_0.7']} ({summary_stats['builds_apfd_gte_0.7']/summary_stats['total_builds']*100:.1f}%)")
    print(f"  Builds with APFD < 0.5:  {summary_stats['builds_apfd_lt_0.5']} ({summary_stats['builds_apfd_lt_0.5']/summary_stats['total_builds']*100:.1f}%)")
    print("="*70)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python apfd_per_build.py <prioritized_csv> [output_csv]")
        print("Example: python apfd_per_build.py ../results/execution_006/prioritized_hybrid.csv apfd_report.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    # Load data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Generate report
    results_df, summary_stats = generate_apfd_report(
        df,
        method_name="filo_priori_v5",
        test_scenario="full",
        output_path=output_csv
    )

    # Print summary
    print_apfd_summary(summary_stats)

    # Show sample results
    print(f"\nSample of results (first 10 builds):")
    print(results_df.head(10).to_string(index=False))
