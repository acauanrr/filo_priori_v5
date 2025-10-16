#!/usr/bin/env python
"""
Script para comparar múltiplas execuções do Filo-Priori V5.

Usage:
    python scripts/compare_executions.py
    python scripts/compare_executions.py --results-dir results
    python scripts/compare_executions.py --export comparison.csv
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import sys


def load_execution_metrics(exec_dir):
    """Load metrics from an execution directory."""
    metrics_file = exec_dir / 'metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        data = json.load(f)

    # Extract key metrics
    metadata = data.get('metadata', {})
    metrics = data.get('metrics', {})
    prob_stats = data.get('probability_stats', {})

    return {
        'execution': exec_dir.name,
        'type': metadata.get('experiment_type', 'unknown'),
        'timestamp': metadata.get('timestamp', 'unknown'),
        'device': metadata.get('device', 'unknown'),
        'train_builds': metadata.get('n_train_builds', 'unknown'),
        'test_builds': metadata.get('n_test_builds', 'unknown'),
        'epochs_executed': metadata.get('n_epochs_executed', 0),
        'early_stopped': metadata.get('early_stopped', False),
        'apfd': metrics.get('apfd', 0.0),
        'apfdc': metrics.get('apfdc', 0.0),
        'auprc': metrics.get('auprc', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1': metrics.get('f1', 0.0),
        'accuracy': metrics.get('accuracy', 0.0),
        'discrimination': metrics.get('discrimination_ratio', 0.0),
        'fail_prob_mean': prob_stats.get('failures_mean', 0.0),
        'pass_prob_mean': prob_stats.get('passes_mean', 0.0),
        'train_total': metadata.get('dataset_stats', {}).get('train_total', 0),
        'train_failures': metadata.get('dataset_stats', {}).get('train_failures', 0),
        'test_total': metadata.get('dataset_stats', {}).get('test_total', 0),
        'test_failures': metadata.get('dataset_stats', {}).get('test_failures', 0)
    }


def compare_executions(results_dir='results', export_file=None):
    """Compare all executions in results directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return

    # Find all execution directories
    exec_dirs = sorted(results_path.glob('execution_*'))

    if not exec_dirs:
        print(f"No execution directories found in '{results_dir}'")
        return

    print(f"Found {len(exec_dirs)} execution(s)\n")

    # Load metrics from all executions
    executions = []
    for exec_dir in exec_dirs:
        metrics = load_execution_metrics(exec_dir)
        if metrics:
            executions.append(metrics)

    if not executions:
        print("No valid metrics found in any execution")
        return

    # Create DataFrame
    df = pd.DataFrame(executions)

    # Display comparison table
    print("="*120)
    print("EXECUTION COMPARISON")
    print("="*120)

    # Key metrics table
    display_cols = ['execution', 'type', 'apfd', 'apfdc', 'auprc',
                   'precision', 'recall', 'f1', 'discrimination']

    print("\n📊 Key Metrics:")
    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Dataset info
    print("\n📁 Dataset Info:")
    dataset_cols = ['execution', 'type', 'train_builds', 'test_builds',
                   'train_total', 'train_failures', 'test_total', 'test_failures']
    print(df[dataset_cols].to_string(index=False))

    # Training info
    print("\n🏋️ Training Info:")
    training_cols = ['execution', 'epochs_executed', 'early_stopped', 'device']
    print(df[training_cols].to_string(index=False))

    # Statistical analysis
    print("\n📈 Statistical Summary:")
    summary_cols = ['apfd', 'apfdc', 'auprc', 'precision', 'recall', 'f1', 'discrimination']
    print(df[summary_cols].describe().round(4))

    # Best performers
    print("\n🏆 Best Performers:")
    for metric in ['apfd', 'apfdc', 'auprc', 'discrimination']:
        best_idx = df[metric].idxmax()
        best = df.loc[best_idx]
        print(f"  Best {metric.upper():15s}: {best['execution']:20s} = {best[metric]:.4f}")

    # Performance comparison with baseline (if execution_001 exists as baseline)
    if len(df) > 1:
        baseline = df.iloc[0]
        print(f"\n📊 Comparison vs Baseline ({baseline['execution']}):")
        for metric in ['apfd', 'auprc', 'discrimination']:
            print(f"\n  {metric.upper()}:")
            for idx, row in df.iterrows():
                diff = row[metric] - baseline[metric]
                pct = (diff / baseline[metric] * 100) if baseline[metric] > 0 else 0
                symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"    {symbol} {row['execution']:20s}: {row[metric]:.4f} ({diff:+.4f}, {pct:+.1f}%)")

    # Export if requested
    if export_file:
        df.to_csv(export_file, index=False)
        print(f"\n💾 Exported to: {export_file}")

    print("\n" + "="*120)


def main():
    parser = argparse.ArgumentParser(
        description='Compare Filo-Priori V5 execution results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results-dir',
        default='results',
        help='Results directory containing execution_* folders (default: results)'
    )

    parser.add_argument(
        '--export',
        help='Export comparison to CSV file'
    )

    args = parser.parse_args()

    compare_executions(args.results_dir, args.export)


if __name__ == '__main__':
    main()
