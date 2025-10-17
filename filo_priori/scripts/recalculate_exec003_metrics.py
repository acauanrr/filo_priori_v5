"""
Script para recalcular e corrigir métricas da execution_003.

Este script:
1. Lê os dados do prioritized_hybrid.csv
2. Recalcula o APFD corretamente (por build, não globalmente)
3. Atualiza metrics.json com valores corretos
4. Regenera summary.txt corrigido

Autor: Filo-Priori V5
Data: 2025-10-17
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.apfd_per_build import generate_apfd_report, print_apfd_summary


def recalculate_metrics(exec_dir):
    """Recalculate and fix metrics for execution_003."""
    exec_path = Path(exec_dir)

    print(f"Processing: {exec_path}")
    print("="*70)

    # 1. Load existing data
    print("\n[1/4] Loading existing data...")
    csv_path = exec_path / 'prioritized_hybrid.csv'
    metrics_path = exec_path / 'metrics.json'

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    df = pd.read_csv(csv_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"Loaded {len(df)} test cases")

    # 2. Recalculate APFD per build
    print("\n[2/4] Recalculating APFD per build...")
    apfd_results_df, apfd_summary = generate_apfd_report(
        df,
        method_name="filo_priori_v5_saint",
        test_scenario=metrics['metadata']['experiment_type'],
        output_path=exec_path / 'apfd_per_build.csv'
    )
    print_apfd_summary(apfd_summary)

    # 3. Update metrics.json
    print("\n[3/4] Updating metrics.json...")

    # Remove old incorrect APFD values from metrics
    if 'apfd' in metrics['metrics']:
        old_apfd = metrics['metrics']['apfd']
        print(f"Removing incorrect global APFD: {old_apfd:.4f}")
        del metrics['metrics']['apfd']
    if 'apfdc' in metrics['metrics']:
        del metrics['metrics']['apfdc']

    # Update with correct APFD summary
    metrics['apfd_per_build_summary'] = apfd_summary

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"✅ Updated metrics.json")

    # 4. Regenerate summary.txt
    print("\n[4/4] Regenerating summary.txt...")

    metadata = metrics['metadata']
    test_metrics = metrics['metrics']
    train_results = metrics['training']
    config = metrics['config']
    prob_stats = metrics['probability_stats']

    summary = f"""
FILO-PRIORI V5 - EXPERIMENT SUMMARY (BGE + SAINT)
{'='*70}

Execution Directory: {exec_path.name}
Timestamp: {metadata['timestamp']}
Mode: {metadata['experiment_type'].upper()}

MODEL
-----
Embeddings: {metadata['embedding_model']} ({metadata['embedding_dim']}D)
Classifier: SAINT Transformer ({metadata['model_params']:,} parameters)
  - Embedding dim: {config['saint']['embedding_dim']}
  - Layers: {config['saint']['num_layers']}
  - Heads: {config['saint']['num_heads']}
  - Intersample attention: {config['saint']['use_intersample']}
Features: {config['saint']['num_continuous']}D (1024 semantic + ~4 temporal)

DATASET
-------
Train: {metadata['dataset_stats']['train_total']} samples, {metadata['dataset_stats']['train_failures']} failures ({metadata['dataset_stats']['train_failure_rate']*100:.2f}%)
Val:   {metadata['dataset_stats']['val_total']} samples
Test:  {metadata['dataset_stats']['test_total']} samples, {metadata['dataset_stats']['test_failures']} failures ({metadata['dataset_stats']['test_failure_rate']*100:.2f}%)

TRAINING
--------
Best Epoch: {train_results['best_epoch']}
Best {train_results['monitor_metric']}: {train_results['best_metric']:.4f}
Total Epochs: {len(train_results['history']['train_loss'])}

TEST RESULTS (Classification)
------------------------------
AUPRC:     {test_metrics['auprc']:.4f}
Precision: {test_metrics['precision']:.4f}
Recall:    {test_metrics['recall']:.4f}
F1:        {test_metrics['f1']:.4f}
Accuracy:  {test_metrics['accuracy']:.4f}
AUC:       {test_metrics['auc']:.4f}

TEST RESULTS (Prioritization)
------------------------------
Mean APFD:   {apfd_summary['mean_apfd']:.4f}
Median APFD: {apfd_summary['median_apfd']:.4f}
Std APFD:    {apfd_summary['std_apfd']:.4f}
Min APFD:    {apfd_summary['min_apfd']:.4f}
Max APFD:    {apfd_summary['max_apfd']:.4f}

APFD Distribution (across {apfd_summary['total_builds']} builds):
  - Builds with APFD = 1.0:  {apfd_summary['builds_apfd_1.0']} ({apfd_summary['builds_apfd_1.0']/apfd_summary['total_builds']*100:.1f}%)
  - Builds with APFD ≥ 0.7:  {apfd_summary['builds_apfd_gte_0.7']} ({apfd_summary['builds_apfd_gte_0.7']/apfd_summary['total_builds']*100:.1f}%)
  - Builds with APFD < 0.5:  {apfd_summary['builds_apfd_lt_0.5']} ({apfd_summary['builds_apfd_lt_0.5']/apfd_summary['total_builds']*100:.1f}%)

PROBABILITY ANALYSIS
--------------------
Failures: mean={prob_stats['failures_mean']:.6f}, std={prob_stats['failures_std']:.6f}
Passes:   mean={prob_stats['passes_mean']:.6f}, std={prob_stats['passes_std']:.6f}
Discrimination: {test_metrics['discrimination_ratio']:.4f}x

FILES SAVED
-----------
- metrics.json              : Complete metrics and results
- config.json               : Experiment configuration
- best_model.pth            : Trained SAINT model (best checkpoint)
- training_history.json     : Per-epoch training metrics
- prioritized_hybrid.csv    : Test predictions with ranks
- apfd_per_build.csv        : APFD calculated per build
- feature_builder.pkl       : Feature engineering artifacts
- embedder/                 : BGE embedder artifacts (scaler only, no PCA)
- summary.txt               : This summary

{'='*70}
"""

    with open(exec_path / 'summary.txt', 'w') as f:
        f.write(summary)
    print(f"✅ Updated summary.txt")

    print("\n" + "="*70)
    print("RECALCULATION COMPLETED!")
    print("="*70)
    print(f"\nCorrected Mean APFD: {apfd_summary['mean_apfd']:.4f}")
    print(f"Builds analyzed: {apfd_summary['total_builds']}")
    print(f"Builds with APFD ≥ 0.7: {apfd_summary['builds_apfd_gte_0.7']} ({apfd_summary['builds_apfd_gte_0.7']/apfd_summary['total_builds']*100:.1f}%)")


if __name__ == "__main__":
    exec_dir = "results/execution_003"

    if len(sys.argv) > 1:
        exec_dir = sys.argv[1]

    recalculate_metrics(exec_dir)
