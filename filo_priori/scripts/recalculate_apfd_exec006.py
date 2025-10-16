"""
Script para Recalcular Ranks e APFD da Execution 006

Este script corrige o problema de ranks globais encontrado na execution_006,
recalculando os ranks POR BUILD e gerando novo relatório de APFD per-build.

Uso:
    python scripts/recalculate_apfd_exec006.py

Autor: Filo-Priori V5
Data: 2025-10-15
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.apfd_per_build import generate_apfd_report, print_apfd_summary

def recalculate_exec006():
    """Recalcula ranks e APFD para execution_006."""

    print("="*70)
    print("RECALCULANDO RANKS E APFD - EXECUTION_006")
    print("="*70)

    # Paths
    exec_dir = Path("results/execution_006")
    input_csv = exec_dir / "prioritized_hybrid.csv"
    output_csv = exec_dir / "prioritized_hybrid_corrected.csv"
    apfd_csv = exec_dir / "apfd_per_build_corrected.csv"
    metrics_json = exec_dir / "metrics.json"

    if not input_csv.exists():
        print(f"\n❌ ERROR: File not found: {input_csv}")
        print("   Make sure execution_006 has been run.")
        return

    print(f"\n📂 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"   Total test cases: {len(df)}")
    print(f"   Total builds: {df['Build_ID'].nunique()}")
    print(f"   Failures: {df['label_binary'].sum():.0f} ({df['label_binary'].mean()*100:.2f}%)")

    # Verify old ranks (should be global)
    print(f"\n🔍 Verificando ranks ANTIGOS (globais):")
    print(f"   Rank mínimo: {df['rank'].min()}")
    print(f"   Rank máximo: {df['rank'].max()}")

    sample_build = df['Build_ID'].iloc[0]
    sample_df = df[df['Build_ID'] == sample_build]
    print(f"\n   Exemplo - Build: {sample_build}")
    print(f"   - Testes: {len(sample_df)}")
    print(f"   - Ranks: {sorted(sample_df['rank'].values)}")
    print(f"   - Esperado: [1, 2, ..., {len(sample_df)}]")

    # Recalculate ranks PER BUILD
    print(f"\n🔧 Recalculando ranks POR BUILD...")
    df['rank_old'] = df['rank']  # Keep old ranks for comparison
    df['rank'] = df.groupby('Build_ID')['probability'] \
                   .rank(method='first', ascending=False) \
                   .astype(int)

    # Verify new ranks
    print(f"\n✅ Ranks NOVOS (por build):")
    print(f"   Rank mínimo: {df['rank'].min()}")
    print(f"   Rank máximo: {df['rank'].max()}")

    sample_df_new = df[df['Build_ID'] == sample_build]
    print(f"\n   Exemplo - Build: {sample_build}")
    print(f"   - Testes: {len(sample_df_new)}")
    print(f"   - Ranks ANTIGOS: {sorted(sample_df_new['rank_old'].values)}")
    print(f"   - Ranks NOVOS: {sorted(sample_df_new['rank'].values)}")

    # Validate ranks
    print(f"\n🧪 Validando correção...")
    all_valid = True
    invalid_builds = []

    for build_id in df['Build_ID'].unique()[:10]:  # Check first 10 builds
        build_df = df[df['Build_ID'] == build_id]
        n_tests = len(build_df)
        max_rank = build_df['rank'].max()

        if max_rank != n_tests:
            all_valid = False
            invalid_builds.append(build_id)

    if all_valid:
        print("   ✅ Validação OK: Ranks estão corretos (max_rank == n_tests)")
    else:
        print(f"   ❌ Validação FALHOU: {len(invalid_builds)} builds com ranks incorretos")
        print(f"      Builds problemáticas: {invalid_builds[:5]}")
        return

    # Recalculate priority score (uses same probabilities, but ranks are now correct)
    print(f"\n📊 Recalculando priority_score...")
    df['priority_score'] = 0.7 * df['probability'] + 0.3 * df['diversity_score']

    # Save corrected predictions
    print(f"\n💾 Salvando predições corrigidas...")
    output_cols = ['Build_ID', 'TC_Key', 'TE_Test_Result', 'label_binary',
                   'probability', 'diversity_score', 'priority_score', 'rank', 'rank_old']
    df[output_cols].to_csv(output_csv, index=False)
    print(f"   ✅ Salvo em: {output_csv}")

    # Generate APFD per-build report
    print(f"\n📈 Gerando relatório APFD per-build...")
    apfd_results_df, apfd_summary = generate_apfd_report(
        df[['Build_ID', 'TC_Key', 'label_binary', 'probability', 'rank']],
        method_name="filo_priori_v5_corrected",
        test_scenario="full_test",
        output_path=apfd_csv
    )

    print_apfd_summary(apfd_summary)

    # Update metrics.json
    print(f"\n📝 Atualizando metrics.json...")
    with open(metrics_json, 'r') as f:
        metrics = json.load(f)

    # Store old APFD summary
    if 'apfd_per_build_summary_old' not in metrics:
        metrics['apfd_per_build_summary_old'] = metrics.get('apfd_per_build_summary', {})

    # Update with new APFD summary
    metrics['apfd_per_build_summary'] = apfd_summary
    metrics['ranks_corrected'] = True
    metrics['correction_timestamp'] = pd.Timestamp.now().isoformat()

    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"   ✅ metrics.json atualizado")

    # Summary comparison
    print(f"\n" + "="*70)
    print("COMPARAÇÃO: ANTES vs DEPOIS")
    print("="*70)

    old_summary = metrics.get('apfd_per_build_summary_old', {})

    if old_summary:
        print(f"\nAPFD per-build:")
        print(f"  ANTES: mean={old_summary.get('mean_apfd', 0):.4f}, "
              f"median={old_summary.get('median_apfd', 0):.4f}")
        print(f"  DEPOIS: mean={apfd_summary['mean_apfd']:.4f}, "
              f"median={apfd_summary['median_apfd']:.4f}")
        print(f"  Melhoria: {(apfd_summary['mean_apfd'] - old_summary.get('mean_apfd', 0)):.4f}")

        print(f"\nBuilds com APFD = 0.0:")
        print(f"  ANTES: {old_summary.get('builds_apfd_lt_0.5', 0)} builds")
        print(f"  DEPOIS: {apfd_summary['builds_apfd_lt_0.5']} builds")

        print(f"\nBuilds com APFD ≥ 0.7:")
        print(f"  ANTES: {old_summary.get('builds_apfd_gte_0.7', 0)} builds")
        print(f"  DEPOIS: {apfd_summary['builds_apfd_gte_0.7']} builds")

    print(f"\n" + "="*70)
    print("RECALCULO CONCLUÍDO!")
    print("="*70)
    print(f"\n✅ Arquivos gerados:")
    print(f"   - {output_csv}")
    print(f"   - {apfd_csv}")
    print(f"   - {metrics_json} (atualizado)")
    print(f"\n📊 APFD per-build médio: {apfd_summary['mean_apfd']:.4f}")
    print(f"   Target: ≥0.70 | Status: {'✅ ATINGIDO' if apfd_summary['mean_apfd'] >= 0.70 else '❌ ABAIXO'}")


if __name__ == "__main__":
    recalculate_exec006()
