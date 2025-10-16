"""
Script para verificar data leakage entre splits train/val/test.

Verifica se há sobreposição de Build_ID entre os conjuntos,
o que causaria vazamento de informação.

Uso:
    python scripts/check_data_leakage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("DATA LEAKAGE VERIFICATION")
print("="*80)

# Load datasets
print("\n📂 Loading datasets...")
train_df = pd.read_csv('../datasets/train.csv')
test_df = pd.read_csv('../datasets/test_full.csv')

print(f"Train: {len(train_df):,} rows")
print(f"Test:  {len(test_df):,} rows")

# Get unique Build_IDs
train_builds = set(train_df['Build_ID'].unique())
test_builds = set(test_df['Build_ID'].unique())

print(f"\nTrain builds: {len(train_builds):,} unique")
print(f"Test builds:  {len(test_builds):,} unique")

# Check overlap
overlap = train_builds & test_builds

print("\n" + "="*80)
print("LEAKAGE CHECK: Train ∩ Test")
print("="*80)

if len(overlap) > 0:
    print(f"🔴 CRITICAL: Found {len(overlap):,} Build_IDs in BOTH train and test!")
    print(f"   Overlap percentage: {len(overlap)/len(train_builds)*100:.2f}% of train builds")
    print(f"   Overlap percentage: {len(overlap)/len(test_builds)*100:.2f}% of test builds")

    # Count affected rows
    train_affected = train_df[train_df['Build_ID'].isin(overlap)]
    test_affected = test_df[test_df['Build_ID'].isin(overlap)]

    print(f"\n   Affected rows in train: {len(train_affected):,} ({len(train_affected)/len(train_df)*100:.2f}%)")
    print(f"   Affected rows in test:  {len(test_affected):,} ({len(test_affected)/len(test_df)*100:.2f}%)")

    print(f"\n🔍 Sample of overlapping Build_IDs:")
    for build_id in list(overlap)[:10]:
        train_count = len(train_df[train_df['Build_ID'] == build_id])
        test_count = len(test_df[test_df['Build_ID'] == build_id])
        print(f"   {build_id}: {train_count} in train, {test_count} in test")

    print("\n⚠️  THIS IS DATA LEAKAGE!")
    print("   The model can memorize patterns from specific builds in training")
    print("   and be tested on the SAME builds, inflating validation metrics.")

else:
    print(f"✅ GOOD: No Build_ID overlap between train and test")
    print("   Train and test sets are properly separated by Build_ID")

# Additional analysis: temporal distribution
print("\n" + "="*80)
print("TEMPORAL DISTRIBUTION ANALYSIS")
print("="*80)

# Assuming Build_ID has temporal ordering (lower ID = older)
train_build_ids = sorted(train_builds)
test_build_ids = sorted(test_builds)

print(f"\nTrain Build_IDs range: {min(train_build_ids)} - {max(train_build_ids)}")
print(f"Test Build_IDs range:  {min(test_build_ids)} - {max(test_build_ids)}")

# Check if test is chronologically after train
train_min = min(train_build_ids)
train_max = max(train_build_ids)
test_min = min(test_build_ids)
test_max = max(test_build_ids)

# Build IDs are strings (not numeric), so we can't reliably compare chronologically
# We'll just report the ranges
print(f"\nℹ️  Note: Build_IDs are strings, chronological ordering may not be reliable")
print(f"   Use this analysis with caution")

# Failure rate comparison
print("\n" + "="*80)
print("FAILURE RATE COMPARISON")
print("="*80)

train_failure_rate = train_df['TE_Test_Result'].value_counts(normalize=True).get('Fail', 0)
test_failure_rate = test_df['TE_Test_Result'].value_counts(normalize=True).get('Fail', 0)

print(f"\nTrain failure rate: {train_failure_rate*100:.2f}%")
print(f"Test failure rate:  {test_failure_rate*100:.2f}%")
print(f"Difference: {abs(train_failure_rate - test_failure_rate)*100:.2f} percentage points")

if abs(train_failure_rate - test_failure_rate) > 0.005:  # > 0.5%
    print(f"\n⚠️  WARNING: Failure rates differ by more than 0.5%")
    print("   This indicates distribution shift between train and test")
else:
    print(f"\n✅ GOOD: Failure rates are similar")

# TC_Key overlap check
print("\n" + "="*80)
print("TC_KEY OVERLAP CHECK")
print("="*80)

train_tcs = set(train_df['TC_Key'].unique())
test_tcs = set(test_df['TC_Key'].unique())
tc_overlap = train_tcs & test_tcs

print(f"\nTrain unique TCs: {len(train_tcs):,}")
print(f"Test unique TCs:  {len(test_tcs):,}")
print(f"TC overlap: {len(tc_overlap):,} ({len(tc_overlap)/len(train_tcs)*100:.1f}% of train TCs)")

if len(tc_overlap) > 0:
    print(f"\n✅ EXPECTED: Same TCs appear in both train and test")
    print("   This is normal - we're testing the same test cases on different builds")
else:
    print(f"\n⚠️  UNEXPECTED: No TC overlap - completely different test cases?")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

issues = []
if len(overlap) > 0:
    issues.append("🔴 Build_ID leakage detected")
if abs(train_failure_rate - test_failure_rate) > 0.005:
    issues.append("⚠️  Failure rate mismatch")

if issues:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
    print("\n🔧 RECOMMENDED ACTIONS:")
    if len(overlap) > 0:
        print("   1. Re-split dataset ensuring NO Build_ID overlap")
        print("   2. Use stratified split by Build_ID, not by rows")
    print("   3. Consider temporal split: train on older builds, test on newer")
else:
    print("\n✅ NO CRITICAL ISSUES FOUND")
    print("   Data splits appear properly configured")
