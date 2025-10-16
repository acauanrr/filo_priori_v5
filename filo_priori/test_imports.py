#!/usr/bin/env python
"""
Test script to verify all imports work correctly.
Run this before executing the main pipeline.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("="*70)
    print("IMPORT VERIFICATION TEST")
    print("="*70)
    print()

    errors = []

    # Test standard libraries
    print("1. Testing standard libraries...")
    try:
        import json, pandas as pd, numpy as np
        import torch, torch.nn as nn
        from sklearn.model_selection import train_test_split
        print("   ✅ Standard libraries OK")
    except ImportError as e:
        errors.append(f"Standard libraries: {e}")
        print(f"   ❌ Error: {e}")

    # Test sentence-transformers
    print("\n2. Testing sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ sentence-transformers OK")
    except ImportError as e:
        errors.append(f"sentence-transformers: {e}")
        print(f"   ❌ Error: {e}")

    # Test utils imports
    print("\n3. Testing utils modules...")
    try:
        from utils.features import FeatureBuilder
        from utils.dataset import TabularDataset, create_balanced_sampler
        from utils.model import DeepMLP
        print("   ✅ utils modules OK")
    except ImportError as e:
        errors.append(f"utils: {e}")
        print(f"   ❌ Error: {e}")

    # Test data_processing imports
    print("\n4. Testing data_processing modules...")
    try:
        import importlib.util

        def load_module(file_path, module_name):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        parse_commit = load_module('data_processing/01_parse_commit.py', 'parse_commit')
        build_text = load_module('data_processing/02_build_text_semantic.py', 'build_text')
        embed_sbert = load_module('data_processing/03_embed_sbert.py', 'embed_sbert')

        assert hasattr(parse_commit, 'process_commits'), "Missing process_commits function"
        assert hasattr(build_text, 'process_text_semantic'), "Missing process_text_semantic function"
        assert hasattr(embed_sbert, 'SBERTEmbedder'), "Missing SBERTEmbedder class"

        print("   ✅ data_processing modules OK")
    except Exception as e:
        errors.append(f"data_processing: {e}")
        print(f"   ❌ Error: {e}")

    # Test paths
    print("\n5. Testing file paths...")
    try:
        train_csv = Path('../datasets/train.csv')
        test_csv = Path('../datasets/test_full.csv')
        results_dir = Path('../results')

        assert train_csv.exists(), f"Train dataset not found: {train_csv.resolve()}"
        assert test_csv.exists(), f"Test dataset not found: {test_csv.resolve()}"
        assert results_dir.exists(), f"Results directory not found: {results_dir.resolve()}"
        assert results_dir.is_dir(), "Results path is not a directory"

        print(f"   ✅ Paths OK")
        print(f"      Train: {train_csv.stat().st_size / 1e9:.2f} GB")
        print(f"      Test:  {test_csv.stat().st_size / 1e9:.2f} GB")
    except AssertionError as e:
        errors.append(f"Paths: {e}")
        print(f"   ❌ Error: {e}")

    # Summary
    print("\n" + "="*70)
    if errors:
        print(f"❌ FAILED - {len(errors)} error(s) found:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        print("="*70)
        return False
    else:
        print("✅ ALL TESTS PASSED - System is ready!")
        print("="*70)
        return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
