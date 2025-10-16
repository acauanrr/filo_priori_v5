"""
Filo-Priori V5 - Script Orquestrador Completo

Executa pipeline end-to-end:
1. Parse commits
2. Build text_semantic
3. Generate SBERT embeddings
4. Build features tabulares
5. Train Deep MLP
6. Evaluate e salvar resultados

Usage:
    # Smoke test
    python run_experiment.py --smoke-train 100 --smoke-test 50 --smoke-epochs 20

    # Full test
    python run_experiment.py --full-test

Autor: Filo-Priori V5
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from data_processing.01_parse_commit import process_commits
from data_processing.02_build_text_semantic import process_text_semantic
from data_processing.03_embed_sbert import SBERTEmbedder, process_embeddings_train, process_embeddings_test
from utils.features import FeatureBuilder
from utils.dataset import TabularDataset, create_balanced_sampler
from utils.model import DeepMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'train_csv': '../filo_priori_v4/datasets/train.csv',
    'test_csv': '../filo_priori_v4/datasets/test_full.csv',
    'output_dir': './artifacts',
    'sbert_target_dim': 128,
    'sbert_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'model_hidden_dims': [512, 256, 128],
    'model_dropout': 0.3,
    'lr': 0.001,
    'batch_size': 128,
    'epochs': 30,
    'patience': 8,
    'pos_weight': 5.0,
    'sampler_positive_fraction': 0.25,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seeds."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_apfd(order, labels):
    """Calculate APFD."""
    labels_arr = np.array(labels)
    n_tests = len(labels_arr)
    fail_indices = np.where(labels_arr >= 0.5)[0]
    fail_count = len(fail_indices)

    if n_tests == 0 or fail_count == 0:
        return 1.0

    order = np.array(order)
    rank_positions = np.empty(n_tests, dtype=np.float64)
    rank_positions[order] = np.arange(1, n_tests + 1, dtype=np.float64)
    fail_positions = rank_positions[labels_arr >= 0.5]

    apfd = 1.0 - fail_positions.sum() / (fail_count * n_tests) + 1.0 / (2.0 * n_tests)
    return float(np.clip(apfd, 0.0, 1.0))


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        continuous = batch['continuous'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(continuous)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(all_labels), np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        continuous = batch['continuous'].to(device)
        labels = batch['label'].to(device)

        logits = model(continuous)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Metrics
    binary_preds = (preds >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average='binary', zero_division=0
    )

    metrics = {
        'loss': total_loss / len(labels),
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
        'auprc': average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    }

    return metrics, preds


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(args):
    """Run complete pipeline."""
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))

    set_seed(config['seed'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("FILO-PRIORI V5 - PIPELINE START")
    logger.info("="*70)
    logger.info(f"Mode: {'SMOKE TEST' if args.smoke_train else 'FULL TEST'}")
    logger.info(f"Device: {config['device']}")

    # ========================================================================
    # STEP 1: Load and parse data
    # ========================================================================
    logger.info("\n[1/7] Loading and parsing commits...")

    df_train = pd.read_csv(config['train_csv'])
    df_test = pd.read_csv(config['test_csv'])

    if args.smoke_train:
        # Smoke test: limit builds
        builds = df_train['Build_ID'].unique()[:args.smoke_train]
        df_train = df_train[df_train['Build_ID'].isin(builds)]

        builds_test = df_test['Build_ID'].unique()[:args.smoke_test]
        df_test = df_test[df_test['Build_ID'].isin(builds_test)]

    logger.info(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")

    df_train = process_commits(df_train)
    df_test = process_commits(df_test)

    # ========================================================================
    # STEP 2: Build text_semantic
    # ========================================================================
    logger.info("\n[2/7] Building text_semantic...")

    df_train = process_text_semantic(df_train)
    df_test = process_text_semantic(df_test)

    # Remove ambiguous labels
    df_train = df_train[df_train['label_binary'].notna()].reset_index(drop=True)
    df_test = df_test[df_test['label_binary'].notna()].reset_index(drop=True)

    logger.info(f"After filtering - Train: {len(df_train)}, Test: {len(df_test)}")

    # ========================================================================
    # STEP 3: Generate embeddings
    # ========================================================================
    logger.info("\n[3/7] Generating SBERT embeddings...")

    embedder = SBERTEmbedder(
        model_name=config['sbert_model'],
        target_dim=config['sbert_target_dim'],
        device=config['device']
    )

    # Train embeddings
    train_texts = df_train['text_semantic'].tolist()
    train_emb = embedder.encode(train_texts)

    if config['sbert_target_dim']:
        embedder.fit_projection(train_emb)
        train_emb = embedder.transform_projection(train_emb)

    embedder.fit_scaler(train_emb)
    train_emb = embedder.transform_scaler(train_emb)

    # Test embeddings
    test_texts = df_test['text_semantic'].tolist()
    test_emb = embedder.encode(test_texts)

    if embedder.pca:
        test_emb = embedder.transform_projection(test_emb)
    test_emb = embedder.transform_scaler(test_emb)

    logger.info(f"Embedding shape: {train_emb.shape}")

    # ========================================================================
    # STEP 4: Build features
    # ========================================================================
    logger.info("\n[4/7] Building tabular features...")

    feature_builder = FeatureBuilder()
    train_cont, train_cat = feature_builder.fit_transform_features(df_train, train_emb)
    test_cont, test_cat = feature_builder.transform_features(df_test, test_emb)

    # Split train/val
    labels_train = df_train['label_binary'].values
    indices = np.arange(len(labels_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=labels_train, random_state=config['seed']
    )

    # Datasets
    train_dataset = TabularDataset(train_cont[train_idx], train_cat[train_idx], labels_train[train_idx])
    val_dataset = TabularDataset(train_cont[val_idx], train_cat[val_idx], labels_train[val_idx])
    test_dataset = TabularDataset(test_cont, test_cat, df_test['label_binary'].values)

    # Samplers
    train_sampler = create_balanced_sampler(labels_train[train_idx], config['sampler_positive_fraction'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ========================================================================
    # STEP 5: Train model
    # ========================================================================
    logger.info("\n[5/7] Training model...")

    model = DeepMLP(
        input_dim=train_cont.shape[1],
        hidden_dims=config['model_hidden_dims'],
        dropout=config['model_dropout']
    ).to(config['device'])

    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Loss with class weight
    pos_weight = torch.tensor([config['pos_weight']]).to(config['device'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    # Training loop
    best_val_auprc = 0
    patience_counter = 0
    history = []

    epochs = args.smoke_epochs if args.smoke_train else config['epochs']

    for epoch in range(epochs):
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, config['device']
        )

        val_metrics, _ = evaluate(model, val_loader, criterion, config['device'])

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss={train_loss:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # Early stopping
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # ========================================================================
    # STEP 6: Evaluate
    # ========================================================================
    logger.info("\n[6/7] Evaluating...")

    test_metrics, test_preds = evaluate(model, test_loader, criterion, config['device'])

    # APFD
    order = np.argsort(-test_preds)
    apfd = calculate_apfd(order, df_test['label_binary'].values)

    logger.info(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"APFD: {apfd:.4f}")

    # ========================================================================
    # STEP 7: Save results
    # ========================================================================
    logger.info("\n[7/7] Saving results...")

    results = {
        'test_metrics': test_metrics,
        'apfd': apfd,
        'config': {k: v for k, v in config.items() if k not in ['device']},
        'history': history
    }

    with open(output_dir / 'metrics_v5.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    df_test['probability_v5'] = test_preds
    df_test['rank_v5'] = np.argsort(-test_preds).argsort() + 1
    df_test[['Build_ID', 'TC_Key', 'label_binary', 'probability_v5', 'rank_v5']].to_csv(
        output_dir / 'predictions_v5.csv', index=False
    )

    logger.info(f"Results saved to {output_dir}")
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filo-Priori V5 Pipeline')

    parser.add_argument('--smoke-train', type=int, help='Smoke test: number of train builds')
    parser.add_argument('--smoke-test', type=int, help='Smoke test: number of test builds')
    parser.add_argument('--smoke-epochs', type=int, default=20, help='Smoke test: number of epochs')
    parser.add_argument('--full-test', action='store_true', help='Run full test')

    parser.add_argument('--output-dir', default='./artifacts', help='Output directory')
    parser.add_argument('--device', default='auto', help='Device: cuda/cpu/auto')

    args = parser.parse_args()

    # Validate
    if not (args.smoke_train or args.full_test):
        parser.error("Must specify either --smoke-train or --full-test")

    if args.smoke_train and not args.smoke_test:
        parser.error("--smoke-test required when using --smoke-train")

    # Run
    try:
        run_pipeline(args)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
