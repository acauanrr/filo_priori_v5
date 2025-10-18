"""
Training utilities for SAINT transformer model.

This module provides training loop, early stopping, and evaluation functions
specifically designed for the SAINT model in the Filo-Priori pipeline.

Author: Filo-Priori V5 (Refactored)
Date: 2025-10-16
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to consider as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current metric value

        Returns:
            True if should stop training, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check for improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create a learning rate scheduler with warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: SAINT model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to use

    Returns:
        (metrics_dict, all_labels, all_probs)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x_continuous = batch['continuous'].to(device)
            x_categorical = batch.get('categorical', None)
            if x_categorical is not None:
                x_categorical = x_categorical.to(device)

            labels = batch['label'].unsqueeze(1).to(device)  # [batch_size, 1]

            # Forward pass
            logits = model(x_continuous, x_categorical)
            loss = criterion(logits, labels)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Store results
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            total_loss += loss.item() * len(labels)

    # Concatenate all results
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    all_probs = np.concatenate(all_probs, axis=0).flatten()
    all_preds = (all_probs >= 0.5).astype(int)

    # Calculate metrics
    avg_loss = total_loss / len(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'auprc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    }

    return metrics, all_labels, all_probs


def train_saint(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str,
    save_dir: Path
) -> Dict:
    """
    Train SAINT model with early stopping and learning rate scheduling.

    Args:
        model: SAINT model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Training configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints

    Returns:
        Dictionary with training history
    """
    logger.info("="*70)
    logger.info("STARTING SAINT TRAINING")
    logger.info("="*70)

    # Extract config
    num_epochs = config.get('num_epochs', 30)
    learning_rate = config.get('learning_rate', 5e-4)
    weight_decay = config.get('weight_decay', 0.01)
    patience = config.get('patience', 8)
    min_delta = config.get('min_delta', 1e-4)  # NEW: minimum improvement threshold
    monitor_metric = config.get('monitor_metric', 'val_auprc')
    warmup_epochs = config.get('warmup_epochs', 3)
    min_lr_ratio = config.get('min_lr_ratio', 0.01)
    gradient_clip = config.get('gradient_clip', 1.0)
    label_smoothing = config.get('label_smoothing', 0.05)
    pos_weight = config.get('pos_weight', None)

    # Move model to device
    model = model.to(device)

    # Calculate class weights if needed
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Label smoothing will be applied manually in the training loop

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'val_auprc': [],
        'learning_rate': []
    }

    best_metric = -float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 70)

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            x_continuous = batch['continuous'].to(device)
            x_categorical = batch.get('categorical', None)
            if x_categorical is not None:
                x_categorical = x_categorical.to(device)

            labels = batch['label'].unsqueeze(1).to(device)  # [batch_size, 1]

            # Apply label smoothing manually
            if label_smoothing > 0:
                labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                labels_smooth = labels

            # Forward pass
            optimizer.zero_grad()
            logits = model(x_continuous, x_categorical)
            loss = criterion(logits, labels_smooth)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            # Track loss
            train_loss += loss.item()
            train_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_batches

        # Validation phase
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val AUPRC: {val_metrics['auprc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['learning_rate'].append(current_lr)

        # Check for best model
        current_metric = val_metrics[monitor_metric.replace('val_', '')]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1

            # Save best model
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                monitor_metric: current_metric,
                'config': config
            }, save_dir / 'best_model.pth')

            logger.info(f"✓ Best model saved (epoch {best_epoch}, {monitor_metric}={best_metric:.4f})")

        # Early stopping check
        if early_stopping(current_metric):
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            logger.info(f"Best epoch: {best_epoch}, Best {monitor_metric}: {best_metric:.4f}")
            break

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best {monitor_metric}: {best_metric:.4f}")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_metric': best_metric,
        'monitor_metric': monitor_metric
    }


def load_saint_checkpoint(model: nn.Module, checkpoint_path: Path, device: str) -> nn.Module:
    """
    Load SAINT model from checkpoint.

    Args:
        model: SAINT model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    # PyTorch 2.6+ requires weights_only=False for pickled objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


def predict_saint(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using SAINT model.

    Args:
        model: Trained SAINT model
        dataloader: DataLoader for inference
        device: Device to use

    Returns:
        (labels, probabilities)
    """
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            x_continuous = batch['continuous'].to(device)
            x_categorical = batch.get('categorical', None)
            if x_categorical is not None:
                x_categorical = x_categorical.to(device)

            labels = batch['label']

            # Forward pass
            logits = model(x_continuous, x_categorical)
            probs = torch.sigmoid(logits)

            # Store results
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all results
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    all_probs = np.concatenate(all_probs, axis=0).flatten()

    return all_labels, all_probs


if __name__ == "__main__":
    # Test training utilities
    print("Testing SAINT training utilities...")

    from filo_priori.models.saint import SAINT
    from filo_priori.utils.dataset import TabularDataset, create_balanced_sampler

    # Create dummy data
    n_samples = 1000
    n_features = 1024  # Semantic-only features

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Create dataset
    dataset = TabularDataset(
        continuous=X,
        categorical=np.array([]).reshape(n_samples, 0),
        labels=y
    )

    # Create dataloader
    sampler = create_balanced_sampler(y, target_positive_fraction=0.3)
    train_loader = DataLoader(dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create model
    model = SAINT(
        num_continuous=n_features,
        embedding_dim=1024,  # ✨ FULL BGE DIMENSION
        num_layers=2,  # Small for testing
        num_heads=8,  # Changed from 4 to 8 (1024/8 = 128 per head)
        dropout=0.1
    )

    # Training config
    config = {
        'num_epochs': 3,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'patience': 5,
        'monitor_metric': 'val_auprc',
        'warmup_epochs': 1,
        'gradient_clip': 1.0,
        'label_smoothing': 0.05,
        'pos_weight': 2.0
    }

    # Train
    save_dir = Path('test_checkpoints')
    results = train_saint(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu',
        save_dir=save_dir
    )

    print(f"\nTraining completed!")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best metric: {results['best_metric']:.4f}")

    # Test prediction
    labels, probs = predict_saint(model, val_loader, 'cpu')
    print(f"\nPrediction shape: {probs.shape}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")

    print("\nSAINT training utilities test passed!")

    # Cleanup
    import shutil
    if save_dir.exists():
        shutil.rmtree(save_dir)
