"""
Probability Calibration Module for Filo-Priori V5.

This module provides calibration methods to improve probability estimates
from the SAINT classifier, which is critical for APFD optimization.

Methods implemented:
- Isotonic Regression: Non-parametric, monotonic calibration
- Platt Scaling: Logistic regression on raw probabilities
- Temperature Scaling: Single-parameter scaling of logits

Author: Filo-Priori V5
Date: 2025-10-17
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Literal
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate model probabilities to improve reliability for APFD optimization.

    Calibration ensures that predicted probabilities match empirical frequencies,
    which is crucial for test case prioritization.

    Example:
        >>> calibrator = ProbabilityCalibrator(method='isotonic')
        >>> calibrator.fit(val_probs, val_labels)
        >>> test_probs_calibrated = calibrator.transform(test_probs)
    """

    def __init__(
        self,
        method: Literal['isotonic', 'platt', 'temperature'] = 'isotonic'
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method
                - 'isotonic': IsotonicRegression (non-parametric, recommended)
                - 'platt': Logistic regression (parametric)
                - 'temperature': Temperature scaling (simplest)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False

        logger.info(f"Initialized ProbabilityCalibrator with method={method}")

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation set.

        Args:
            probs: Predicted probabilities [N,]
            labels: True binary labels [N,]

        Returns:
            Self for chaining
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()

        if len(probs) != len(labels):
            raise ValueError(f"Length mismatch: probs={len(probs)}, labels={len(labels)}")

        n_samples = len(probs)
        n_positive = labels.sum()

        logger.info(f"Fitting calibrator on {n_samples} samples ({n_positive} positive, {n_positive/n_samples*100:.2f}%)")

        if self.method == 'isotonic':
            # Non-parametric, monotonic calibration
            self.calibrator = IsotonicRegression(
                out_of_bounds='clip',  # Clip to [0, 1]
                y_min=0.0,
                y_max=1.0
            )
            self.calibrator.fit(probs, labels)

        elif self.method == 'platt':
            # Logistic regression on probabilities
            self.calibrator = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            # Reshape for sklearn
            self.calibrator.fit(probs.reshape(-1, 1), labels)

        elif self.method == 'temperature':
            # Temperature scaling: find optimal temperature T
            # Calibrated_prob = sigmoid(logit / T)
            from scipy.optimize import minimize_scalar

            def negative_log_likelihood(temperature):
                """NLL loss for temperature scaling."""
                # Convert probs to logits
                eps = 1e-7
                probs_clipped = np.clip(probs, eps, 1 - eps)
                logits = np.log(probs_clipped / (1 - probs_clipped))

                # Scale logits
                scaled_logits = logits / temperature

                # Convert back to probs
                calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
                calibrated_probs = np.clip(calibrated_probs, eps, 1 - eps)

                # Negative log-likelihood
                nll = -np.mean(
                    labels * np.log(calibrated_probs) +
                    (1 - labels) * np.log(1 - calibrated_probs)
                )
                return nll

            # Find optimal temperature
            result = minimize_scalar(
                negative_log_likelihood,
                bounds=(0.1, 10.0),
                method='bounded'
            )

            self.calibrator = result.x  # Store optimal temperature
            logger.info(f"Optimal temperature: {self.calibrator:.4f}")

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True

        # Compute calibration metrics
        calibrated_probs = self.transform(probs)
        self._log_calibration_quality(probs, calibrated_probs, labels)

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            probs: Raw probabilities [N,]

        Returns:
            Calibrated probabilities [N,]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        probs = np.asarray(probs).flatten()

        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(probs)

        elif self.method == 'platt':
            calibrated = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

        elif self.method == 'temperature':
            # Convert to logits, scale, convert back
            eps = 1e-7
            probs_clipped = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs_clipped / (1 - probs_clipped))
            scaled_logits = logits / self.calibrator
            calibrated = 1 / (1 + np.exp(-scaled_logits))

        # Ensure valid probabilities
        calibrated = np.clip(calibrated, 0.0, 1.0)

        return calibrated

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit calibrator and transform in one step.

        Args:
            probs: Predicted probabilities [N,]
            labels: True binary labels [N,]

        Returns:
            Calibrated probabilities [N,]
        """
        return self.fit(probs, labels).transform(probs)

    def _log_calibration_quality(
        self,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        labels: np.ndarray
    ):
        """Log calibration quality metrics."""
        from sklearn.metrics import brier_score_loss, log_loss

        # Brier score (lower is better)
        brier_raw = brier_score_loss(labels, raw_probs)
        brier_calibrated = brier_score_loss(labels, calibrated_probs)

        # Log loss (lower is better)
        eps = 1e-7
        logloss_raw = log_loss(labels, np.clip(raw_probs, eps, 1-eps))
        logloss_calibrated = log_loss(labels, np.clip(calibrated_probs, eps, 1-eps))

        # Discrimination (failures vs passes mean probabilities)
        fail_idx = labels == 1
        pass_idx = labels == 0

        if fail_idx.sum() > 0 and pass_idx.sum() > 0:
            disc_raw = raw_probs[fail_idx].mean() / raw_probs[pass_idx].mean()
            disc_cal = calibrated_probs[fail_idx].mean() / calibrated_probs[pass_idx].mean()
        else:
            disc_raw = disc_cal = 1.0

        logger.info("=" * 70)
        logger.info("CALIBRATION QUALITY")
        logger.info("=" * 70)
        logger.info(f"Brier Score:      {brier_raw:.4f} → {brier_calibrated:.4f} ({(brier_calibrated-brier_raw)/brier_raw*100:+.1f}%)")
        logger.info(f"Log Loss:         {logloss_raw:.4f} → {logloss_calibrated:.4f} ({(logloss_calibrated-logloss_raw)/logloss_raw*100:+.1f}%)")
        logger.info(f"Discrimination:   {disc_raw:.4f}x → {disc_cal:.4f}x ({(disc_cal-disc_raw)/disc_raw*100:+.1f}%)")
        logger.info("=" * 70)

    def save(self, path: Path):
        """
        Save calibrator to disk.

        Args:
            path: Path to save calibrator
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted
            }, f)

        logger.info(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ProbabilityCalibrator':
        """
        Load calibrator from disk.

        Args:
            path: Path to load calibrator from

        Returns:
            Loaded calibrator
        """
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls(method=data['method'])
        calibrator.calibrator = data['calibrator']
        calibrator.is_fitted = data['is_fitted']

        logger.info(f"Calibrator loaded from {path}")

        return calibrator


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and
    empirical frequencies across probability bins.

    Args:
        probs: Predicted probabilities [N,]
        labels: True binary labels [N,]
        n_bins: Number of probability bins

    Returns:
        ECE score (0 = perfectly calibrated, 1 = worst)
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx

        if mask.sum() == 0:
            continue

        # Average predicted probability in bin
        avg_pred = probs[mask].mean()

        # Empirical frequency in bin
        avg_true = labels[mask].mean()

        # Weighted contribution to ECE
        bin_weight = mask.sum() / len(probs)
        ece += bin_weight * abs(avg_pred - avg_true)

    return ece


if __name__ == "__main__":
    # Test calibration module
    print("Testing Probability Calibration Module...")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Simulate poorly calibrated probabilities (overconfident)
    true_labels = np.random.binomial(1, 0.3, n_samples)
    raw_probs = np.random.beta(2, 5, n_samples)  # Biased probabilities

    # Add correlation with labels
    raw_probs[true_labels == 1] += 0.2
    raw_probs = np.clip(raw_probs, 0, 1)

    print(f"Dataset: {n_samples} samples, {true_labels.sum()} positive ({true_labels.mean()*100:.1f}%)")
    print(f"Raw probabilities: mean={raw_probs.mean():.4f}, std={raw_probs.std():.4f}")

    # Compute ECE before calibration
    ece_before = compute_expected_calibration_error(raw_probs, true_labels)
    print(f"ECE (before): {ece_before:.4f}")

    # Test each calibration method
    for method in ['isotonic', 'platt', 'temperature']:
        print(f"\n{'='*70}")
        print(f"Testing {method.upper()} calibration...")
        print(f"{'='*70}")

        calibrator = ProbabilityCalibrator(method=method)
        calibrated_probs = calibrator.fit_transform(raw_probs, true_labels)

        # Compute ECE after calibration
        ece_after = compute_expected_calibration_error(calibrated_probs, true_labels)
        print(f"ECE (after):  {ece_after:.4f} (improvement: {(ece_before-ece_after)/ece_before*100:.1f}%)")

        # Test save/load
        save_path = Path(f'test_calibrator_{method}.pkl')
        calibrator.save(save_path)
        loaded_calibrator = ProbabilityCalibrator.load(save_path)

        # Verify loaded calibrator works
        calibrated_probs_loaded = loaded_calibrator.transform(raw_probs)
        assert np.allclose(calibrated_probs, calibrated_probs_loaded), "Save/load mismatch!"

        # Cleanup
        save_path.unlink()

        print(f"✓ {method.upper()} calibration test passed!")

    print("\n" + "="*70)
    print("All calibration tests passed!")
    print("="*70)
