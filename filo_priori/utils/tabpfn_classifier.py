"""
TabPFN Classifier Wrapper for Filo-Priori V5.

TabPFN é um transformer pré-treinado para classificação tabular que não requer
treinamento tradicional. Basta chamar fit() e predict_proba().

Autor: Filo-Priori V5 (Refactored)
Data: 2025-10-15
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    raise ImportError(
        "TabPFN not installed. Install it with: pip install tabpfn"
    )

logger = logging.getLogger(__name__)


class FiloPrioriTabPFN:
    """
    Wrapper para TabPFN adaptado ao pipeline Filo-Priori.

    TabPFN é um modelo pré-treinado que não requer loop de treinamento,
    otimizador, ou early stopping. Basta fit() nos dados.
    """

    def __init__(self, device: str = 'cpu', n_estimators: int = 4):
        """
        Args:
            device: 'cpu' ou 'cuda' (TabPFN suporta GPU mas funciona bem em CPU)
            n_estimators: Número de ensembles (default: 4, recomendado 1-8)
        """
        self.device = device
        self.n_estimators = n_estimators

        logger.info(f"Initializing TabPFN with device={device}, n_estimators={n_estimators}")

        # TabPFN não precisa de device explícito na inicialização
        # O modelo detecta automaticamente
        self.model = TabPFNClassifier(
            device=device,
            N_ensemble_configurations=n_estimators
        )

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FiloPrioriTabPFN':
        """
        Treina (fit) o TabPFN nos dados.

        IMPORTANTE: TabPFN não tem loop de treinamento. O "fit" é apenas
        armazenar os dados de treino para usar na inferência.

        Args:
            X: Features [N, D] - até 10000 samples, até 1000 features
            y: Labels binárias [N]

        Returns:
            self
        """
        logger.info(f"Fitting TabPFN on {len(X)} samples with {X.shape[1]} features")

        # Validações
        if len(X) > 10000:
            logger.warning(
                f"TabPFN works best with ≤10000 samples. "
                f"You have {len(X)} samples. Consider subsampling."
            )

        if X.shape[1] > 1000:
            logger.warning(
                f"TabPFN works best with ≤1000 features. "
                f"You have {X.shape[1]} features. Consider PCA."
            )

        # Converter para numpy se necessário
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        # Fit
        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("TabPFN fitted successfully")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz probabilidades.

        Args:
            X: Features [N, D]

        Returns:
            Probabilidades [N, 2] para classes [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        logger.info(f"Predicting probabilities for {len(X)} samples")
        proba = self.model.predict_proba(X)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes binárias.

        Args:
            X: Features [N, D]

        Returns:
            Classes binárias [N]
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def save(self, path: Path):
        """
        Salva modelo usando pickle.

        NOTA: TabPFN salva os dados de treino, então o arquivo pode ser grande.

        Args:
            path: Caminho para salvar (ex: models/tabpfn_model.pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"TabPFN model saved to {path}")

    @staticmethod
    def load(path: Path) -> 'FiloPrioriTabPFN':
        """
        Carrega modelo salvo.

        Args:
            path: Caminho do modelo salvo

        Returns:
            Instância de FiloPrioriTabPFN
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"TabPFN model loaded from {path}")
        return model


def train_tabpfn(X_train: np.ndarray,
                 y_train: np.ndarray,
                 device: str = 'cpu',
                 n_estimators: int = 4) -> Tuple[FiloPrioriTabPFN, dict]:
    """
    Helper function para treinar TabPFN.

    Não há loop de épocas, early stopping, ou métricas de validação.
    TabPFN é "zero-shot" - já foi pré-treinado em milhares de datasets.

    Args:
        X_train: Features de treino [N, D]
        y_train: Labels de treino [N]
        device: 'cpu' ou 'cuda'
        n_estimators: Número de ensembles

    Returns:
        (model, info_dict)
    """
    logger.info("="*70)
    logger.info("TRAINING TABPFN (NO EPOCHS, PRE-TRAINED MODEL)")
    logger.info("="*70)

    # Criar e treinar
    model = FiloPrioriTabPFN(device=device, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Info
    info = {
        'n_train_samples': len(X_train),
        'n_features': X_train.shape[1],
        'device': device,
        'n_estimators': n_estimators,
        'class_distribution': {
            'n_positive': int(y_train.sum()),
            'n_negative': int(len(y_train) - y_train.sum()),
            'positive_ratio': float(y_train.mean())
        }
    }

    logger.info(f"Training complete. Samples: {info['n_train_samples']}, Features: {info['n_features']}")
    logger.info(f"Class distribution: {info['class_distribution']['n_positive']} positive, "
                f"{info['class_distribution']['n_negative']} negative "
                f"({info['class_distribution']['positive_ratio']*100:.2f}% positive)")

    return model, info
