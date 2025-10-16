"""
Feature engineering - Combina embeddings + features tabulares.

Autor: Filo-Priori V5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple
import pickle


class FeatureBuilder:
    """Constrói features numéricas e categóricas."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.categorical_cols = []
        self.numerical_cols = []

    def fit_transform_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit e transform features (TRAIN).

        Returns:
            (continuous_features, categorical_features)
        """
        # Numéricas
        numerical = []

        # Contagens de commit
        for col in ['commit_n_msgs', 'commit_n_apis', 'commit_n_issues',
                    'commit_n_modules', 'commit_n_packages', 'commit_n_flags', 'commit_n_errors']:
            if col in df.columns:
                self.numerical_cols.append(col)
                numerical.append(df[col].fillna(0).values.reshape(-1, 1))

        # Concat numéricas
        if numerical:
            num_array = np.concatenate(numerical, axis=1)
            # Padronizar
            scaler = StandardScaler()
            num_array = scaler.fit_transform(num_array)
            self.scalers['numerical'] = scaler
        else:
            num_array = np.zeros((len(df), 0))

        # Categóricas
        categorical = []
        cat_cols = ['CR_Resolution', 'CR_Component_Name', 'CR_Type']

        for col in cat_cols:
            if col in df.columns:
                self.categorical_cols.append(col)
                le = LabelEncoder()
                # Trata NaN
                values = df[col].fillna('MISSING').astype(str)
                encoded = le.fit_transform(values)
                categorical.append(encoded.reshape(-1, 1))
                self.label_encoders[col] = le

        if categorical:
            cat_array = np.concatenate(categorical, axis=1)
        else:
            cat_array = np.zeros((len(df), 0), dtype=int)

        # Continuous: embeddings + numéricas
        continuous = np.concatenate([embeddings, num_array], axis=1)

        print(f"Features shape - Continuous: {continuous.shape}, Categorical: {cat_array.shape}")
        return continuous, cat_array

    def transform_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform features (VAL/TEST)."""
        # Numéricas
        numerical = []
        for col in self.numerical_cols:
            numerical.append(df[col].fillna(0).values.reshape(-1, 1))

        if numerical:
            num_array = np.concatenate(numerical, axis=1)
            num_array = self.scalers['numerical'].transform(num_array)
        else:
            num_array = np.zeros((len(df), 0))

        # Categóricas
        categorical = []
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            values = df[col].fillna('MISSING').astype(str)
            # Handle unseen categories
            encoded = []
            for v in values:
                try:
                    encoded.append(le.transform([v])[0])
                except ValueError:
                    encoded.append(0)  # Unknown -> 0
            categorical.append(np.array(encoded).reshape(-1, 1))

        if categorical:
            cat_array = np.concatenate(categorical, axis=1)
        else:
            cat_array = np.zeros((len(df), 0), dtype=int)

        continuous = np.concatenate([embeddings, num_array], axis=1)
        return continuous, cat_array

    def save(self, path):
        """Salva encoders e scalers."""
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scalers': self.scalers,
                'categorical_cols': self.categorical_cols,
                'numerical_cols': self.numerical_cols
            }, f)

    def load(self, path):
        """Carrega encoders e scalers."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['label_encoders']
            self.scalers = data['scalers']
            self.categorical_cols = data['categorical_cols']
            self.numerical_cols = data['numerical_cols']
