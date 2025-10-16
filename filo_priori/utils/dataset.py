"""Dataset com balanceamento classe-ciente."""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np


class TabularDataset(Dataset):
    """Dataset para features contínuas + categóricas."""

    def __init__(self, continuous, categorical, labels):
        self.continuous = torch.FloatTensor(continuous)
        self.categorical = torch.LongTensor(categorical) if categorical.shape[1] > 0 else None
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'continuous': self.continuous[idx],
            'label': self.labels[idx]
        }
        if self.categorical is not None:
            item['categorical'] = self.categorical[idx]
        return item


def create_balanced_sampler(labels, target_positive_fraction=0.25):
    """Cria sampler balanceado por classe."""
    labels = np.array(labels)
    pos_count = float(labels.sum())
    neg_count = float(len(labels) - pos_count)

    if pos_count == 0 or neg_count == 0:
        return None

    # Pesos para atingir proporção alvo
    pos_weight = target_positive_fraction / pos_count
    neg_weight = (1 - target_positive_fraction) / neg_count

    weights = np.where(labels > 0, pos_weight, neg_weight)
    weights = torch.DoubleTensor(weights)

    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
