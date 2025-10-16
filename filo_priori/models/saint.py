"""
SAINT (Self-Attention and Intersample Attention Transformer) for Filo-Priori V5.

SAINT is a transformer-based model designed for tabular data that combines:
1. Self-attention within each sample (standard transformer)
2. Intersample attention across samples in a batch (novel mechanism)

This implementation is adapted for test case prioritization with binary classification.

Reference: "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"
           (Somepalli et al., 2021)

Author: Filo-Priori V5 (Refactored)
Date: 2025-10-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for mixed continuous and categorical features.

    For tabular data with mostly continuous features (like our 1028D input),
    we use a linear projection to create embeddings.
    """

    def __init__(self,
                 num_continuous: int,
                 num_categorical: int = 0,
                 embedding_dim: int = 128,
                 categorical_dims: Optional[list] = None):
        """
        Args:
            num_continuous: Number of continuous features (1028 in our case)
            num_categorical: Number of categorical features (0 in our case)
            embedding_dim: Dimension of embeddings
            categorical_dims: List of cardinalities for each categorical feature
        """
        super().__init__()

        self.num_continuous = num_continuous
        self.num_categorical = num_categorical
        self.embedding_dim = embedding_dim

        # Continuous features: each feature gets projected to embedding_dim
        self.continuous_embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_continuous)
        ])

        # Categorical features (if any)
        if num_categorical > 0 and categorical_dims is not None:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(dim, embedding_dim) for dim in categorical_dims
            ])
        else:
            self.categorical_embeddings = None

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x_continuous: torch.Tensor, x_categorical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_continuous: [batch_size, num_continuous] continuous features
            x_categorical: [batch_size, num_categorical] categorical features (optional)

        Returns:
            embeddings: [batch_size, num_features, embedding_dim]
        """
        batch_size = x_continuous.size(0)

        # Embed continuous features
        continuous_embeds = []
        for i, embed_layer in enumerate(self.continuous_embeddings):
            # Extract i-th feature: [batch_size, 1]
            feat = x_continuous[:, i:i+1]
            # Embed: [batch_size, embedding_dim]
            emb = embed_layer(feat)
            continuous_embeds.append(emb)

        # Stack: [batch_size, num_continuous, embedding_dim]
        continuous_embeds = torch.stack(continuous_embeds, dim=1)

        # Combine with categorical if present
        if self.categorical_embeddings is not None and x_categorical is not None:
            categorical_embeds = []
            for i, embed_layer in enumerate(self.categorical_embeddings):
                # Extract i-th categorical feature: [batch_size]
                feat = x_categorical[:, i].long()
                # Embed: [batch_size, embedding_dim]
                emb = embed_layer(feat)
                categorical_embeds.append(emb)

            # Stack: [batch_size, num_categorical, embedding_dim]
            categorical_embeds = torch.stack(categorical_embeds, dim=1)

            # Concatenate along feature dimension
            all_embeds = torch.cat([continuous_embeds, categorical_embeds], dim=1)
        else:
            all_embeds = continuous_embeds

        # Apply layer norm
        all_embeds = self.layer_norm(all_embeds)

        return all_embeds


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention for within-sample feature interactions."""

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [batch_size, seq_len, embedding_dim * 3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embedding_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class IntersampleAttention(nn.Module):
    """
    Intersample attention: attention across different samples in a batch.

    This is the key innovation of SAINT. Instead of only attending to features
    within a sample, we also attend to features across different samples in the batch.
    """

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]

        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Transpose to [seq_len, batch_size, embedding_dim] for intersample attention
        x = x.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]

        # Compute Q, K, V
        qkv = self.qkv(x)  # [seq_len, batch_size, embedding_dim * 3]
        qkv = qkv.reshape(seq_len, batch_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, seq_len, num_heads, batch_size, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention (across batch dimension)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [seq_len, num_heads, batch_size, batch_size]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [seq_len, num_heads, batch_size, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [seq_len, batch_size, num_heads, head_dim]
        attn_output = attn_output.reshape(seq_len, batch_size, self.embedding_dim)

        # Transpose back to [batch_size, seq_len, embedding_dim]
        attn_output = attn_output.transpose(0, 1)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class SAINTBlock(nn.Module):
    """
    SAINT transformer block combining self-attention and intersample attention.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int = 8,
                 mlp_hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_intersample: bool = True):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            mlp_hidden_dim: Hidden dimension of MLP (default: 4 * embedding_dim)
            dropout: Dropout rate
            use_intersample: Whether to use intersample attention
        """
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = 4 * embedding_dim

        self.use_intersample = use_intersample

        # Self-attention (within-sample)
        self.self_attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(embedding_dim)

        # Intersample attention (across-sample)
        if use_intersample:
            self.intersample_attn = IntersampleAttention(embedding_dim, num_heads, dropout)
            self.intersample_attn_norm = nn.LayerNorm(embedding_dim)

        # Feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]

        Returns:
            output: [batch_size, seq_len, embedding_dim]
        """
        # Self-attention with residual connection
        x = x + self.self_attn(self.self_attn_norm(x))

        # Intersample attention with residual connection
        if self.use_intersample:
            x = x + self.intersample_attn(self.intersample_attn_norm(x))

        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))

        return x


class SAINT(nn.Module):
    """
    SAINT: Self-Attention and Intersample Attention Transformer for tabular data.

    Architecture:
    1. Embedding layer for continuous/categorical features
    2. Stack of SAINT transformer blocks
    3. Pooling and classification head
    """

    def __init__(self,
                 num_continuous: int,
                 num_categorical: int = 0,
                 categorical_dims: Optional[list] = None,
                 embedding_dim: int = 128,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_intersample: bool = True,
                 num_classes: int = 1):
        """
        Args:
            num_continuous: Number of continuous features (1028 in our case)
            num_categorical: Number of categorical features
            categorical_dims: List of cardinalities for categorical features
            embedding_dim: Dimension of feature embeddings
            num_layers: Number of SAINT transformer blocks
            num_heads: Number of attention heads
            mlp_hidden_dim: Hidden dimension of MLP in transformer blocks
            dropout: Dropout rate
            use_intersample: Whether to use intersample attention
            num_classes: Number of output classes (1 for binary classification with BCE)
        """
        super().__init__()

        self.num_continuous = num_continuous
        self.num_categorical = num_categorical
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_intersample = use_intersample

        # Embedding layer
        self.embedding = EmbeddingLayer(
            num_continuous=num_continuous,
            num_categorical=num_categorical,
            embedding_dim=embedding_dim,
            categorical_dims=categorical_dims
        )

        # SAINT transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SAINTBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                use_intersample=use_intersample
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool across features
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, x_continuous: torch.Tensor, x_categorical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_continuous: [batch_size, num_continuous] continuous features
            x_categorical: [batch_size, num_categorical] categorical features (optional)

        Returns:
            logits: [batch_size, num_classes] (or [batch_size, 1] for binary)
        """
        # Embed features: [batch_size, num_features, embedding_dim]
        x = self.embedding(x_continuous, x_categorical)

        # Apply SAINT transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Pool across features: [batch_size, embedding_dim, num_features] -> [batch_size, embedding_dim, 1]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, num_features]
        x = self.pooling(x)  # [batch_size, embedding_dim, 1]
        x = x.squeeze(-1)  # [batch_size, embedding_dim]

        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]

        return logits


def create_saint_model(config: dict) -> SAINT:
    """
    Factory function to create SAINT model from config.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        SAINT model instance
    """
    model_config = config.get('saint', {})

    model = SAINT(
        num_continuous=model_config.get('num_continuous', 1028),
        num_categorical=model_config.get('num_categorical', 0),
        categorical_dims=model_config.get('categorical_dims', None),
        embedding_dim=model_config.get('embedding_dim', 128),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        mlp_hidden_dim=model_config.get('mlp_hidden_dim', None),
        dropout=model_config.get('dropout', 0.1),
        use_intersample=model_config.get('use_intersample', True),
        num_classes=1  # Binary classification with BCE loss
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing SAINT model...")

    # Create dummy data
    batch_size = 16
    num_continuous = 1028

    x_continuous = torch.randn(batch_size, num_continuous)

    # Create model
    model = SAINT(
        num_continuous=num_continuous,
        embedding_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_intersample=True
    )

    # Forward pass
    logits = model(x_continuous)

    print(f"Input shape: {x_continuous.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with BCE loss
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)

    print(f"BCE loss: {loss.item():.4f}")
    print("\nSAINT model test passed!")
