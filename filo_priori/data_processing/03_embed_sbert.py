"""
Geração de embeddings usando BGE-large-en-v1.5 (1024D).

IMPORTANTE: PCA foi removido! BGE-large gera embeddings de 1024 dimensões
que são usados diretamente sem redução de dimensionalidade.

Autor: Filo-Priori V5 (Refactored)
Data: 2025-10-15
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


class SBERTEmbedder:
    """
    Classe para gerar e processar embeddings usando BGE-large-en-v1.5.

    NOTA: target_dim é mantido para compatibilidade retroativa mas é ignorado.
    BGE-large gera embeddings de 1024D que são usados diretamente.
    """

    def __init__(self,
                 model_name: str = 'BAAI/bge-large-en-v1.5',
                 target_dim: Optional[int] = None,  # Mantido por compatibilidade, mas ignorado
                 batch_size: int = 32,
                 device: str = 'cuda'):
        """
        Args:
            model_name: Nome do modelo (default: BAAI/bge-large-en-v1.5)
            target_dim: OBSOLETO - mantido por compatibilidade, mas ignorado
            batch_size: Tamanho do batch para encoding
            device: 'cuda' ou 'cpu'
        """
        self.model_name = model_name
        self.target_dim = None  # Sempre None - PCA removido
        self.batch_size = batch_size
        self.device = device

        if target_dim is not None:
            print(f"WARNING: target_dim={target_dim} is ignored. BGE-large uses native 1024D embeddings.")

        print(f"Loading BGE model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {embedding_dim}")

        if embedding_dim != 1024:
            print(f"WARNING: Expected 1024D for BGE-large, got {embedding_dim}D")

        self.scaler = None

    def encode(self, texts: list, show_progress: bool = True) -> np.ndarray:
        """
        Encoda textos em embeddings de 1024D.

        Args:
            texts: Lista de strings
            show_progress: Mostrar barra de progresso

        Returns:
            Array numpy [N, 1024] com embeddings
        """
        print(f"Encoding {len(texts)} texts with BGE-large...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # Vamos normalizar depois com StandardScaler
            show_progress_bar=show_progress
        )

        print(f"Encoded shape: {embeddings.shape}")
        return embeddings

    def fit_scaler(self, embeddings: np.ndarray):
        """
        Ajusta StandardScaler nos embeddings de treino.

        Args:
            embeddings: Array [N, D] de embeddings
        """
        print("Fitting StandardScaler...")
        self.scaler = StandardScaler()
        self.scaler.fit(embeddings)
        print(f"Scaler fitted. Mean: {self.scaler.mean_[:5]}, Std: {self.scaler.scale_[:5]}")

    def transform_scaler(self, embeddings: np.ndarray) -> np.ndarray:
        """Aplica padronização."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")

        print("Applying StandardScaler...")
        return self.scaler.transform(embeddings)

    def save_artifacts(self, output_dir: Path):
        """
        Salva Scaler (PCA removido).

        Args:
            output_dir: Diretório para salvar artefatos
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.scaler is not None:
            scaler_path = output_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {scaler_path}")

    def load_artifacts(self, artifacts_dir: Path):
        """
        Carrega Scaler salvo (PCA removido).

        Args:
            artifacts_dir: Diretório com artefatos
        """
        artifacts_dir = Path(artifacts_dir)

        scaler_path = artifacts_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")


def process_embeddings_train(df: pd.DataFrame,
                             text_column: str = 'text_semantic',
                             target_dim: Optional[int] = None,
                             output_dir: Optional[Path] = None) -> Tuple[np.ndarray, SBERTEmbedder]:
    """
    Processa embeddings para conjunto de TREINO usando BGE-large (1024D).

    NOTA: target_dim é ignorado - BGE-large sempre produz 1024D.

    Args:
        df: Dataframe com text_column
        text_column: Nome da coluna com textos
        target_dim: OBSOLETO - ignorado
        output_dir: Diretório para salvar artefatos

    Returns:
        (embeddings_scaled, embedder)
    """
    print("="*70)
    print("PROCESSING TRAIN EMBEDDINGS WITH BGE-LARGE")
    print("="*70)

    # Criar embedder (target_dim é ignorado internamente)
    embedder = SBERTEmbedder(target_dim=target_dim)

    # Encode - produz 1024D diretamente
    texts = df[text_column].tolist()
    embeddings = embedder.encode(texts)

    # Scaler (PCA removido)
    embedder.fit_scaler(embeddings)
    embeddings_scaled = embedder.transform_scaler(embeddings)

    # Salvar artefatos
    if output_dir is not None:
        embedder.save_artifacts(output_dir)

        # Salvar embeddings
        emb_path = Path(output_dir) / 'embeddings_train.npy'
        np.save(emb_path, embeddings_scaled)
        print(f"Embeddings saved to {emb_path}")

    print(f"\nFinal embedding shape: {embeddings_scaled.shape} (expected: [N, 1024])")
    return embeddings_scaled, embedder


def process_embeddings_test(df: pd.DataFrame,
                            embedder: SBERTEmbedder,
                            text_column: str = 'text_semantic',
                            output_path: Optional[Path] = None) -> np.ndarray:
    """
    Processa embeddings para conjunto de TESTE/VALIDAÇÃO.

    Usa Scaler já ajustado no treino (PCA removido).

    Args:
        df: Dataframe com text_column
        embedder: SBERTEmbedder já treinado
        text_column: Nome da coluna com textos
        output_path: Path para salvar embeddings

    Returns:
        embeddings_scaled
    """
    print("="*70)
    print("PROCESSING TEST/VAL EMBEDDINGS WITH BGE-LARGE")
    print("="*70)

    # Encode - produz 1024D diretamente
    texts = df[text_column].tolist()
    embeddings = embedder.encode(texts)

    # Scaler (PCA removido)
    embeddings_scaled = embedder.transform_scaler(embeddings)

    # Salvar
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings_scaled)
        print(f"Embeddings saved to {output_path}")

    print(f"\nFinal embedding shape: {embeddings_scaled.shape} (expected: [N, 1024])")
    return embeddings_scaled


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 03_embed_sbert.py <semantic_csv> [target_dim] [output_dir]")
        print("Example: python 03_embed_sbert.py ../artifacts/train_semantic.csv 128 ../artifacts/embeddings/")
        print("\ntarget_dim: Dimensão alvo para PCA (default: None, usa 384D original)")
        sys.exit(1)

    input_path = sys.argv[1]
    target_dim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('../artifacts/embeddings/')

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Shape: {df.shape}")
    print(f"Target dimension: {target_dim if target_dim else 'None (384D original)'}")

    # Process
    embeddings, embedder = process_embeddings_train(
        df,
        target_dim=target_dim,
        output_dir=output_dir
    )

    print("\n" + "="*70)
    print("EMBEDDING STATISTICS")
    print("="*70)
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"Min: {embeddings.min():.6f}")
    print(f"Max: {embeddings.max():.6f}")

    print("\nDone!")
