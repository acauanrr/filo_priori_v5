"""
Construção da coluna text_semantic para embeddings SBERT.

Combina:
- TE_Summary (limpo)
- TC_Steps (compactado)
- commit_text (do módulo 01)

Autor: Filo-Priori V5
Data: 2025-10-15
"""

import re
import pandas as pd
import numpy as np
from typing import Optional


def clean_text(text) -> str:
    """
    Limpa e normaliza texto.

    Remove:
    - HTML/markup
    - Múltiplos espaços
    - Caracteres especiais desnecessários
    """
    if pd.isna(text) or not text:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '[URL]', text)

    # Remove IDs longos (já estão em commit_text)
    text = re.sub(r'\b[A-Z0-9]{20,}\b', '[ID]', text)

    # Normaliza espaços
    text = re.sub(r'\s+', ' ', text)

    # Remove bullets/markers
    text = re.sub(r'^\s*[\-\*\•]\s*', '', text, flags=re.MULTILINE)

    return text.strip()


def compact_steps(text, max_chars: int = 800) -> str:
    """
    Compacta TC_Steps mantendo informação relevante.

    Formato: "Step 1: ... Step 2: ..."
    """
    if pd.isna(text) or not text:
        return ""

    text = clean_text(text)

    # Tenta identificar steps numerados
    steps = re.findall(r'(?:Step\s+\d+|^\d+[\.\)])\s*:?\s*([^\n]+)', text, re.MULTILINE | re.IGNORECASE)

    if steps:
        # Reconstrói com formato consistente
        formatted = []
        for i, step in enumerate(steps[:10], 1):  # Max 10 steps
            formatted.append(f"Step {i}: {step.strip()}")
        text = " ".join(formatted)

    # Trunca se necessário
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def build_text_semantic(row: pd.Series, max_length: int = 2048) -> str:
    """
    Constrói text_semantic combinando TE_Summary, TC_Steps e commit_text.

    Args:
        row: Linha do dataframe com colunas necessárias
        max_length: Tamanho máximo do texto final

    Returns:
        String formatada para embedding
    """
    parts = []

    # TE Summary
    if 'TE_Summary' in row.index and row['TE_Summary']:
        te_summary = clean_text(row['TE_Summary'])
        if te_summary:
            parts.append(f"[TE Summary] {te_summary}.")

    # TC Steps
    if 'TC_Steps' in row.index and row['TC_Steps']:
        tc_steps = compact_steps(row['TC_Steps'])
        if tc_steps:
            parts.append(f"[TC Steps] {tc_steps}.")

    # Commit text
    if 'commit_text' in row.index and row['commit_text']:
        commit_text = str(row['commit_text'])
        if commit_text and commit_text != "No commit info.":
            parts.append(commit_text)

    # Combina tudo
    text = "\n".join(parts)

    # Trunca se necessário
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # Fallback
    if not text.strip():
        text = "No information available."

    return text


def normalize_label(label) -> Optional[int]:
    """
    Normaliza o rótulo do teste.

    Returns:
        1: Fail
        0: Pass/Conditional Pass
        None: Blocked/Pending/Delete (remover do treino)
    """
    if pd.isna(label):
        return None

    label_str = str(label).lower().strip()

    # Falha
    if any(x in label_str for x in ['fail', 'failed', 'failure']):
        return 1

    # Sucesso
    if any(x in label_str for x in ['pass', 'passed', 'conditional pass']):
        return 0

    # Estados ambíguos - remover
    if any(x in label_str for x in ['blocked', 'pending', 'delete', 'na', 'skip']):
        return None

    # Default: considerar pass se não identificado
    return 0


def process_text_semantic(df: pd.DataFrame,
                          label_column: str = 'TE_Test_Result',
                          max_length: int = 2048) -> pd.DataFrame:
    """
    Processa dataframe completo para criar text_semantic e normalizar labels.

    Args:
        df: Dataframe com commit_text já gerado
        label_column: Nome da coluna de rótulos
        max_length: Tamanho máximo do text_semantic

    Returns:
        Dataframe com novas colunas: text_semantic, label_binary
    """
    print(f"Processing {len(df)} rows...")

    # Verifica colunas necessárias
    required = ['commit_text']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Run 01_parse_commit.py first.")

    # Cria text_semantic
    print("Building text_semantic...")
    df['text_semantic'] = df.apply(
        lambda row: build_text_semantic(row, max_length=max_length),
        axis=1
    )

    # Normaliza labels
    if label_column in df.columns:
        print(f"Normalizing labels from column '{label_column}'...")
        df['label_binary'] = df[label_column].apply(normalize_label)

        # Estatísticas
        n_total = len(df)
        n_fail = (df['label_binary'] == 1).sum()
        n_pass = (df['label_binary'] == 0).sum()
        n_removed = df['label_binary'].isna().sum()

        print("\n" + "="*70)
        print("LABEL STATISTICS")
        print("="*70)
        print(f"Total rows: {n_total}")
        print(f"Failures (1): {n_fail} ({n_fail/n_total*100:.2f}%)")
        print(f"Passes (0): {n_pass} ({n_pass/n_total*100:.2f}%)")
        print(f"Ambiguous (removed): {n_removed} ({n_removed/n_total*100:.2f}%)")
        if n_fail > 0:
            print(f"Imbalance ratio: {n_pass/n_fail:.1f}:1")

    # Estatísticas de text_semantic
    print("\n" + "="*70)
    print("TEXT_SEMANTIC STATISTICS")
    print("="*70)
    lengths = df['text_semantic'].str.len()
    print(f"Average length: {lengths.mean():.0f} chars")
    print(f"Median length: {lengths.median():.0f} chars")
    print(f"Max length: {lengths.max():.0f} chars")
    print(f"Truncated (>{max_length}): {(lengths > max_length).sum()} ({(lengths > max_length).mean()*100:.1f}%)")

    # Amostra
    print("\n" + "="*70)
    print("SAMPLE text_semantic (first non-empty):")
    print("="*70)
    for idx, text in df['text_semantic'].items():
        if text and text != "No information available.":
            print(text[:500] + "..." if len(text) > 500 else text)
            break

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 02_build_text_semantic.py <parsed_csv> [output_csv]")
        print("Example: python 02_build_text_semantic.py ../artifacts/train_parsed.csv ../artifacts/train_semantic.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('_parsed.csv', '_semantic.csv')

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")

    # Process
    result = process_text_semantic(df)

    # Remove rows with ambiguous labels if label_binary exists
    if 'label_binary' in result.columns:
        before = len(result)
        result = result[result['label_binary'].notna()].copy()
        after = len(result)
        print(f"\nRemoved {before - after} rows with ambiguous labels")

    # Save
    print(f"\nSaving to {output_path}...")
    result.to_csv(output_path, index=False)

    print(f"\nDone! Output shape: {result.shape}")
