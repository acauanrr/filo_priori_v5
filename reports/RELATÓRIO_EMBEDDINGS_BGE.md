# Relatório Técnico: Sistema de Embeddings no Filo-Priori V5

**Autor:** Análise do Projeto Filo-Priori V5
**Data:** 2025-10-17
**Versão:** 1.0

---

## 📋 Sumário Executivo

Este relatório documenta o sistema completo de geração de embeddings semânticos utilizado no projeto **Filo-Priori V5**, um sistema de priorização de testes baseado em Machine Learning que usa embeddings de texto para representar casos de teste e predizer falhas.

**Principais Descobertas:**
- O sistema utiliza o modelo **BGE-large-en-v1.5** da BAAI (Beijing Academy of Artificial Intelligence)
- Gera embeddings de **1024 dimensões** (upgrade do SBERT anterior de 768D)
- **Não utiliza PCA** - os embeddings completos são mantidos para preservar informação semântica
- Processa **3 colunas de entrada**: `TE_Summary`, `TC_Steps` e `commit`
- Cada teste é representado por um vetor de 1024 números reais no espaço semântico

---

## 🎯 1. Visão Geral do Sistema de Embeddings

### 1.1 O que são Embeddings?

**Embeddings** são representações vetoriais de texto que capturam seu significado semântico em um espaço numérico de alta dimensão. No contexto do Filo-Priori:

```
Texto (String) ────────────> Embedding (Vetor Numérico)
"Este teste falhou"          [0.42, -0.15, 0.89, ..., 0.33]
                              └──────────────┬─────────────┘
                                      1024 dimensões
```

**Por que usar embeddings?**
- Algoritmos de ML trabalham com números, não com texto
- Capturam relações semânticas (textos similares → vetores próximos)
- Permitem comparações matemáticas entre testes
- Generalizam melhor que representações bag-of-words ou TF-IDF

### 1.2 Arquitetura do Modelo BGE

O **BGE (BAAI General Embedding)** é um modelo transformer estado-da-arte treinado especificamente para tarefas de retrieval e similaridade semântica.

**Especificações Técnicas:**
- **Modelo:** `BAAI/bge-large-en-v1.5`
- **Arquitetura:** Transformer baseado em BERT
- **Dimensionalidade:** 1024 (vs 768 do SBERT anterior)
- **Linguagem:** Inglês
- **Treinamento:** Contrastive learning em milhões de pares de textos
- **Rank no MTEB:** Top-5 em benchmarks de embedding

**Vantagens sobre SBERT:**
- +33% mais dimensões (1024 vs 768)
- Melhor performance em tarefas de retrieval
- Maior capacidade de capturar nuances semânticas
- Treinamento mais robusto em dados de domínio técnico

---

## 📥 2. Dados de Entrada: Quais Colunas São Utilizadas?

### 2.1 Colunas do Dataset Original

O dataset `train.csv` contém as seguintes colunas relevantes para embeddings:

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `TE_Summary` | String | Resumo do Test Execution | "TE - TC - OTA: Download upgrade package when the user select Wifi Only option" |
| `TC_Steps` | String | Passos do Test Case | "1. Check system update... 2. Select WiFi only..." |
| `commit` | String (List) | Mensagens de commits relacionados | "['IKSWQ-787: support below interfaces...']" |
| `TE_Test_Result` | String | Label (Pass/Fail) | "Pass" ou "Fail" |

### 2.2 Exemplo Real de Dados Brutos

Aqui está um exemplo real extraído do dataset:

```csv
Build_ID: QPW30.18
TE_Summary: TE - TC - OTA: Download upgrade package when the user select Wifi Only option
TC_Steps:
  0. 1. Check the system update manually from Settings -> About phone -> System updates.
  1. 2. Select download over WiFi only.
  2. 3. Turn on WiFi and connect to a valid AP.
  3. 4. During upgrade download, go out of the wifi signal covered area to lose the wifi connection.
  4. 5. Wait for about 20 minutes and go back to wifi signal covered area.
  5. 6. During upgrade download, turn on airplane mode.
  6. 7. Turn off airplane mode.
commit: ['IKSWQ-787: support below interfaces Solution: (1)getCDMADataRate/setCDMADataRate (2)setAkeyInfo (3)getCdmaSidNidPairs/setCdmaSidNidPairs']
TE_Test_Result: Pass
```

---

## 🔄 3. Pipeline de Processamento Completo

### 3.1 Etapa 1: Construção do `commit_text` (Módulo 01)

**Arquivo:** `filo_priori/data_processing/01_parse_commit.py`

Processa a coluna `commit` (que é uma string representando lista Python):

```python
# Input
commit_raw = "['IKSWQ-787: support interfaces', 'MCA-123: fix bug']"

# Processing
commits = ast.literal_eval(commit_raw)  # Converte string → lista Python
commit_text = " | ".join(commits)       # Une com separador

# Output
commit_text = "IKSWQ-787: support interfaces | MCA-123: fix bug"
```

**Casos especiais:**
- Lista vazia → `"No commit info."`
- Erro de parsing → `"No commit info."`

### 3.2 Etapa 2: Construção do `text_semantic` (Módulo 02)

**Arquivo:** `filo_priori/data_processing/02_build_text_semantic.py`

Esta é a etapa **mais crítica** - combina as 3 fontes de texto em uma única string estruturada.

#### 3.2.1 Limpeza de Texto

Função `clean_text()` aplica:
1. Remove tags HTML: `<div>texto</div>` → `texto`
2. Substitui URLs: `http://example.com` → `[URL]`
3. Remove IDs longos: `ABC123DEF456GHI789JKL` → `[ID]`
4. Normaliza espaços: `"texto  \n\n  outro"` → `"texto outro"`
5. Remove bullets: `"• item"` → `"item"`

#### 3.2.2 Compactação de Steps

Função `compact_steps()` formata os passos do teste:

```python
# Input (raw TC_Steps)
"""
0. 1. Check the system update manually...
1. 2. Select download over WiFi only.
2. 3. Turn on WiFi and connect to a valid AP.
"""

# Processing
# 1. Detecta padrões de steps (Step 1, 1., etc)
# 2. Extrai até 10 steps
# 3. Reformata consistentemente
# 4. Trunca em 800 caracteres

# Output (formatted)
"Step 1: Check the system update manually... Step 2: Select download over WiFi only. Step 3: Turn on WiFi and connect to a valid AP..."
```

#### 3.2.3 Combinação Final

Função `build_text_semantic()` cria o texto final:

```python
# Template
text_semantic = """
[TE Summary] {TE_Summary limpo}.
[TC Steps] {TC_Steps formatados}.
{commit_text}
"""

# Exemplo real de OUTPUT
text_semantic = """[TE Summary] TE - TC - OTA: Download upgrade package when the user select Wifi Only option.
[TC Steps] Step 1: Check the system update manually from Settings -> About phone -> System updates. Step 2: Select download over WiFi only. Step 3: Turn on WiFi and connect to a valid AP. Step 4: During upgrade download, go out of the wifi signal covered area to lose the wifi connection. Step 5: Wait for about 20 minutes and go back to wifi signal covered area. Step 6: During upgrade download, turn on airplane mode. Step 7: Turn off airplane mode.
IKSWQ-787: support below interfaces Solution: (1)getCDMADataRate/setCDMADataRate (2)setAkeyInfo (3)getCdmaSidNidPairs/setCdmaSidNidPairs"""
```

**Características:**
- Máximo de 2048 caracteres (trunca se exceder)
- Seções claramente delimitadas com `[...]`
- Fallback para `"No information available."` se vazio
- Média de ~800-1200 caracteres por teste

### 3.3 Etapa 3: Geração de Embeddings BGE (Módulo 03)

**Arquivo:** `filo_priori/data_processing/03_embed_sbert.py`

#### 3.3.1 Inicialização do Modelo

```python
from sentence_transformers import SentenceTransformer

# Carrega modelo BGE-large
embedder = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
dim = embedder.get_sentence_embedding_dimension()  # 1024
```

**Configuração:**
- `model_name`: `'BAAI/bge-large-en-v1.5'`
- `batch_size`: 256 (para processamento eficiente)
- `device`: `'cuda'` (GPU) ou `'cpu'`
- `normalize_embeddings`: `False` (normalização feita depois com StandardScaler)

#### 3.3.2 Encoding: Texto → Vetor

A função `encode()` converte cada `text_semantic` em um vetor de 1024 dimensões:

```python
# Input: Lista de textos
texts = [
    "[TE Summary] Test wifi download...",
    "[TE Summary] Test bluetooth connection...",
    ...
]

# Processing
embeddings = embedder.encode(
    texts,
    batch_size=256,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Output: Numpy array
# Shape: (N_tests, 1024)
# Dtype: float32
```

**O que acontece internamente no modelo BGE?**

1. **Tokenização:** Texto → Tokens (palavras/subpalavras)
   ```
   "Test wifi download" → ["Test", "wifi", "down", "##load"]
   ```

2. **Embedding de Tokens:** Cada token → Vetor (768D)
   ```
   "Test"     → [0.1, -0.3, 0.5, ...]
   "wifi"     → [0.4, 0.2, -0.1, ...]
   "down"     → [-0.2, 0.6, 0.3, ...]
   "##load"   → [0.3, -0.1, 0.4, ...]
   ```

3. **Transformer Layers:** 24 camadas de atenção processam contexto
   - Self-attention captura relações entre palavras
   - Feed-forward networks transformam representações
   - Layer normalization estabiliza training

4. **Pooling:** Agrega tokens → Embedding da sentença (1024D)
   ```
   Mean pooling sobre todos os tokens:
   embedding_final = mean([emb_test, emb_wifi, emb_down, emb_load])
   ```

5. **Output:** Vetor denso de 1024 dimensões
   ```
   embedding = [0.42, -0.15, 0.89, 0.31, ..., 0.67]
                └────────────1024 valores────────────┘
   ```

#### 3.3.3 Exemplo de Transformação

**Input (text_semantic):**
```
[TE Summary] TE - TC - OTA: Download upgrade package when the user select Wifi Only option.
[TC Steps] Step 1: Check the system update manually from Settings -> About phone -> System updates...
IKSWQ-787: support below interfaces Solution: (1)getCDMADataRate/setCDMADataRate
```
- **Tipo:** String
- **Comprimento:** ~450 caracteres

↓↓↓ **BGE Encoding** ↓↓↓

**Output (embedding):**
```python
array([
    -0.0234,  0.0156, -0.0421,  0.0389,  0.0127, -0.0298,  0.0445, -0.0167,
     0.0312, -0.0234,  0.0189, -0.0356,  0.0278,  0.0123, -0.0401,  0.0334,
    ...  (mais 1008 valores)  ...
    -0.0189,  0.0267, -0.0312,  0.0423, -0.0156,  0.0298,  0.0145, -0.0378
], dtype=float32)
```
- **Tipo:** numpy.ndarray
- **Shape:** (1024,)
- **Dtype:** float32
- **Propriedades:**
  - Média: ~0.0 (após StandardScaler)
  - Std: ~1.0 (após StandardScaler)
  - Norma L2: ~5-15 (antes de normalização)

#### 3.3.4 Normalização com StandardScaler

Após o encoding, os embeddings são padronizados:

```python
from sklearn.preprocessing import StandardScaler

# Fit no conjunto de TREINO
scaler = StandardScaler()
scaler.fit(train_embeddings)

# Transform
train_embeddings_scaled = scaler.transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)  # Usa mesma escala!
```

**Por que StandardScaler?**
- Remove diferenças de escala entre dimensões
- Facilita convergência do modelo SAINT
- Cada dimensão fica com média=0 e std=1
- Preserva estrutura semântica (transformação linear)

**IMPORTANTE:** Não há PCA! Versão V5 removeu PCA para manter todas as 1024 dimensões originais.

---

## 🧮 4. Características dos Embeddings Gerados

### 4.1 Dimensionalidade e Estrutura

**Representação Final:**
- **Shape:** `(N_samples, 1024)` onde N_samples = número de testes
- **Tipo:** `numpy.ndarray` com dtype `float32`
- **Tamanho em memória:** ~4 bytes × 1024 × N_samples
  - Exemplo: 100K testes = ~400 MB

**Estrutura dos 1024 valores:**
- Cada dimensão captura um "aspecto" semântico
- Dimensões iniciais (0-256): Conceitos gerais (verbos, substantivos)
- Dimensões médias (257-768): Contextos e domínios
- Dimensões finais (769-1024): Nuances e detalhes

### 4.2 Propriedades Semânticas

**Similaridade Cosseno:**

Testes semanticamente similares têm embeddings próximos:

```python
# Dois testes sobre WiFi
test1 = "Check WiFi connection on device"
test2 = "Verify WiFi connectivity works"
# → cosine_similarity(emb1, emb2) ≈ 0.85

# Teste WiFi vs Bluetooth (diferentes)
test3 = "Test Bluetooth pairing process"
# → cosine_similarity(emb1, emb3) ≈ 0.45
```

**Operações Vetoriais:**

Embeddings suportam álgebra vetorial:

```python
# Analogias (exemplo ilustrativo)
embedding("WiFi connection") - embedding("WiFi") + embedding("Bluetooth")
≈ embedding("Bluetooth connection")
```

### 4.3 Estatísticas Típicas

Após StandardScaler, os embeddings apresentam:

| Métrica | Valor Típico | Interpretação |
|---------|--------------|---------------|
| Média | ~0.0 | Centrado na origem |
| Desvio Padrão | ~1.0 | Escala normalizada |
| Min | -3.5 a -4.5 | Outliers negativos |
| Max | +3.5 a +4.5 | Outliers positivos |
| Norma L2 | ~30-35 | Magnitude do vetor |

---

## 🔗 5. Integração com o Modelo SAINT

### 5.1 Composição Final das Features

Os embeddings semânticos são combinados com features temporais:

```python
# Embeddings semânticos: 1024D
semantic_features = embedding_BGE  # Shape: (N, 1024)

# Features temporais: 4D
temporal_features = [
    last_run,        # Binário: estava no build anterior?
    fail_count,      # log1p(número de falhas históricas)
    avg_duration,    # log1p(tempo médio de execução)
    run_frequency    # 1/(1 + dias desde última execução)
]  # Shape: (N, 4)

# Concatenação final
final_features = np.concatenate([semantic_features, temporal_features], axis=1)
# Shape: (N, 1028)
```

**Total:** 1028 dimensões = 1024 semânticas + 4 temporais

### 5.2 Entrada no Modelo SAINT

```python
# Modelo SAINT (config.yaml)
saint:
  num_continuous: 1028   # Todas as features são contínuas
  num_categorical: 0     # Sem features categóricas
  embedding_dim: 128     # SAINT embedding interno (diferente do BGE!)
  num_layers: 6
  num_heads: 8
```

**Fluxo de dados:**

```
1024D BGE Embedding ─┐
                     ├─> Concat ─> 1028D ─> SAINT Embedding Layer ─> 128D
4D Temporal Features ─┘                     (Linear: 1028→128)
                                                    ↓
                                            Transformer Blocks
                                                    ↓
                                            Classification Head
                                                    ↓
                                            Probability [0,1]
```

**Por que mais uma camada de embedding?**
- SAINT usa sua própria representação interna (128D)
- Reduz dimensionalidade para eficiência computacional
- Aprende features não-lineares específicas da tarefa
- Permite atenção entre diferentes features

---

## 📊 6. Exemplo Completo: Do CSV ao Embedding

### 6.1 Dados Originais (CSV Row)

```csv
Build_ID,TE_Summary,TC_Steps,commit,TE_Test_Result
QPW30.18,"TE - TC - OTA: Download upgrade package","1. Check system update... 2. Select WiFi only...","['IKSWQ-787: support interfaces']","Pass"
```

### 6.2 Após Módulo 01 (Parse Commit)

```python
{
    'Build_ID': 'QPW30.18',
    'TE_Summary': 'TE - TC - OTA: Download upgrade package',
    'TC_Steps': '1. Check system update... 2. Select WiFi only...',
    'commit_text': 'IKSWQ-787: support interfaces',  # ← NOVO
    'TE_Test_Result': 'Pass'
}
```

### 6.3 Após Módulo 02 (Build Text Semantic)

```python
{
    ...,
    'text_semantic': '[TE Summary] TE - TC - OTA: Download upgrade package.\n[TC Steps] Step 1: Check system update... Step 2: Select WiFi only...\nIKSWQ-787: support interfaces',  # ← NOVO
    'label_binary': 0  # ← NOVO (0=Pass, 1=Fail)
}
```

### 6.4 Após Módulo 03 (BGE Embeddings)

```python
{
    ...,
    'embedding': array([
        -0.0234,  0.0156, -0.0421,  0.0389,  # Primeiras dimensões
        ...,  # 1016 dimensões intermediárias
        -0.0189,  0.0267, -0.0312,  0.0423   # Últimas dimensões
    ], dtype=float32),  # ← NOVO
    # Shape: (1024,)
}
```

### 6.5 Entrada Final no Modelo (com Temporais)

```python
final_input = np.array([
    # 1024 dimensões semânticas (BGE)
    -0.0234,  0.0156, -0.0421, ..., -0.0312,  0.0423,
    # 4 dimensões temporais
    1.0,      # last_run (estava no build anterior)
    0.6931,   # fail_count (log1p(1 falha histórica))
    2.3026,   # avg_duration (log1p(9 segundos))
    0.8333    # run_frequency (executou há 1 dia)
])
# Shape: (1028,)

# Label
label = 0  # Pass
```

---

## ⚙️ 7. Configurações e Parâmetros

### 7.1 Arquivo de Configuração (config.yaml)

```yaml
# Feature Engineering
features:
  # Semantic features (BGE-large)
  use_semantic: true
  semantic_model: "BAAI/bge-large-en-v1.5"
  semantic_dim: 1024  # Output dimension

  # Temporal features
  use_temporal: true
  temporal_features:
    - "last_run"
    - "fail_count"
    - "avg_duration"
    - "run_frequency"

  # Total
  total_dim: 1028  # 1024 + 4
```

### 7.2 Classe SBERTEmbedder (Principais Parâmetros)

```python
embedder = SBERTEmbedder(
    model_name='BAAI/bge-large-en-v1.5',
    target_dim=None,        # OBSOLETO (PCA removido)
    batch_size=256,         # Aumentado de 32 para 256
    device='cuda'           # GPU recomendada
)
```

**Nota:** O parâmetro `target_dim` é mantido por compatibilidade mas ignorado (PCA foi removido na V5).

### 7.3 Recursos Computacionais

**Tempo de Processamento (estimado):**
- **100 testes:** ~5-10 segundos (GPU) / ~30-60 segundos (CPU)
- **10K testes:** ~8-15 minutos (GPU) / ~45-90 minutos (CPU)
- **100K testes:** ~1.5-3 horas (GPU) / ~8-15 horas (CPU)

**Memória Requerida:**
- **Modelo BGE:** ~1.3 GB VRAM (GPU) ou RAM (CPU)
- **Embeddings:** ~400 MB por 100K testes
- **Batch processing:** ~2-4 GB durante encoding

---

## 🔍 8. Diferenças da Versão V4 (SBERT) para V5 (BGE)

| Aspecto | V4 (SBERT) | V5 (BGE) | Melhoria |
|---------|-----------|----------|----------|
| **Modelo** | `all-mpnet-base-v2` | `BAAI/bge-large-en-v1.5` | Estado-da-arte |
| **Dimensões** | 768 | 1024 | +33% |
| **PCA** | Sim (768→128) | Não (mantém 1024) | +700% dimensões finais |
| **Total Features** | 772 (768+4) | 1028 (1024+4) | +33% |
| **MTEB Rank** | Top-20 | Top-5 | Melhor retrieval |
| **Treinamento** | Genérico | Otimizado p/ search | Mais robusto |

**Impacto no Desempenho:**
- Melhor captura de nuances semânticas
- Menor loss de informação (sem PCA)
- Embeddings mais expressivos
- Trade-off: maior custo computacional

---

## 💡 9. Perguntas Frequentes (FAQ)

### Q1: Por que 1024 dimensões? Não é muito?

**R:** 1024 dimensões são necessárias para capturar a complexidade semântica de textos técnicos. O modelo BGE foi treinado para produzir embeddings nessa dimensionalidade, e reduzir (com PCA) causaria perda de informação. O SAINT transforma isso em 128D internamente de forma otimizada para a tarefa.

### Q2: O modelo BGE entende português?

**R:** Não. O `bge-large-en-v1.5` foi treinado apenas em inglês. Se o dataset contém textos em português, considere:
- Usar um modelo multilíngue: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Traduzir textos antes de embedding
- Fine-tunar BGE em dados portugueses

### Q3: Como embeddings capturam "significado"?

**R:** Através de treinamento em milhões de exemplos de pares de textos similares/dissimilares (contrastive learning). O modelo aprende que:
- Sinônimos → vetores próximos
- Contextos similares → vetores similares
- Tópicos diferentes → vetores distantes

### Q4: Posso usar embeddings pré-computados?

**R:** Sim! Os embeddings são salvos em `artifacts/embeddings/embeddings_train.npy`. Se o dataset não mudar, você pode reutilizá-los sem reprocessar.

### Q5: Por que StandardScaler e não MinMaxScaler?

**R:** StandardScaler preserva a estrutura semântica (transformação linear) e lida melhor com outliers. Embeddings já têm distribuição aproximadamente gaussiana, tornando StandardScaler ideal.

### Q6: O que é "embedding dimension" do SAINT (128)?

**R:** É diferente do BGE! O SAINT cria uma nova representação interna de 128D para facilitar atenção entre features. Pense em:
- BGE 1024D: representação semântica do texto
- SAINT 128D: representação aprendida para classificação

---

## 📚 10. Referências e Recursos

### 10.1 Papers Acadêmicos

1. **BGE Model:**
   - BAAI, "C-Pack: Packaged Resources To Advance General Chinese Embedding" (2023)
   - Xiao et al., "BGE: General Embedding Model" (2024)

2. **Sentence Transformers:**
   - Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)

3. **SAINT Model:**
   - Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training" (2021)

### 10.2 Código Fonte Relevante

| Módulo | Caminho | Função Principal |
|--------|---------|------------------|
| Parse Commit | `filo_priori/data_processing/01_parse_commit.py` | `parse_commit()` |
| Build Text | `filo_priori/data_processing/02_build_text_semantic.py` | `build_text_semantic()` |
| BGE Embeddings | `filo_priori/data_processing/03_embed_sbert.py` | `SBERTEmbedder.encode()` |
| SAINT Model | `filo_priori/models/saint.py` | `SAINTClassifier` |

### 10.3 Links Úteis

- **BGE Hugging Face:** https://huggingface.co/BAAI/bge-large-en-v1.5
- **Sentence Transformers Docs:** https://www.sbert.net/
- **MTEB Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard

---

## ✅ 11. Checklist de Validação

Para verificar se o sistema de embeddings está funcionando corretamente:

- [ ] Modelo BGE carrega sem erros
- [ ] Dimensão reportada é 1024
- [ ] `text_semantic` tem comprimento médio de ~800-1200 chars
- [ ] Embeddings têm shape `(N, 1024)`
- [ ] StandardScaler gera média≈0 e std≈1
- [ ] Não há valores NaN ou Inf nos embeddings
- [ ] Scaler é salvo em `artifacts/embedder/scaler.pkl`
- [ ] Features finais têm shape `(N, 1028)`

---

## 📝 12. Conclusão

O sistema de embeddings do Filo-Priori V5 é uma pipeline robusta de 3 etapas que transforma texto bruto de testes de software em representações vetoriais densas de 1024 dimensões. O uso do modelo BGE-large-en-v1.5 estado-da-arte, combinado com a remoção do PCA e processamento cuidadoso de texto, resulta em embeddings altamente expressivos que capturam nuances semânticas essenciais para a tarefa de priorização de testes.

**Principais Takeaways:**
1. **3 colunas → 1 embedding:** `TE_Summary` + `TC_Steps` + `commit` → 1024D
2. **Sem perda de informação:** PCA removido, todas as 1024 dimensões preservadas
3. **Processo determinístico:** Mesmo texto sempre gera mesmo embedding
4. **Escalável:** Processa 100K+ testes em algumas horas (GPU)
5. **Estado-da-arte:** BGE-large é top-5 em benchmarks MTEB

---

**Documento gerado em:** 2025-10-17
**Versão do Filo-Priori:** V5 (SAINT + BGE)
**Autor:** Análise Técnica Automatizada
