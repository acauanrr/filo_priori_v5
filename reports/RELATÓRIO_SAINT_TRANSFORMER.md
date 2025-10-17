# Relatório Técnico: Modelo SAINT Transformer no Filo-Priori V5

**Autor:** Análise do Projeto Filo-Priori V5
**Data:** 2025-10-17
**Versão:** 1.0

---

## 📋 Sumário Executivo

Este relatório documenta em detalhes a arquitetura e funcionamento do **SAINT (Self-Attention and Intersample Attention Transformer)**, o modelo de Deep Learning utilizado no Filo-Priori V5 para classificar casos de teste e predizer falhas.

**Principais Descobertas:**
- SAINT é um transformer especializado para **dados tabulares** (não texto)
- Combina **self-attention** (dentro de cada amostra) + **intersample attention** (entre amostras)
- Modelo com **~1,86 milhões de parâmetros**
- Arquitetura: 6 camadas, 8 cabeças de atenção, embedding interno de 128D
- Treinado com **early stopping** e **cosine learning rate schedule**

---

## 🎯 1. O que é o SAINT?

### 1.1 Visão Geral

**SAINT** (Self-Attention and Intersample Attention Transformer) é uma arquitetura transformer criada especificamente para dados tabulares, proposta no paper:

> Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training" (NeurIPS 2021)

**Por que SAINT para dados tabulares?**

Transformers tradicionalmente foram criados para sequências (texto, áudio), onde a ordem importa. Em dados tabulares:
- **Não há ordem natural** entre features (coluna 1 não é "antes" da coluna 2)
- Features são **heterogêneas** (dimensões diferentes, escalas diferentes)
- Relações entre **amostras** também importam (não só entre features)

SAINT resolve esses problemas com duas inovações:
1. **Feature Embedding Layer**: Cada feature é projetada independentemente
2. **Intersample Attention**: Atenção entre diferentes amostras no batch

### 1.2 Comparação: SAINT vs MLP vs Transformer Tradicional

| Aspecto | MLP Tradicional | Transformer (BERT-like) | SAINT |
|---------|-----------------|-------------------------|-------|
| **Entrada** | Vetor concatenado | Sequência de tokens | Features independentes |
| **Atenção** | Nenhuma | Self-attention (sequência) | Self + Intersample |
| **Relações** | Não-lineares fixas | Entre tokens | Entre features + amostras |
| **Posicionalidade** | N/A | Essencial | Não utilizada |
| **Batch** | Independente | Independente | **Dependente (intersample)** |

**Vantagem chave do SAINT:** Aprende a comparar amostras durante treinamento, capturando padrões coletivos que MLPs não conseguem.

---

## 🏗️ 2. Arquitetura Completa do SAINT

### 2.1 Visão Geral em Camadas

```
Input: 1028 features (1024 BGE + 4 temporais)
       ↓
┌──────────────────────────────────────┐
│  1. EMBEDDING LAYER                  │
│  1028 features → 1028 embeddings 128D│
│  Total: 1028 × (1×128) = 131,584 params │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  2. SAINT BLOCK #1                   │
│  - Self-Attention (within sample)    │
│  - Intersample Attention (across)    │
│  - Feed-Forward MLP                  │
└──────────────────────────────────────┘
       ↓
     [... 4 blocos intermediários ...]
       ↓
┌──────────────────────────────────────┐
│  3. SAINT BLOCK #6                   │
│  (mesma estrutura)                   │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  4. POOLING LAYER                    │
│  Adaptive Average Pooling            │
│  [batch, 1028, 128] → [batch, 128]   │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  5. CLASSIFICATION HEAD              │
│  Linear(128 → 64) + GELU             │
│  Linear(64 → 1)                      │
└──────────────────────────────────────┘
       ↓
Output: Logit [batch, 1] → Sigmoid → Probability [0,1]
```

### 2.2 Contagem de Parâmetros

**Total: ~1,860,225 parâmetros**

Distribuição:
- **Embedding Layer:** ~131K parâmetros
- **6 SAINT Blocks:** ~1,700K parâmetros
- **Classification Head:** ~29K parâmetros

Cálculo detalhado por componente está na Seção 4.

---

## 🧩 3. Componentes Detalhados

### 3.1 Embedding Layer

**Objetivo:** Transformar cada feature numérica em um vetor denso de 128 dimensões.

#### 3.1.1 Arquitetura

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, num_continuous=1028, embedding_dim=128):
        # Cria 1028 Linear layers independentes
        self.continuous_embeddings = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_continuous)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)
```

**Por que Linear(1, 128) para cada feature?**
- Cada feature é tratada **independentemente**
- Aprende transformação específica para aquela dimensão
- Permite features com diferentes escalas/significados

#### 3.1.2 Forward Pass

```python
# Input: [batch_size, 1028]
x = torch.randn(16, 1028)

# Para cada feature i:
for i in range(1028):
    feat = x[:, i:i+1]           # [16, 1] - extrai feature i
    emb = embed_layer[i](feat)   # [16, 128] - projeta para 128D
    embeddings.append(emb)

# Stack todas as embeddings
embeddings = torch.stack(embeddings, dim=1)  # [16, 1028, 128]
embeddings = layer_norm(embeddings)          # Normaliza
```

**Output:** `[batch_size, 1028, 128]` - 1028 features, cada uma representada por 128 valores

#### 3.1.3 Exemplo Numérico

```
Input (1 amostra, primeiras 5 features):
x = [0.42, -0.15, 0.89, 0.31, -0.67, ...]
    └─┬──┘
      │
Feature 1 (0.42) → Linear_1 → [0.12, -0.34, 0.56, ..., 0.78]  (128 dims)
Feature 2 (-0.15) → Linear_2 → [-0.23, 0.45, -0.12, ..., 0.34] (128 dims)
Feature 3 (0.89) → Linear_3 → [0.67, -0.89, 0.23, ..., -0.45] (128 dims)
...

Output:
embeddings = [
  [0.12, -0.34, 0.56, ..., 0.78],  # Feature 1 embedding
  [-0.23, 0.45, -0.12, ..., 0.34], # Feature 2 embedding
  [0.67, -0.89, 0.23, ..., -0.45], # Feature 3 embedding
  ...
]  # Shape: [1028, 128]
```

---

### 3.2 Multi-Head Self-Attention

**Objetivo:** Capturar relações entre diferentes features **dentro da mesma amostra**.

#### 3.2.1 O que é Self-Attention?

Self-attention permite que cada feature "olhe" para todas as outras features e decida quais são relevantes.

**Exemplo intuitivo:**
- Feature "BGE_dim_50" (semântica de "WiFi") pode atender a "BGE_dim_120" (semântica de "rede")
- Feature "fail_count" pode atender a "avg_duration" (testes longos que falharam antes)

#### 3.2.2 Mecanismo Matemático

```python
# Input: [batch, seq_len=1028, emb_dim=128]
x = embeddings

# 1. Projeta para Q, K, V
Q = Linear_Q(x)  # [batch, 1028, 128] - Queries
K = Linear_K(x)  # [batch, 1028, 128] - Keys
V = Linear_V(x)  # [batch, 1028, 128] - Values

# 2. Multi-head: divide em 8 cabeças
# [batch, 1028, 128] → [batch, 8, 1028, 16]
Q = Q.view(batch, 1028, 8, 16).transpose(1, 2)
K = K.view(batch, 1028, 8, 16).transpose(1, 2)
V = V.view(batch, 1028, 8, 16).transpose(1, 2)

# 3. Calcula scores de atenção
# Q × K^T: quanto cada feature se relaciona com as outras
scores = (Q @ K.transpose(-2, -1)) / sqrt(16)  # [batch, 8, 1028, 1028]

# 4. Softmax: normaliza para [0,1]
attn_weights = softmax(scores, dim=-1)  # [batch, 8, 1028, 1028]

# 5. Aplica atenção aos valores
output = attn_weights @ V  # [batch, 8, 1028, 16]

# 6. Concatena cabeças
output = output.transpose(1, 2).reshape(batch, 1028, 128)
```

#### 3.2.3 Visualização da Matriz de Atenção

```
Attention Weights (1 cabeça, simplificado):
           Feature_1  Feature_2  Feature_3  ...  Feature_1028
Feature_1  [  0.15      0.32      0.05    ...     0.02     ]
Feature_2  [  0.08      0.41      0.23    ...     0.01     ]
Feature_3  [  0.22      0.19      0.35    ...     0.04     ]
...
Feature_1028[ 0.03      0.07      0.12    ...     0.38     ]

Interpretação:
- Feature_1 "atende muito" a Feature_2 (peso 0.32)
- Feature_2 "atende a si mesma" fortemente (0.41 - self-attention)
- Pesos somam 1.0 em cada linha (softmax)
```

#### 3.2.4 Por que Multi-Head (8 cabeças)?

Cada cabeça aprende um "tipo" diferente de relação:
- **Cabeça 1:** Relações semânticas (BGE features entre si)
- **Cabeça 2:** Relações temporais (histórico + semântica)
- **Cabeça 3:** Correlações numéricas (duração + falhas)
- ...

Dividir 128D em 8 × 16D permite especialização.

---

### 3.3 Intersample Attention (Inovação do SAINT)

**Objetivo:** Capturar relações entre diferentes amostras no batch.

#### 3.3.1 Diferença Fundamental

```
Self-Attention:
- Compara Feature_i com Feature_j na MESMA amostra
- Matriz de atenção: [1028 features × 1028 features]

Intersample Attention:
- Compara Amostra_i com Amostra_j para a MESMA feature
- Matriz de atenção: [batch_size × batch_size]
```

#### 3.3.2 Por que é útil?

**Exemplo prático:**

```
Batch de 16 testes:
- Test_1: "WiFi download" - fail_count=5
- Test_2: "WiFi upload"   - fail_count=0
- Test_3: "WiFi speed"    - fail_count=4
...
- Test_16: "Bluetooth"    - fail_count=0

Intersample Attention permite:
- Test_1 "perceber" que Test_3 é similar (ambos WiFi + falharam)
- Test_1 "ignorar" Test_16 (domínio diferente)
- Aprender padrões: "testes WiFi tendem a falhar juntos"
```

#### 3.3.3 Implementação

```python
# Input: [batch_size, 1028, 128]
x = embeddings

# 1. TRANSPOSE para [seq_len=1028, batch_size, 128]
x = x.transpose(0, 1)

# 2. Projeta Q, K, V (mesmo conceito)
Q = Linear_Q(x)  # [1028, batch, 128]
K = Linear_K(x)
V = Linear_V(x)

# 3. Multi-head reshape
Q = Q.view(1028, batch, 8, 16).permute(0, 2, 1, 3)  # [1028, 8, batch, 16]

# 4. Atenção ENTRE AMOSTRAS (dim=-2 agora é batch)
scores = (Q @ K.transpose(-2, -1)) / sqrt(16)  # [1028, 8, batch, batch]
attn_weights = softmax(scores, dim=-1)

# 5. Output: [1028, 8, batch, 16] → transpose → [batch, 1028, 128]
output = attn_weights @ V
output = output.permute(2, 0, 1, 3).reshape(batch, 1028, 128)
```

#### 3.3.4 Visualização da Matriz Intersample

```
Batch size = 4 (simplificado)
Feature_1 intersample attention:

           Sample_1  Sample_2  Sample_3  Sample_4
Sample_1   [  0.40     0.35     0.15     0.10   ]
Sample_2   [  0.30     0.45     0.20     0.05   ]
Sample_3   [  0.18     0.22     0.50     0.10   ]
Sample_4   [  0.05     0.08     0.12     0.75   ]

Interpretação:
- Sample_1 e Sample_2 se "atendem" mutuamente (0.35, 0.30)
- Sample_4 é mais isolada (0.75 self-attention, baixa com outras)
- Cada FEATURE tem sua própria matriz de atenção intersample
```

**Poder do Intersample:** O modelo aprende que "se outros testes similares falharam, eu também posso falhar" - generalização coletiva!

---

### 3.4 SAINT Block Completo

Um bloco SAINT combina:
1. Self-attention (features within sample)
2. Intersample attention (samples across batch)
3. Feed-Forward MLP
4. Residual connections + Layer Normalization

#### 3.4.1 Arquitetura do Bloco

```python
class SAINTBlock(nn.Module):
    def forward(self, x):  # x: [batch, 1028, 128]

        # 1. Self-Attention com residual
        x = x + self.self_attn(self.norm1(x))

        # 2. Intersample Attention com residual (se habilitado)
        if self.use_intersample:
            x = x + self.intersample_attn(self.norm2(x))

        # 3. Feed-Forward MLP com residual
        x = x + self.mlp(self.norm3(x))

        return x  # [batch, 1028, 128]
```

#### 3.4.2 Feed-Forward MLP

```python
# MLP de 2 camadas com GELU
self.mlp = nn.Sequential(
    nn.Linear(128, 512),    # Expande 4x
    nn.GELU(),              # Ativação suave
    nn.Dropout(0.1),
    nn.Linear(512, 128),    # Comprime de volta
    nn.Dropout(0.1)
)
```

**Por que 4x expansion?**
- Padrão em transformers (BERT usa 4×, GPT também)
- Permite representações mais ricas temporariamente
- Comprime de volta para manter dimensionalidade

#### 3.4.3 Residual Connections

```python
# Sem residual (ruim - gradientes somem)
x = self.mlp(self.norm(x))

# Com residual (bom - gradientes fluem)
x = x + self.mlp(self.norm(x))
```

**Benefícios:**
- **Gradient flow:** Gradientes chegam às camadas iniciais
- **Identity mapping:** Camada pode aprender "não fazer nada" se necessário
- **Estabilidade:** Evita explosão/desaparecimento de gradientes

---

### 3.5 Classification Head

Após 6 blocos SAINT, temos `[batch, 1028, 128]`. Precisamos de `[batch, 1]` para classificação.

#### 3.5.1 Pooling

```python
# Input: [batch, 1028, 128]

# Transpose para [batch, 128, 1028]
x = x.transpose(1, 2)

# Adaptive Average Pooling: média sobre features
x = AdaptiveAvgPool1d(1)(x)  # [batch, 128, 1]

# Remove dimensão extra
x = x.squeeze(-1)  # [batch, 128]
```

**Por que average pooling?**
- Agrega informação de todas as 1028 features
- Invariante à ordem (features não têm ordem natural)
- Reduz overfitting (vs. pegar só 1 feature)

#### 3.5.2 Classifier

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 64),      # Reduz dimensionalidade
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1)         # Saída binária
)

# Output: [batch, 1] - logit
logit = classifier(x)

# Converter para probabilidade
prob = torch.sigmoid(logit)  # [0, 1]
```

**Interpretação:**
- `logit > 0` → `prob > 0.5` → Predição = FAIL
- `logit < 0` → `prob < 0.5` → Predição = PASS

---

## 🔢 4. Matemática e Parâmetros

### 4.1 Cálculo de Parâmetros por Componente

#### Embedding Layer
```
Cada feature: Linear(1, 128) = 1 × 128 + 128 = 256 parâmetros
Total: 1028 × 256 = 263,168 parâmetros
LayerNorm(128): 128 × 2 = 256 parâmetros
Subtotal: 263,424 parâmetros
```

#### Multi-Head Self-Attention (1 módulo)
```
Q, K, V projections: Linear(128, 128 × 3) = 128 × 384 + 384 = 49,536
Output projection: Linear(128, 128) = 128 × 128 + 128 = 16,512
Subtotal: 66,048 parâmetros
```

#### Intersample Attention (1 módulo)
```
Mesma estrutura do self-attention: 66,048 parâmetros
```

#### Feed-Forward MLP (1 módulo)
```
Linear(128, 512): 128 × 512 + 512 = 66,048
Linear(512, 128): 512 × 128 + 128 = 65,664
Subtotal: 131,712 parâmetros
```

#### SAINT Block Completo (1 bloco)
```
Self-attention: 66,048
Intersample attention: 66,048
MLP: 131,712
3 × LayerNorm(128): 3 × 256 = 768
Subtotal por bloco: 264,576 parâmetros
```

#### 6 SAINT Blocks
```
6 × 264,576 = 1,587,456 parâmetros
```

#### Classification Head
```
LayerNorm(128): 256
Linear(128, 64): 128 × 64 + 64 = 8,256
Linear(64, 1): 64 × 1 + 1 = 65
Subtotal: 8,577 parâmetros
```

#### **TOTAL GERAL**
```
Embedding:      263,424
6 Blocks:     1,587,456
Classifier:       8,577
─────────────────────────
TOTAL:        1,859,457 parâmetros ≈ 1.86M
```

### 4.2 Memória e Computação

**Memória do modelo (FP32):**
```
1,860,000 params × 4 bytes = 7.44 MB
```

**Memória durante treinamento (batch_size=16):**
```
Forward pass:
- Embeddings: 16 × 1028 × 128 × 4 bytes = 8.4 MB
- 6 Blocks intermediários: ~50 MB
- Gradientes: ~7.4 MB
- Optimizer states (AdamW): ~15 MB

Total: ~80-100 MB por batch (estimativa)
```

**FLOPs por forward pass (estimativa):**
```
Embedding: 1028 × 128 × batch = 131K × batch
Self-attention (6 layers): 6 × (1028² × 128) × batch ≈ 800M × batch
MLP (6 layers): 6 × (128 × 512 × 2) × batch ≈ 0.8M × batch

Total: ~800M FLOPs por amostra
Para batch=16: ~12.8 GFLOPs
```

---

## ⚙️ 5. Configuração e Hiperparâmetros

### 5.1 Configuração do Modelo (config.yaml)

```yaml
saint:
  num_continuous: 1028          # Features de entrada
  num_categorical: 0            # Sem features categóricas
  categorical_dims: null
  embedding_dim: 128            # Dimensão interna do SAINT
  num_layers: 6                 # Número de SAINT blocks
  num_heads: 8                  # Cabeças de atenção
  mlp_hidden_dim: null          # Default: 4 × embedding_dim = 512
  dropout: 0.1                  # Dropout rate
  use_intersample: true         # Ativar intersample attention
```

### 5.2 Configuração de Treinamento

```yaml
training:
  num_epochs: 30
  learning_rate: 0.0005         # 5e-4 (conservador)
  weight_decay: 0.01            # L2 regularization
  batch_size: 16
  gradient_clip: 1.0            # Gradient clipping

  # Class imbalance
  pos_weight: 5.0               # BCE loss weight para classe positiva
  use_balanced_sampler: true    # WeightedRandomSampler
  target_positive_fraction: 0.20  # 20% positivos por batch

  # Learning rate schedule
  lr_schedule: "cosine"         # Cosine annealing
  warmup_epochs: 3              # Warmup steps
  min_lr_ratio: 0.01            # Min LR = 0.01 × initial LR

  # Early stopping
  patience: 8                   # Epochs sem melhoria
  monitor_metric: "val_auprc"   # Métrica a monitorar
  validation_split: 0.15        # 15% para validação

  # Regularization
  label_smoothing: 0.05         # Previne overconfidence
```

### 5.3 Hiperparâmetros Importantes

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `embedding_dim: 128` | 128D | Balance entre capacidade e eficiência |
| `num_layers: 6` | 6 blocos | Deep enough para padrões complexos |
| `num_heads: 8` | 8 cabeças | 128/8 = 16D por cabeça (divisível) |
| `dropout: 0.1` | 10% | Regularização moderada |
| `learning_rate: 5e-4` | 0.0005 | Conservador para estabilidade |
| `pos_weight: 5.0` | 5× | Compensa desbalanceamento (~5% fails) |
| `label_smoothing: 0.05` | 5% | Previne overfit em labels |
| `patience: 8` | 8 epochs | Permite explorar platôs |
| `monitor_metric: val_auprc` | AUPRC | Melhor que accuracy para dados desbalanceados |

---

## 🚀 6. Forward Pass Completo - Exemplo Passo a Passo

Vamos seguir **1 amostra** pelo modelo inteiro:

### Input

```python
# 1 teste com 1028 features (1024 BGE + 4 temporais)
x = torch.tensor([[
    # BGE embeddings (1024 dims)
    -0.0234,  0.0156, -0.0421, ...,  0.0423,  # (primeiras/últimas)
    # Temporal features (4 dims)
    1.0,      # last_run (estava no build anterior)
    0.6931,   # fail_count (log1p(1) = ln(2))
    2.3026,   # avg_duration (log1p(9) = ln(10))
    0.8333    # run_frequency
]])  # Shape: [1, 1028]
```

### Passo 1: Embedding Layer

```python
# Cada feature é projetada independentemente
embeddings = []
for i in range(1028):
    feat = x[:, i:i+1]              # [1, 1]
    emb = linear_layers[i](feat)    # [1, 128]
    embeddings.append(emb)

embeddings = torch.stack(embeddings, dim=1)  # [1, 1028, 128]
embeddings = layer_norm(embeddings)

# Agora temos 1028 vetores de 128D cada
```

### Passo 2: SAINT Block #1

```python
# Self-Attention
attn_output_1 = multi_head_self_attn(embeddings)
embeddings = embeddings + attn_output_1  # Residual

# Intersample Attention (batch_size=1, não faz muito aqui)
attn_output_2 = intersample_attn(embeddings)
embeddings = embeddings + attn_output_2  # Residual

# Feed-Forward MLP
mlp_output = mlp(embeddings)
embeddings = embeddings + mlp_output  # Residual

# Output: [1, 1028, 128]
```

### Passo 3-7: SAINT Blocks #2-6

```python
# Repete a mesma estrutura 5 vezes
# Cada bloco refina a representação
for block in saint_blocks[1:6]:
    embeddings = block(embeddings)

# Output: [1, 1028, 128]
```

### Passo 8: Pooling

```python
# Transpose: [1, 1028, 128] → [1, 128, 1028]
embeddings_t = embeddings.transpose(1, 2)

# Average pooling sobre features: [1, 128, 1028] → [1, 128, 1]
pooled = adaptive_avg_pool(embeddings_t)

# Squeeze: [1, 128, 1] → [1, 128]
pooled = pooled.squeeze(-1)
```

### Passo 9: Classification Head

```python
# Input: [1, 128]
x = pooled

# Layer 1: Linear(128, 64)
x = linear1(x)  # [1, 64]
x = gelu(x)
x = dropout(x)

# Layer 2: Linear(64, 1)
logit = linear2(x)  # [1, 1]

# Valor exemplo: logit = tensor([0.8234])
```

### Output Final

```python
# Converter logit para probabilidade
prob = torch.sigmoid(logit)  # [1, 1]

# Resultado: prob = tensor([0.6950])
# Interpretação: 69.5% de chance de FALHA
```

---

## 📊 7. Treinamento do SAINT

### 7.1 Loop de Treinamento

```python
def train_saint(model, train_loader, val_loader, config):
    # Setup
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    scheduler = CosineScheduleWithWarmup(optimizer, warmup_epochs=3)
    early_stopping = EarlyStopping(patience=8, monitor='val_auprc')

    for epoch in range(30):
        # TRAINING
        model.train()
        for batch in train_loader:
            x = batch['continuous']  # [16, 1028]
            y = batch['label']       # [16]

            # Label smoothing
            y_smooth = y * 0.95 + 0.5 * 0.05  # [0→0.025, 1→0.975]

            # Forward
            logits = model(x)        # [16, 1]
            loss = criterion(logits, y_smooth.unsqueeze(1))

            # Backward
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate(model, val_loader)

        # Early stopping
        if early_stopping(val_metrics['auprc']):
            break

        # Save best model
        if val_metrics['auprc'] > best_auprc:
            torch.save(model.state_dict(), 'best_model.pth')
```

### 7.2 Learning Rate Schedule

```python
def cosine_schedule_with_warmup(step, warmup=900, total=9000):
    # Warmup phase (primeiras 3 epochs)
    if step < warmup:
        return step / warmup * 5e-4

    # Cosine decay phase
    progress = (step - warmup) / (total - warmup)
    cosine_decay = 0.5 * (1 + cos(π * progress))
    min_lr = 5e-6  # 0.01 × 5e-4
    return min_lr + (5e-4 - min_lr) * cosine_decay

# Exemplo:
# Step 0: LR = 0.0
# Step 450 (meio do warmup): LR = 2.5e-4
# Step 900 (fim do warmup): LR = 5e-4
# Step 4950 (meio do treino): LR ≈ 2.5e-4
# Step 9000 (fim): LR = 5e-6
```

**Benefícios:**
- **Warmup:** Previne gradientes instáveis no início
- **Cosine decay:** Permite explorar melhor no final (LR baixa)
- **Min LR ratio:** Nunca para completamente (0.01 × original)

### 7.3 Estratégias contra Desbalanceamento

**Problema:** Dataset tem ~95% PASS, 5% FAIL

**Soluções aplicadas:**

1. **Pos_weight na loss:**
```python
criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
# Fail: peso 5.0
# Pass: peso 1.0
# Loss_fail contribui 5× mais que Loss_pass
```

2. **WeightedRandomSampler:**
```python
# Calcula pesos: inverso da frequência
weights = [1/count_pass if y==0 else 1/count_fail if y==1 for y in labels]
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))

# Resultado: ~20% FAIL em cada batch (vs 5% no dataset)
```

3. **AUPRC como métrica:**
```python
monitor_metric = 'val_auprc'  # Area Under Precision-Recall Curve
# AUPRC é mais sensível a classe minoritária que AUC-ROC
```

### 7.4 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=8, mode='max'):
        self.patience = patience
        self.best_score = -inf
        self.counter = 0

    def __call__(self, score):
        if score > self.best_score + 1e-4:  # Melhoria significativa
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Parar treinamento
        return False

# Exemplo de execução:
# Epoch 1: AUPRC=0.015 → best=0.015, counter=0
# Epoch 2: AUPRC=0.018 → best=0.018, counter=0 (melhorou!)
# Epoch 3: AUPRC=0.019 → best=0.019, counter=0
# Epoch 4: AUPRC=0.018 → counter=1 (piorou)
# Epoch 5: AUPRC=0.017 → counter=2
# ...
# Epoch 12: AUPRC=0.019 → counter=8 → STOP!
```

---

## 🎯 8. Inferência e Predição

### 8.1 Carregando Modelo Treinado

```python
# Criar modelo com mesma arquitetura
model = SAINT(
    num_continuous=1028,
    embedding_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    use_intersample=True
)

# Carregar pesos salvos
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Modo de inferência
```

### 8.2 Predição em Batch

```python
def predict(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():  # Desativa gradientes
        for batch in test_loader:
            x = batch['continuous'].to(device)  # [batch, 1028]

            # Forward pass
            logits = model(x)           # [batch, 1]
            probs = torch.sigmoid(logits)  # [batch, 1]

            all_probs.append(probs.cpu().numpy())

    # Concatena todos os batches
    all_probs = np.concatenate(all_probs)  # [N_test]

    return all_probs

# Uso:
test_probs = predict(model, test_loader)
# Output: array([0.123, 0.856, 0.034, ..., 0.678])
```

### 8.3 Priorização de Testes (APFD)

```python
# Após obter probabilidades para todos os testes
test_df['prob_fail'] = test_probs

# Ordenar por probabilidade decrescente
test_df_sorted = test_df.sort_values('prob_fail', ascending=False)

# Priorização:
# 1º: Teste com prob=0.92 (mais provável de falhar)
# 2º: Teste com prob=0.87
# 3º: Teste com prob=0.85
# ...
# Último: Teste com prob=0.01 (menos provável)

# Calcular APFD (Average Percentage of Faults Detected)
apfd = calculate_apfd(test_df_sorted['actual_result'])
print(f"APFD: {apfd:.4f}")  # Objetivo: ≥ 0.70
```

---

## 💡 9. Interpretabilidade do SAINT

### 9.1 Atenção como Explicação

Podemos extrair pesos de atenção para entender decisões:

```python
# Durante forward pass, salvar attention weights
model.eval()
attentions = []

def hook_fn(module, input, output):
    attentions.append(output[1])  # Salva pesos de atenção

# Registrar hook
model.transformer_blocks[0].self_attn.register_forward_hook(hook_fn)

# Forward pass
with torch.no_grad():
    logits = model(x)

# Analisar atenção
attn_weights = attentions[0]  # [batch, num_heads, seq_len, seq_len]

# Para amostra 0, cabeça 0
attn_sample = attn_weights[0, 0]  # [1028, 1028]

# Top-5 features que "Feature_0" atende
top5_indices = attn_sample[0].argsort(descending=True)[:5]
print(f"Feature 0 atende mais a features: {top5_indices}")
# Output: [0, 523, 87, 1015, 342]
# → Feature 0 olha para features 523, 87, 1015, 342
```

### 9.2 Feature Importance

```python
# Método 1: Permutation importance
def feature_importance(model, val_loader, baseline_metric):
    importances = []

    for feat_idx in range(1028):
        # Embaralha feature i
        for batch in val_loader:
            batch['continuous'][:, feat_idx] = batch['continuous'][:, feat_idx][torch.randperm(len(batch))]

        # Reavaliar
        new_metric = evaluate(model, val_loader)['auprc']

        # Importância = queda no desempenho
        importances.append(baseline_metric - new_metric)

    return np.array(importances)

# Features mais importantes:
# - BGE_dim_50: 0.023 (alta importância)
# - fail_count: 0.018
# - avg_duration: 0.012
# - BGE_dim_120: 0.009
```

---

## 🔍 10. Análise Comparativa

### 10.1 SAINT vs MLP

| Aspecto | MLP (V4) | SAINT (V5) |
|---------|----------|------------|
| **Arquitetura** | 3 camadas densas | 6 transformer blocks |
| **Parâmetros** | ~250K | ~1.86M |
| **Relações** | Aprende implicitamente | Explícitas (attention) |
| **Batch dependence** | Independente | Intersample attention |
| **Interpretabilidade** | Baixa | Média (attention weights) |
| **Overfitting** | Moderado | Maior risco (mais params) |
| **Tempo treino (GPU)** | ~5 min | ~30-60 min |

### 10.2 Vantagens do SAINT

1. **Modelagem de relações complexas:**
   - MLP: Linear → ReLU → Linear (não-linearidade fixa)
   - SAINT: Atenção adaptativa (aprende quais features combinar)

2. **Generalização entre amostras:**
   - MLP: Cada amostra é independente
   - SAINT: Intersample attention captura padrões coletivos

3. **Escalabilidade:**
   - MLP: Linear com número de features (772→1028 ok)
   - SAINT: Quadrático, mas paralelo (GPU eficiente)

### 10.3 Desvantagens do SAINT

1. **Custo computacional:**
   - Self-attention: O(1028² × 128) por camada
   - Intersample: O(batch² × 128) por feature

2. **Overfitting em datasets pequenos:**
   - 1.86M params precisam de muitos dados
   - Smoke test (100 builds) é insuficiente

3. **Complexidade de tuning:**
   - MLP: 3-4 hiperparâmetros
   - SAINT: 10+ hiperparâmetros

---

## 📚 11. Perguntas Frequentes (FAQ)

### Q1: Por que SAINT e não BERT/GPT?

**R:** BERT/GPT foram criados para sequências com ordem temporal (texto). Dados tabulares não têm ordem - a coluna 1 não vem "antes" da coluna 2. SAINT:
- Remove positional embeddings
- Trata cada feature independentemente
- Adiciona intersample attention (BERT não tem)

### Q2: Intersample attention funciona com batch_size=1?

**R:** Tecnicamente sim, mas não traz benefício (compara amostra consigo mesma). Batches maiores (16-32) são ideais para intersample aprender padrões entre amostras.

### Q3: Como SAINT lida com features heterogêneas?

**R:** A Embedding Layer projeta cada feature separadamente via Linear(1, 128). Assim:
- Feature 1 (BGE semantic) aprende sua projeção
- Feature 1028 (run_frequency) aprende outra projeção
- Não há "mistura" prematura - só depois no attention

### Q4: Posso desabilitar intersample attention?

**R:** Sim, mude `use_intersample: false` no config. Resultado:
- SAINT vira um transformer "normal" (só self-attention)
- Perde capacidade de generalização coletiva
- ~30% mais rápido (1 attention layer a menos por bloco)

### Q5: Como escolher `embedding_dim`?

**R:** Trade-off capacidade vs eficiência:
- **Menor (64D):** Mais rápido, pode underfit
- **Padrão (128D):** Balance (usado neste projeto)
- **Maior (256D):** Mais expressivo, risco de overfit

Regra: `embedding_dim × num_features ≈ tamanho do dataset / 10`
- 128 × 1028 = 131K
- Dataset: ~500K amostras → ok!

### Q6: SAINT precisa de normalização de features?

**R:** Sim! As 1028 features já vêm normalizadas:
- BGE embeddings: StandardScaler aplicado
- Temporais: log1p + escalas [0,1]

Sem normalização, self-attention dominaria features com maior magnitude.

### Q7: Por que 6 camadas e não 12 (como BERT)?

**R:** BERT processa sequências de 512 tokens complexos (palavras). Nossos "tokens" são features numéricas mais simples:
- 6 camadas: suficiente para capturar relações
- 12 camadas: risco de overfit (dados tabulares < texto)
- Teste empírico mostrou 6 ideal

---

## ✅ 12. Checklist de Validação

Para verificar se o SAINT está funcionando corretamente:

- [ ] **Arquitetura**
  - [ ] Modelo carrega sem erros
  - [ ] Parâmetros totais ≈ 1.86M
  - [ ] Input shape: [batch, 1028]
  - [ ] Output shape: [batch, 1]

- [ ] **Embedding Layer**
  - [ ] 1028 Linear layers criadas
  - [ ] Output: [batch, 1028, 128]
  - [ ] LayerNorm aplicado

- [ ] **Attention**
  - [ ] Self-attention funciona
  - [ ] Intersample attention ativa (se `use_intersample=true`)
  - [ ] Attention weights somam 1.0 (softmax)

- [ ] **Treinamento**
  - [ ] Loss diminui ao longo das épocas
  - [ ] Val AUPRC monitorado
  - [ ] Early stopping ativa após patience epochs
  - [ ] Melhor modelo salvo em `best_model.pth`

- [ ] **Inferência**
  - [ ] Modelo carrega de checkpoint
  - [ ] Probabilidades no range [0, 1]
  - [ ] Batch inference funciona

---

## 📈 13. Limitações e Melhorias Futuras

### 13.1 Limitações Atuais

1. **Tamanho do modelo:**
   - 1.86M params pode overfit em datasets pequenos
   - Smoke test (100 builds) insuficiente

2. **Custo computacional:**
   - O(N² M) onde N=1028 features, M=batch
   - Lento em CPU (~10× mais lento que MLP)

3. **Interpretabilidade limitada:**
   - Attention weights difíceis de visualizar (1028×1028)
   - Não é tão explícito quanto árvores de decisão

4. **Desbalanceamento de classes:**
   - Apesar de pos_weight, ainda luta com 95/5 split
   - AUPRC ~0.02-0.04 (baixo)

### 13.2 Melhorias Propostas

**Curto prazo:**
1. **Reduzir modelo para smoke test:**
   - `num_layers: 2` ao invés de 6
   - `embedding_dim: 64` ao invés de 128
   - ~300K params (vs 1.86M)

2. **Experimentar outras losses:**
   - Focal Loss (foca em exemplos difíceis)
   - Dice Loss (otimiza F1 diretamente)

**Médio prazo:**
3. **Feature selection:**
   - Reduzir de 1028 para top-500 features mais importantes
   - Acelera 4× (500² vs 1028²)

4. **Ensemble:**
   - Combinar SAINT + MLP + XGBoost
   - Votar por consenso

**Longo prazo:**
5. **Contrastive pre-training:**
   - Paper original do SAINT usa pre-training
   - Aprender representações antes de fine-tuning

6. **Arquitetura adaptativa:**
   - Attention esparsa (Top-K attention)
   - Pruning de camadas menos importantes

---

## 📝 14. Conclusão

O **SAINT Transformer** é uma arquitetura poderosa e inovadora para dados tabulares, que combina:

1. **Self-Attention:** Captura relações entre features dentro de cada amostra
2. **Intersample Attention:** Aprende padrões comparando diferentes amostras
3. **Deep Architecture:** 6 camadas para modelar complexidade

**Principais Insights:**

✅ **Forças:**
- Modelagem explícita de relações (vs. MLP implícito)
- Generalização coletiva via intersample attention
- Escalável para muitas features (1028)

⚠️ **Desafios:**
- Requer dataset grande (1.86M params)
- Computacionalmente intensivo (O(N² M))
- Prone a overfit em smoke tests

**Quando usar SAINT:**
- Dataset grande (>50K amostras)
- Muitas features (>100)
- Relações complexas entre amostras (testes similares falham juntos)
- GPU disponível

**Quando usar MLP:**
- Dataset pequeno (<10K amostras)
- Poucas features (<100)
- Prioridade: velocidade de treinamento
- CPU apenas

O Filo-Priori V5 escolheu SAINT apostando em seu potencial de generalização superior. Os resultados finais (APFD) determinarão se a complexidade adicional vale a pena.

---

**Documento gerado em:** 2025-10-17
**Versão do Filo-Priori:** V5 (SAINT + BGE)
**Referência:** Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training", NeurIPS 2021
**Implementação:** `filo_priori/models/saint.py` (456 linhas)
