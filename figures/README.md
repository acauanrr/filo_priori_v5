# Filo-Priori V5 - Diagrams

This directory contains Mermaid diagrams documenting the Filo-Priori V5 architecture, pipeline, and innovations.

## Diagrams Overview

### 1. `filo_priori_executive_summary.mmd` ⭐ **START HERE**
**Purpose**: High-level executive summary with all key information in one view

**Content**:
- V5 core architecture (BGE + SAINT + Calibration)
- Feature breakdown (1034D total)
- Training strategy
- Execution 006 results
- V4 vs V5 comparison table
- Next steps and current hypothesis
- Project status

**Best for**: Quick overview, presentations, stakeholder communication

---

### 2. `filo_priori_simplified_pipeline.mmd`
**Purpose**: Simplified end-to-end pipeline with innovations highlighted

**Content**:
- Input → Features → Model → Training → Calibration → Output
- Key innovations marked with 🌟
- Current challenges marked with ⚠️
- Clear visual flow

**Best for**: Understanding the complete pipeline flow

---

### 3. `filo_priori_v5_architecture.mmd`
**Purpose**: Detailed architecture breakdown

**Content**:
- Feature engineering details (semantic, commit, categorical, temporal)
- SAINT transformer architecture
- Training strategy (class imbalance handling, optimization)
- Probability calibration process
- Output metrics
- Legend of key innovations

**Best for**: Technical deep dive into architecture components

---

### 4. `filo_priori_v4_vs_v5.mmd`
**Purpose**: Side-by-side comparison of V4 and V5

**Content**:
- V4 architecture (baseline)
- V5 architecture (current)
- Key differences table
- Execution 006 results
- Visual upgrade flow

**Best for**: Understanding what changed from V4 to V5

---

### 5. `saint_transformer_detail.mmd`
**Purpose**: Deep dive into SAINT transformer architecture

**Content**:
- Input embedding layers
- SAINT block structure (4 layers)
  - Self-attention mechanism
  - Intersample attention mechanism (unique to SAINT)
  - Feed-forward network
- Classification head
- Model statistics (~69.8M parameters)
- Why SAINT for test prioritization
- Key innovation explanation

**Best for**: Understanding the SAINT transformer and intersample attention

---

### 6. `filo_priori_pipeline.mmd` (Original)
**Purpose**: Complete step-by-step pipeline (8 steps)

**Content**:
- Step 1: Load & Parse Commits
- Step 2: Build Text Semantic
- Step 3: Generate BGE Embeddings
- Step 4: Build Tabular Features
- Step 5: Split & Prepare Data
- Step 6: Train SAINT Transformer
- Step 7: Calibrate & Evaluate
- Step 8: Save Results

**Best for**: Detailed implementation understanding, code walkthroughs

---

## Key Concepts

### 🌟 V5 Innovations
1. **BGE-large embeddings**: 1024D (vs SBERT 768D)
2. **No PCA**: Full embedding dimension preserved
3. **SAINT Transformer**: Self-attention + intersample attention
4. **Full 1024D embedding**: In model (vs 128D reduction in V4)
5. **Probability calibration**: Isotonic regression
6. **Aggressive class imbalance handling**: pos_weight=15.0, 40% positive sampling
7. **69.8M parameters**: 279x larger than V4

### ⚠️ Current Challenges (Execution 006)
- APFD: 0.574 (target: 0.70) - **18% below target**
- AUPRC: 0.048 (target: 0.20) - **76% below target**
- Early overfitting: best epoch 2/30
- Weak discrimination: 1.34x (failures vs passes)

### 💡 Current Hypothesis
Model is **too large** (69.8M params) for **too few failures** (~1,654 train samples)
→ Overfits quickly (epoch 2), then degrades
→ Needs either: (a) more data, (b) smaller model, or (c) stronger regularization

## Viewing Diagrams

### Option 1: VS Code + Mermaid Extension
1. Install "Markdown Preview Mermaid Support" extension
2. Open any `.mmd` file
3. Right-click → "Open Preview"

### Option 2: Online Mermaid Editor
1. Go to https://mermaid.live/
2. Copy-paste content from `.mmd` file
3. View rendered diagram

### Option 3: GitHub
- GitHub automatically renders Mermaid diagrams in markdown files
- View in repository browser

## Recommended Reading Order

For **new team members**:
1. `filo_priori_executive_summary.mmd` - Get the big picture
2. `filo_priori_v4_vs_v5.mmd` - Understand what changed
3. `filo_priori_simplified_pipeline.mmd` - See the flow

For **technical deep dive**:
1. `saint_transformer_detail.mmd` - Understand SAINT architecture
2. `filo_priori_v5_architecture.mmd` - See full architecture
3. `filo_priori_pipeline.mmd` - Follow step-by-step implementation

For **debugging/improvement**:
1. `filo_priori_executive_summary.mmd` - See current results and hypothesis
2. `saint_transformer_detail.mmd` - Understand model complexity
3. Review execution_006 metrics in `filo_priori/results/execution_006/`

## File Sizes & Complexity

| Diagram | Lines | Complexity | Purpose |
|---------|-------|------------|---------|
| `filo_priori_executive_summary.mmd` | ~100 | High | Executive overview |
| `filo_priori_simplified_pipeline.mmd` | ~80 | Medium | Quick pipeline view |
| `filo_priori_v5_architecture.mmd` | ~120 | High | Detailed architecture |
| `filo_priori_v4_vs_v5.mmd` | ~100 | Medium | Comparison |
| `saint_transformer_detail.mmd` | ~140 | High | SAINT deep dive |
| `filo_priori_pipeline.mmd` | ~60 | Medium | Step-by-step flow |

## Updates

**Last Updated**: 2025-10-19
**Based on**: Execution 006 results
**Status**: 🔴 Under Development - Performance Below Target

All diagrams reflect the **actual implementation** as verified in:
- `filo_priori/scripts/core/run_experiment_server.py`
- `filo_priori/results/execution_006/`
- `config.yaml`
- Model code in `filo_priori/models/saint.py`

## Contributing

When updating diagrams:
1. Ensure consistency with actual code implementation
2. Verify metrics against latest execution results
3. Update this README if adding new diagrams
4. Use consistent color coding:
   - 🔴 Red (#e74c3c): Key innovations, critical issues
   - 🟢 Green (#27ae60): Good results, advantages
   - 🟡 Yellow (#f39c12): Warnings, attention needed
   - 🔵 Blue (#3498db): Information, comparisons
   - 🟣 Purple (#9b59b6): Statistics, metadata
