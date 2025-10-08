# Hyperparameter Tuning Documentation

## Overview

This document details the comprehensive hyperparameter exploration conducted for the T5-small finance chatbot, demonstrating systematic optimization to achieve ≥10% improvement over baseline performance.

## Methodology

### 1. Baseline Establishment

**Zero-Shot Evaluation**: Before fine-tuning, we evaluate the pretrained T5-small model on our finance validation set to establish a performance baseline.

\`\`\`bash
python scripts/02_train_t5_tf.py --baseline --track
\`\`\`

This provides the performance floor that fine-tuning must exceed.

### 2. Hyperparameter Search Space

We systematically explore 8 key hyperparameters:

| Hyperparameter | Range | Purpose |
|----------------|-------|---------|
| **Learning Rate** | [1e-4, 3e-4, 5e-4] | Controls optimization step size; too high causes instability, too low slows convergence |
| **Batch Size** | [8, 16, 32] | Affects gradient estimation quality and memory usage |
| **Epochs** | [5, 8] | Training duration; more epochs risk overfitting |
| **Label Smoothing** | [0.0, 0.1, 0.2] | Regularization technique preventing overconfident predictions |
| **Warmup Ratio** | [0.0, 0.05, 0.1] | Gradual learning rate increase for stable training |
| **Dropout** | [0.1, 0.2] | Prevents overfitting by randomly dropping neurons |
| **Weight Decay** | [0.0, 0.01] | L2 regularization for weight magnitude control |
| **Gradient Clipping** | [0.5, 1.0, 2.0] | Prevents exploding gradients |

### 3. Systematic Exploration

We test 7 predefined configurations covering different training strategies:

1. **Baseline Configuration**: Standard hyperparameters
2. **Higher Learning Rate**: Faster convergence test
3. **Larger Batch Size**: Gradient estimation quality
4. **Strong Regularization**: Overfitting prevention
5. **Lower LR + Longer Training**: Conservative approach
6. **Aggressive Training**: Maximum performance push
7. **Conservative Approach**: Stability-focused

### 4. Automated Search

Run the complete hyperparameter search:

\`\`\`bash
python scripts/05_hyperparameter_search.py
\`\`\`

This automatically:
- Evaluates baseline
- Trains all 7 configurations
- Tracks results in `experiments/runs.csv`
- Generates comparison report
- Identifies best configuration

## Evaluation Metrics

### Primary Metrics

1. **Perplexity**: Measures model confidence (lower is better)
   - Formula: `exp(cross_entropy_loss)`
   - Interpretable as "effective vocabulary size" for next token prediction

2. **Validation Loss**: Cross-entropy loss on validation set

3. **Improvement Percentage**: 
   \`\`\`
   improvement = ((baseline_ppl - finetuned_ppl) / baseline_ppl) × 100%
   \`\`\`

### Success Criteria

- **Target**: ≥10% improvement over baseline perplexity
- **Validation**: Performance on held-out test set
- **Generalization**: Consistent performance across validation batches

## Results Interpretation

### Expected Baseline Performance
- **Perplexity**: ~3.5-4.0 (zero-shot T5-small)
- **Loss**: ~1.2-1.4

### Target Fine-tuned Performance
- **Perplexity**: ≤3.15 (≥10% improvement)
- **Loss**: ≤1.15

### Hyperparameter Impact Analysis

**Learning Rate**:
- Too low (1e-5): Slow convergence, may not reach optimum
- Optimal (3e-4): Balanced convergence and stability
- Too high (1e-3): Training instability, divergence risk

**Batch Size**:
- Small (8): Noisy gradients, better generalization, slower training
- Medium (16): Good balance
- Large (32): Stable gradients, faster training, may overfit

**Label Smoothing**:
- None (0.0): Sharp predictions, may overfit
- Light (0.1): Balanced regularization
- Heavy (0.2): Strong regularization, may underfit

**Warmup**:
- None (0.0): Risk of early instability
- Light (0.05): Smooth start
- Heavy (0.1): Very gradual, may slow initial progress

## Experiment Tracking

All experiments are logged in `experiments/runs.csv` with:
- Run ID and timestamp
- All hyperparameter values
- Validation metrics
- Improvement percentage
- Configuration notes

### View Results

\`\`\`bash
# Print summary
python scripts/02_train_t5_tf.py --summary

# View detailed comparison
cat experiments/comparison_table.csv

# Check best configuration
cat experiments/best_run.json
\`\`\`

## Best Practices

1. **Always evaluate baseline first** to establish comparison point
2. **Track all experiments** for reproducibility
3. **Use validation set** for hyperparameter selection
4. **Test on held-out test set** for final evaluation
5. **Document configuration rationale** in notes field
6. **Monitor for overfitting** via train/val loss divergence
7. **Use early stopping** to prevent overtraining
8. **Save best model** based on validation performance

## Reproducing Results

### Quick Start (Best Configuration)

\`\`\`bash
# 1. Evaluate baseline
python scripts/02_train_t5_tf.py --baseline --track

# 2. Train with best hyperparameters (from search results)
python scripts/02_train_t5_tf.py \
  --lr 3e-4 \
  --batch_size 16 \
  --epochs 5 \
  --label_smoothing 0.1 \
  --warmup_ratio 0.05 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --track \
  --notes "Best configuration"

# 3. View results
python scripts/02_train_t5_tf.py --summary
\`\`\`

### Full Hyperparameter Search

\`\`\`bash
# Run complete automated search (7 configurations)
python scripts/05_hyperparameter_search.py
\`\`\`

This takes approximately 2-4 hours depending on hardware.

## Troubleshooting

**Issue**: Training loss not decreasing
- **Solution**: Increase learning rate or reduce regularization

**Issue**: Validation loss increasing (overfitting)
- **Solution**: Increase dropout, label smoothing, or reduce epochs

**Issue**: Training unstable/diverging
- **Solution**: Decrease learning rate, increase warmup, reduce gradient clipping threshold

**Issue**: Slow convergence
- **Solution**: Increase learning rate or batch size

## References

- T5 Paper: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Label Smoothing: Szegedy et al., "Rethinking the Inception Architecture"
- Learning Rate Warmup: Goyal et al., "Accurate, Large Minibatch SGD"
