# Bidirectional-Language-Translation

This project focuses on **Bilingual Lexicon Induction (BLI)** using **Deep Bidirectional Backpropagation (Deep B-BP)**. It implements a multi-layer nonlinear mapping with shared transposed weights to learn bidirectional representations between languages.

## Overview

The core methodology extends traditional linear mapping (like Procrustes) into a deep, bidirectional framework. Instead of learning two independent mappings (Source → Target and Target → Source), this approach uses a single set of shared weights that are used in their original form for the forward pass and transposed for the backward pass.

## Architecture (Deep B-BP)

The model is implemented in pure NumPy and features:
- **Forward (src -> tgt)**: `x -> W1 -> σ -> [W2 -> σ ->] W_out -> normalize -> y_hat`
- **Backward (tgt -> src)**: `z -> W_out^T -> σ -> [W2^T -> σ ->] W1^T -> normalize -> x_hat`
- **Shared Weights**: Weights are shared and transposed between directions.
- **Direction-specific Biases**: Separate biases for forward and backward passes.
- **Activation**: LeakyReLU.
- **Optimization**: Adam optimizer with a joint loss function incorporating forward loss, backward loss, and an orthogonality penalty.

## Key Files

- `deep_bbp_bli.py`: Implementation of the Deep Bidirectional Backpropagation model.
- `bbp_bli.py`: Baseline or variant implementation of Bidirectional Backpropagation.
- `xing_orthogonal.py`: Implementation of Xing's orthogonal transform (Procrustes).
- `setup_data.sh`: Script to download and prepare word embeddings (FastText) and dictionaries.
- `nllb_200_distilled_600M.ipynb`: Notebook for experiments with NLLB-200 models.

## Metrics

The project evaluates performance using:
- **P@k (Precision at k)**: Standard BLI metric.
- **BC (Bidirectional Capability)**: Average P@1 across both directions.
- **DG (Directional Gap)**: Absolute difference between forward and backward P@1.
- **Round-trip Similarity**: Measuring how well a word is reconstructed after a forward and backward pass.

## Usage

To train and evaluate the Deep B-BP model:

```bash
python deep_bbp_bli.py \
    --src_emb path/to/source.vec \
    --tgt_emb path/to/target.vec \
    --dict_train path/to/train_dict.txt \
    --dict_test path/to/test_dict.txt \
    --hidden_dims 512 \
    --n_epochs 200
```

## Results Summary

Initial experiments show significant improvements over standard Procrustes baselines in terms of both induction accuracy and bidirectional consistency. Detailed logs can be found in `results.txt`.
