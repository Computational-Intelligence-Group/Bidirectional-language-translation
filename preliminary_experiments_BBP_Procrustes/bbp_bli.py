"""
Bidirectional Backpropagation (B-BP) for Bilingual Lexicon Induction

Based on Adigun & Kosko (2020) B-BP framework, applied to cross-lingual
word embedding alignment as a replacement for Xing et al.'s orthogonal transform.

Key differences from Xing et al.:
- Trains BOTH directions jointly (forward: W, backward: W^T)
- Soft orthogonal penalty instead of hard SVD re-orthogonalization
- Joint loss: L = alpha * L_fwd + beta * L_bwd + gamma * ||W^TW - I||_F
- W is free to deviate from orthogonality if it helps

Usage:
    python bbp_bli.py \
        --src_emb wiki.en.vec \
        --tgt_emb wiki.es.vec \
        --dict_train dict_en_es_train.txt \
        --dict_test dict_en_es_test.txt \
        --init procrustes \
        --W_init W_procrustes.npy
"""

import numpy as np
import argparse
import time
from collections import defaultdict

# ---- PyTorch for automatic differentiation ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    exit(1)


# =============================================================================
# 1. LOADING EMBEDDINGS (same as xing_orthogonal.py)
# =============================================================================

def load_embeddings(filepath, max_vocab=200000):
    words = []
    vectors = []
    dim = None

    print(f"Loading embeddings from {filepath} ...")
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            tokens = line.rstrip().split(' ')
            if i == 0 and len(tokens) == 2:
                try:
                    int(tokens[0])
                    int(tokens[1])
                    print(f"  Detected header: {tokens[0]} words, {tokens[1]} dimensions")
                    continue
                except ValueError:
                    pass
            if len(tokens) < 3:
                continue
            word = tokens[0]
            try:
                vec = np.array(tokens[1:], dtype=np.float32)
            except ValueError:
                continue
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                continue
            words.append(word)
            vectors.append(vec)
            if len(words) >= max_vocab:
                break

    vectors = np.array(vectors, dtype=np.float32)
    word2idx = {w: i for i, w in enumerate(words)}
    print(f"  Loaded {len(words)} words, dimension={dim}")
    return words, vectors, word2idx


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms


def load_dictionary(filepath, src_word2idx, tgt_word2idx):
    src_indices = []
    tgt_indices = []
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            src_word, tgt_word = parts[0], parts[1]
            if src_word in src_word2idx and tgt_word in tgt_word2idx:
                src_indices.append(src_word2idx[src_word])
                tgt_indices.append(tgt_word2idx[tgt_word])
            else:
                skipped += 1
    print(f"  Loaded {len(src_indices)} pairs, skipped {skipped} (OOV)")
    return np.array(src_indices), np.array(tgt_indices)


# =============================================================================
# 2. B-BP MODEL
# =============================================================================

class BBP_Linear(nn.Module):
    """
    Bidirectional Backpropagation linear mapping for BLI.

    Forward direction:  y_hat = W @ x       (source -> target)
    Backward direction: x_hat = W^T @ z     (target -> source)

    Both directions share the SAME weight matrix W.
    W^T is not a separate parameter — it's the actual transpose of W.
    """

    def __init__(self, dim, W_init=None):
        super().__init__()

        if W_init is not None:
            # Initialize from Procrustes solution (warm start)
            self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32))
        else:
            # Random orthogonal initialization
            random_matrix = torch.randn(dim, dim)
            U, _, Vt = torch.linalg.svd(random_matrix)
            self.W = nn.Parameter(U @ Vt)

    def forward_pass(self, x):
        """Map source -> target: y_hat = W @ x (batched: X @ W^T)"""
        return x @ self.W.T

    def backward_pass(self, z):
        """Map target -> source: x_hat = W^T @ z (batched: Z @ W)"""
        return z @ self.W


# =============================================================================
# 3. LOSS FUNCTIONS
# =============================================================================

def cosine_distance_loss(predicted, target):
    """
    Mean cosine distance between predicted and target vectors.
    cosine_distance = 1 - cosine_similarity

    Both inputs should be (batch_size, dim).
    We normalize before computing to ensure we measure direction only.
    """
    pred_norm = predicted / (predicted.norm(dim=1, keepdim=True) + 1e-8)
    tgt_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity: dot product of normalized vectors
    cos_sim = (pred_norm * tgt_norm).sum(dim=1)  # (batch_size,)

    # Cosine distance = 1 - cosine_similarity
    cos_dist = 1.0 - cos_sim

    return cos_dist.mean()


def orthogonal_penalty(W):
    """
    Soft orthogonality regularizer: ||W^T W - I||_F

    This encourages W to stay near-orthogonal without forcing it.
    If gamma=0, W is completely free. If gamma is large, W is pushed
    toward orthogonality (approaching Xing's hard constraint).
    """
    dim = W.shape[0]
    WtW = W.T @ W
    I = torch.eye(dim, device=W.device, dtype=W.dtype)
    return torch.norm(WtW - I, p='fro')


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_bbp(model, X_train_np, Z_train_np,
              alpha=1.0, beta=1.0, gamma=0.01,
              lr=0.001, optimizer_type='adam',
              n_epochs=100, batch_size=512,
              verbose=True):
    """
    Train B-BP model with joint bidirectional loss.

    L = alpha * L_fwd + beta * L_bwd + gamma * ||W^TW - I||_F

    Parameters:
        model: BBP_Linear instance
        X_train_np: (n, d) normalized source training embeddings
        Z_train_np: (n, d) normalized target training embeddings
        alpha: weight on forward loss (source -> target)
        beta: weight on backward loss (target -> source)
        gamma: weight on orthogonal penalty
        lr: learning rate
        optimizer_type: 'adam' or 'sgd'
        n_epochs: number of training epochs
        batch_size: mini-batch size
    """
    device = next(model.parameters()).device

    # Convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    Z_train = torch.tensor(Z_train_np, dtype=torch.float32, device=device)

    n_train = X_train.shape[0]

    # Setup optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Training history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_fwd': [],
        'loss_bwd': [],
        'loss_orth': [],
        'orth_error': [],
    }

    for epoch in range(n_epochs):
        # Shuffle training data
        perm = torch.randperm(n_train, device=device)
        X_shuffled = X_train[perm]
        Z_shuffled = Z_train[perm]

        epoch_loss = 0.0
        epoch_loss_fwd = 0.0
        epoch_loss_bwd = 0.0
        epoch_loss_orth = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_shuffled[start:end]
            Z_batch = Z_shuffled[start:end]

            optimizer.zero_grad()

            # --- Forward pass: source -> target ---
            y_hat = model.forward_pass(X_batch)      # y_hat = W @ x
            L_fwd = cosine_distance_loss(y_hat, Z_batch)

            # --- Backward pass: target -> source ---
            x_hat = model.backward_pass(Z_batch)     # x_hat = W^T @ z
            L_bwd = cosine_distance_loss(x_hat, X_batch)

            # --- Orthogonal penalty ---
            L_orth = orthogonal_penalty(model.W)

            # --- Joint B-BP loss ---
            L_total = alpha * L_fwd + beta * L_bwd + gamma * L_orth

            # --- Backpropagate and update ---
            L_total.backward()
            optimizer.step()

            # NO re-orthogonalization here. That's the key difference from Xing.

            epoch_loss += L_total.item()
            epoch_loss_fwd += L_fwd.item()
            epoch_loss_bwd += L_bwd.item()
            epoch_loss_orth += L_orth.item()
            n_batches += 1

        # Epoch averages
        avg_loss = epoch_loss / n_batches
        avg_fwd = epoch_loss_fwd / n_batches
        avg_bwd = epoch_loss_bwd / n_batches
        avg_orth = epoch_loss_orth / n_batches

        # Check how orthogonal W currently is
        with torch.no_grad():
            W_np = model.W.detach().cpu().numpy()
            orth_err = np.linalg.norm(W_np.T @ W_np - np.eye(W_np.shape[0]))

        history['epoch'].append(epoch + 1)
        history['loss_total'].append(avg_loss)
        history['loss_fwd'].append(avg_fwd)
        history['loss_bwd'].append(avg_bwd)
        history['loss_orth'].append(avg_orth)
        history['orth_error'].append(orth_err)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Fwd: {avg_fwd:.6f} | "
                  f"Bwd: {avg_bwd:.6f} | "
                  f"Orth: {avg_orth:.4f} | "
                  f"||W^TW-I||: {orth_err:.6f}")

    return history


# =============================================================================
# 5. EVALUATION (numpy-based, same as xing_orthogonal.py)
# =============================================================================

def evaluate(W_np, src_vectors, tgt_vectors, src_indices, tgt_indices, direction='forward', k_values=[1, 5]):
    """
    Evaluate word translation accuracy.
    direction='forward':  use W    (src -> tgt)
    direction='backward': use W^T  (tgt -> src)
    """
    n_test = len(src_indices)

    if direction == 'forward':
        X_test = src_vectors[src_indices]
        mapped = X_test @ W_np.T                          # y_hat = W x
        mapped = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
        similarities = mapped @ tgt_vectors.T
        gold_indices = tgt_indices
    else:
        Z_test = tgt_vectors[tgt_indices]
        mapped = Z_test @ W_np                            # x_hat = W^T z
        mapped = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
        similarities = mapped @ src_vectors.T
        gold_indices = src_indices

    correct = {k: 0 for k in k_values}
    max_k = max(k_values)

    for i in range(n_test):
        sim_row = similarities[i]
        top_k_idx = np.argpartition(-sim_row, max_k)[:max_k]
        top_k_idx = top_k_idx[np.argsort(-sim_row[top_k_idx])]

        for k in k_values:
            if gold_indices[i] in top_k_idx[:k]:
                correct[k] += 1

    results = {}
    for k in k_values:
        acc = correct[k] / n_test * 100
        results[k] = acc
        print(f"    P@{k}: {acc:.2f}% ({correct[k]}/{n_test})")

    return results


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="B-BP for Bilingual Lexicon Induction")

    # Data arguments
    parser.add_argument('--src_emb', type=str, required=True)
    parser.add_argument('--tgt_emb', type=str, required=True)
    parser.add_argument('--dict_train', type=str, required=True)
    parser.add_argument('--dict_test', type=str, required=True)
    parser.add_argument('--max_vocab', type=int, default=200000)

    # Initialization
    parser.add_argument('--init', type=str, default='procrustes',
                        choices=['random', 'procrustes', 'file'],
                        help='How to initialize W')
    parser.add_argument('--W_init', type=str, default=None,
                        help='Path to .npy file for W initialization (used with --init file)')

    # B-BP hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight on forward loss (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight on backward loss (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Weight on orthogonal penalty (default: 0.01)')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Mini-batch size (default: 512)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    src_words, src_vectors, src_word2idx = load_embeddings(args.src_emb, args.max_vocab)
    tgt_words, tgt_vectors, tgt_word2idx = load_embeddings(args.tgt_emb, args.max_vocab)

    src_vectors_norm = normalize_vectors(src_vectors)
    tgt_vectors_norm = normalize_vectors(tgt_vectors)

    print("\nTraining dictionary:")
    train_src_idx, train_tgt_idx = load_dictionary(args.dict_train, src_word2idx, tgt_word2idx)
    print("Test dictionary:")
    test_src_idx, test_tgt_idx = load_dictionary(args.dict_test, src_word2idx, tgt_word2idx)

    X_train = src_vectors_norm[train_src_idx]
    Z_train = tgt_vectors_norm[train_tgt_idx]

    dim = X_train.shape[1]
    print(f"\nTraining pairs: {X_train.shape[0]}, Dim: {dim}")

    # -------------------------------------------------------------------------
    # Initialize W
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INITIALIZING W")
    print("=" * 60)

    if args.init == 'file' and args.W_init is not None:
        print(f"  Loading W from {args.W_init}")
        W_init = np.load(args.W_init)
        assert W_init.shape == (dim, dim), f"W shape mismatch: {W_init.shape} vs ({dim},{dim})"

    elif args.init == 'procrustes':
        print("  Computing Procrustes solution as warm start ...")
        M = Z_train.T @ X_train
        U, S, Vt = np.linalg.svd(M, full_matrices=True)
        W_init = U @ Vt
        orth_err = np.linalg.norm(W_init.T @ W_init - np.eye(dim))
        print(f"  Procrustes ||W^TW - I||_F = {orth_err:.6e}")

    else:  # random
        print("  Using random orthogonal initialization")
        W_init = None

    # -------------------------------------------------------------------------
    # Evaluate initialization before training
    # -------------------------------------------------------------------------
    if W_init is not None:
        print("\n  --- Pre-training evaluation (init W) ---")
        print("  Forward (EN -> ES):")
        evaluate(W_init, src_vectors_norm, tgt_vectors_norm,
                 test_src_idx, test_tgt_idx, direction='forward')
        print("  Backward (ES -> EN):")
        evaluate(W_init, src_vectors_norm, tgt_vectors_norm,
                 test_src_idx, test_tgt_idx, direction='backward')

    # -------------------------------------------------------------------------
    # Create model and train
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("B-BP TRAINING")
    print(f"  alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"  optimizer={args.optimizer}, lr={args.lr}")
    print(f"  epochs={args.n_epochs}, batch_size={args.batch_size}")
    print("=" * 60)

    model = BBP_Linear(dim, W_init=W_init).to(device)

    t0 = time.time()
    history = train_bbp(
        model, X_train, Z_train,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lr=args.lr,
        optimizer_type=args.optimizer,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        verbose=True
    )
    train_time = time.time() - t0
    print(f"\n  Total training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Extract final W and evaluate
    # -------------------------------------------------------------------------
    W_final = model.W.detach().cpu().numpy()

    # Check orthogonality of final W
    orth_err = np.linalg.norm(W_final.T @ W_final - np.eye(dim))
    print(f"\n  Final ||W^TW - I||_F = {orth_err:.6f}")

    # Check how much W changed from initialization
    if W_init is not None:
        delta = np.linalg.norm(W_final - W_init)
        print(f"  ||W_final - W_init||_F = {delta:.6f}")

    print("\n" + "=" * 60)
    print("B-BP EVALUATION")
    print("=" * 60)

    print("\n  Forward (EN -> ES):")
    fwd_results = evaluate(W_final, src_vectors_norm, tgt_vectors_norm,
                           test_src_idx, test_tgt_idx, direction='forward')

    print("\n  Backward (ES -> EN) via W^T:")
    bwd_results = evaluate(W_final, src_vectors_norm, tgt_vectors_norm,
                           test_src_idx, test_tgt_idx, direction='backward')

    # -------------------------------------------------------------------------
    # Bidirectional consistency metrics (novel to B-BP)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL METRICS")
    print("=" * 60)

    # Bidirectional Consistency (BC): average of forward and backward P@1
    bc = (fwd_results[1] + bwd_results[1]) / 2
    print(f"  Bidirectional Consistency (BC): {bc:.2f}%")
    print(f"    = (P@1_fwd + P@1_bwd) / 2 = ({fwd_results[1]:.2f} + {bwd_results[1]:.2f}) / 2")

    # Direction Gap (DG): absolute difference between forward and backward P@1
    dg = abs(fwd_results[1] - bwd_results[1])
    print(f"\n  Direction Gap (DG): {dg:.2f}%")
    print(f"    = |P@1_fwd - P@1_bwd| = |{fwd_results[1]:.2f} - {bwd_results[1]:.2f}|")

    # Round-trip consistency: map x -> y_hat -> x_reconstructed, measure similarity
    print("\n  Round-trip consistency (x -> Wx -> W^T(Wx)):")
    X_test = src_vectors_norm[test_src_idx]
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat = model.forward_pass(X_test_t)
        x_recon = model.backward_pass(y_hat)
    x_recon_np = x_recon.cpu().numpy()
    x_recon_np = x_recon_np / (np.linalg.norm(x_recon_np, axis=1, keepdims=True) + 1e-8)
    roundtrip_sim = np.mean(np.sum(X_test * x_recon_np, axis=1))
    print(f"    Mean cosine similarity(x, W^T W x): {roundtrip_sim:.6f}")
    print(f"    (1.0 = perfect round-trip, Xing achieves ~1.0 due to orthogonality)")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    np.save('W_bbp.npy', W_final)
    print(f"\nSaved: W_bbp.npy")

    # Save training history
    np.savez('bbp_history.npz', **{k: np.array(v) for k, v in history.items()})
    print(f"Saved: bbp_history.npz")

    # -------------------------------------------------------------------------
    # Summary comparison with Xing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Xing (Procrustes)':<20} {'B-BP':<20}")
    print("-" * 70)

    # Load Xing results if available
    try:
        W_xing = np.load('W_procrustes.npy')
        print("  (Loading Xing W from W_procrustes.npy for comparison)")
        print("\n  Evaluating Xing for comparison...")

        print("  Xing Forward:")
        xing_fwd = evaluate(W_xing, src_vectors_norm, tgt_vectors_norm,
                            test_src_idx, test_tgt_idx, direction='forward')
        print("  Xing Backward:")
        xing_bwd = evaluate(W_xing, src_vectors_norm, tgt_vectors_norm,
                            test_src_idx, test_tgt_idx, direction='backward')

        xing_bc = (xing_fwd[1] + xing_bwd[1]) / 2
        xing_dg = abs(xing_fwd[1] - xing_bwd[1])
        xing_orth = np.linalg.norm(W_xing.T @ W_xing - np.eye(dim))

        print(f"\n{'Metric':<30} {'Xing':<20} {'B-BP':<20}")
        print("-" * 70)
        print(f"{'P@1 Forward (EN->ES)':<30} {xing_fwd[1]:<20.2f} {fwd_results[1]:<20.2f}")
        print(f"{'P@5 Forward (EN->ES)':<30} {xing_fwd[5]:<20.2f} {fwd_results[5]:<20.2f}")
        print(f"{'P@1 Backward (ES->EN)':<30} {xing_bwd[1]:<20.2f} {bwd_results[1]:<20.2f}")
        print(f"{'P@5 Backward (ES->EN)':<30} {xing_bwd[5]:<20.2f} {bwd_results[5]:<20.2f}")
        print(f"{'BC (avg P@1)':<30} {xing_bc:<20.2f} {bc:<20.2f}")
        print(f"{'DG (|fwd-bwd| P@1)':<30} {xing_dg:<20.2f} {dg:<20.2f}")
        print(f"{'||W^TW - I||_F':<30} {xing_orth:<20.6f} {orth_err:<20.6f}")
        print(f"{'Round-trip sim':<30} {'~1.000000':<20} {roundtrip_sim:<20.6f}")

    except FileNotFoundError:
        print("  W_procrustes.npy not found. Run xing_orthogonal.py first for comparison.")
        print(f"\n{'P@1 Forward (EN->ES)':<30} {'N/A':<20} {fwd_results[1]:<20.2f}")
        print(f"{'P@5 Forward (EN->ES)':<30} {'N/A':<20} {fwd_results[5]:<20.2f}")
        print(f"{'P@1 Backward (ES->EN)':<30} {'N/A':<20} {bwd_results[1]:<20.2f}")
        print(f"{'P@5 Backward (ES->EN)':<30} {'N/A':<20} {bwd_results[5]:<20.2f}")


if __name__ == '__main__':
    main()
