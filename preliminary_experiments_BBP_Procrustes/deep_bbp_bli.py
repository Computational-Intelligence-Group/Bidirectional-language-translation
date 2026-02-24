"""
Deep Bidirectional Backpropagation (Deep B-BP) for Bilingual Lexicon Induction
Variant 3: Multi-layer nonlinear mapping with shared transposed weights.
Pure NumPy — no PyTorch required.

Architecture:
  Forward (src -> tgt):  x -> W1 -> σ -> [W2 -> σ ->] W_out -> normalize -> y_hat
  Backward (tgt -> src): z -> W_out^T -> σ -> [W2^T -> σ ->] W1^T -> normalize -> x_hat

  Weight matrices are SHARED (transposed) between directions.
  Biases are direction-specific: b_k for forward, c_k for backward.

Usage:
    python deep_bbp_bli.py \
        --src_emb wiki.en.vec --tgt_emb wiki.fi.vec \
        --dict_train dict_en_fi_train.txt --dict_test dict_en_fi_test.txt \
        --hidden_dims 512 --n_epochs 200
"""

import numpy as np
import argparse
import time


# =============================================================================
# 1. DATA LOADING
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
    return vectors / np.maximum(norms, 1e-8)


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
# 2. ACTIVATION FUNCTIONS
# =============================================================================

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


# =============================================================================
# 3. ADAM OPTIMIZER (handles multiple parameter arrays)
# =============================================================================

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.states = {}
        self.t = 0

    def step(self, name, param, grad):
        """Update a named parameter given its gradient."""
        if name not in self.states:
            self.states[name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
            }
        # t is shared across all params (incremented once per batch)
        s = self.states[name]
        s['m'] = self.beta1 * s['m'] + (1 - self.beta1) * grad
        s['v'] = self.beta2 * s['v'] + (1 - self.beta2) * (grad ** 2)
        m_hat = s['m'] / (1 - self.beta1 ** self.t)
        v_hat = s['v'] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def increment_t(self):
        self.t += 1


# =============================================================================
# 4. DEEP B-BP MODEL
# =============================================================================

class DeepBBP:
    """
    Deep Bidirectional Backpropagation model.

    Layer dims example: [300, 512, 300] means:
        W0: (512, 300) - maps 300 -> 512
        W1: (300, 512) - maps 512 -> 300

    Forward:  x -> W0.T -> σ -> W1.T -> normalize
    Backward: z -> W1   -> σ -> W0   -> normalize

    Shared weights, separate biases per direction.
    """

    def __init__(self, layer_dims):
        """
        layer_dims: list of dimensions, e.g. [300, 512, 300] or [300, 512, 512, 300]
                    First = input dim (source), Last = output dim (target)
        """
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        # Initialize weight matrices with Xavier initialization
        self.weights = []
        for k in range(self.n_layers):
            fan_in = layer_dims[k]
            fan_out = layer_dims[k + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_out, fan_in)).astype(np.float32)
            self.weights.append(W)

        # Forward biases (one per layer)
        self.fwd_biases = [np.zeros(layer_dims[k + 1], dtype=np.float32)
                           for k in range(self.n_layers)]

        # Backward biases (one per layer in reverse direction)
        # bwd_biases[0] corresponds to the first layer of the backward pass
        # which uses weights[n_layers-1]
        self.bwd_biases = [None] * self.n_layers
        for step in range(self.n_layers):
            rev_idx = self.n_layers - 1 - step
            bias_dim = self.weights[rev_idx].shape[1] if step == self.n_layers - 1 \
                else self.weights[rev_idx].shape[0]
            # Actually, let me compute output dim of each backward step
            # step 0: z @ W[n-1], output dim = W[n-1].shape[1]... no
            # Let me think: W[rev_idx] has shape (layer_dims[rev_idx+1], layer_dims[rev_idx])
            # z @ W[rev_idx] gives (n, layer_dims[rev_idx])
            self.bwd_biases[step] = np.zeros(layer_dims[rev_idx], dtype=np.float32)

    def forward_pass(self, X):
        """
        Forward pass: source -> target
        Returns y_hat (normalized) and cache for gradient computation.
        """
        cache = {
            'h_input': [],      # input to each layer
            'pre_act': [],      # pre-activation values
        }

        h = X.copy()
        for k in range(self.n_layers):
            cache['h_input'].append(h)
            pre = h @ self.weights[k].T + self.fwd_biases[k]  # (n, dims[k+1])
            cache['pre_act'].append(pre)

            if k < self.n_layers - 1:
                h = leaky_relu(pre)
            else:
                h = pre  # no activation on output layer

        # Normalize output
        y_raw = h
        y_norms = np.linalg.norm(y_raw, axis=1, keepdims=True)
        y_norms = np.maximum(y_norms, 1e-8)
        y_hat = y_raw / y_norms

        cache['y_raw'] = y_raw
        cache['y_norms'] = y_norms
        cache['y_hat'] = y_hat

        return y_hat, cache

    def backward_pass(self, Z):
        """
        Backward pass: target -> source (using transposed weights)
        Returns x_hat (normalized) and cache for gradient computation.
        """
        cache = {
            'g_input': [],
            'pre_act': [],
        }

        h = Z.copy()
        for step in range(self.n_layers):
            cache['g_input'].append(h)
            rev_idx = self.n_layers - 1 - step
            # Use weight directly (not transposed) = equivalent to W^T in column form
            pre = h @ self.weights[rev_idx] + self.bwd_biases[step]
            cache['pre_act'].append(pre)

            if step < self.n_layers - 1:
                h = leaky_relu(pre)
            else:
                h = pre  # no activation on output layer

        x_raw = h
        x_norms = np.linalg.norm(x_raw, axis=1, keepdims=True)
        x_norms = np.maximum(x_norms, 1e-8)
        x_hat = x_raw / x_norms

        cache['x_raw'] = x_raw
        cache['x_norms'] = x_norms
        cache['x_hat'] = x_hat

        return x_hat, cache

    def compute_loss_and_gradients(self, X_batch, Z_batch, alpha, beta, gamma):
        """
        Compute joint B-BP loss and gradients for all parameters.

        L = alpha * L_fwd + beta * L_bwd + gamma * sum_k ||Wk^T Wk - I||_F
        """
        n = X_batch.shape[0]

        # ==============================================================
        # FORWARD PASS AND LOSS
        # ==============================================================
        y_hat, fwd_cache = self.forward_pass(X_batch)
        Z_norm = normalize_vectors(Z_batch)

        cos_sim_fwd = np.sum(y_hat * Z_norm, axis=1)
        L_fwd = np.mean(1.0 - cos_sim_fwd)

        # Gradient of L_fwd w.r.t. y_raw (through normalization)
        cos_fwd_col = cos_sim_fwd.reshape(-1, 1)
        delta_fwd_out = -(1.0 / n) * (Z_norm - cos_fwd_col * y_hat) / fwd_cache['y_norms']

        # ==============================================================
        # BACKWARD (REVERSE) PASS AND LOSS
        # ==============================================================
        x_hat, bwd_cache = self.backward_pass(Z_batch)
        X_norm = normalize_vectors(X_batch)

        cos_sim_bwd = np.sum(x_hat * X_norm, axis=1)
        L_bwd = np.mean(1.0 - cos_sim_bwd)

        # Gradient of L_bwd w.r.t. x_raw (through normalization)
        cos_bwd_col = cos_sim_bwd.reshape(-1, 1)
        delta_bwd_out = -(1.0 / n) * (X_norm - cos_bwd_col * x_hat) / bwd_cache['x_norms']

        # ==============================================================
        # BACKPROP THROUGH FORWARD PASS -> gradients for weights and fwd_biases
        # ==============================================================
        grad_W_fwd = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.fwd_biases]

        delta = delta_fwd_out  # (n, dims[-1])
        for k in range(self.n_layers - 1, -1, -1):
            # Layer k: pre = h_input[k] @ W[k].T + b[k]
            grad_W_fwd[k] = delta.T @ fwd_cache['h_input'][k]
            grad_b[k] = np.sum(delta, axis=0)

            if k > 0:
                delta = delta @ self.weights[k]  # backprop through W[k].T
                delta = delta * leaky_relu_deriv(fwd_cache['pre_act'][k - 1])

        # ==============================================================
        # BACKPROP THROUGH REVERSE PASS -> gradients for weights and bwd_biases
        # ==============================================================
        grad_W_bwd = [np.zeros_like(w) for w in self.weights]
        grad_c = [np.zeros_like(b) for b in self.bwd_biases]

        delta = delta_bwd_out  # (n, dims[0])
        for step in range(self.n_layers - 1, -1, -1):
            rev_idx = self.n_layers - 1 - step
            # Layer at this step: pre = g_input[step] @ W[rev_idx] + c[step]
            grad_W_bwd[rev_idx] = bwd_cache['g_input'][step].T @ delta
            grad_c[step] = np.sum(delta, axis=0)

            if step > 0:
                delta = delta @ self.weights[rev_idx].T
                delta = delta * leaky_relu_deriv(bwd_cache['pre_act'][step - 1])

        # ==============================================================
        # ORTHOGONAL PENALTY: sum_k ||Wk^T Wk - I||_F
        # ==============================================================
        L_orth = 0.0
        grad_W_orth = [np.zeros_like(w) for w in self.weights]

        for k in range(self.n_layers):
            W = self.weights[k]
            WtW = W.T @ W
            # I size matches WtW: (dims[k], dims[k])
            I_k = np.eye(WtW.shape[0], dtype=np.float32)
            diff = WtW - I_k
            norm_diff = np.linalg.norm(diff, 'fro')
            L_orth += norm_diff

            if norm_diff > 1e-10:
                grad_W_orth[k] = 2.0 * (W @ diff) / norm_diff

        # ==============================================================
        # COMBINE ALL GRADIENTS
        # ==============================================================
        L_total = alpha * L_fwd + beta * L_bwd + gamma * L_orth

        grad_W_total = []
        for k in range(self.n_layers):
            g = alpha * grad_W_fwd[k] + beta * grad_W_bwd[k] + gamma * grad_W_orth[k]
            grad_W_total.append(g)

        grad_b_total = [alpha * g for g in grad_b]
        grad_c_total = [beta * g for g in grad_c]

        return L_total, L_fwd, L_bwd, L_orth, grad_W_total, grad_b_total, grad_c_total


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_deep_bbp(model, X_train, Z_train,
                   alpha=1.0, beta=1.0, gamma=0.01,
                   lr=0.001, n_epochs=200, batch_size=512,
                   verbose=True):
    """Train Deep B-BP model with Adam optimizer."""
    n_train = X_train.shape[0]
    optimizer = AdamOptimizer(lr=lr)

    history = {
        'epoch': [], 'loss_total': [], 'loss_fwd': [],
        'loss_bwd': [], 'loss_orth': [],
    }

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        X_shuffled = X_train[perm]
        Z_shuffled = Z_train[perm]

        epoch_loss = 0.0
        epoch_fwd = 0.0
        epoch_bwd = 0.0
        epoch_orth = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_shuffled[start:end]
            Z_batch = Z_shuffled[start:end]

            L_total, L_fwd, L_bwd, L_orth, grad_W, grad_b, grad_c = \
                model.compute_loss_and_gradients(X_batch, Z_batch, alpha, beta, gamma)

            # Update all parameters via Adam
            optimizer.increment_t()

            for k in range(model.n_layers):
                model.weights[k] = optimizer.step(f'W{k}', model.weights[k], grad_W[k])
                model.fwd_biases[k] = optimizer.step(f'b{k}', model.fwd_biases[k], grad_b[k])
                model.bwd_biases[k] = optimizer.step(f'c{k}', model.bwd_biases[k], grad_c[k])

            epoch_loss += L_total
            epoch_fwd += L_fwd
            epoch_bwd += L_bwd
            epoch_orth += L_orth
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_fwd = epoch_fwd / n_batches
        avg_bwd = epoch_bwd / n_batches
        avg_orth = epoch_orth / n_batches

        history['epoch'].append(epoch + 1)
        history['loss_total'].append(avg_loss)
        history['loss_fwd'].append(avg_fwd)
        history['loss_bwd'].append(avg_bwd)
        history['loss_orth'].append(avg_orth)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Fwd: {avg_fwd:.6f} | "
                  f"Bwd: {avg_bwd:.6f} | "
                  f"Orth: {avg_orth:.4f}")

    return history


# =============================================================================
# 6. EVALUATION
# =============================================================================

def evaluate_model(model, src_vectors, tgt_vectors, src_indices, tgt_indices,
                   direction='forward', k_values=[1, 5]):
    """Evaluate using the deep model's forward or backward pass."""
    n_test = len(src_indices)

    if direction == 'forward':
        X_test = src_vectors[src_indices]
        y_hat, _ = model.forward_pass(X_test)
        similarities = y_hat @ tgt_vectors.T
        gold_indices = tgt_indices
    else:
        Z_test = tgt_vectors[tgt_indices]
        x_hat, _ = model.backward_pass(Z_test)
        similarities = x_hat @ src_vectors.T
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


def evaluate_procrustes(W, src_vectors, tgt_vectors, src_indices, tgt_indices,
                        direction='forward', k_values=[1, 5]):
    """Evaluate linear Procrustes baseline."""
    n_test = len(src_indices)

    if direction == 'forward':
        X_test = src_vectors[src_indices]
        mapped = X_test @ W.T
        mapped = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
        similarities = mapped @ tgt_vectors.T
        gold_indices = tgt_indices
    else:
        Z_test = tgt_vectors[tgt_indices]
        mapped = Z_test @ W
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
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep B-BP for BLI (NumPy)")

    parser.add_argument('--src_emb', type=str, required=True)
    parser.add_argument('--tgt_emb', type=str, required=True)
    parser.add_argument('--dict_train', type=str, required=True)
    parser.add_argument('--dict_test', type=str, required=True)
    parser.add_argument('--max_vocab', type=int, default=200000)

    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512],
                        help='Hidden layer dimensions (default: 512)')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("=" * 60)
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
    d = X_train.shape[1]

    print(f"\nTraining pairs: {X_train.shape[0]}, Dim: {d}")

    # -------------------------------------------------------------------------
    # Procrustes baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROCRUSTES BASELINE")
    print("=" * 60)

    M = Z_train.T @ X_train
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    W_procrustes = (U @ Vt).astype(np.float32)

    print("\n  Forward (src -> tgt):")
    proc_fwd = evaluate_procrustes(W_procrustes, src_vectors_norm, tgt_vectors_norm,
                                   test_src_idx, test_tgt_idx, direction='forward')
    print("  Backward (tgt -> src):")
    proc_bwd = evaluate_procrustes(W_procrustes, src_vectors_norm, tgt_vectors_norm,
                                   test_src_idx, test_tgt_idx, direction='backward')

    # -------------------------------------------------------------------------
    # Create Deep B-BP model
    # -------------------------------------------------------------------------
    layer_dims = [d] + args.hidden_dims + [d]

    print("\n" + "=" * 60)
    print("DEEP B-BP MODEL")
    print(f"  Architecture: {' -> '.join(str(x) for x in layer_dims)}")
    print(f"  Activation: LeakyReLU (alpha=0.01)")
    n_params = sum(w.shape[0] * w.shape[1] for w in [np.zeros((layer_dims[k+1], layer_dims[k]))
                                                       for k in range(len(layer_dims)-1)])
    n_bias_params = sum(layer_dims[k+1] for k in range(len(layer_dims)-1))
    print(f"  Weight params: {n_params:,}")
    print(f"  Bias params: {2 * n_bias_params:,} (forward + backward)")
    print(f"  Total params: {n_params + 2 * n_bias_params:,}")
    print(f"  alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"  lr={args.lr}, epochs={args.n_epochs}, batch_size={args.batch_size}")
    print("=" * 60)

    model = DeepBBP(layer_dims)

    # Evaluate before training
    print("\n  --- Pre-training evaluation ---")
    print("  Forward:")
    pre_fwd = evaluate_model(model, src_vectors_norm, tgt_vectors_norm,
                             test_src_idx, test_tgt_idx, direction='forward')
    print("  Backward:")
    pre_bwd = evaluate_model(model, src_vectors_norm, tgt_vectors_norm,
                             test_src_idx, test_tgt_idx, direction='backward')

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("\n  --- Training ---")
    t0 = time.time()
    history = train_deep_bbp(
        model, X_train, Z_train,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size,
        verbose=True
    )
    train_time = time.time() - t0
    print(f"\n  Total training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Post-training evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    print("\n  Forward (src -> tgt):")
    post_fwd = evaluate_model(model, src_vectors_norm, tgt_vectors_norm,
                              test_src_idx, test_tgt_idx, direction='forward')

    print("  Backward (tgt -> src):")
    post_bwd = evaluate_model(model, src_vectors_norm, tgt_vectors_norm,
                              test_src_idx, test_tgt_idx, direction='backward')

    # -------------------------------------------------------------------------
    # Bidirectional metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL METRICS")
    print("=" * 60)

    bc = (post_fwd[1] + post_bwd[1]) / 2
    dg = abs(post_fwd[1] - post_bwd[1])
    proc_bc = (proc_fwd[1] + proc_bwd[1]) / 2
    proc_dg = abs(proc_fwd[1] - proc_bwd[1])

    print(f"  BC (Deep B-BP): {bc:.2f}%")
    print(f"  DG (Deep B-BP): {dg:.2f}%")

    # Round-trip
    X_test = src_vectors_norm[test_src_idx]
    y_hat, _ = model.forward_pass(X_test)
    x_recon, _ = model.backward_pass(y_hat)
    roundtrip_sim = np.mean(np.sum(X_test * x_recon, axis=1))
    print(f"  Round-trip similarity: {roundtrip_sim:.6f}")

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: PROCRUSTES vs DEEP B-BP")
    print("=" * 60)

    pre_bc = (pre_fwd[1] + pre_bwd[1]) / 2

    print(f"\n{'Metric':<30} {'Procrustes':<15} {'Pre-train':<15} {'Deep B-BP':<15} {'Delta':<10}")
    print("-" * 85)
    print(f"{'P@1 Forward':<30} {proc_fwd[1]:<15.2f} {pre_fwd[1]:<15.2f} {post_fwd[1]:<15.2f} {post_fwd[1]-proc_fwd[1]:>+.2f}")
    print(f"{'P@5 Forward':<30} {proc_fwd[5]:<15.2f} {pre_fwd[5]:<15.2f} {post_fwd[5]:<15.2f} {post_fwd[5]-proc_fwd[5]:>+.2f}")
    print(f"{'P@1 Backward':<30} {proc_bwd[1]:<15.2f} {pre_bwd[1]:<15.2f} {post_bwd[1]:<15.2f} {post_bwd[1]-proc_bwd[1]:>+.2f}")
    print(f"{'P@5 Backward':<30} {proc_bwd[5]:<15.2f} {pre_bwd[5]:<15.2f} {post_bwd[5]:<15.2f} {post_bwd[5]-proc_bwd[5]:>+.2f}")
    print(f"{'BC (avg P@1)':<30} {proc_bc:<15.2f} {pre_bc:<15.2f} {bc:<15.2f} {bc-proc_bc:>+.2f}")
    print(f"{'DG (|fwd-bwd| P@1)':<30} {proc_dg:<15.2f} {'—':<15} {dg:<15.2f} {dg-proc_dg:>+.2f}")
    print(f"{'Round-trip sim':<30} {'~1.000':<15} {'—':<15} {roundtrip_sim:<15.6f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    save_dict = {}
    for k in range(model.n_layers):
        save_dict[f'W{k}'] = model.weights[k]
        save_dict[f'b{k}'] = model.fwd_biases[k]
        save_dict[f'c{k}'] = model.bwd_biases[k]
    save_dict['layer_dims'] = np.array(model.layer_dims)

    np.savez('deep_bbp_model.npz', **save_dict)
    np.savez('deep_bbp_history.npz', **{k: np.array(v) for k, v in history.items()})
    print(f"\nSaved: deep_bbp_model.npz, deep_bbp_history.npz")


if __name__ == '__main__':
    main()
