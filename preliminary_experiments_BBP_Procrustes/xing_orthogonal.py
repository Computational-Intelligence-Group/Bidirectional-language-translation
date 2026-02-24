"""
Xing et al. (2015) - Normalized Word Embedding and Orthogonal Transform
for Bilingual Word Translation

Faithful implementation of the paper's approach:
1. Normalize word vectors to unit length
2. Objective: max_W  sum_i (W x_i)^T z_i   [Eq. 4]
3. Gradient: nabla_W = sum_i x_i z_i^T        [Eq. 5]
4. Update:   W = W + alpha * nabla_W            [Eq. 6]
5. Re-orthogonalize: SVD(W) -> replace singular values with 1  [Eq. 7]
6. Evaluate with cosine similarity for P@1, P@5

Also includes:
- Mikolov (2013) linear transform baseline [Eq. 3]: min_W sum ||Wx_i - z_i||^2
- Closed-form Procrustes (equivalent to Xing in the limit)

Usage:
    python xing_orthogonal.py \
        --src_emb path/to/en.vec \
        --tgt_emb path/to/es.vec \
        --dict_train path/to/train_dict.txt \
        --dict_test path/to/test_dict.txt
"""

import numpy as np
import argparse
import time
from collections import defaultdict


# =============================================================================
# 1. LOADING EMBEDDINGS
# =============================================================================

def load_embeddings(filepath, max_vocab=200000):
    """
    Load word embeddings from text format (word2vec / FastText .vec format).
    Each line: word dim1 dim2 ... dimN
    First line may be: vocab_size dimension (FastText header) - we detect and skip it.
    """
    words = []
    vectors = []
    dim = None

    print(f"Loading embeddings from {filepath} ...")
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            tokens = line.rstrip().split(' ')

            # Detect FastText header line: "vocab_size dimension"
            if i == 0 and len(tokens) == 2:
                try:
                    int(tokens[0])
                    int(tokens[1])
                    print(f"  Detected header: {tokens[0]} words, {tokens[1]} dimensions")
                    continue
                except ValueError:
                    pass  # Not a header, treat as normal line

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

    vectors = np.array(vectors, dtype=np.float32)  # shape: (vocab_size, dim)
    word2idx = {w: i for i, w in enumerate(words)}

    print(f"  Loaded {len(words)} words, dimension={dim}")
    return words, vectors, word2idx


# =============================================================================
# 2. NORMALIZATION
# =============================================================================

def normalize_vectors(vectors):
    """
    Normalize each vector to unit length (L2 norm = 1).
    After this, inner product = cosine similarity.
    This is Xing et al. Section 3.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # avoid division by zero
    return vectors / norms


# =============================================================================
# 3. LOADING BILINGUAL DICTIONARY
# =============================================================================

def load_dictionary(filepath, src_word2idx, tgt_word2idx):
    """
    Load bilingual dictionary. Each line: source_word target_word
    Only keep pairs where both words are in their respective vocabularies.
    Returns arrays of index pairs.
    """
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
# 4. MIKOLOV BASELINE: LINEAR TRANSFORM (Eq. 3)
# =============================================================================

def learn_mikolov_linear(X_train, Z_train):
    """
    Mikolov et al. (2013b) baseline.
    min_W  sum_i ||W x_i - z_i||^2

    Closed-form solution: W = Z^T X (X^T X)^{-1}
    where X is (n, d_src) and Z is (n, d_tgt).

    Note: W maps src -> tgt, so W is (d_tgt, d_src).
    We solve: W = (X^T X)^{-1} X^T Z then transpose,
    or more directly: W = Z.T @ X @ inv(X.T @ X)
    
    Actually the standard least squares for W x_i ≈ z_i with
    x_i as column vectors means:
    W = Z.T @ pinv(X.T)  or equivalently  W = Z.T @ X @ inv(X.T @ X)
    
    But with X as (n, d) row-vectors: we want W (d_tgt x d_src) such that
    X @ W.T ≈ Z  =>  W.T = pinv(X) @ Z  =>  W = Z.T @ pinv(X.T)
    Simplest: W.T = np.linalg.lstsq(X, Z)
    """
    # X_train: (n, d), Z_train: (n, d)
    # We want W such that W @ x = z (x is column vector)
    # Equivalently: X @ W.T = Z
    W_T, _, _, _ = np.linalg.lstsq(X_train, Z_train, rcond=None)
    W = W_T.T  # (d_tgt, d_src)
    return W


# =============================================================================
# 5. XING ORTHOGONAL TRANSFORM - ITERATIVE (Eq. 4-7)
# =============================================================================

def learn_xing_iterative(X_train, Z_train, lr=1.0, n_iters=100, verbose=True):
    """
    Xing et al. (2015) iterative orthogonal transform.

    Objective [Eq. 4]:  max_W  sum_i (W x_i)^T z_i
    Gradient  [Eq. 5]:  nabla_W = sum_i x_i z_i^T
                         BUT NOTE: in matrix form with X (n,d) rows and Z (n,d) rows:
                         nabla_W = X^T @ Z  (this gives d x d matrix)
                         Since we want W (d x d) mapping columns: nabla_W = Z.T @ X
                         Wait - let me be precise.

    The paper writes: nabla_W = sum_i x_i y_i^T  [Eq. 5]
    where x_i is source (column), y_i is target (column).
    So nabla_W = X_cols @ Z_cols^T
    With X as (n, d) row-matrix: X_cols = X.T (d, n), Z_cols = Z.T (d, n)
    nabla_W = X.T @ Z ... wait no.
    
    sum_i x_i y_i^T where x_i is (d,1) and y_i is (d,1):
    = [x_1 x_2 ... x_n] @ [y_1 y_2 ... y_n]^T  ... no
    
    x_i y_i^T is (d, d). Sum of n such matrices.
    In matrix form: sum_i x_i y_i^T = X_cols @ Y_cols^T = X.T @ Z
    where X.T is (d, n) and Z is (n, d).
    Result: (d, d). This is nabla_W.

    But W maps x -> Wx. The objective is sum (Wx_i)^T z_i = sum x_i^T W^T z_i
    = trace(W^T Z^T X) ... derivative w.r.t. W is Z^T @ X ... hmm.
    
    Let me just derive it carefully.
    f(W) = sum_i (W x_i)^T z_i = sum_i x_i^T W^T z_i = trace(X^T W^T Z)  [X,Z are n x d]
         = trace(W^T Z X^T)  ... no. Let me use index notation.
    
    Actually: sum_i z_i^T W x_i = trace(Z^T W X)  ... no.
    
    sum_i (W x_i)^T z_i = sum_i x_i^T W^T z_i
    
    With X = [x_1^T; x_2^T; ...] (n x d) and Z = [z_1^T; z_2^T; ...] (n x d):
    sum_i x_i^T W^T z_i = trace(X W^T Z^T) ... hmm this isn't clean.
    
    More directly: df/dW = sum_i d/dW (W x_i)^T z_i = sum_i z_i x_i^T
    Because d/dW (Wx)^T z = z x^T  (standard matrix calculus identity).
    
    So nabla_W = sum_i z_i x_i^T = Z.T @ X   (with Z.T as (d,n), X as (n,d))
    
    But the paper writes nabla_W = sum_i x_i y_i^T  [Eq. 5].
    That would give X.T @ Z.
    
    Hmm, these are transposes of each other. The paper might have a different 
    convention. Let's check: the paper's Eq. 5 says nabla_W = sum x_i y_i^T.
    
    If the objective is max sum (Wx)^T z, the gradient is sum z x^T = Z.T @ X.
    If the paper says sum x y^T = X.T @ Z, then maybe they define the mapping
    as x^T W (row-vector convention)?
    
    For implementation, both conventions converge to the same orthogonal W after
    SVD re-orthogonalization. I'll use the standard convention and verify.
    
    Actually, it doesn't matter much because after SVD orthogonalization,
    the iterative process converges to the Procrustes solution regardless.
    Let me just use the correct gradient: nabla_W = Z.T @ X for max sum (Wx)^T z.
    
    Parameters:
        X_train: (n, d) normalized source embeddings for training pairs
        Z_train: (n, d) normalized target embeddings for training pairs
        lr: learning rate (alpha in Eq. 6)
        n_iters: number of gradient ascent iterations
    
    Returns:
        W: (d, d) orthogonal matrix
    """
    d = X_train.shape[1]

    # Initialize W as identity (a valid orthogonal matrix)
    W = np.eye(d, dtype=np.float32)

    for t in range(n_iters):
        # --- Eq. 5: Compute gradient ---
        # nabla_W = sum_i z_i x_i^T = Z^T @ X
        grad = Z_train.T @ X_train  # (d, d)

        # --- Eq. 6: Gradient ascent update ---
        W = W + lr * grad

        # --- Eq. 7: Re-orthogonalize via SVD ---
        # Find closest orthogonal matrix: min ||W - W_bar|| s.t. W_bar^T W_bar = I
        # Solution: SVD(W) = U Sigma V^T, then W_bar = U V^T
        U, S, Vt = np.linalg.svd(W, full_matrices=True)
        W = U @ Vt

        if verbose and (t + 1) % 10 == 0:
            # Compute objective value: sum (Wx_i)^T z_i
            mapped = X_train @ W.T  # (n, d)
            obj = np.sum(mapped * Z_train)
            print(f"  Iter {t+1}/{n_iters}: objective = {obj:.4f}")

    return W


# =============================================================================
# 6. PROCRUSTES CLOSED-FORM (equivalent to Xing in the limit)
# =============================================================================

def learn_procrustes(X_train, Z_train):
    """
    Closed-form orthogonal Procrustes solution.
    
    max_W  sum_i (W x_i)^T z_i  s.t. W^T W = I
    
    Equivalent to: max_W trace(W^T Z^T X) s.t. W^T W = I
    
    Solution: M = Z^T X, SVD(M) = U Sigma V^T, then W* = U V^T
    
    This gives the same result as Xing's iterative approach 
    but in one shot.
    """
    M = Z_train.T @ X_train  # (d, d)
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    W = U @ Vt
    return W


# =============================================================================
# 7. EVALUATION
# =============================================================================

def evaluate(W, src_vectors, tgt_vectors, src_indices, tgt_indices, tgt_words, k_values=[1, 5]):
    """
    Evaluate bilingual word translation.
    
    For each source word in the test set:
    1. Map it to target space: y_hat = W @ x
    2. Find nearest neighbors in target space by cosine similarity
       (since all vectors are normalized, cosine sim = dot product)
    3. Check if the correct translation is in the top-k
    
    Reports P@1 and P@5.
    """
    # Get test embeddings
    X_test = src_vectors[src_indices]  # (n_test, d)
    
    # Map source to target space
    mapped = X_test @ W.T  # (n_test, d)
    
    # Normalize mapped vectors (W is orthogonal so this should be ~unit already,
    # but normalize to be safe for cosine computation)
    mapped_norms = np.linalg.norm(mapped, axis=1, keepdims=True)
    mapped = mapped / np.maximum(mapped_norms, 1e-8)
    
    # Compute cosine similarity with ALL target vectors
    # Since both are normalized: cosine_sim = dot product
    # mapped: (n_test, d), tgt_vectors: (vocab_tgt, d)
    # similarities: (n_test, vocab_tgt)
    similarities = mapped @ tgt_vectors.T
    
    # For each test pair, check if correct target is in top-k
    results = {k: 0 for k in k_values}
    n_test = len(src_indices)
    
    # Build ground truth: for each source word, collect all valid target indices
    # (a source word might have multiple valid translations)
    src_to_tgt = defaultdict(set)
    for si, ti in zip(src_indices, tgt_indices):
        src_to_tgt[si].add(ti)
    
    # Deduplicate: evaluate each unique source word once
    unique_src = list(src_to_tgt.keys())
    
    correct = {k: 0 for k in k_values}
    
    for idx_in_batch, src_idx in enumerate(src_indices):
        sim_row = similarities[idx_in_batch]
        
        # Get top-k indices (descending similarity)
        max_k = max(k_values)
        top_k_indices = np.argpartition(-sim_row, max_k)[:max_k]
        top_k_indices = top_k_indices[np.argsort(-sim_row[top_k_indices])]
        
        gold_tgt = tgt_indices[idx_in_batch]
        
        for k in k_values:
            if gold_tgt in top_k_indices[:k]:
                correct[k] += 1
    
    for k in k_values:
        acc = correct[k] / n_test * 100
        print(f"  P@{k}: {acc:.2f}% ({correct[k]}/{n_test})")
    
    return {k: correct[k] / n_test * 100 for k in k_values}


def evaluate_backward(W, src_vectors, tgt_vectors, src_indices, tgt_indices, src_words, k_values=[1, 5]):
    """
    Evaluate reverse direction: target -> source using W^T.
    Since W is orthogonal, W^T = W^{-1}, so this is the free reverse mapping.
    """
    Z_test = tgt_vectors[tgt_indices]
    
    # Map target to source space using W^T
    mapped = Z_test @ W  # (n_test, d) -- this is W^T @ z for each z
    
    # Normalize
    mapped_norms = np.linalg.norm(mapped, axis=1, keepdims=True)
    mapped = mapped / np.maximum(mapped_norms, 1e-8)
    
    # Cosine similarity with all source vectors
    similarities = mapped @ src_vectors.T
    
    n_test = len(tgt_indices)
    correct = {k: 0 for k in k_values}
    
    for idx_in_batch, tgt_idx in enumerate(tgt_indices):
        sim_row = similarities[idx_in_batch]
        max_k = max(k_values)
        top_k_indices = np.argpartition(-sim_row, max_k)[:max_k]
        top_k_indices = top_k_indices[np.argsort(-sim_row[top_k_indices])]
        
        gold_src = src_indices[idx_in_batch]
        
        for k in k_values:
            if gold_src in top_k_indices[:k]:
                correct[k] += 1
    
    for k in k_values:
        acc = correct[k] / n_test * 100
        print(f"  P@{k}: {acc:.2f}% ({correct[k]}/{n_test})")
    
    return {k: correct[k] / n_test * 100 for k in k_values}


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Xing et al. (2015) Orthogonal Transform")
    parser.add_argument('--src_emb', type=str, required=True,
                        help='Path to source language embeddings (.vec or .txt)')
    parser.add_argument('--tgt_emb', type=str, required=True,
                        help='Path to target language embeddings (.vec or .txt)')
    parser.add_argument('--dict_train', type=str, required=True,
                        help='Path to training dictionary (src_word tgt_word per line)')
    parser.add_argument('--dict_test', type=str, required=True,
                        help='Path to test dictionary (src_word tgt_word per line)')
    parser.add_argument('--max_vocab', type=int, default=200000,
                        help='Max vocabulary size to load (default: 200000)')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate for Xing iterative (default: 1.0)')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='Number of iterations for Xing iterative (default: 100)')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Load embeddings
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading embeddings")
    print("=" * 60)
    src_words, src_vectors, src_word2idx = load_embeddings(args.src_emb, args.max_vocab)
    tgt_words, tgt_vectors, tgt_word2idx = load_embeddings(args.tgt_emb, args.max_vocab)

    # -------------------------------------------------------------------------
    # Step 2: Normalize to unit length (Xing Section 3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Normalizing vectors to unit length")
    print("=" * 60)
    src_vectors_norm = normalize_vectors(src_vectors)
    tgt_vectors_norm = normalize_vectors(tgt_vectors)

    # Verify normalization
    src_norms = np.linalg.norm(src_vectors_norm[:10], axis=1)
    print(f"  Sample source norms after normalization: {src_norms[:5]}")

    # -------------------------------------------------------------------------
    # Step 3: Load dictionaries
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Loading bilingual dictionaries")
    print("=" * 60)
    print("Training dictionary:")
    train_src_idx, train_tgt_idx = load_dictionary(args.dict_train, src_word2idx, tgt_word2idx)
    print("Test dictionary:")
    test_src_idx, test_tgt_idx = load_dictionary(args.dict_test, src_word2idx, tgt_word2idx)

    # Extract training pair matrices
    X_train = src_vectors_norm[train_src_idx]  # (n_train, d)
    Z_train = tgt_vectors_norm[train_tgt_idx]  # (n_train, d)

    print(f"\n  Training pairs: {X_train.shape[0]}")
    print(f"  Test pairs: {len(test_src_idx)}")
    print(f"  Source dim: {X_train.shape[1]}, Target dim: {Z_train.shape[1]}")

    assert X_train.shape[1] == Z_train.shape[1], \
        "Source and target dimensions must match! (Use same dim embeddings)"

    # =====================================================================
    # METHOD 1: Mikolov (2013) Linear Transform Baseline [Eq. 3]
    # =====================================================================
    print("\n" + "=" * 60)
    print("METHOD 1: Mikolov (2013) Linear Transform (MSE, unnormalized)")
    print("  Objective: min_W sum ||Wx_i - z_i||^2")
    print("  Using UNNORMALIZED embeddings (as in original paper)")
    print("=" * 60)

    X_train_unnorm = src_vectors[train_src_idx]
    Z_train_unnorm = tgt_vectors[train_tgt_idx]

    t0 = time.time()
    W_mikolov = learn_mikolov_linear(X_train_unnorm, Z_train_unnorm)
    print(f"  Training time: {time.time() - t0:.3f}s")

    print("\n  Forward (EN -> ES):")
    evaluate(W_mikolov, src_vectors, tgt_vectors, test_src_idx, test_tgt_idx, tgt_words)
    
    print("\n  Backward (ES -> EN) via W^T:")
    evaluate_backward(W_mikolov, src_vectors, tgt_vectors, test_src_idx, test_tgt_idx, src_words)

    # =====================================================================
    # METHOD 2: Xing (2015) Orthogonal Transform - Closed-Form Procrustes
    # =====================================================================
    print("\n" + "=" * 60)
    print("METHOD 2: Xing (2015) Orthogonal Transform - Procrustes (closed-form)")
    print("  Objective: max_W sum (Wx_i)^T z_i  s.t. W^TW = I")
    print("  Using NORMALIZED embeddings")
    print("=" * 60)

    t0 = time.time()
    W_procrustes = learn_procrustes(X_train, Z_train)
    print(f"  Training time: {time.time() - t0:.3f}s")

    # Verify orthogonality
    orth_error = np.linalg.norm(W_procrustes.T @ W_procrustes - np.eye(W_procrustes.shape[0]))
    print(f"  Orthogonality check ||W^TW - I||_F = {orth_error:.6e}")

    print("\n  Forward (EN -> ES):")
    evaluate(W_procrustes, src_vectors_norm, tgt_vectors_norm,
             test_src_idx, test_tgt_idx, tgt_words)
    
    print("\n  Backward (ES -> EN) via W^T (free from orthogonality):")
    evaluate_backward(W_procrustes, src_vectors_norm, tgt_vectors_norm,
                      test_src_idx, test_tgt_idx, src_words)

    # =====================================================================
    # METHOD 3: Xing (2015) Orthogonal Transform - Iterative [Eq. 4-7]
    # =====================================================================
    print("\n" + "=" * 60)
    print("METHOD 3: Xing (2015) Orthogonal Transform - Iterative")
    print(f"  lr={args.lr}, n_iters={args.n_iters}")
    print("  Gradient ascent + SVD re-orthogonalization each step")
    print("=" * 60)

    t0 = time.time()
    W_xing_iter = learn_xing_iterative(
        X_train, Z_train,
        lr=args.lr,
        n_iters=args.n_iters,
        verbose=True
    )
    print(f"  Training time: {time.time() - t0:.3f}s")

    orth_error = np.linalg.norm(W_xing_iter.T @ W_xing_iter - np.eye(W_xing_iter.shape[0]))
    print(f"  Orthogonality check ||W^TW - I||_F = {orth_error:.6e}")

    print("\n  Forward (EN -> ES):")
    evaluate(W_xing_iter, src_vectors_norm, tgt_vectors_norm,
             test_src_idx, test_tgt_idx, tgt_words)
    
    print("\n  Backward (ES -> EN) via W^T:")
    evaluate_backward(W_xing_iter, src_vectors_norm, tgt_vectors_norm,
                      test_src_idx, test_tgt_idx, src_words)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Method 1 (Mikolov): MSE loss, unnormalized, unconstrained W")
    print("Method 2 (Procrustes): Cosine loss, normalized, orthogonal W (closed-form)")
    print("Method 3 (Xing iter): Cosine loss, normalized, orthogonal W (iterative)")
    print("\nMethods 2 and 3 should give nearly identical results.")
    print("Both should outperform Method 1 (as shown in Xing Table 1 vs 2).")

    # Save W matrices for later use with B-BP
    np.save('W_procrustes.npy', W_procrustes)
    np.save('W_xing_iterative.npy', W_xing_iter)
    np.save('W_mikolov.npy', W_mikolov)
    print("\nSaved W matrices: W_procrustes.npy, W_xing_iterative.npy, W_mikolov.npy")


if __name__ == '__main__':
    main()
