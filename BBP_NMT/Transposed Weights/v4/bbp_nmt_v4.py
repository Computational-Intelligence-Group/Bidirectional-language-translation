"""
B-BP NMT v4 — Adigun-Kosko Framework
Two Surgical Fixes: Untied Output + No Input Feeding
=====================================================
Two fixes from v3 (fixed-point repetition collapse, zero BLEU):

  FIX 1 — Untied output projections
    v3 used tied weights: logit = out @ E_tgt.weight.T
    The embedding matrix had to simultaneously:
      (a) map token IDs to vectors for encoder/decoder input
      (b) satisfy B-BP inversion constraint (U^T compatibility)
      (c) project decoder states to sharp output distributions
    These three objectives compete — output distribution stays near-uniform
    (max prob 8.7%), argmax always picks "die".
    v4: separate W_out_en = Linear(d, V_en) and W_out_de = Linear(d, V_de)
    Embeddings are now free to focus on representation; output layers score.

  FIX 2 — Remove input feeding
    v3 decoder: x_t = Embed(y_{t-1}) + context_{t-1}
    Once the decoder collapsed to "die" at step 1:
      context_1 ≈ context_2 ≈ ... (attention over same h)
      Embed("die") + context ≈ constant input at every step
      → fixed point: h_t ≈ h_{t-1} → identical output forever
    v4 decoder: x_t = Embed(y_{t-1})  (context used only for output, not input)
    Context still used at output: logit = W_out(h_t + ctx_proj(context_t))
    Removes fixed-point mechanism while keeping attention benefit.

UNCHANGED FROM V3:
  - B-BP cell: h_t = tanh(h_{t-1} @ U + x + b) / tanh(h_{t-1} @ U^T + x + b)
  - 2-layer stacked encoder (both layers share U)
  - BPE tokenization (8k vocab per language)
  - Scaled dot-product attention (no learned scoring params)
  - Cycle warmup: ep1-5 λ=0.00 | ep6-10 λ=0.05 | ep11+ λ=0.10
  - lr=1e-3, hid=256, batch=64

"""
# ── 0. Imports ─────────────────────────────────────────────────────────────────
import math, time, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import sentencepiece as spm
import sacrebleu

# ── 1. Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 2. Hyperparameters ─────────────────────────────────────────────────────────
BPE_VOCAB  = 8_000     # sentencepiece vocab size per language
MAX_LEN    = 40        # max tokens per sentence (post-BPE)
TRAIN_CAP  = 200_000   # subsample from WMT14 (same as v1 for fair comparison)
BATCH      = 64
HID        = 256       # hidden / embedding dimension
LR         = 1e-3      # FIX 1: was 3e-4
EPOCHS     = 20
ENC_LAYERS = 2         # FIX 3: stacked encoder

# FIX 2: cycle warmup schedule
def get_lambda(epoch: int) -> float:
    if epoch <= 5:  return 0.00
    if epoch <= 10: return 0.05
    return 0.10
SEED       = 42

random.seed(SEED)
torch.manual_seed(SEED)

# SentencePiece special token indices (set via bos_id / eos_id / pad_id args)
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# ── 3. Load WMT14 ──────────────────────────────────────────────────────────────
print("Loading WMT14 en-de ...")
ds        = load_dataset("wmt14", "de-en")
train_raw = ds["train"]
val_raw   = ds["validation"]
test_raw  = ds["test"]
print(f"Raw  train={len(train_raw):,}  val={len(val_raw):,}  test={len(test_raw):,}")

def get_pair(sample):
    t = sample.get("translation", {})
    return t.get("en", "").strip(), t.get("de", "").strip()

# Subsample training indices
all_idx = list(range(len(train_raw)))
random.shuffle(all_idx)
sub_idx = all_idx[:TRAIN_CAP]

# ── 4. Train BPE models ────────────────────────────────────────────────────────
print(f"\nTraining BPE models (vocab={BPE_VOCAB}) on {TRAIN_CAP:,} pairs ...")

en_lines, de_lines = [], []
for idx in sub_idx:
    en, de = get_pair(train_raw[idx])
    if en and de:
        en_lines.append(en.lower())
        de_lines.append(de.lower())

with open("/tmp/train_en.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(en_lines))
with open("/tmp/train_de.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(de_lines))

# Train EN BPE
spm.SentencePieceTrainer.train(
    input="/tmp/train_en.txt",
    model_prefix="/tmp/spm_en",
    vocab_size=BPE_VOCAB,
    character_coverage=1.0,     # Latin script → full coverage
    model_type="bpe",
    pad_id=PAD_IDX, pad_piece="<pad>",
    unk_id=UNK_IDX, unk_piece="<unk>",
    bos_id=SOS_IDX, bos_piece="<sos>",
    eos_id=EOS_IDX, eos_piece="<eos>",
    shuffle_input_sentence=True,
)

# Train DE BPE
spm.SentencePieceTrainer.train(
    input="/tmp/train_de.txt",
    model_prefix="/tmp/spm_de",
    vocab_size=BPE_VOCAB,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=PAD_IDX, pad_piece="<pad>",
    unk_id=UNK_IDX, unk_piece="<unk>",
    bos_id=SOS_IDX, bos_piece="<sos>",
    eos_id=EOS_IDX, eos_piece="<eos>",
    shuffle_input_sentence=True,
)

sp_en = spm.SentencePieceProcessor(model_file="/tmp/spm_en.model")
sp_de = spm.SentencePieceProcessor(model_file="/tmp/spm_de.model")

EN_VOCAB = sp_en.vocab_size()
DE_VOCAB = sp_de.vocab_size()
print(f"EN BPE vocab: {EN_VOCAB:,}  |  DE BPE vocab: {DE_VOCAB:,}")

# ── 5. Dataset encoding ─────────────────────────────────────────────────────────
def batch_encode(sp, texts):
    """Batch encode all texts at once — 100x faster than one-by-one."""
    return sp.encode([t.lower().strip() for t in texts])

def make_pairs_batch(en_texts, de_texts):
    """Encode full list in one batch call, filter by length."""
    print(f"  Batch encoding {len(en_texts):,} pairs ...", flush=True)
    all_en = batch_encode(sp_en, en_texts)
    all_de = batch_encode(sp_de, de_texts)
    pairs = []
    for en_ids, de_ids in zip(all_en, all_de):
        if not (1 <= len(en_ids) <= MAX_LEN and 1 <= len(de_ids) <= MAX_LEN):
            continue
        en_t = torch.tensor([SOS_IDX] + en_ids + [EOS_IDX], dtype=torch.long)
        de_t = torch.tensor([SOS_IDX] + de_ids + [EOS_IDX], dtype=torch.long)
        pairs.append((en_t, de_t))
    return pairs

def make_pairs_from_raw(raw_split):
    """Encode a raw HF dataset split using batch encoding."""
    en_texts, de_texts = [], []
    for s in raw_split:
        en, de = get_pair(s)
        if en and de:
            en_texts.append(en)
            de_texts.append(de)
    return make_pairs_batch(en_texts, de_texts)

def collate(batch):
    en_seqs, de_seqs = zip(*batch)
    return (pad_sequence(en_seqs, padding_value=PAD_IDX),
            pad_sequence(de_seqs, padding_value=PAD_IDX))

print("\nEncoding pairs with BPE (batch mode) ...")
train_pairs = make_pairs_batch(en_lines, de_lines)
val_pairs   = make_pairs_from_raw(val_raw)
test_pairs  = make_pairs_from_raw(test_raw)
print(f"Train: {len(train_pairs):,}  Val: {len(val_pairs):,}  Test: {len(test_pairs):,}")

train_loader = DataLoader(train_pairs, batch_size=BATCH, shuffle=True,
                          collate_fn=collate, drop_last=True)
val_loader   = DataLoader(val_pairs,   batch_size=BATCH, shuffle=False,
                          collate_fn=collate)
test_loader  = DataLoader(test_pairs,  batch_size=BATCH, shuffle=False,
                          collate_fn=collate)

# ── 6. B-BP RNN Cell (UNCHANGED FROM V1) ──────────────────────────────────────
class BBPRNNCell(nn.Module):
    """
    Single matrix U. Two directions via transpose.

    Forward  (bbp=False): h_t = tanh(h_{t-1} @ U   + x_t + b)
    Backward (bbp=True):  h_t = tanh(h_{t-1} @ U^T + x_t + b)

    This is the Adigun-Kosko inversion: U used as U^T in the reverse direction.
    Mathematically: if U is orthogonal, U^T = U^{-1} and the backward path
    is the true inverse of the forward path.
    """
    def __init__(self, dim: int):
        super().__init__()
        k = 1.0 / math.sqrt(dim)
        self.U = nn.Parameter(torch.empty(dim, dim).uniform_(-k, k))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, h: torch.Tensor, bbp: bool = False):
        W = self.U.T if bbp else self.U
        return torch.tanh(h @ W + x + self.b)

# ── 7. Scaled Dot-Product Attention ───────────────────────────────────────────
class ScaledDotAttention(nn.Module):
    """
    No additional scoring parameters — pure dot product scaled by sqrt(d).

    This keeps the B-BP spirit: the only learned recurrent parameter is U.
    Attention is a retrieval mechanism over encoder states, not a new
    learned transformation.

    scores_i = (h_dec · h_enc_i) / sqrt(d)
    alpha    = softmax(scores, masked at PAD positions)
    context  = alpha @ H_enc
    """
    def forward(self,
                h_dec: torch.Tensor,
                H_enc: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        h_dec    : (B, d)
        H_enc    : (T_src, B, d)
        src_mask : (B, T_src) — True at PAD positions

        returns  : context (B, d)
        """
        # (B, T_src, d) @ (B, d, 1) → (B, T_src, 1) → (B, T_src)
        scores = torch.bmm(
            H_enc.permute(1, 0, 2),       # (B, T, d)
            h_dec.unsqueeze(2)             # (B, d, 1)
        ).squeeze(2) / math.sqrt(h_dec.shape[-1])

        if src_mask is not None:
            # Mask PAD positions to a large negative number
            # Using -1e9 instead of -inf to avoid NaN in edge cases
            scores = scores.masked_fill(src_mask, -1e9)

        alpha = F.softmax(scores, dim=1)   # (B, T)

        # (B, 1, T) @ (B, T, d) → (B, 1, d) → (B, d)
        context = torch.bmm(
            alpha.unsqueeze(1),
            H_enc.permute(1, 0, 2)
        ).squeeze(1)

        return context

# ── 8. B-BP Seq2Seq with Attention ────────────────────────────────────────────
class BBPSeq2SeqAttn(nn.Module):
    """
    B-BP Seq2Seq with attention.

    SHARED PARAMETERS
    ─────────────────
      cell.U    : single recurrent matrix (d × d), used as U and U^T
      E_en      : EN embedding (tied to EN output projection)
      E_de      : DE embedding (tied to DE output projection)
      ctx_proj  : Linear(d, d) — projects attention context before output
                  (one small matrix; does NOT affect the B-BP cell)

    ENCODER
    ───────
      Stores ALL hidden states H = [h_1, ..., h_T]
      Returns (h_T, H, src_mask) for decoder use

    DECODER — Input Feeding
    ───────────────────────
      At each step t:
        inp_t   = E_tgt[y_t] + context_{t-1}    ← input feeding
        h_t     = tanh(h_{t-1} @ W + inp_t + b_dec)
        ctx_t   = attention(h_t, H, mask)
        logit_t = (h_t + ctx_proj(ctx_t)) @ E_tgt^T

    FOUR INFERENCE MODES
    ────────────────────
      EN→DE:       encode(U),   decode(U)   — forward B-BP path
      DE→EN:       encode(U^T), decode(U^T) — backward B-BP path (inversion)
      EN cycle:    encode(U),   decode(U^T) — inversion self-test
      DE cycle:    encode(U^T), decode(U)   — inversion self-test
    """

    def __init__(self, en_vocab: int, de_vocab: int,
                 hid_dim: int, dropout: float = 0.3):
        super().__init__()
        self.hid_dim = hid_dim

        # ── Core B-BP cell — THE shared recurrent parameter ──────────────────
        self.cell = BBPRNNCell(hid_dim)

        # ── Language-specific embeddings (tied to output projections) ─────────
        self.E_en = nn.Embedding(en_vocab, hid_dim, padding_idx=PAD_IDX)
        self.E_de = nn.Embedding(de_vocab, hid_dim, padding_idx=PAD_IDX)

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = ScaledDotAttention()

        # ── Context projection for output combination ─────────────────────────
        self.ctx_proj = nn.Linear(hid_dim, hid_dim, bias=False)

        # ── FIX 1: Untied output projections (separate from embeddings) ────────
        # v3 used E_tgt.weight.T — embedding and output competed for same matrix
        # v4: dedicated output layer per language, embeddings free to represent
        self.W_out_en = nn.Linear(hid_dim, en_vocab, bias=False)
        self.W_out_de = nn.Linear(hid_dim, de_vocab, bias=False)

        # ── Per-layer encoder biases + decoder bias ──────────────────────────
        self.b_enc1 = nn.Parameter(torch.zeros(hid_dim))  # encoder layer 1
        self.b_enc2 = nn.Parameter(torch.zeros(hid_dim))  # encoder layer 2
        self.b_dec  = nn.Parameter(torch.zeros(hid_dim))  # decoder

        self.drop = nn.Dropout(dropout)

    # ── Internal step helpers ─────────────────────────────────────────────────
    def _enc_step1(self, x: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        return torch.tanh(h @ W + x + self.b_enc1)

    def _enc_step2(self, x: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        return torch.tanh(h @ W + x + self.b_enc2)

    def _dec_step(self, x: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        return torch.tanh(h @ W + x + self.b_dec)

    # ── Encoder ──────────────────────────────────────────────────────────────
    def encode(self, src: torch.Tensor, emb: nn.Embedding, bbp: bool):
        """
        src : (T_src, B)
        bbp : False → use U (forward EN→DE)
              True  → use U^T (backward DE→EN)

        Returns
        -------
        h_T      : (B, d)       — final hidden state (decoder init)
        H        : (T_src, B, d) — all encoder states (for attention)
        src_mask : (B, T_src)   — True at PAD positions
        """
        T, B = src.shape
        W  = self.cell.U.T if bbp else self.cell.U
        h1 = torch.zeros(B, self.hid_dim, device=src.device)
        h2 = torch.zeros(B, self.hid_dim, device=src.device)
        H2 = []
        for t in range(T):
            x  = self.drop(emb(src[t]))      # (B, d)
            h1 = self._enc_step1(x,  h1, W)  # layer 1: reads embedding
            h2 = self._enc_step2(h1, h2, W)  # layer 2: reads layer-1 output
            H2.append(h2)
        H2       = torch.stack(H2)               # (T, B, d)
        src_mask = (src == PAD_IDX).T            # (B, T)
        return h2, H2, src_mask

    # ── Decoder with attention + input feeding ─────────────────────────────
    def decode(self, tgt_in: torch.Tensor, h: torch.Tensor,
               H_enc: torch.Tensor, out_emb: nn.Embedding,
               out_proj: nn.Linear,
               bbp: bool, src_mask: torch.Tensor = None):
        """
        tgt_in  : (T_tgt, B)     — shifted-right target tokens
        h       : (B, d)         — initial hidden state (= h_T from encoder)
        H_enc   : (T_src, B, d)  — all encoder hidden states
        out_emb : embedding module for target language (input side only)
        out_proj: Linear(d, V)   — untied output projection (FIX 1)
        bbp     : False → U (forward), True → U^T (backward)
        src_mask: (B, T_src)     — PAD mask for attention

        Returns
        -------
        logits  : (T_tgt, B, V)
        """
        T, B = tgt_in.shape
        W = self.cell.U.T if bbp else self.cell.U

        # No input feeding in v4 — context initialisation not needed
        context = torch.zeros(B, self.hid_dim, device=tgt_in.device)
        logits  = []

        for t in range(T):
            # FIX 2: No input feeding — embed only, no context addition
            # v3 added context_{t-1} to embedding → caused fixed-point collapse
            x = self.drop(out_emb(tgt_in[t]))                    # (B, d)

            # B-BP recurrent step (forward or backward)
            h = self._dec_step(x, h, W)

            # Attend over all encoder states
            context = self.attention(h, H_enc, src_mask)         # (B, d)

            # FIX 1: Untied output — use dedicated W_out, not E_tgt.weight.T
            # Context still contributes to output (attention benefit preserved)
            out   = h + self.ctx_proj(context)                   # (B, d)
            logit = self.drop(out_proj(out))                     # (B, V)
            logits.append(logit)

        return torch.stack(logits)  # (T_tgt, B, V)

    # ── Translation forward passes ────────────────────────────────────────────
    def forward_en_de(self, src_en, tgt_de_in):
        """Forward B-BP path: encode U, decode U"""
        h, H, mask = self.encode(src_en, self.E_en, bbp=False)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de, bbp=False, src_mask=mask)

    def forward_de_en(self, src_de, tgt_en_in):
        """Backward B-BP path: encode U^T, decode U^T (network inverts itself)"""
        h, H, mask = self.encode(src_de, self.E_de, bbp=True)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en, bbp=True, src_mask=mask)

    # ── Cycle consistency paths ───────────────────────────────────────────────
    def cycle_en(self, src_en, tgt_en_in):
        """
        EN → encode(U) → attend(H_en) → decode(U^T) → EN_recon
        Purest B-BP test: encode forward, reconstruct via inversion.
        """
        h, H, mask = self.encode(src_en, self.E_en, bbp=False)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en, bbp=True, src_mask=mask)

    def cycle_de(self, src_de, tgt_de_in):
        """
        DE → encode(U^T) → attend(H_de) → decode(U) → DE_recon
        """
        h, H, mask = self.encode(src_de, self.E_de, bbp=True)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de, bbp=False, src_mask=mask)

    # ── Greedy inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def translate(self, src: torch.Tensor, src_emb: nn.Embedding,
                  out_emb: nn.Embedding, out_proj: nn.Linear,
                  bbp: bool, max_len: int = 50):
        """
        Greedy decode with untied output projection, no input feeding.
        Returns token ids (B, out_len).
        """
        self.eval()
        h, H, mask = self.encode(src, src_emb, bbp)
        B = src.shape[1]
        W = self.cell.U.T if bbp else self.cell.U

        tok    = torch.full((B,), SOS_IDX, dtype=torch.long, device=src.device)
        output = []

        for _ in range(max_len):
            # FIX 2: No input feeding — embed only
            x       = self.drop(out_emb(tok))              # (B, d)
            h       = self._dec_step(x, h, W)
            context = self.attention(h, H, mask)
            out     = h + self.ctx_proj(context)
            # FIX 1: Untied output projection
            tok     = out_proj(out).argmax(-1)             # (B,)
            output.append(tok)
            if (tok == EOS_IDX).all():
                break

        return torch.stack(output).T  # (B, out_len)

    def translate_en_de(self, src_en, max_len=50):
        return self.translate(src_en, self.E_en, self.E_de,
                              self.W_out_de, bbp=False, max_len=max_len)

    def translate_de_en(self, src_de, max_len=50):
        return self.translate(src_de, self.E_de, self.E_en,
                              self.W_out_en, bbp=True, max_len=max_len)

# ── 9. Loss function ──────────────────────────────────────────────────────────
def xloss(logits, tgt_out, pad_idx=PAD_IDX):
    """Cross-entropy loss, ignoring PAD positions."""
    V = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, V),
        tgt_out.reshape(-1),
        ignore_index=pad_idx,
    )

# ── 10. BLEU computation ──────────────────────────────────────────────────────
def compute_bleu(model, loader, direction, sp_hyp, sp_ref, max_batches=30):
    """
    Compute corpus BLEU using sentencepiece detokenisation.
    sp_hyp: sentencepiece model for hypothesis language
    sp_ref: sentencepiece model for reference language
    """
    model.eval()
    hyps, refs = [], []

    with torch.no_grad():
        for i, (en_b, de_b) in enumerate(loader):
            if i >= max_batches:
                break
            en_b, de_b = en_b.to(device), de_b.to(device)

            if direction == "en_de":
                preds   = model.translate_en_de(en_b)    # (B, out_len)
                ref_ids = de_b.T                          # (B, T)
                sp_h, sp_r = sp_hyp, sp_ref
            else:
                preds   = model.translate_de_en(de_b)
                ref_ids = en_b.T
                sp_h, sp_r = sp_hyp, sp_ref

            for p, r in zip(preds.tolist(), ref_ids.tolist()):
                # Strip EOS and PAD from hypothesis
                clean_p = []
                for tok in p:
                    if tok == EOS_IDX:
                        break
                    if tok not in (PAD_IDX, EOS_IDX, SOS_IDX):
                        clean_p.append(tok)

                # Strip special tokens from reference
                clean_r = [tok for tok in r
                           if tok not in (PAD_IDX, EOS_IDX, SOS_IDX)]

                hyps.append(sp_h.decode(clean_p))
                refs.append([sp_r.decode(clean_r)])

    return sacrebleu.corpus_bleu(hyps, list(zip(*refs))).score

# ── 11. Training loop ─────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, lam=0.0):
    model.train()
    tot = nmt_tot = cyc_tot = n = 0

    for en_b, de_b in loader:
        en_b, de_b = en_b.to(device), de_b.to(device)
        opt.zero_grad()

        # ── EN→DE  (forward, uses U) ────────────────────────────────────────
        l_en_de = xloss(model.forward_en_de(en_b, de_b[:-1]), de_b[1:])

        # ── DE→EN  (backward, uses U^T) ─────────────────────────────────────
        l_de_en = xloss(model.forward_de_en(de_b, en_b[:-1]), en_b[1:])

        # ── EN cycle: encode(U) → decode(U^T) → EN_recon ────────────────────
        l_cyc_en = xloss(model.cycle_en(en_b, en_b[:-1]), en_b[1:])

        # ── DE cycle: encode(U^T) → decode(U) → DE_recon ────────────────────
        l_cyc_de = xloss(model.cycle_de(de_b, de_b[:-1]), de_b[1:])

        nmt  = l_en_de + l_de_en
        cyc  = l_cyc_en + l_cyc_de
        loss = nmt + lam * cyc

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        tot     += loss.item()
        nmt_tot += nmt.item()
        cyc_tot += cyc.item()
        n       += 1

    return tot / n, nmt_tot / n, cyc_tot / n

@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    tot, n = 0., 0
    for en_b, de_b in loader:
        en_b, de_b = en_b.to(device), de_b.to(device)
        l1 = xloss(model.forward_en_de(en_b, de_b[:-1]), de_b[1:])
        l2 = xloss(model.forward_de_en(de_b, en_b[:-1]), en_b[1:])
        tot += (l1 + l2).item()
        n   += 1
    return tot / n

# ── 12. Main ──────────────────────────────────────────────────────────────────
def main():
    # ── Model ──────────────────────────────────────────────────────────────────
    model = BBPSeq2SeqAttn(
        en_vocab=EN_VOCAB,
        de_vocab=DE_VOCAB,
        hid_dim=HID,
        dropout=0.3,
    ).to(device)

    # Parameter breakdown
    total    = sum(p.numel() for p in model.parameters())
    rec_p    = model.cell.U.numel()
    emb_p    = model.E_en.weight.numel() + model.E_de.weight.numel()
    out_p    = model.W_out_en.weight.numel() + model.W_out_de.weight.numel()
    ctx_p    = model.ctx_proj.weight.numel()
    other_p  = total - rec_p - emb_p - out_p - ctx_p

    print(f"\n{'='*70}")
    print(f"  B-BP NMT v4  |  Untied Output + No Input Feeding  |  Adigun-Kosko")
    print(f"{'='*70}")
    print(f"  Total params         : {total:,}")
    print(f"  cell.U (recurrent)   : {rec_p:,}   <- shared: enc-L1, enc-L2, dec")
    print(f"  E_en + E_de (input)  : {emb_p:,}   <- BPE 8k, input side only")
    print(f"  W_out_en + W_out_de  : {out_p:,}   <- FIX 1: untied output projections")
    print(f"  ctx_proj             : {ctx_p:,}   <- attention context combiner")
    print(f"  biases + other       : {other_p:,}")
    print(f"  hid={HID}  batch={BATCH}  lr={LR}  enc_layers={ENC_LAYERS}  epochs={EPOCHS}")
    print(f"  input_feeding=False (FIX 2)  cycle_warmup=True")

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5
    )

    best_val      = 1e9
    best_en_de    = 0.0
    best_de_en    = 0.0
    epoch_records = []

    # ── Result directory ───────────────────────────────────────────────────────
    os.makedirs("bbp_v4_results", exist_ok=True)

    # CSV log flushed after every epoch — survives VM crashes
    csv_file = open("bbp_v4_results/training_log.csv", "w")
    csv_file.write("epoch,train_loss,nmt_loss,cyc_loss,val_loss,"
                   "bleu_en_de,bleu_de_en,lambda,elapsed_s\n")
    csv_file.flush()

    print(f"\n{'Ep':>3} | {'Train':>8} | {'NMT':>8} | {'Cyc':>8} | "
          f"{'Val':>8} | {'EN→DE':>7} | {'DE→EN':>7} | {'λ':>5} | {'s':>5}")
    print("-" * 80)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        lam        = get_lambda(ep)
        tr, nt, cy = train_epoch(model, train_loader, opt, lam)
        vl         = eval_loss(model, val_loader)
        sched.step(vl)

        b_en_de = compute_bleu(model, val_loader, "en_de", sp_de, sp_de)
        b_de_en = compute_bleu(model, val_loader, "de_en", sp_en, sp_en)
        elapsed = time.time() - t0

        mark = " ◀" if vl < best_val else ""
        print(f"{ep:>3} | {tr:>8.4f} | {nt:>8.4f} | {cy:>8.4f} | "
              f"{vl:>8.4f} | {b_en_de:>7.2f} | {b_de_en:>7.2f} | "
              f"{lam:.2f} | {elapsed:>5.0f}{mark}")

        # Flush CSV row immediately — one row per epoch, never lost
        csv_file.write(f"{ep},{tr:.4f},{nt:.4f},{cy:.4f},{vl:.4f},"
                       f"{b_en_de:.2f},{b_de_en:.2f},{lam:.2f},{elapsed:.0f}\n")
        csv_file.flush()

        # Full checkpoint every epoch (model + optimizer state = resumable)
        torch.save({
            "epoch"      : ep,
            "model_state": model.state_dict(),
            "opt_state"  : opt.state_dict(),
            "val_loss"   : vl,
            "bleu_en_de" : b_en_de,
            "bleu_de_en" : b_de_en,
        }, f"bbp_v4_results/checkpoint_ep{ep:02d}.pt")

        # Delete checkpoint from 3 epochs ago to save disk
        # Keeps: best.pt + last 3 epoch checkpoints always
        old = f"bbp_v4_results/checkpoint_ep{ep-3:02d}.pt"
        if ep > 3 and os.path.exists(old) and vl >= best_val:
            os.remove(old)

        # Save best separately — never deleted
        if vl < best_val:
            best_val   = vl
            best_en_de = b_en_de
            best_de_en = b_de_en
            torch.save({
                "epoch"      : ep,
                "model_state": model.state_dict(),
                "opt_state"  : opt.state_dict(),
                "val_loss"   : vl,
                "bleu_en_de" : b_en_de,
                "bleu_de_en" : b_de_en,
            }, "bbp_v4_results/best.pt")

        epoch_records.append({
            "epoch": ep, "train": tr, "nmt": nt, "cyc": cy,
            "val": vl, "en_de": b_en_de, "de_en": b_de_en,
        })

    csv_file.close()

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    best_ckpt = torch.load("bbp_v4_results/best.pt")
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Best checkpoint: epoch {best_ckpt['epoch']}  "
          f"val_loss={best_ckpt['val_loss']:.4f}")

    test_en_de = compute_bleu(model, test_loader, "en_de",
                              sp_de, sp_de, max_batches=9999)
    test_de_en = compute_bleu(model, test_loader, "de_en",
                              sp_en, sp_en, max_batches=9999)

    print(f"\n{'='*70}")
    print(f"  FINAL TEST RESULTS — B-BP NMT v4")
    print(f"{'='*70}")
    print(f"  BLEU EN→DE  (forward,  uses U)    : {test_en_de:.2f}")
    print(f"  BLEU DE→EN  (backward, uses U^T)  : {test_de_en:.2f}")
    print(f"  Average BLEU                      : {(test_en_de+test_de_en)/2:.2f}")
    print(f"{'='*70}")

    # ── V1 vs V2 comparison ────────────────────────────────────────────────────
    print(f"\n  V1 vs V2 Comparison")
    print(f"  {'Metric':<30} {'V1':>8} {'V2':>8}")
    print(f"  {'-'*48}")
    print(f"  {'EN→DE BLEU (test)':<30} {'5.37':>8} {test_en_de:>8.2f}")
    print(f"  {'DE→EN BLEU (test)':<30} {'3.47':>8} {test_de_en:>8.2f}")
    print(f"  {'Avg BLEU (test)':<30} {'4.42':>8} {(test_en_de+test_de_en)/2:>8.2f}")
    print(f"  {'Tokenisation':<30} {'word-10k':>8} {'BPE-8k':>8}")
    print(f"  {'Attention':<30} {'none':>8} {'dot':>8}")

    # ── Save final summary JSON ────────────────────────────────────────────────
    import json
    summary = {
        "model": "B-BP NMT v4",
        "total_params" : sum(p.numel() for p in model.parameters()),
        "recurrent_U"  : model.cell.U.numel(),
        "best_epoch"   : best_ckpt["epoch"],
        "best_val_loss": float(best_val),
        "test_bleu_en_de": float(test_en_de),
        "test_bleu_de_en": float(test_de_en),
        "test_bleu_avg"  : float((test_en_de + test_de_en) / 2),
        "training_curve" : epoch_records,
    }
    with open("bbp_v4_results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll results saved to bbp_v4_results/")
    print("  training_log.csv   — epoch-by-epoch metrics")
    print("  best.pt            — best model checkpoint")
    print("  summary.json       — final summary with full training curve")
    print("  checkpoint_ep*.pt  — last 3 epoch checkpoints (resumable)")


if __name__ == "__main__":
    main()
