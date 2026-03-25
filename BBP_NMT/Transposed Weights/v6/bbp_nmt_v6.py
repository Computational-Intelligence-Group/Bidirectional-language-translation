"""
B-BP NMT v6 — Adigun-Kosko Framework
B-BP Bidirectional Encoder + hid=512 + 32k Shared BPE + Stronger Ortho
=======================================================================
Five improvements from v5 full dataset (EN->DE 8.37 / DE->EN 11.42 BLEU):

  CHANGE 1 — B-BP Bidirectional Encoder
    v5 encoder read source left-to-right only (unidirectional).
    Each hidden state h_t only knew tokens 1..t, not t+1..T.
    v6: bidirectional encoder using the SAME single matrix U:
      Forward  (left to right):  h_fwd_t = GRU(embed(x_t), h_fwd_{t-1}, W=U)
      Backward (right to left):  h_bwd_t = GRU(embed(x_t), h_bwd_{t+1}, W=U^T)
      Combined: H_t = bi_proj( cat(h_fwd_t, h_bwd_t) )   Linear(2d -> d)
    U reads the sentence forward, U^T reads it backward.
    Extends the Adigun-Kosko inversion to sequence reading direction.
    Encoder is now direction-agnostic — only the decoder selects U vs U^T.

  CHANGE 2 — Hidden dimension 256 -> 512
    With full WMT14 data (3.5M pairs), architecture is now the bottleneck.
    Doubling hid quadruples recurrent capacity (U: 65k -> 262k params).
    Embeddings and output projections scale accordingly.

  CHANGE 3 — 32k Shared BPE vocabulary
    v5 used 8k separate BPE models per language.
    German morphological compounds split into 3-4 pieces at 8k, causing
    output corruption (e.g. "verhaftet" -> "arhaftet").
    v6: single SentencePiece model trained on combined EN+DE text, vocab=32k.
    Cross-lingual subword sharing helps with cognates and named entities.

  CHANGE 4 — Stronger orthogonality regulariser (mu=0.01 -> 0.05)
    v5 full dataset converged to ortho error 0.73. More regularisation
    pressure should tighten U further toward exact inversion.

  CHANGE 5 — Beam search length penalty (alpha=0.6)
    v5 beam search had no length normalisation.
    Length penalty: score = log_prob / (length ** 0.6)
    Prevents beam from preferring short outputs. Typically +0.5-1 BLEU.

UNCHANGED FROM V5 FULL:
  - Full WMT14 training data (TRAIN_CAP=None)
  - GRU B-BP cell (shared U, ortho init)
  - Scaled dot-product attention (no learned scoring params)
  - No input feeding
  - Untied output projections (W_out_en, W_out_de)
  - Cycle warmup: ep1-5 lam=0.00 | ep6-10 lam=0.05 | ep11+ lam=0.10
  - batch=256, lr=1e-3, epochs=30
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import math, time, random, os, json
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
BPE_VOCAB      = 32_000     # CHANGE 3: shared vocab, was 8k per-language
MAX_LEN        = 40
TRAIN_CAP      = None       # full WMT14
BATCH          = 256
HID            = 512        # CHANGE 2: was 256
LR             = 1e-3
EPOCHS         = 30
MU_ORTHO       = 0.05       # CHANGE 4: was 0.01
BEAM_SIZE      = 5          # CHANGE 5: was 4
LENGTH_PENALTY = 0.6        # CHANGE 5: beam length penalty alpha
SEED           = 42

random.seed(SEED)
torch.manual_seed(SEED)

# Cycle warmup — unchanged
def get_lambda(epoch):
    if epoch <= 5:  return 0.00
    if epoch <= 10: return 0.05
    return 0.10

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

all_idx = list(range(len(train_raw)))
random.shuffle(all_idx)
sub_idx = all_idx if TRAIN_CAP is None else all_idx[:TRAIN_CAP]

# ── 4. Train SHARED BPE model (CHANGE 3) ──────────────────────────────────────
cap_str = "full dataset" if TRAIN_CAP is None else f"{TRAIN_CAP:,}"
print(f"\nBuilding shared BPE vocab={BPE_VOCAB} on {cap_str} ...")

en_lines, de_lines = [], []
for idx in sub_idx:
    en, de = get_pair(train_raw[idx])
    if en and de:
        en_lines.append(en.lower())
        de_lines.append(de.lower())

# Write combined EN+DE text for shared BPE training
with open("/tmp/train_shared.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(en_lines + de_lines))

spm.SentencePieceTrainer.train(
    input="/tmp/train_shared.txt",
    model_prefix="/tmp/spm_shared",
    vocab_size=BPE_VOCAB,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=PAD_IDX, pad_piece="<pad>",
    unk_id=UNK_IDX, unk_piece="<unk>",
    bos_id=SOS_IDX, bos_piece="<sos>",
    eos_id=EOS_IDX, eos_piece="<eos>",
    shuffle_input_sentence=True,
)

# Both languages use the same tokenizer
sp = spm.SentencePieceProcessor(model_file="/tmp/spm_shared.model")
sp_en = sp
sp_de = sp
VOCAB = sp.vocab_size()
EN_VOCAB = VOCAB
DE_VOCAB = VOCAB
print(f"Shared vocab size: {VOCAB:,}")

# ── 5. Dataset encoding ────────────────────────────────────────────────────────
def batch_encode(sp_model, texts):
    return sp_model.encode([t.lower().strip() for t in texts])

def make_pairs_batch(en_texts, de_texts):
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

print("\nEncoding pairs with shared BPE ...")
train_pairs = make_pairs_batch(en_lines, de_lines)
val_pairs   = make_pairs_from_raw(val_raw)
test_pairs  = make_pairs_from_raw(test_raw)
print(f"Train: {len(train_pairs):,}  Val: {len(val_pairs):,}  Test: {len(test_pairs):,}")

train_loader = DataLoader(train_pairs, batch_size=BATCH, shuffle=True,
                          collate_fn=collate, drop_last=True,
                          num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_pairs, batch_size=BATCH, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_pairs, batch_size=BATCH, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)

# ── 6. B-BP GRU Cell (unchanged from v5, dim=512) ─────────────────────────────
class BBPGRUCell(nn.Module):
    """
    GRU-style B-BP cell. Single shared recurrent matrix U.
    Forward (bbp=False): W = U   | Backward (bbp=True): W = U^T
    Gates: z (update), r (reset), n (candidate)
    All three gates use the same U / U^T as recurrent weight.
    W_z, W_r, W_n are input-side projections — not subject to inversion.
    U is orthogonally initialised.
    """
    def __init__(self, dim):
        super().__init__()
        self.U   = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.U)
        self.W_z = nn.Linear(dim, dim, bias=True)
        self.W_r = nn.Linear(dim, dim, bias=True)
        self.W_n = nn.Linear(dim, dim, bias=True)

    def forward(self, x, h, bbp=False):
        W = self.U.T if bbp else self.U
        z = torch.sigmoid(self.W_z(x) + h @ W)
        r = torch.sigmoid(self.W_r(x) + h @ W)
        n = torch.tanh(self.W_n(x) + (r * h) @ W)
        return (1.0 - z) * h + z * n

# ── 7. Scaled Dot-Product Attention (unchanged) ───────────────────────────────
class ScaledDotAttention(nn.Module):
    def forward(self, h_dec, H_enc, src_mask=None):
        """
        h_dec   : (B, d)
        H_enc   : (T, B, d)
        returns : context (B, d)
        """
        scores = torch.bmm(
            H_enc.permute(1, 0, 2),      # (B, T, d)
            h_dec.unsqueeze(2)            # (B, d, 1)
        ).squeeze(2) / math.sqrt(h_dec.shape[-1])
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        alpha   = F.softmax(scores, dim=1)
        context = torch.bmm(
            alpha.unsqueeze(1),
            H_enc.permute(1, 0, 2)
        ).squeeze(1)
        return context

# ── 8. B-BP Seq2Seq Model v6 ──────────────────────────────────────────────────
class BBPSeq2SeqAttn(nn.Module):
    """
    B-BP NMT v6.

    Encoder — B-BP Bidirectional (CHANGE 1):
      Forward  (left->right): h_fwd_t = GRU(embed(x_t), h_fwd_{t-1}, W=U)
      Backward (right->left): h_bwd_t = GRU(embed(x_t), h_bwd_{t+1}, W=U^T)
      Combined: H_t = bi_proj( [h_fwd_t ; h_bwd_t] )   shape (T, B, d)
      Decoder init: bi_proj( [h_fwd_T ; h_bwd_0] )      shape (B, d)

    The encoder is direction-agnostic: always uses U forward and U^T backward.
    Only the decoder selects U vs U^T to distinguish translation direction.

    Decoder (unchanged from v5):
      x_t = embed(y_{t-1})                           no input feeding
      h_t = GRU(x_t, h_{t-1}, bbp)                  U or U^T
      ctx = attention(h_t, H)
      o_t = h_t + ctx_proj(ctx)
      logit = W_out_tgt(o_t)                         untied

    Four inference modes:
      EN->DE: encode(EN), decode(U)   -> W_out_de
      DE->EN: encode(DE), decode(U^T) -> W_out_en
      EN cyc: encode(EN), decode(U^T) -> W_out_en
      DE cyc: encode(DE), decode(U)   -> W_out_de
    """
    def __init__(self, vocab_size, hid_dim, dropout=0.3):
        super().__init__()
        self.hid_dim   = hid_dim
        self.cell      = BBPGRUCell(hid_dim)
        # Separate embeddings per language — both use shared vocab
        self.E_en      = nn.Embedding(vocab_size, hid_dim, padding_idx=PAD_IDX)
        self.E_de      = nn.Embedding(vocab_size, hid_dim, padding_idx=PAD_IDX)
        # Untied output projections
        self.W_out_en  = nn.Linear(hid_dim, vocab_size, bias=False)
        self.W_out_de  = nn.Linear(hid_dim, vocab_size, bias=False)
        # CHANGE 1: bidirectional projection (2*hid -> hid)
        self.bi_proj   = nn.Linear(hid_dim * 2, hid_dim, bias=False)
        # Attention context combiner
        self.ctx_proj  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.attention = ScaledDotAttention()
        self.drop      = nn.Dropout(dropout)

    def ortho_loss(self):
        """CHANGE 4: mu=0.05. Penalises U for deviating from U^T U = I."""
        U = self.cell.U
        I = torch.eye(U.shape[0], device=U.device)
        return torch.norm(U.T @ U - I, p='fro') ** 2

    def encode(self, src, emb):
        """
        CHANGE 1: B-BP Bidirectional Encoder.
        Forward pass (left->right) uses W=U.
        Backward pass (right->left) uses W=U^T.
        Both passes share the same U — no extra parameters for inversion.

        src : (T, B)
        emb : embedding module for source language
        Returns: h_T (B, d), H (T, B, d), src_mask (B, T)
        """
        T, B = src.shape

        # ── Forward pass: left to right, W=U ─────────────────────────────────
        h_fwd = torch.zeros(B, self.hid_dim, device=src.device)
        H_fwd = []
        for t in range(T):
            x     = self.drop(emb(src[t]))
            h_fwd = self.cell(x, h_fwd, bbp=False)   # W = U
            H_fwd.append(h_fwd)
        # H_fwd[t] has seen tokens 0..t (left context)

        # ── Backward pass: right to left, W=U^T ──────────────────────────────
        h_bwd = torch.zeros(B, self.hid_dim, device=src.device)
        H_bwd = [None] * T
        for t in range(T - 1, -1, -1):
            x     = self.drop(emb(src[t]))
            h_bwd = self.cell(x, h_bwd, bbp=True)    # W = U^T
            H_bwd[t] = h_bwd
        # H_bwd[t] has seen tokens t..T-1 (right context)

        # ── Combine forward and backward at each position ─────────────────────
        H_combined = []
        for t in range(T):
            cat_t = torch.cat([H_fwd[t], H_bwd[t]], dim=-1)  # (B, 2*d)
            H_combined.append(self.bi_proj(cat_t))             # (B, d)
        H = torch.stack(H_combined)  # (T, B, d)

        # Decoder init: combine final forward state with final backward state
        # h_fwd_T = H_fwd[-1] (has seen full sentence left-to-right)
        # h_bwd_0 = H_bwd[0]  (has seen full sentence right-to-left)
        h_init = self.bi_proj(
            torch.cat([H_fwd[-1], H_bwd[0]], dim=-1)
        )  # (B, d)

        src_mask = (src == PAD_IDX).T  # (B, T)
        return h_init, H, src_mask

    def decode(self, tgt_in, h, H_enc, out_emb, out_proj, bbp, src_mask=None):
        """
        Decoder unchanged from v5.
        tgt_in  : (T_tgt, B)
        h       : (B, d)    decoder init from encoder
        H_enc   : (T, B, d) encoder memory for attention
        out_emb : source-side embedding (input only)
        out_proj: untied output projection
        bbp     : False=U (EN->DE), True=U^T (DE->EN)
        """
        T, B   = tgt_in.shape
        logits = []
        for t in range(T):
            x     = self.drop(out_emb(tgt_in[t]))     # no input feeding
            h     = self.cell(x, h, bbp=bbp)
            ctx   = self.attention(h, H_enc, src_mask)
            out   = h + self.ctx_proj(ctx)
            logit = self.drop(out_proj(out))
            logits.append(logit)
        return torch.stack(logits)  # (T, B, V)

    # ── Four forward passes ───────────────────────────────────────────────────
    def forward_en_de(self, src_en, tgt_de_in):
        """EN->DE: bidir encode EN, decode with U -> German"""
        h, H, mask = self.encode(src_en, self.E_en)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de,
                           bbp=False, src_mask=mask)

    def forward_de_en(self, src_de, tgt_en_in):
        """DE->EN: bidir encode DE, decode with U^T -> English"""
        h, H, mask = self.encode(src_de, self.E_de)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en,
                           bbp=True, src_mask=mask)

    def cycle_en(self, src_en, tgt_en_in):
        """EN cycle: bidir encode EN, decode with U^T -> reconstruct EN"""
        h, H, mask = self.encode(src_en, self.E_en)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en,
                           bbp=True, src_mask=mask)

    def cycle_de(self, src_de, tgt_de_in):
        """DE cycle: bidir encode DE, decode with U -> reconstruct DE"""
        h, H, mask = self.encode(src_de, self.E_de)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de,
                           bbp=False, src_mask=mask)

    # ── Beam search with length penalty (CHANGE 5) ───────────────────────────
    @torch.no_grad()
    def beam_decode(self, src, src_emb, out_emb, out_proj, bbp,
                    beam_size=5, max_len=50, length_penalty=0.6):
        """
        Beam search with length penalty.
        Length-normalised score: log_prob / (len ** length_penalty)
        src: (T, 1) — single sentence
        """
        self.eval()
        h_init, H, mask = self.encode(src, src_emb)

        # beams: (raw_log_score, token_list, hidden_state)
        beams     = [(0.0, [SOS_IDX], h_init)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break
            all_cands = []
            for score, tokens, h in beams:
                if tokens[-1] == EOS_IDX:
                    completed.append((score, tokens))
                    continue
                tok = torch.tensor([tokens[-1]], device=src.device)
                x   = self.drop(out_emb(tok))
                h   = self.cell(x, h, bbp=bbp)
                ctx = self.attention(h, H, mask)
                out = h + self.ctx_proj(ctx)
                log_probs = F.log_softmax(out_proj(out), dim=-1)
                topk_scores, topk_toks = log_probs[0].topk(beam_size)
                for s, t in zip(topk_scores.tolist(), topk_toks.tolist()):
                    all_cands.append((score + s, tokens + [t], h))

            if not all_cands:
                break
            # Sort by length-normalised score for pruning
            all_cands.sort(
                key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty),
                reverse=True
            )
            beams = all_cands[:beam_size]
            if all(b[1][-1] == EOS_IDX for b in beams):
                for b in beams:
                    completed.append((b[0], b[1]))
                break

        for b in beams:
            completed.append((b[0], b[1]))

        if not completed:
            return []

        # Select best by length-normalised score
        def norm_score(item):
            score, tokens = item
            length = max(len(tokens) - 1, 1)  # exclude SOS
            return score / (length ** length_penalty)

        best = sorted(completed, key=norm_score, reverse=True)[0][1]
        return [t for t in best if t not in (SOS_IDX, EOS_IDX, PAD_IDX)]

    def beam_en_de(self, src_en, beam_size=BEAM_SIZE, max_len=50):
        return self.beam_decode(src_en, self.E_en, self.E_de, self.W_out_de,
                                bbp=False, beam_size=beam_size, max_len=max_len,
                                length_penalty=LENGTH_PENALTY)

    def beam_de_en(self, src_de, beam_size=BEAM_SIZE, max_len=50):
        return self.beam_decode(src_de, self.E_de, self.E_en, self.W_out_en,
                                bbp=True, beam_size=beam_size, max_len=max_len,
                                length_penalty=LENGTH_PENALTY)

# ── 9. Loss ───────────────────────────────────────────────────────────────────
def xloss(logits, tgt_out, pad_idx=PAD_IDX):
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V),
                           tgt_out.reshape(-1),
                           ignore_index=pad_idx)

# ── 10. BLEU with beam search ─────────────────────────────────────────────────
def compute_bleu(model, loader, direction, sp_hyp, sp_ref, max_batches=30):
    """Both sp_hyp and sp_ref are the shared tokenizer in v6."""
    model.eval()
    hyps, refs = [], []
    count = 0
    with torch.no_grad():
        for en_b, de_b in loader:
            if count >= max_batches:
                break
            en_b, de_b = en_b.to(device), de_b.to(device)
            B = en_b.shape[1]
            for i in range(B):
                if direction == "en_de":
                    src     = en_b[:, i:i+1]
                    ref_ids = de_b[:, i].tolist()
                    pred    = model.beam_en_de(src)
                else:
                    src     = de_b[:, i:i+1]
                    ref_ids = en_b[:, i].tolist()
                    pred    = model.beam_de_en(src)
                clean_r = [t for t in ref_ids
                           if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                hyps.append(sp_hyp.decode(pred))
                refs.append([sp_ref.decode(clean_r)])
            count += 1
    return sacrebleu.corpus_bleu(hyps, list(zip(*refs))).score

# ── 11. Training loop ─────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, lam=0.0):
    model.train()
    tot = nmt_tot = cyc_tot = ortho_tot = n = 0
    for en_b, de_b in loader:
        en_b, de_b = en_b.to(device), de_b.to(device)
        opt.zero_grad()

        l_en_de  = xloss(model.forward_en_de(en_b, de_b[:-1]), de_b[1:])
        l_de_en  = xloss(model.forward_de_en(de_b, en_b[:-1]), en_b[1:])
        l_cyc_en = xloss(model.cycle_en(en_b, en_b[:-1]), en_b[1:])
        l_cyc_de = xloss(model.cycle_de(de_b, de_b[:-1]), de_b[1:])

        nmt   = l_en_de + l_de_en
        cyc   = l_cyc_en + l_cyc_de
        ortho = model.ortho_loss()
        loss  = nmt + lam * cyc + MU_ORTHO * ortho

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        tot       += loss.item()
        nmt_tot   += nmt.item()
        cyc_tot   += cyc.item()
        ortho_tot += ortho.item()
        n         += 1

    return tot / n, nmt_tot / n, cyc_tot / n, ortho_tot / n

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
    model = BBPSeq2SeqAttn(
        vocab_size=VOCAB,
        hid_dim=HID,
        dropout=0.3,
    ).to(device)

    # Parameter breakdown
    total    = sum(p.numel() for p in model.parameters())
    rec_p    = model.cell.U.numel()
    gate_p   = (sum(p.numel() for p in model.cell.W_z.parameters()) +
                sum(p.numel() for p in model.cell.W_r.parameters()) +
                sum(p.numel() for p in model.cell.W_n.parameters()))
    biproj_p = model.bi_proj.weight.numel()
    emb_p    = model.E_en.weight.numel() + model.E_de.weight.numel()
    out_p    = model.W_out_en.weight.numel() + model.W_out_de.weight.numel()
    ctx_p    = model.ctx_proj.weight.numel()
    other_p  = total - rec_p - gate_p - biproj_p - emb_p - out_p - ctx_p

    print(f"\n{'='*72}")
    print(f"  B-BP NMT v6  |  BiDir Encoder + hid=512 + 32k BPE  |  Adigun-Kosko")
    print(f"{'='*72}")
    print(f"  Total params              : {total:,}")
    print(f"  cell.U (recurrent)        : {rec_p:,}   <- shared U, ortho init")
    print(f"  GRU gates (W_z,W_r,W_n)  : {gate_p:,}   <- input projections")
    print(f"  bi_proj (2*hid -> hid)    : {biproj_p:,}   <- bidir combiner (NEW)")
    print(f"  E_en + E_de (input emb)   : {emb_p:,}   <- 32k shared vocab")
    print(f"  W_out_en + W_out_de       : {out_p:,}   <- untied output")
    print(f"  ctx_proj                  : {ctx_p:,}   <- attention combiner")
    print(f"  biases + other            : {other_p:,}")
    print(f"  hid={HID}  batch={BATCH}  lr={LR}  epochs={EPOCHS}  beam_k={BEAM_SIZE}")
    print(f"  mu_ortho={MU_ORTHO}  length_penalty={LENGTH_PENALTY}  cycle_warmup=True")
    print(f"  encoder=B-BP-BiDir  input_feeding=False  vocab=32k-shared")

    with torch.no_grad():
        U = model.cell.U
        I = torch.eye(U.shape[0], device=U.device)
        init_ortho = torch.norm(U.T @ U - I, p='fro').item()
    print(f"  Initial ||U^T U - I||_F   : {init_ortho:.4f}  (0.00 = perfect ortho)")

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    best_val      = 1e9
    epoch_records = []
    os.makedirs("bbp_v6_results", exist_ok=True)

    csv_file = open("bbp_v6_results/training_log.csv", "w")
    csv_file.write("epoch,train_loss,nmt_loss,cyc_loss,ortho_loss,"
                   "val_loss,bleu_en_de,bleu_de_en,lambda,elapsed_s\n")
    csv_file.flush()

    print(f"\n{'Ep':>3} | {'Train':>8} | {'NMT':>8} | {'Cyc':>8} | {'Ortho':>7} | "
          f"{'Val':>8} | {'EN->DE':>6} | {'DE->EN':>6} | {'lam':>4} | {'s':>6}")
    print("-" * 94)

    for ep in range(1, EPOCHS + 1):
        t0               = time.time()
        lam              = get_lambda(ep)
        tr, nt, cy, orth = train_epoch(model, train_loader, opt, lam)
        vl               = eval_loss(model, val_loader)
        sched.step(vl)
        b_en_de = compute_bleu(model, val_loader, "en_de", sp_de, sp_de)
        b_de_en = compute_bleu(model, val_loader, "de_en", sp_en, sp_en)
        elapsed = time.time() - t0

        mark = " <" if vl < best_val else ""
        print(f"{ep:>3} | {tr:>8.4f} | {nt:>8.4f} | {cy:>8.4f} | {orth:>7.4f} | "
              f"{vl:>8.4f} | {b_en_de:>6.2f} | {b_de_en:>6.2f} | "
              f"{lam:.2f} | {elapsed:>6.0f}{mark}", flush=True)

        csv_file.write(f"{ep},{tr:.4f},{nt:.4f},{cy:.4f},{orth:.4f},{vl:.4f},"
                       f"{b_en_de:.2f},{b_de_en:.2f},{lam:.2f},{elapsed:.0f}\n")
        csv_file.flush()

        torch.save({
            "epoch": ep, "model_state": model.state_dict(),
            "opt_state": opt.state_dict(), "val_loss": vl,
            "bleu_en_de": b_en_de, "bleu_de_en": b_de_en,
        }, f"bbp_v6_results/checkpoint_ep{ep:02d}.pt")

        old = f"bbp_v6_results/checkpoint_ep{ep-3:02d}.pt"
        if ep > 3 and os.path.exists(old) and vl >= best_val:
            os.remove(old)

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(), "val_loss": vl,
                "bleu_en_de": b_en_de, "bleu_de_en": b_de_en,
            }, "bbp_v6_results/best.pt")

        epoch_records.append({
            "epoch": ep, "train": tr, "nmt": nt, "cyc": cy,
            "ortho": orth, "val": vl, "en_de": b_en_de, "de_en": b_de_en,
        })

    csv_file.close()

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    best_ckpt = torch.load("bbp_v6_results/best.pt")
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Best checkpoint: epoch {best_ckpt['epoch']}  "
          f"val_loss={best_ckpt['val_loss']:.4f}")

    with torch.no_grad():
        U = model.cell.U
        I = torch.eye(U.shape[0], device=U.device)
        final_ortho = torch.norm(U.T @ U - I, p='fro').item()

    test_en_de = compute_bleu(model, test_loader, "en_de",
                              sp_de, sp_de, max_batches=9999)
    test_de_en = compute_bleu(model, test_loader, "de_en",
                              sp_en, sp_en, max_batches=9999)

    print(f"\n{'='*72}")
    print(f"  FINAL TEST RESULTS — B-BP NMT v6")
    print(f"{'='*72}")
    print(f"  BLEU EN->DE  (forward,  uses U)    : {test_en_de:.2f}")
    print(f"  BLEU DE->EN  (backward, uses U^T)  : {test_de_en:.2f}")
    print(f"  Average BLEU                       : {(test_en_de+test_de_en)/2:.2f}")
    print(f"  Final ||U^T U - I||_F              : {final_ortho:.4f}")
    print(f"{'='*72}")

    print(f"\n  V4 vs V5-131k vs V5-full vs V6")
    print(f"  {'Metric':<32} {'V4':>6} {'V5-131k':>8} {'V5-full':>8} {'V6':>6}")
    print(f"  {'-'*62}")
    print(f"  {'EN->DE BLEU (test)':<32} {'4.10':>6} {'4.70':>8} {'8.37':>8} {test_en_de:>6.2f}")
    print(f"  {'DE->EN BLEU (test)':<32} {'6.44':>6} {'7.39':>8} {'11.42':>8} {test_de_en:>6.2f}")
    print(f"  {'Avg BLEU (test)':<32} {'5.27':>6} {'6.05':>8} {'9.90':>8} {(test_en_de+test_de_en)/2:>6.2f}")
    print(f"  {'Encoder':<32} {'uni':>6} {'uni':>8} {'uni':>8} {'bidir':>6}")
    print(f"  {'HID':<32} {'256':>6} {'256':>8} {'256':>8} {'512':>6}")
    print(f"  {'BPE vocab':<32} {'8k':>6} {'8k':>8} {'8k':>8} {'32k':>6}")

    with open("bbp_v6_results/summary.json", "w") as f:
        json.dump({
            "model": "B-BP NMT v6",
            "total_params": total,
            "recurrent_U": rec_p,
            "gru_gate_params": gate_p,
            "bidir_proj_params": biproj_p,
            "hid_dim": HID,
            "vocab_size": VOCAB,
            "beam_size": BEAM_SIZE,
            "length_penalty": LENGTH_PENALTY,
            "mu_ortho": MU_ORTHO,
            "init_ortho_error": float(init_ortho),
            "final_ortho_error": float(final_ortho),
            "best_epoch": best_ckpt["epoch"],
            "best_val_loss": float(best_val),
            "test_bleu_en_de": float(test_en_de),
            "test_bleu_de_en": float(test_de_en),
            "test_bleu_avg": float((test_en_de + test_de_en) / 2),
            "training_curve": epoch_records,
        }, f, indent=2)

    print("\nAll results saved to bbp_v6_results/")
    print("  training_log.csv   — epoch metrics")
    print("  best.pt            — best model checkpoint")
    print("  summary.json       — full summary")
    print("  checkpoint_ep*.pt  — last 3 epoch checkpoints (resumable)")


if __name__ == "__main__":
    main()
