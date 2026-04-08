"""
A100 Optimized Luong Simulation (EN <-> DE)
============================================
Architecture: 2-Layer Bi-GRU Encoder + Standard GRU Decoder
Attention:    Post-GRU + Scaled Multiplicative + Input Feeding
Direction:    EN→DE uses U for both BiGRU directions;
              DE→EN uses U^T for both BiGRU directions.
              W (input projection) is shared across directions via joint BPE vocab.
Schedules:    Eval at Start, Every 2 Epochs, and End
Tracking:     BLEU, chrF (both directions), Peak VRAM (MB), Metric Logging (CSV)
Attention:    Attention heatmap saved at every eval step for one probe sentence
"""

import math, time, random, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import sentencepiece as spm
import sacrebleu
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe on headless A100 nodes
import matplotlib.pyplot as plt

# ── 1. Device & A100 Hardware Optimization ──────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Hyperparameters ────────────────────────────────────────────────────────
BPE_VOCAB = 16_000         
MAX_LEN   = 40
TRAIN_CAP = 1_000_000      
BATCH     = 1024           
HID       = 512        
LR        = 5e-4           
EPOCHS    = 50
BEAM_SIZE = 5          
SEED      = 42
EXP_DIR   = "./luong_simulation_final"
os.makedirs(EXP_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# ── 3. Dataset & Tokenizer Setup ──────────────────────────────────────────────
spm_model_path = os.path.join(EXP_DIR, "spm_shared.model")

print(f"Loading WMT14 en-de (Subsetting {TRAIN_CAP:,} pairs)...")
ds = load_dataset("wmt14", "de-en")
train_raw, val_raw = ds["train"], ds["validation"]

def get_pair(sample):
    t = sample.get("translation", {})
    return t.get("en", "").strip(), t.get("de", "").strip()

# Create 1M training subset
all_idx = list(range(len(train_raw)))
random.shuffle(all_idx)
sub_idx = all_idx[:TRAIN_CAP]

en_lines, de_lines = [], []
for idx in sub_idx:
    en, de = get_pair(train_raw[idx])
    if en and de:
        en_lines.append(en.lower())
        de_lines.append(de.lower())

# Train Tokenizer
shared_txt_path = os.path.join(EXP_DIR, "train_shared.txt")
with open(shared_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(en_lines + de_lines))

if not os.path.exists(spm_model_path):
    spm.SentencePieceTrainer.train(
        input=shared_txt_path, model_prefix=os.path.join(EXP_DIR, "spm_shared"),
        vocab_size=BPE_VOCAB, character_coverage=1.0, model_type="bpe",
        pad_id=PAD_IDX, unk_id=UNK_IDX, bos_id=SOS_IDX, eos_id=EOS_IDX
    )

sp = spm.SentencePieceProcessor(model_file=spm_model_path)
VOCAB = sp.vocab_size()

def batch_encode(raw_split, cap=None):
    pairs = []
    count = 0
    for s in raw_split:
        en, de = get_pair(s)
        en_ids, de_ids = sp.encode(en.lower()), sp.encode(de.lower())
        if 1 <= len(en_ids) <= MAX_LEN and 1 <= len(de_ids) <= MAX_LEN:
            pairs.append((torch.tensor([SOS_IDX] + en_ids + [EOS_IDX]),
                          torch.tensor([SOS_IDX] + de_ids + [EOS_IDX])))
            count += 1
        if cap and count >= cap: break
    return pairs

def collate(batch):
    en_seqs, de_seqs = zip(*batch)
    return (pad_sequence(en_seqs, padding_value=PAD_IDX),
            pad_sequence(de_seqs, padding_value=PAD_IDX))

train_pairs = batch_encode(train_raw, cap=TRAIN_CAP)
val_pairs   = batch_encode(val_raw, cap=1000) # Smaller eval for speed

train_loader = DataLoader(train_pairs, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_pairs, batch_size=BATCH, shuffle=False, collate_fn=collate)

# ── 4. Custom GRU Primitives ──────────────────────────────────────────────────

class CustomGRUCell(nn.Module):
    """
    GRU cell with fused W (input) projection and per-gate square U (hidden) matrices.

    Why per-gate square U instead of the previous fused W_rzh (H→2H)?
    ─────────────────────────────────────────────────────────────────
    CustomBiGRU shares U between the forward and backward directions, with the
    backward direction using U^T.  Transposing a non-square fused matrix (H→2H)
    produces a (2H→H) matrix whose columns no longer correspond to individual
    gates — the transpose is mathematically undefined per gate.  Square (H×H)
    matrices transpose cleanly: U_r^T, U_z^T, U_n^T each remain (H×H) and map
    h → (B, H) identically to their originals, just with mirrored dynamics.

    The transpose is applied via F.linear(h, U.weight.T) which reads the
    existing weight storage transposed — zero extra memory, no .t() copy.

    _step() accepts a transpose_U flag so the same cell object drives both
    the forward pass (transpose_U=False) and the backward pass (transpose_U=True)
    inside CustomBiGRU.  The decoder always calls forward() which defaults to
    transpose_U=False and is unaffected by this change.

    W projection stays fused (input_size → 3H): 1 matmul covers all three gates.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Fused input projection: one matmul for all three gates  (B, 3H)
        self.W_ih = nn.Linear(input_size,   3 * hidden_size, bias=True)

        # Per-gate square hidden matrices  (H × H each)
        # Square shape is required for U / U^T weight-sharing in CustomBiGRU.
        self.U_r  = nn.Linear(hidden_size, hidden_size, bias=False)  # reset gate
        self.U_z  = nn.Linear(hidden_size, hidden_size, bias=False)  # update gate
        self.U_n  = nn.Linear(hidden_size, hidden_size, bias=True)   # new gate

    def _step(self, x_proj: torch.Tensor, h: torch.Tensor,
              transpose_U: bool = False) -> torch.Tensor:
        """
        Core GRU recurrence given pre-projected input x_proj (B, 3H).

        transpose_U=False  →  forward direction  (U_r, U_z, U_n)
        transpose_U=True   →  backward direction (U_r^T, U_z^T, U_n^T)

        F.linear(h, W.T) reads the stored weight transposed; no copy is made.
        """
        r_x, z_x, n_x = x_proj.chunk(3, dim=-1)          # each (B, H)

        if transpose_U:
            r_h = F.linear(h, self.U_r.weight.T)           # (B, H)
            z_h = F.linear(h, self.U_z.weight.T)           # (B, H)
            n_h = F.linear(h, self.U_n.weight.T, self.U_n.bias)  # (B, H)
        else:
            r_h = self.U_r(h)                              # (B, H)
            z_h = self.U_z(h)                              # (B, H)
            n_h = self.U_n(h)                              # (B, H)

        r = torch.sigmoid(r_x + r_h)                      # reset gate
        z = torch.sigmoid(z_x + z_h)                      # update gate
        n = torch.tanh(n_x + r * n_h)                     # new gate (r gates hidden)

        return (1.0 - z) * n + z * h                      # (B, H)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Standard single-step interface used by the decoder."""
        return self._step(self.W_ih(x), h, transpose_U=False)


class CustomBiGRU(nn.Module):
    """
    Bidirectional GRU encoder with shared W and direction-controlled U weights.

    Weight-sharing scheme (per layer)
    ──────────────────────────────────
      W  (input projection, W_ih):  fully shared — one matmul serves both directions
                                    and both translation directions (joint BPE vocab).
      U  (hidden projection):
          EN→DE (use_transpose_U=False): both fwd (L→R) and bwd (R→L) use U
          DE→EN (use_transpose_U=True):  both fwd (L→R) and bwd (R→L) use U^T

    Why the same U (or U^T) for both fwd and bwd within a direction?
    ─────────────────────────────────────────────────────────────────
    Within a single translation direction the BiGRU is reading the same language
    in two temporal directions.  Using the same U for both keeps the hidden-state
    dynamics symmetric around time: the forward pass encodes "what came before"
    via U, and the backward pass encodes "what comes after" via the same U applied
    to a reversed sequence.  This is the natural generalisation of weight-tying
    within one language direction.

    Across translation directions (EN→DE vs DE→EN) the U/U^T relationship means
    any feature pattern the model learns to propagate in the EN→DE direction is
    immediately available in transposed form for DE→EN — the two tasks share
    structure rather than learning completely independent hidden dynamics.

    Efficiency
    ──────────
      - W_ih is computed ONCE per layer: one (T*B, input) @ (input, 3H) matmul
        covers both the forward and backward passes.
      - Output buffers pre-allocated; no list/stack inside loops.
      - Dropout applied once per layer boundary on the full (T, B, 2H) tensor.

    Interface matches nn.GRU(bidirectional=True):
        output : (T, B, hidden_size * 2)
        hidden : (num_layers * 2, B, hidden_size)  [fwd_l0, bwd_l0, fwd_l1, bwd_l1]
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.drop        = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # One shared cell per layer — W and U are shared between fwd/bwd directions.
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * 2
            self.cells.append(CustomGRUCell(in_size, hidden_size))

    def forward(
        self,
        x:               torch.Tensor,              # (T, B, input_size)
        h_0:             torch.Tensor | None = None, # (num_layers*2, B, H) or None
        use_transpose_U: bool = False,               # False=EN→DE, True=DE→EN
    ) -> tuple[torch.Tensor, torch.Tensor]:

        T, B, _ = x.shape
        H = self.hidden_size

        if h_0 is None:
            h_0 = x.new_zeros(self.num_layers * 2, B, H)

        layer_in     = x
        final_hidden = []

        for layer in range(self.num_layers):
            cell = self.cells[layer]

            # ── Single shared W projection for both temporal directions ───────
            # One (T*B, input) @ (input, 3H) matmul covers both fwd and bwd.
            x_proj = cell.W_ih(layer_in)                    # (T, B, 3H)

            # ── Pre-allocate output buffers ───────────────────────────────────
            fwd_out = layer_in.new_empty(T, B, H)
            bwd_out = layer_in.new_empty(T, B, H)

            # ── Forward pass (L→R): U or U^T depending on translation dir ────
            h = h_0[layer * 2]
            for t in range(T):
                h = cell._step(x_proj[t], h, transpose_U=use_transpose_U)
                fwd_out[t] = h
            final_hidden.append(h)

            # ── Backward pass (R→L): same U choice as forward ─────────────────
            h = h_0[layer * 2 + 1]
            for t in range(T - 1, -1, -1):
                h = cell._step(x_proj[t], h, transpose_U=use_transpose_U)
                bwd_out[t] = h
            final_hidden.append(h)

            # ── Concatenate directions & apply inter-layer dropout ─────────────
            layer_out = torch.cat([fwd_out, bwd_out], dim=-1)   # (T, B, 2H)
            layer_in  = self.drop(layer_out) if layer < self.num_layers - 1 else layer_out

        output = layer_in                           # (T, B, 2H)
        hidden = torch.stack(final_hidden)          # (num_layers*2, B, H)
        return output, hidden


# ── 5. Model Architecture (Stabilized Luong) ───────────────────────────────────

class LuongScaledAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W_a = nn.Linear(hid_dim, hid_dim * 2, bias=False)

    def forward(self, h_dec, H_enc, src_mask=None):
        H_enc  = H_enc.permute(1, 0, 2).contiguous()                          # (B, T, 2H)
        query  = self.W_a(h_dec).unsqueeze(1)                                  # (B, 1, 2H)
        scores = torch.bmm(query, H_enc.permute(0, 2, 1)).squeeze(1)          # (B, T)
        scores = scores / math.sqrt(self.W_a.out_features)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        alpha   = F.softmax(scores, dim=1)                                     # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), H_enc).squeeze(1)             # (B, 2H)
        return context, alpha   # alpha returned so translate() can build attn matrix

class LuongSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hid_dim, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.emb = nn.Embedding(vocab_size, hid_dim, padding_idx=PAD_IDX)
        self.encoder = CustomBiGRU(hid_dim, hid_dim, num_layers=2, dropout=dropout)
        self.fc_init = nn.Linear(hid_dim * 2, hid_dim)
        self.attention = LuongScaledAttention(hid_dim)
        self.attn_combine = nn.Linear(hid_dim + (hid_dim * 2), hid_dim, bias=False)
        self.decoder = CustomGRUCell(hid_dim * 2, hid_dim)
        self.out_proj = nn.Linear(hid_dim, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def encode(self, src, direction: str = 'en_de'):
        embedded = self.drop(self.emb(src))
        # EN→DE: both BiGRU directions use U
        # DE→EN: both BiGRU directions use U^T
        H_enc, hidden = self.encoder(embedded, use_transpose_U=(direction == 'de_en'))
        h_init   = torch.tanh(self.fc_init(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        src_mask = (src == PAD_IDX).T
        return h_init, H_enc, src_mask

    def forward(self, src, tgt, direction: str = 'en_de'):
        h, H_enc, src_mask = self.encode(src, direction)
        T, B = tgt.shape
        logits  = []
        h_tilde = torch.zeros(B, self.hid_dim, device=src.device)

        for t in range(T):
            x          = self.drop(self.emb(tgt[t]))
            gru_input  = torch.cat((x, h_tilde), dim=1)        # input feeding
            h          = self.decoder(gru_input, h)
            ctx, _     = self.attention(h, H_enc, src_mask)    # discard alpha in training
            h_tilde    = torch.tanh(self.attn_combine(torch.cat((h, ctx), dim=1)))
            logits.append(self.out_proj(self.drop(h_tilde)))

        return torch.stack(logits)

    @torch.no_grad()
    def translate(self, src, direction: str = 'en_de'):
        """
        Beam-search translation.  Returns (token_ids, attn_matrix).
          token_ids  : list[int]  — best beam including SOS/EOS
          attn_matrix: np.ndarray — shape (T_tgt_steps, T_src) — attention weights
                       of the best beam at each decoder step, for plotting.

        Beam tuple: (score, tokens, h_s, ht_s, attn_rows)
          attn_rows is a list of (T_src,) CPU tensors, one per decoded step.
        """
        self.eval()
        h, H, mask = self.encode(src, direction)
        B          = src.size(1)
        h_tilde    = torch.zeros(B, self.hid_dim, device=src.device)
        beams      = [(0.0, [SOS_IDX], h, h_tilde, [])]

        for _ in range(MAX_LEN):
            all_cands = []
            for score, tokens, h_s, ht_s, attn_rows in beams:
                if tokens[-1] == EOS_IDX:
                    all_cands.append((score, tokens, h_s, ht_s, attn_rows))
                    continue
                tok       = torch.tensor([tokens[-1]], device=src.device)
                x         = self.emb(tok)
                h_s       = self.decoder(torch.cat((x, ht_s), dim=1), h_s)
                ctx, alpha = self.attention(h_s, H, mask)           # alpha: (1, T_src)
                ht_s      = torch.tanh(self.attn_combine(torch.cat((h_s, ctx), dim=1)))
                probs     = F.log_softmax(self.out_proj(ht_s), dim=-1)
                s_vals, t_vals = probs[0].topk(BEAM_SIZE)
                alpha_row = alpha[0].cpu()                          # (T_src,)
                for s, t in zip(s_vals.tolist(), t_vals.tolist()):
                    all_cands.append((score + s, tokens + [t], h_s, ht_s,
                                      attn_rows + [alpha_row]))
            beams = sorted(all_cands, key=lambda x: x[0], reverse=True)[:BEAM_SIZE]
            if all(b[1][-1] == EOS_IDX for b in beams):
                break

        best_score, best_tokens, _, _, best_attn_rows = beams[0]
        attn_matrix = torch.stack(best_attn_rows).numpy()  # (T_tgt_steps, T_src)
        return best_tokens, attn_matrix

# ── 6. Simulation Logic ────────────────────────────────────────────────────────

def init_weights(m):
    # recurse=False ensures each parameter is visited exactly once.
    for name, param in m.named_parameters(recurse=False):
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.1, 0.1)
        else:
            nn.init.zeros_(param.data)


def plot_attention(
    attn_en_de: "np.ndarray",  # (T_tgt, T_src_en)
    src_en_ids: list,
    pred_de_ids: list,
    attn_de_en: "np.ndarray",  # (T_tgt, T_src_de)
    src_de_ids: list,
    pred_en_ids: list,
    epoch: int,
    save_dir: str,
) -> None:
    """
    Save a side-by-side attention heatmap for one probe sentence.
    Left panel:  EN→DE  (source=EN tokens, y-axis=predicted DE tokens)
    Right panel: DE→EN  (source=DE tokens, y-axis=predicted EN tokens)
    Rows = decoder steps (target tokens), Columns = source tokens.
    """
    def ids_to_pieces(ids):
        # sp.id_to_piece gives the raw BPE subword piece for each token id.
        return [sp.id_to_piece(i) for i in ids if i not in (PAD_IDX, SOS_IDX)]

    src_en_toks  = ids_to_pieces(src_en_ids)
    pred_de_toks = ids_to_pieces(pred_de_ids)
    src_de_toks  = ids_to_pieces(src_de_ids)
    pred_en_toks = ids_to_pieces(pred_en_ids)

    # Trim attn matrices to match decoded lengths (beam may produce fewer rows
    # than MAX_LEN if EOS was hit early).
    attn_en_de = attn_en_de[:len(pred_de_toks), :len(src_en_toks)]
    attn_de_en = attn_de_en[:len(pred_en_toks), :len(src_de_toks)]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(src_en_toks)),
                                            max(6,  len(pred_de_toks) // 2 + 4)))
    for ax, attn, src_toks, tgt_toks, title in [
        (axes[0], attn_en_de, src_en_toks,  pred_de_toks, "EN → DE"),
        (axes[1], attn_de_en, src_de_toks,  pred_en_toks, "DE → EN"),
    ]:
        im = ax.imshow(attn, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(src_toks)))
        ax.set_xticklabels(src_toks, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(tgt_toks)))
        ax.set_yticklabels(tgt_toks, fontsize=7)
        ax.set_xlabel("Source tokens", fontsize=9)
        ax.set_ylabel("Predicted target tokens (decoder steps)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Luong Attention — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"attention_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"    Attention plot → {path}")


@torch.no_grad()
def run_eval(model, loader, epoch, csv_path, peak_mem=0.0):
    """
    Computes BLEU + chrF for both EN→DE and DE→EN.
    Picks the first sentence from the loader as a probe and saves attention plots.
    """
    model.eval()

    hyps_en_de, refs_en_de = [], []
    hyps_de_en, refs_de_en = [], []
    probe_set = False
    probe_en_ids = probe_de_ids = None

    for en_b, de_b in loader:
        for i in range(min(en_b.shape[1], 100)):
            src_en = en_b[:, i:i+1].to(device)
            src_de = de_b[:, i:i+1].to(device)

            # EN→DE
            pred_de, attn_en_de = model.translate(src_en, direction='en_de')
            hyps_en_de.append(sp.decode(pred_de))
            refs_en_de.append([sp.decode(de_b[:, i].tolist())])

            # DE→EN
            pred_en, attn_de_en = model.translate(src_de, direction='de_en')
            hyps_de_en.append(sp.decode(pred_en))
            refs_de_en.append([sp.decode(en_b[:, i].tolist())])

            # Save the first sentence as the attention probe for this eval step
            if not probe_set:
                probe_en_ids  = en_b[:, i].tolist()
                probe_de_ids  = de_b[:, i].tolist()
                probe_attn_en_de = attn_en_de
                probe_pred_de    = pred_de
                probe_attn_de_en = attn_de_en
                probe_pred_en    = pred_en
                probe_set = True

    bleu_en_de = sacrebleu.corpus_bleu(hyps_en_de, list(zip(*refs_en_de))).score
    chrf_en_de = sacrebleu.corpus_chrf(hyps_en_de, list(zip(*refs_en_de))).score
    bleu_de_en = sacrebleu.corpus_bleu(hyps_de_en, list(zip(*refs_de_en))).score
    chrf_de_en = sacrebleu.corpus_chrf(hyps_de_en, list(zip(*refs_de_en))).score

    print(f"\n>>> [EVAL EPOCH {epoch}]"
          f"  EN→DE  BLEU: {bleu_en_de:.2f}  chrF: {chrf_en_de:.2f}"
          f"  |  DE→EN  BLEU: {bleu_de_en:.2f}  chrF: {chrf_de_en:.2f}"
          f"  |  Mem: {peak_mem:.0f} MB")

    with open(csv_path, "a") as f:
        f.write(f"{epoch},{bleu_en_de:.2f},{chrf_en_de:.2f},"
                f"{bleu_de_en:.2f},{chrf_de_en:.2f},{peak_mem:.0f}\n")

    # Attention heatmap for the probe sentence
    if probe_set:
        plot_attention(
            probe_attn_en_de, probe_en_ids,  probe_pred_de,
            probe_attn_de_en, probe_de_ids,  probe_pred_en,
            epoch, EXP_DIR,
        )

    return bleu_en_de


def main():
    model = LuongSeq2Seq(VOCAB, HID).to(device)
    model.apply(init_weights)
    model = torch.compile(model)

    opt  = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    csv_path = os.path.join(EXP_DIR, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,bleu_en_de,chrf_en_de,bleu_de_en,chrf_de_en,peak_mem_mb\n")

    # Evaluation: BEGINNING
    run_eval(model, val_loader, 0, csv_path, 0.0)

    for ep in range(1, EPOCHS + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)

        for en_b, de_b in train_loader:
            en_b, de_b = en_b.to(device), de_b.to(device)
            opt.zero_grad(set_to_none=True)

            # ── Randomly choose translation direction for this batch ───────────
            # 50/50 split: EN→DE uses U; DE→EN uses U^T.
            # W is shared across both because joint BPE embedding is shared.
            if random.random() < 0.5:
                src, tgt, direction = en_b, de_b, 'en_de'
            else:
                src, tgt, direction = de_b, en_b, 'de_en'

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = crit(
                    model(src, tgt[:-1], direction).view(-1, VOCAB),
                    tgt[1:].view(-1)
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        peak_mem = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"Epoch {ep} Done. Peak Mem: {peak_mem:.0f} MB")

        # Evaluation: EVERY TWO EPOCHS
        if ep % 2 == 0:
            run_eval(model, val_loader, ep, csv_path, peak_mem)

    # Evaluation: END
    run_eval(model, val_loader, EPOCHS, csv_path, peak_mem)
    print("Simulation Complete.")

if __name__ == "__main__":
    main()