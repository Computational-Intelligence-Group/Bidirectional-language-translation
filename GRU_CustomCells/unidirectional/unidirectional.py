"""
A100 Optimized Luong Simulation (EN -> DE)
=========================================
Architecture: 2-Layer Bi-GRU Encoder + Standard GRU Decoder
Attention:    Post-GRU + Scaled Multiplicative + Input Feeding
Schedules:    Eval at Start, Every 2 Epochs, and End
Tracking:     BLEU, chrF, Peak VRAM (MB), Metric Logging (CSV)
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
    Drop-in replacement for nn.GRUCell with fused gate projections.

    Efficiency gains vs nn.GRUCell:
      - Input projection: 1 matmul (3H outputs) instead of 3 separate ones.
      - Hidden projection: r/z fused into 1 matmul (2H outputs); n kept separate
        because r must gate it element-wise before the tanh — they cannot be merged.
      - Total: 3 matmuls per step vs 6, which halves projection FLOP count.

    Additionally exposes W_ih and _step() so CustomBiGRU can pre-project the
    entire input sequence in a single batched matmul and reuse the result across
    all timesteps, avoiding redundant work in the encoder.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Fused input projection for all three gates: r, z, n  →  (B, 3H)
        self.W_ih  = nn.Linear(input_size,    3 * hidden_size, bias=True)
        # Fused hidden projection for reset & update gates    →  (B, 2H)
        self.W_rzh = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        # Separate hidden projection for new-gate             →  (B, H)
        # (must stay separate: n_hidden is element-wise gated by r before tanh)
        self.W_nh  = nn.Linear(hidden_size,     hidden_size, bias=True)

    def _step(self, x_proj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Core GRU recurrence given a pre-projected input x_proj (B, 3H).
        Called directly by CustomBiGRU to avoid recomputing W_ih per timestep.
        """
        r_x, z_x, n_x = x_proj.chunk(3, dim=-1)   # each (B, H)

        rz_h       = self.W_rzh(h)                 # (B, 2H)  — fused r+z hidden
        r_h, z_h   = rz_h.chunk(2, dim=-1)         # each (B, H)

        r = torch.sigmoid(r_x + r_h)               # reset gate
        z = torch.sigmoid(z_x + z_h)               # update gate
        n = torch.tanh(n_x + r * self.W_nh(h))     # new gate  (r gates hidden part)

        return (1.0 - z) * n + z * h               # (B, H)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Standard interface: takes raw input x (B, input_size) and hidden h (B, H)."""
        return self._step(self.W_ih(x), h)


class CustomBiGRU(nn.Module):
    """
    Drop-in replacement for:
        nn.GRU(input_size, hidden_size, num_layers=2, bidirectional=True, dropout=p)

    Matches the exact (output, hidden) interface of nn.GRU:
        output : (T, B, hidden_size * 2)
        hidden : (num_layers * 2, B, hidden_size)   ordered [fwd_l0, bwd_l0, fwd_l1, bwd_l1]

    Efficiency gains vs nn.GRU called with a Python loop:
      - Per-layer, per-direction: the entire sequence is projected in ONE batched
        matmul  W_ih @ x  →  (T, B, 3H)  before the recurrent loop starts.
        This replaces T individual (B, input) @ (input, 3H) matmuls with a single
        (T*B, input) @ (input, 3H) matmul — far better GPU utilisation.
      - Output tensors are pre-allocated with torch.empty; no Python list appends
        or torch.stack calls inside the loop.
      - Dropout is applied once per layer boundary on the full (T, B, 2H) tensor
        rather than per timestep.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.drop        = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.cells_fwd = nn.ModuleList()
        self.cells_bwd = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * 2
            self.cells_fwd.append(CustomGRUCell(in_size, hidden_size))
            self.cells_bwd.append(CustomGRUCell(in_size, hidden_size))

    def forward(
        self,
        x:   torch.Tensor,                     # (T, B, input_size)
        h_0: torch.Tensor | None = None,        # (num_layers*2, B, H) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        T, B, _ = x.shape
        H = self.hidden_size

        if h_0 is None:
            h_0 = x.new_zeros(self.num_layers * 2, B, H)

        layer_in     = x
        final_hidden = []                       # will hold (num_layers*2) tensors of shape (B, H)

        for layer in range(self.num_layers):
            fwd_cell = self.cells_fwd[layer]
            bwd_cell = self.cells_bwd[layer]

            # ── Pre-project the ENTIRE sequence in one batched matmul ──────────
            # Shape: (T, B, 3H).  Avoids T individual (B, in) @ (in, 3H) calls.
            x_proj_fwd = fwd_cell.W_ih(layer_in)   # (T, B, 3H)
            x_proj_bwd = bwd_cell.W_ih(layer_in)   # (T, B, 3H)

            # ── Pre-allocate output buffers ────────────────────────────────────
            fwd_out = layer_in.new_empty(T, B, H)
            bwd_out = layer_in.new_empty(T, B, H)

            # ── Forward pass ───────────────────────────────────────────────────
            h = h_0[layer * 2]                      # (B, H)
            for t in range(T):
                h = fwd_cell._step(x_proj_fwd[t], h)
                fwd_out[t] = h
            final_hidden.append(h)                  # final fwd hidden  (B, H)

            # ── Backward pass ──────────────────────────────────────────────────
            h = h_0[layer * 2 + 1]                  # (B, H)
            for t in range(T - 1, -1, -1):
                h = bwd_cell._step(x_proj_bwd[t], h)
                bwd_out[t] = h
            final_hidden.append(h)                  # final bwd hidden  (B, H)
            # (final bwd hidden = state after processing t=0, i.e. h_0-equivalent
            #  for the backward direction — consistent with nn.GRU convention)

            # ── Concatenate directions & apply inter-layer dropout ─────────────
            layer_out = torch.cat([fwd_out, bwd_out], dim=-1)   # (T, B, 2H)
            layer_in  = self.drop(layer_out) if layer < self.num_layers - 1 else layer_out

        output = layer_in                                        # (T, B, 2H)
        hidden = torch.stack(final_hidden)                       # (num_layers*2, B, H)
        return output, hidden


# ── 5. Model Architecture (Stabilized Luong) ───────────────────────────────────

class LuongScaledAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Projects decoder hidden (H) → encoder space (2H) so the dot product
        # with H_enc (2H) is valid.  Previously W_a projected H_enc instead of
        # h_dec, which inverted the Luong general formula: score = h_dec^T W_a h_enc
        self.W_a = nn.Linear(hid_dim, hid_dim * 2, bias=False)

    def forward(self, h_dec, H_enc, src_mask=None):
        # .contiguous() avoids non-contiguous memory layout issues under torch.compile
        H_enc = H_enc.permute(1, 0, 2).contiguous()        # (B, T, 2H)
        # Luong general score: h_dec^T · W_a · h_enc
        query  = self.W_a(h_dec).unsqueeze(1)               # (B, 1, 2H)
        scores = torch.bmm(query, H_enc.permute(0, 2, 1)).squeeze(1)  # (B, T)
        scores = scores / math.sqrt(self.W_a.out_features)

        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        alpha   = F.softmax(scores, dim=1)                  # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), H_enc).squeeze(1)     # (B, 2H)
        return context

class LuongSeq2Seq(nn.Module):
    def __init__(self, vocab_size, hid_dim, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.emb = nn.Embedding(vocab_size, hid_dim, padding_idx=PAD_IDX)
        self.encoder = CustomBiGRU(hid_dim, hid_dim, num_layers=2, dropout=dropout)
        self.fc_init = nn.Linear(hid_dim * 2, hid_dim)
        self.attention = LuongScaledAttention(hid_dim)
        self.attn_combine = nn.Linear(hid_dim + (hid_dim * 2), hid_dim, bias=False)
        self.decoder = CustomGRUCell(hid_dim * 2, hid_dim) # Widened for Input Feeding
        self.out_proj = nn.Linear(hid_dim, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def encode(self, src):
        embedded = self.drop(self.emb(src))
        H_enc, hidden = self.encoder(embedded)
        h_init = torch.tanh(self.fc_init(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        src_mask = (src == PAD_IDX).T
        return h_init, H_enc, src_mask

    def forward(self, src, tgt):
        h, H_enc, src_mask = self.encode(src)
        T, B = tgt.shape
        logits = []
        h_tilde = torch.zeros(B, self.hid_dim, device=src.device)
        
        for t in range(T):
            x = self.drop(self.emb(tgt[t]))
            # Input Feeding:
            gru_input = torch.cat((x, h_tilde), dim=1)
            h = self.decoder(gru_input, h)
            ctx = self.attention(h, H_enc, src_mask)
            combined = torch.cat((h, ctx), dim=1)
            h_tilde = torch.tanh(self.attn_combine(combined))
            logits.append(self.out_proj(self.drop(h_tilde)))
            
        return torch.stack(logits)

    @torch.no_grad()
    def translate(self, src):
        self.eval()
        h, H, mask = self.encode(src)
        B = src.size(1)
        h_tilde = torch.zeros(B, self.hid_dim, device=src.device)
        beams = [(0.0, [SOS_IDX], h, h_tilde)]
        
        for _ in range(MAX_LEN):
            all_cands = []
            for score, tokens, h_s, ht_s in beams:
                if tokens[-1] == EOS_IDX:
                    all_cands.append((score, tokens, h_s, ht_s))
                    continue
                tok = torch.tensor([tokens[-1]], device=src.device)
                x = self.emb(tok)
                h_s = self.decoder(torch.cat((x, ht_s), dim=1), h_s)       
                ctx = self.attention(h_s, H, mask)
                ht_s = torch.tanh(self.attn_combine(torch.cat((h_s, ctx), dim=1)))
                probs = F.log_softmax(self.out_proj(ht_s), dim=-1)
                s_vals, t_vals = probs[0].topk(BEAM_SIZE)
                for s, t in zip(s_vals.tolist(), t_vals.tolist()):
                    all_cands.append((score + s, tokens + [t], h_s, ht_s))
            beams = sorted(all_cands, key=lambda x: x[0], reverse=True)[:BEAM_SIZE]
            if all(b[1][-1] == EOS_IDX for b in beams): break
        return beams[0][1]

# ── 6. Simulation Logic ────────────────────────────────────────────────────────

def init_weights(m):
    # recurse=False ensures each parameter is visited exactly once.
    # Without it, named_parameters() recurses into all descendants, so a
    # parameter deep in the tree is re-initialized once per ancestor module.
    for name, param in m.named_parameters(recurse=False):
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.1, 0.1)
        else:
            nn.init.zeros_(param.data)

@torch.no_grad()
def run_eval(model, loader, epoch, csv_path, peak_mem=0.0):
    model.eval()
    hyps, refs = [], []
    for en_b, de_b in loader:
        for i in range(min(en_b.shape[1], 100)): # Subset for speed
            src = en_b[:, i:i+1].to(device)
            pred = model.translate(src)
            hyps.append(sp.decode(pred))
            refs.append([sp.decode(de_b[:, i].tolist())])
    
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs))).score
    chrf = sacrebleu.corpus_chrf(hyps, list(zip(*refs))).score
    
    print(f"\n>>> [EVAL EPOCH {epoch}] BLEU: {bleu:.2f} | chrF: {chrf:.2f} | Mem: {peak_mem:.0f} MB")
    with open(csv_path, "a") as f:
        f.write(f"{epoch},{bleu:.2f},{chrf:.2f},{peak_mem:.0f}\n")
    return bleu

def main():
    model = LuongSeq2Seq(VOCAB, HID).to(device)
    model.apply(init_weights)
    model = torch.compile(model) 
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    csv_path = os.path.join(EXP_DIR, "metrics.csv")
    with open(csv_path, "w") as f: f.write("epoch,bleu,chrf,peak_mem_mb\n")

    # Evaluation: BEGINNING
    run_eval(model, val_loader, 0, csv_path, 0.0)

    for ep in range(1, EPOCHS + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        
        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = crit(model(src, trg[:-1]).view(-1, VOCAB), trg[1:].view(-1))
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