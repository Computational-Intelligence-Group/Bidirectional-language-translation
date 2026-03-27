"""
B-BP NMT Baseline — Forward Loss Only
======================================
Baseline experiment for Dr. Adigun.
Only L_en_de (forward EN->DE cross-entropy) is backpropagated.
L_de_en, L_cyc, L_ortho are computed and logged every epoch
but do NOT contribute to gradient updates.

This answers: what does U^T produce as a zero-shot inverse
when U is trained purely on forward translation with no
inversion supervision (no backward loss, no cycle loss,
no orthogonality regulariser)?

Comparison target: v5 full dataset
  EN->DE: 8.37  DE->EN: 11.42  Avg: 9.90
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
BPE_VOCAB  = 8_000
MAX_LEN    = 40
TRAIN_CAP  = None #200_000
BATCH      = 256 #64
HID        = 1024 #256
LR         = 1e-3
EPOCHS     = 30          # CHANGE 5: was 20
ENC_LAYERS = 2
MU_ORTHO   = 0.01        # CHANGE 3: orthogonality regulariser weight
BEAM_SIZE  = 4           # CHANGE 4: beam search width
SEED       = 42

random.seed(SEED)
torch.manual_seed(SEED)

# Cycle warmup schedule — unchanged from v4
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
#sub_idx = all_idx[:TRAIN_CAP]
sub_idx = all_idx if TRAIN_CAP is None else all_idx[:TRAIN_CAP]

# ── 4. Train BPE models ────────────────────────────────────────────────────────
cap_str = "full dataset" if TRAIN_CAP is None else f"{TRAIN_CAP:,}"
print(f"\nTraining BPE models (vocab={BPE_VOCAB}) on {cap_str} pairs ...")

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

for lang, prefix in [("en", "/tmp/spm_en"), ("de", "/tmp/spm_de")]:
    spm.SentencePieceTrainer.train(
        input=f"/tmp/train_{lang}.txt",
        model_prefix=prefix,
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
print(f"EN vocab: {EN_VOCAB:,}  |  DE vocab: {DE_VOCAB:,}")

# ── 5. Dataset encoding ────────────────────────────────────────────────────────
def batch_encode(sp, texts):
    return sp.encode([t.lower().strip() for t in texts])

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

print("\nEncoding pairs with BPE (batch mode) ...")
train_pairs = make_pairs_batch(en_lines, de_lines)
val_pairs   = make_pairs_from_raw(val_raw)
test_pairs  = make_pairs_from_raw(test_raw)
print(f"Train: {len(train_pairs):,}  Val: {len(val_pairs):,}  Test: {len(test_pairs):,}")

train_loader = DataLoader(train_pairs, batch_size=BATCH, shuffle=True,
                          collate_fn=collate, drop_last=True,
                          num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_pairs,   batch_size=BATCH, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_pairs,  batch_size=BATCH, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)

# ── 6. CHANGE 1+2: BBPGRUCell with orthogonal init ────────────────────────────
class BBPGRUCell(nn.Module):
    """
    GRU-style B-BP cell. Single shared recurrent matrix U governs both
    forward (W=U) and backward (W=U^T) passes. B-BP claim preserved.

    Gates:
      z_t = sigmoid( W_z(x_t) + h_{t-1} @ W )         update gate
      r_t = sigmoid( W_r(x_t) + h_{t-1} @ W )         reset gate
      n_t = tanh(    W_n(x_t) + (r_t * h_{t-1}) @ W ) candidate
      h_t = (1 - z_t) * h_{t-1} + z_t * n_t

    W_z, W_r, W_n map x_t only — not subject to inversion constraint.
    U is orthogonally initialised (CHANGE 2): starts at U^T U = I.
    """
    def __init__(self, dim):
        super().__init__()
        self.U   = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.U)      # CHANGE 2: orthogonal init
        self.W_z = nn.Linear(dim, dim, bias=True)
        self.W_r = nn.Linear(dim, dim, bias=True)
        self.W_n = nn.Linear(dim, dim, bias=True)

    def forward(self, x, h, bbp=False):
        W = self.U.T if bbp else self.U
        z = torch.sigmoid(self.W_z(x) + h @ W)
        r = torch.sigmoid(self.W_r(x) + h @ W)
        n = torch.tanh(self.W_n(x) + (r * h) @ W)
        return (1.0 - z) * h + z * n

# ── 7. Scaled Dot-Product Attention ───────────────────────────────────────────
class ScaledDotAttention(nn.Module):
    def forward(self, h_dec, H_enc, src_mask=None):
        scores = torch.bmm(
            H_enc.permute(1, 0, 2),
            h_dec.unsqueeze(2)
        ).squeeze(2) / math.sqrt(h_dec.shape[-1])
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        alpha   = F.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1),
                            H_enc.permute(1, 0, 2)).squeeze(1)
        return context

# ── 8. B-BP Seq2Seq Model ─────────────────────────────────────────────────────
class BBPSeq2SeqAttn(nn.Module):
    def __init__(self, en_vocab, de_vocab, hid_dim, dropout=0.3):
        super().__init__()
        self.hid_dim   = hid_dim
        self.cell      = BBPGRUCell(hid_dim)          # CHANGE 1+2
        self.E_en      = nn.Embedding(en_vocab, hid_dim, padding_idx=PAD_IDX)
        self.E_de      = nn.Embedding(de_vocab, hid_dim, padding_idx=PAD_IDX)
        self.W_out_en  = nn.Linear(hid_dim, en_vocab, bias=False)
        self.W_out_de  = nn.Linear(hid_dim, de_vocab, bias=False)
        self.ctx_proj  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.attention = ScaledDotAttention()
        self.drop      = nn.Dropout(dropout)

    # CHANGE 3: orthogonality regulariser
    def ortho_loss(self):
        U = self.cell.U
        I = torch.eye(U.shape[0], device=U.device)
        return torch.norm(U.T @ U - I, p='fro') ** 2

    def encode(self, src, emb, bbp):
        T, B = src.shape
        h1 = torch.zeros(B, self.hid_dim, device=src.device)
        h2 = torch.zeros(B, self.hid_dim, device=src.device)
        H2 = []
        for t in range(T):
            x  = self.drop(emb(src[t]))
            h1 = self.cell(x,  h1, bbp=bbp)   # encoder layer 1
            h2 = self.cell(h1, h2, bbp=bbp)   # encoder layer 2
            H2.append(h2)
        H2       = torch.stack(H2)
        src_mask = (src == PAD_IDX).T
        return h2, H2, src_mask

    def decode(self, tgt_in, h, H_enc, out_emb, out_proj, bbp, src_mask=None):
        T, B   = tgt_in.shape
        logits = []
        for t in range(T):
            x     = self.drop(out_emb(tgt_in[t]))      # no input feeding
            h     = self.cell(x, h, bbp=bbp)
            ctx   = self.attention(h, H_enc, src_mask)
            out   = h + self.ctx_proj(ctx)
            logit = self.drop(out_proj(out))
            logits.append(logit)
        return torch.stack(logits)

    def forward_en_de(self, src_en, tgt_de_in):
        h, H, mask = self.encode(src_en, self.E_en, bbp=False)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de,
                           bbp=False, src_mask=mask)

    def forward_de_en(self, src_de, tgt_en_in):
        h, H, mask = self.encode(src_de, self.E_de, bbp=True)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en,
                           bbp=True, src_mask=mask)

    def cycle_en(self, src_en, tgt_en_in):
        h, H, mask = self.encode(src_en, self.E_en, bbp=False)
        return self.decode(tgt_en_in, h, H, self.E_en, self.W_out_en,
                           bbp=True, src_mask=mask)

    def cycle_de(self, src_de, tgt_de_in):
        h, H, mask = self.encode(src_de, self.E_de, bbp=True)
        return self.decode(tgt_de_in, h, H, self.E_de, self.W_out_de,
                           bbp=False, src_mask=mask)

    # CHANGE 4: beam search decode (single sentence, batch=1)
    @torch.no_grad()
    def beam_decode(self, src, src_emb, out_emb, out_proj, bbp,
                    beam_size=4, max_len=50):
        self.eval()
        h_init, H, mask = self.encode(src, src_emb, bbp)

        # beams: list of (log_score, token_list, hidden_state)
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
            all_cands.sort(key=lambda x: x[0], reverse=True)
            beams = all_cands[:beam_size]
            if all(b[1][-1] == EOS_IDX for b in beams):
                for b in beams:
                    completed.append((b[0], b[1]))
                break

        for b in beams:
            completed.append((b[0], b[1]))

        if not completed:
            return []

        best = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
        return [t for t in best if t not in (SOS_IDX, EOS_IDX, PAD_IDX)]

    def beam_en_de(self, src_en, beam_size=BEAM_SIZE, max_len=50):
        return self.beam_decode(src_en, self.E_en, self.E_de, self.W_out_de,
                                bbp=False, beam_size=beam_size, max_len=max_len)

    def beam_de_en(self, src_de, beam_size=BEAM_SIZE, max_len=50):
        return self.beam_decode(src_de, self.E_de, self.E_en, self.W_out_en,
                                bbp=True, beam_size=beam_size, max_len=max_len)

# ── 9. Loss ───────────────────────────────────────────────────────────────────
def xloss(logits, tgt_out, pad_idx=PAD_IDX):
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V),
                           tgt_out.reshape(-1),
                           ignore_index=pad_idx)

# ── 10. BLEU with beam search ─────────────────────────────────────────────────
def compute_bleu(model, loader, direction, sp_hyp, sp_ref, max_batches=30):
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
                    sp_h, sp_r = sp_hyp, sp_ref
                else:
                    src     = de_b[:, i:i+1]
                    ref_ids = en_b[:, i].tolist()
                    pred    = model.beam_de_en(src)
                    sp_h, sp_r = sp_hyp, sp_ref
                clean_r = [t for t in ref_ids
                           if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                hyps.append(sp_h.decode(pred))
                refs.append([sp_r.decode(clean_r)])
            count += 1
    return sacrebleu.corpus_bleu(hyps, list(zip(*refs))).score

# ── 11. Training loop — BASELINE: forward loss only ──────────────────────────
def train_epoch(model, loader, opt, lam=0.0):
    """
    Baseline run: only L_en_de (forward EN->DE) is backpropagated.
    L_de_en, L_cyc, L_ortho are computed and logged but NOT in the loss.
    This shows what U^T produces as a zero-shot inverse when U is trained
    purely on forward translation with no inversion supervision.
    """
    model.train()
    tot = fw_tot = bw_tot = cyc_tot = ortho_tot = n = 0
    for en_b, de_b in loader:
        en_b, de_b = en_b.to(device), de_b.to(device)
        opt.zero_grad()

        l_en_de  = xloss(model.forward_en_de(en_b, de_b[:-1]), de_b[1:])
        l_de_en  = xloss(model.forward_de_en(de_b, en_b[:-1]), en_b[1:])
        l_cyc_en = xloss(model.cycle_en(en_b, en_b[:-1]), en_b[1:])
        l_cyc_de = xloss(model.cycle_de(de_b, de_b[:-1]), de_b[1:])

        cyc   = l_cyc_en + l_cyc_de
        ortho = model.ortho_loss()

        # BASELINE: only forward EN->DE loss is backpropagated
        loss = l_en_de

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        tot       += loss.item()
        fw_tot    += l_en_de.item()
        bw_tot    += l_de_en.item()
        cyc_tot   += cyc.item()
        ortho_tot += ortho.item()
        n         += 1

    return tot/n, fw_tot/n, bw_tot/n, cyc_tot/n, ortho_tot/n

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
        en_vocab=EN_VOCAB, de_vocab=DE_VOCAB,
        hid_dim=HID, dropout=0.3,
    ).to(device)

    total   = sum(p.numel() for p in model.parameters())
    rec_p   = model.cell.U.numel()
    gate_p  = (sum(p.numel() for p in model.cell.W_z.parameters()) +
               sum(p.numel() for p in model.cell.W_r.parameters()) +
               sum(p.numel() for p in model.cell.W_n.parameters()))
    emb_p   = model.E_en.weight.numel() + model.E_de.weight.numel()
    out_p   = model.W_out_en.weight.numel() + model.W_out_de.weight.numel()
    ctx_p   = model.ctx_proj.weight.numel()
    other_p = total - rec_p - gate_p - emb_p - out_p - ctx_p

    print(f"\n{'='*70}")
    print(f"  B-BP NMT Baseline  |  Forward Loss Only  |  Adigun-Kosko")
    print(f"{'='*70}")
    print(f"  Total params             : {total:,}")
    print(f"  cell.U (recurrent)       : {rec_p:,}   <- shared U, ortho init")
    print(f"  GRU gates (W_z,W_r,W_n)  : {gate_p:,}   <- input projections")
    print(f"  E_en + E_de (input emb)  : {emb_p:,}   <- BPE 8k")
    print(f"  W_out_en + W_out_de      : {out_p:,}   <- untied output")
    print(f"  ctx_proj                 : {ctx_p:,}   <- attention combiner")
    print(f"  biases + other           : {other_p:,}")
    print(f"  hid={HID}  batch={BATCH}  lr={LR}  epochs={EPOCHS}  beam_k={BEAM_SIZE}")
    print(f"  BASELINE: loss=L_en_de only  |  bw/cyc/ortho logged but NOT backpropagated")

    with torch.no_grad():
        U = model.cell.U
        I = torch.eye(U.shape[0], device=U.device)
        init_ortho = torch.norm(U.T @ U - I, p='fro').item()
    print(f"  Initial ||U^T U - I||_F  : {init_ortho:.4f}  (0.00 = perfect ortho)")

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    best_val = 1e9
    epoch_records = []
    os.makedirs("bbp_baseline_fw_results", exist_ok=True)

    csv_file = open("bbp_baseline_fw_results/training_log.csv", "w")
    csv_file.write("epoch,train_loss,fw_loss,bw_loss,cyc_loss,ortho_loss,"
                   "val_loss,bleu_en_de,bleu_de_en,lambda,elapsed_s\n")
    csv_file.flush()

    print(f"\n{'Ep':>3} | {'Train':>8} | {'FW_Loss':>8} | {'BW_Loss':>8} | {'Cyc':>8} | {'Ortho':>7} | "
          f"{'Val':>8} | {'EN->DE':>6} | {'DE->EN':>6} | {'lam':>4} | {'s':>5}")
    print("-" * 100)

    for ep in range(1, EPOCHS + 1):
        t0               = time.time()
        lam                      = get_lambda(ep)
        tr, fw, bw, cy, orth     = train_epoch(model, train_loader, opt, lam)
        vl                       = eval_loss(model, val_loader)
        sched.step(vl)
        b_en_de = compute_bleu(model, val_loader, "en_de", sp_de, sp_de)
        b_de_en = compute_bleu(model, val_loader, "de_en", sp_en, sp_en)
        elapsed = time.time() - t0

        mark = " <" if vl < best_val else ""
        print(f"{ep:>3} | {tr:>8.4f} | {fw:>8.4f} | {bw:>8.4f} | {cy:>8.4f} | {orth:>7.4f} | "
              f"{vl:>8.4f} | {b_en_de:>6.2f} | {b_de_en:>6.2f} | "
              f"{lam:.3f} | {elapsed:>5.0f}{mark}", flush=True)

        csv_file.write(f"{ep},{tr:.4f},{fw:.4f},{bw:.4f},{cy:.4f},{orth:.4f},{vl:.4f},"
                       f"{b_en_de:.2f},{b_de_en:.2f},{lam:.4f},{elapsed:.0f}\n")
        csv_file.flush()

        torch.save({
            "epoch": ep, "model_state": model.state_dict(),
            "opt_state": opt.state_dict(), "val_loss": vl,
            "bleu_en_de": b_en_de, "bleu_de_en": b_de_en,
        }, f"bbp_baseline_fw_results/checkpoint_ep{ep:02d}.pt")

        old = f"bbp_baseline_fw_results/checkpoint_ep{ep-3:02d}.pt"
        if ep > 3 and os.path.exists(old) and vl >= best_val:
            os.remove(old)

        if vl < best_val:
            best_val = vl
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(), "val_loss": vl,
                "bleu_en_de": b_en_de, "bleu_de_en": b_de_en,
            }, "bbp_baseline_fw_results/best.pt")

        epoch_records.append({
            "epoch": ep, "train": tr, "fw": fw, "bw": bw,
            "cyc": cy, "ortho": orth, "val": vl,
            "en_de": b_en_de, "de_en": b_de_en,
        })

    csv_file.close()

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    best_ckpt = torch.load("bbp_baseline_fw_results/best.pt")
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

    print(f"\n{'='*70}")
    print(f"  FINAL TEST RESULTS — B-BP NMT v5")
    print(f"{'='*70}")
    print(f"  BLEU EN->DE  (forward,  uses U)    : {test_en_de:.2f}")
    print(f"  BLEU DE->EN  (backward, uses U^T)  : {test_de_en:.2f}")
    print(f"  Average BLEU                       : {(test_en_de+test_de_en)/2:.2f}")
    print(f"  Final ||U^T U - I||_F              : {final_ortho:.4f}")
    print(f"{'='*70}")
    print(f"\n  V1 vs V2 vs V3 vs V4 vs V5")
    print(f"  {'Metric':<32} {'V1':>6} {'V2':>6} {'V3':>6} {'V4':>6} {'V5':>6}")
    print(f"  {'-'*62}")
    print(f"  {'EN->DE BLEU (test)':<32} {'5.37':>6} {'~0':>6} {'0.01':>6} {'4.10':>6} {test_en_de:>6.2f}")
    print(f"  {'DE->EN BLEU (test)':<32} {'3.47':>6} {'~0':>6} {'0.00':>6} {'6.44':>6} {test_de_en:>6.2f}")
    print(f"  {'Avg BLEU (test)':<32} {'4.42':>6} {'~0':>6} {'0.01':>6} {'5.27':>6} {(test_en_de+test_de_en)/2:>6.2f}")
    print(f"  {'Cell':<32} {'tanh':>6} {'tanh':>6} {'tanh':>6} {'tanh':>6} {'GRU':>6}")
    print(f"  {'Decoding':<32} {'greedy':>6} {'greedy':>6} {'greedy':>6} {'greedy':>6} {'beam4':>6}")

    with open("bbp_baseline_fw_results/summary.json", "w") as f:
        json.dump({
            "model": "B-BP NMT v5",
            "total_params": total,
            "recurrent_U": rec_p,
            "gru_gate_params": gate_p,
            "beam_size": BEAM_SIZE,
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

    print("\nAll results saved to bbp_baseline_fw_results/")
    print("  training_log.csv   — epoch metrics (includes ortho_loss column)")
    print("  best.pt            — best model checkpoint")
    print("  summary.json       — full summary with init + final ortho error")
    print("  checkpoint_ep*.pt  — last 3 epoch checkpoints (resumable)")


if __name__ == "__main__":
    main()
