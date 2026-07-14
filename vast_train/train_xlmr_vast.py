# =============================================================================
# KIRALIK A100 icin XLM-R LARGE egitimi (72B etiketli, DOGRU hiperparametre).
# Onceki cokme sebebi: yuksek LR + fp16. Cozum: bf16 + LR 1e-5 + warmup.
# GEREKLI (ayni klasor): train_text.parquet, test_text.parquet
# CIKTI: xlmrL_test.npy (3.36M test skoru) + xlmrL_holdout.npy
#   pip install -q torch transformers pandas pyarrow numpy
#   python train_xlmr_vast.py
# =============================================================================
import os, time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "xlm-roberta-large"
MAXLEN, TRB, INB, EPOCHS, LR, HOLDOUT = 160, 32, 128, 2, 1e-5, 4
dev = "cuda"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class DS(Dataset):
    def __init__(s, q, d, y=None): s.q, s.d, s.y = q, d, y
    def __len__(s): return len(s.q)
    def __getitem__(s, i): return (s.q[i], s.d[i]) if s.y is None else (s.q[i], s.d[i], s.y[i])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    def coll_tr(b):
        q, d, y = zip(*b)
        e = tok(list(q), list(d), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        e["labels"] = torch.tensor(y, dtype=torch.long); return e
    def coll_te(b):
        q, d = zip(*b)
        return tok(list(q), list(d), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

    log("Veri...")
    tr = pd.read_parquet(os.path.join(HERE, "train_text.parquet"))
    te = pd.read_parquet(os.path.join(HERE, "test_text.parquet"))
    trn = tr[tr.fold != HOLDOUT].reset_index(drop=True)
    hol = tr[tr.fold == HOLDOUT].reset_index(drop=True)
    log(f"train={len(trn):,} holdout={len(hol):,} test={len(te):,}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    dl = DataLoader(DS(trn.q.tolist(), trn.d.tolist(), trn.label.to_numpy()),
                    batch_size=TRB, shuffle=True, collate_fn=coll_tr, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(dl) * EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.CrossEntropyLoss()
    log(f"Egitim {EPOCHS} epoch, {len(dl)} step/epoch, LR={LR} bf16")
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = lossf(out.logits, b["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if i % 500 == 0:
                log(f"  ep{ep+1} {i}/{len(dl)} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

    @torch.no_grad()
    def score(df):
        model.eval()
        dl = DataLoader(DS(df.q.tolist(), df.d.tolist()), batch_size=INB, collate_fn=coll_te,
                        num_workers=4, pin_memory=True)
        out = np.zeros(len(df), np.float32); p = 0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**b).logits
            out[p:p+lg.shape[0]] = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy(); p += lg.shape[0]
        return out

    log("Holdout skorla...")
    ho = score(hol); np.save(os.path.join(HERE, "xlmrL_holdout.npy"), ho)
    from sklearn.metrics import f1_score
    y = hol.label.to_numpy()
    best = max((f1_score(y, (ho >= t).astype(int), average="macro"), t) for t in np.arange(0.3, 0.71, 0.05))
    log(f"HOLDOUT macroF1={best[0]:.4f} @ {best[1]:.2f}  (0.34 ise yine coktu; 0.90+ ise BASARILI)")
    log("Test skorla (3.36M, ~40dk)...")
    ts = score(te); np.save(os.path.join(HERE, "xlmrL_test.npy"), ts)
    log("BITTI -> xlmrL_test.npy + xlmrL_holdout.npy")


if __name__ == "__main__":
    main()
