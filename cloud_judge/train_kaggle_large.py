# =============================================================================
# KAGGLE (2xT4 GPU) UZERINDE UCRETSIZ VE KARARLI LARGE MODEL EGITIMI
#
# Bu script, XLM-Roberta-Large veya mDeBERTa-v3-Large modellerinin egitim sirasinda
# cokmesini (gradient collapse) engellemek icin su iyilestirmeleri icerir:
#   1. bfloat16 kullanildiginda GradScaler devredisi birakilmistir (bf16 icin gerekmez ve cökme yaratir).
#   2. Ogrenme orani (LR) daha karali olan 7e-6 seviyesine cekilmistir.
#   3. Buyuk modeller icin Gradient Accumulation eklenerek daha kararli gradient adimlari saglanmistir.
#
# KULLANIM:
#   1. Kaggle'da yeni notebook acin.
#   2. Accelerator: GPU T4 x2, Internet: ON, Persistence: Files.
#   3. Bu kodu tek hucreye yapistirip calistirin.
#   4. Ciktilari (/kaggle/working/ altindaki .npy dosyalarini) indirip repoya yukleyin.
# =============================================================================
import subprocess, sys, os
try:
    import hf_transfer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf-transfer"], check=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

# Model ve hiperparametre ayarları
MODEL = "xlm-roberta-large"
MAXLEN = 160
BATCH_SIZE = 16          # Cift GPU (2x T4) icin uygun batch size
ACCUMULATION_STEPS = 2   # Efektif batch size = 16 * 2 * 2 (GPUs) = 64
EPOCHS = 2
LR = 7e-6                # Kararli ogrenme orani
HOLDOUT = 4
dev = "cuda"

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class DS(Dataset):
    def __init__(self, q, d, y=None):
        self.q, self.d, self.y = q, d, y
    def __len__(self):
        return len(self.q)
    def __getitem__(self, i):
        return (self.q[i], self.d[i]) if self.y is None else (self.q[i], self.d[i], self.y[i])

def main():
    # Kaggle input yollarini bulma
    COMP = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "items.csv" in files and "submission_pairs.csv" in files:
            COMP = root
            break
            
    # Eger train_text.parquet dosyasi kaggle'a yuklenmisse yolunu ayarla
    TRAIN_PARQUET_PATH = None
    TEST_PARQUET_BASE = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_text.parquet" in files:
            TRAIN_PARQUET_PATH = os.path.join(root, "train_text.parquet")
        if "test_text_part0.parquet" in files:
            TEST_PARQUET_BASE = root

    assert COMP, "Yarisma verileri bulunamadi."
    assert TRAIN_PARQUET_PATH, "train_text.parquet bulunamadi. Lutfen Kaggle'a dataset olarak ekleyin."
    
    log(f"Model yukleniyor: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    
    def coll_tr(b):
        q, d, y = zip(*b)
        e = tok(list(q), list(d), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        e["labels"] = torch.tensor(y, dtype=torch.long)
        return e
        
    def coll_te(b):
        q, d = zip(*b)
        return tok(list(q), list(d), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

    log("Veri yukleniyor...")
    tr = pd.read_parquet(TRAIN_PARQUET_PATH)
    te = pd.concat([pd.read_parquet(os.path.join(TEST_PARQUET_BASE, f"test_text_part{k}.parquet")) for k in range(4)], ignore_index=True)
    
    trn = tr[tr.fold != HOLDOUT].reset_index(drop=True)
    hol = tr[tr.fold == HOLDOUT].reset_index(drop=True)
    log(f"Train={len(trn):,} | Holdout={len(hol):,} | Test={len(te):,}")

    # Model yukleme
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    
    # Cift GPU (DataParallel) kullanimi
    if torch.cuda.device_count() > 1:
        log(f"Cift GPU algilandi: {torch.cuda.device_count()} GPUs. DataParallel aktif.")
        model = torch.nn.DataParallel(model)
    model = model.to(dev)
    
    dl = DataLoader(DS(trn.q.tolist(), trn.d.tolist(), trn.label.to_numpy()),
                    batch_size=BATCH_SIZE, shuffle=True, collate_fn=coll_tr, num_workers=0, pin_memory=True)
                    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(dl) * EPOCHS // ACCUMULATION_STEPS
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss()
    
    log(f"Egitim basliyor: {EPOCHS} epoch, BatchSize={BATCH_SIZE}, Accumulation={ACCUMULATION_STEPS}")
    
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        opt.zero_grad()
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            
            # T4 kartlari bfloat16 desteklemediginden float16 autocast kullanilir
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = lossf(out.logits, b["labels"])
                loss = loss / ACCUMULATION_STEPS
                
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                
            if i % 200 == 0:
                log(f"  ep{ep+1} {i}/{len(dl)} loss={loss.item() * ACCUMULATION_STEPS:.4f} ({time.time()-t0:.0f}s)")
                
    @torch.no_grad()
    def score(df):
        model.eval()
        dl = DataLoader(DS(df.q.tolist(), df.d.tolist()), batch_size=BATCH_SIZE * 4, collate_fn=coll_te,
                        num_workers=0, pin_memory=True)
        out = np.zeros(len(df), np.float32)
        p = 0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**b)
                lg = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            out[p:p+lg.shape[0]] = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
            p += lg.shape[0]
        return out

    log("Holdout setini skorlama...")
    ho = score(hol)
    np.save("/kaggle/working/large_holdout.npy", ho)
    
    from sklearn.metrics import f1_score
    y = hol.label.to_numpy()
    best = max((f1_score(y, (ho >= t).astype(int), average="macro"), t) for t in np.arange(0.3, 0.71, 0.05))
    log(f"Holdout macro-F1: {best[0]:.5f} at threshold: {best[1]:.2f}")
    
    log("Test setini skorlama (3.36M)...")
    ts = score(te)
    np.save("/kaggle/working/large_test.npy", ts)
    log("TUM ISLEMLER TAMAMLANDI! Ciktilar hazır: /kaggle/working/large_test.npy")

if __name__ == "__main__":
    main()
