# =============================================================================
# KAGGLE (T4 GPU) UZERINDE UCRETSIZ mDEBERTa-v3-base EGITIMI
#
# Bu script, microsoft/mdeberta-v3-base modelini Kaggle uzerinde egitip
# test/holdout tahminlerini üretir. Kilitlenmeleri engellemek icin
# tek GPU (cuda:0) uzerinde calisir ve hf_transfer / offline dataset arama destekler.
# =============================================================================
import subprocess, sys, os
try:
    import hf_transfer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf-transfer"], check=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = "microsoft/mdeberta-v3-base"
MAXLEN = 160
BATCH_SIZE = 32          # mDeBERTa-base icin tek T4 GPU'da 32 gayet kararlidir
EPOCHS = 2
LR = 2e-5
HOLDOUT = 4
dev = "cuda:0"

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
            
    # train_text.parquet yolunu bulma
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
    
    # Arka planda indirme boyutunu izleyen thread (Görünürlük saglamak icin)
    import threading
    stop_monitor = threading.Event()
    def monitor_download():
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)
        last_size = 0
        t0 = time.time()
        while not stop_monitor.is_set():
            size = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(cache_dir) for f in files)
            dt = time.time() - t0
            speed = (size - last_size) / dt if dt > 0 else 0
            log(f"  [Monitor] HF Cache size: {size/(1024*1024):.2f} MB (+{(size-last_size)/(1024*1024):.2f} MB) | Speed: {speed/(1024*1024):.2f} MB/s")
            last_size = size
            t0 = time.time()
            time.sleep(5)
            
    monitor_thread = threading.Thread(target=monitor_download, daemon=True)
    monitor_thread.start()

    # Kaggle input altinda mdeberta-v3-base dataset'i ekli ise yerel yoldan yukle (internetsiz)
    model_path = MODEL
    for root, dirs, files in os.walk("/kaggle/input"):
        if ("pytorch_model.bin" in files or "model.safetensors" in files) and "config.json" in files:
            r_low = root.lower()
            if "deberta-v3-base" in r_low or "mdeberta-v3-base" in r_low or "mdeberta-base" in r_low:
                model_path = root
                log(f"Yerel model dosyalari bulundu: {model_path} (Egitim internetsiz/offline baslayacak)")
                break

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=(model_path != MODEL))
    
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
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, local_files_only=(model_path != MODEL))
    stop_monitor.set()
    
    # Tek GPU kullanimi (Kaggle'da DataParallel NCCL kilitlenmelerini engellemek icin)
    model = model.to(dev)
    
    dl = DataLoader(DS(trn.q.tolist(), trn.d.tolist(), trn.label.to_numpy()),
                    batch_size=BATCH_SIZE, shuffle=True, collate_fn=coll_tr, num_workers=0, pin_memory=True)
                    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler('cuda')
    lossf = torch.nn.CrossEntropyLoss()
    
    log(f"Egitim basliyor: {EPOCHS} epoch, BatchSize={BATCH_SIZE}")
    
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = lossf(out.logits, b["labels"])
                
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
                
            if i % 200 == 0:
                log(f"  ep{ep+1} {i}/{len(dl)} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
                
    @torch.no_grad()
    def score(df):
        model.eval()
        dl = DataLoader(DS(df.q.tolist(), df.d.tolist()), batch_size=BATCH_SIZE * 4, collate_fn=coll_te,
                        num_workers=0, pin_memory=True)
        out = np.zeros(len(df), np.float32)
        p = 0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**b)
                lg = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            out[p:p+lg.shape[0]] = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
            p += lg.shape[0]
        return out

    log("Holdout setini skorlama...")
    ho = score(hol)
    np.save("/kaggle/working/mdeberta_holdout.npy", ho)
    
    from sklearn.metrics import f1_score
    y = hol.label.to_numpy()
    best = max((f1_score(y, (ho >= t).astype(int), average="macro"), t) for t in np.arange(0.3, 0.71, 0.05))
    log(f"Holdout macro-F1: {best[0]:.5f} at threshold: {best[1]:.2f}")
    
    log("Test setini skorlama (3.36M)...")
    ts = score(te)
    np.save("/kaggle/working/mdeberta_test.npy", ts)
    log("TUM ISLEMLER TAMAMLANDI!")

if __name__ == "__main__":
    main()
