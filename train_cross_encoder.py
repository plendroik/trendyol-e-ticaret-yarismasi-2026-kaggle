"""
Cross-encoder relevance scorer — the semantic NLP track. Parametrized so several
backbones/seeds can be trained and later blended.

Reads the SAME training pairs/folds that fast_submit.py produced
(artifacts/train_pairs.parquet) so scores blend cleanly with the GBDT.
Serializes structured fields into the product side (KDD'22 winner style):
  query  [SEP]  title . marka <brand> kategori <leaf_cat> renk <color> materyal <mat>

Usage:
  python train_cross_encoder.py --model dbmdz/bert-base-turkish-cased --suffix berturk
  python train_cross_encoder.py --model microsoft/mdeberta-v3-base --suffix mdeberta --epochs 2

Outputs (artifacts/):
  ce_holdout_<suffix>.npy   scores on holdout-fold rows
  ce_test_<suffix>.npy      scores on submission_pairs rows (same order)
  ce_holdout_meta.npy       (label, row_index) shared across runs (same holdout)
  ce_model_<suffix>/        saved fine-tuned model + tokenizer
"""
import os
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART_DIR = os.path.join(DATA_DIR, "artifacts")
HOLDOUT_FOLD = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Discriminative attribute keys to surface (front-loaded so truncation keeps them)
_ATTR_KEYS = ["renk", "materyal", "desen", "kumaş tipi", "ortam", "stil",
              "kol tipi", "yaka tipi", "boy", "kalıp"]
_ATTR_RE = {k: re.compile(re.escape(k) + r":\s*([^,]+)") for k in _ATTR_KEYS}


_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")


def build_docs(items, rich=True):
    """rich=True: full category path + many attributes. rich=False: the basic
    serialization that produced the LB-0.80 model (title+brand+leaf+renk+materyal)."""
    titles = items["title"].fillna("").astype(str).tolist()
    brands = items["brand"].fillna("").astype(str).tolist()
    cats = items["category"].fillna("").astype(str).tolist()
    genders = items["gender"].fillna("").astype(str).tolist()
    ages = items["age_group"].fillna("").astype(str).tolist()
    attrs = items["attributes"].fillna("").astype(str).tolist()
    docs = []
    for t, b, c, g, ag, a in zip(titles, brands, cats, genders, ages, attrs):
        al = a.lower()
        if rich:
            cat_full = c.replace("/", " > ") if c else ""
            parts = [f"{t}", f"marka {b}", f"kategori {cat_full}", f"{g} {ag}"]
            for k in _ATTR_KEYS:
                m = _ATTR_RE[k].search(al)
                if m:
                    parts.append(f"{k} {m.group(1).strip()}")
            docs.append(" . ".join(parts))
        else:
            leaf = c.split("/")[-1] if c else ""
            cm = _C.search(al); mm = _M.search(al)
            docs.append(f"{t} . marka {b} kategori {leaf} renk {cm.group(1).strip() if cm else ''} "
                        f"materyal {mm.group(1).strip() if mm else ''}")
    return docs


class PairDataset(Dataset):
    def __init__(self, queries, docs, labels=None):
        self.q = queries; self.d = docs; self.y = labels

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        if self.y is None:
            return self.q[i], self.d[i]
        return self.q[i], self.d[i], self.y[i]


def make_collate(tokenizer, has_label, max_len):
    def collate(batch):
        if has_label:
            qs, ds, ys = zip(*batch)
        else:
            qs, ds = zip(*batch)
        enc = tokenizer(list(qs), list(ds), truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        if has_label:
            enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc
    return collate


class FGM:
    """Fast Gradient Method: perturb the word-embedding weights along their
    gradient to smooth decision boundaries (better cold-start generalization)."""
    def __init__(self, model, eps=1.0, emb_name="word_embeddings"):
        self.model = model; self.eps = eps; self.emb_name = emb_name; self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


@torch.no_grad()
def predict(model, tokenizer, queries, docs, max_len, infer_batch, tag=""):
    model.eval()
    dl = DataLoader(PairDataset(queries, docs), batch_size=infer_batch, shuffle=False,
                    collate_fn=make_collate(tokenizer, False, max_len), num_workers=0, pin_memory=True)
    out = np.zeros(len(queries), dtype=np.float32)
    pos = 0; t0 = time.time()
    for bi, enc in enumerate(dl):
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        out[pos:pos + len(p)] = p; pos += len(p)
        if bi % 500 == 0:
            log(f"  predict[{tag}] {pos}/{len(queries)} ({pos/len(queries)*100:.1f}%, {time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--suffix", default="berturk")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--train_batch", type=int, default=64)
    ap.add_argument("--infer_batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=0, help="cap training rows (0=all)")
    ap.add_argument("--fgm", action="store_true", help="FGM adversarial training on embeddings")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--basic_docs", action="store_true", help="basic serialization (LB-0.80 recipe)")
    ap.add_argument("--terms_file", default=None, help="override terms.csv (e.g. expanded_terms.csv)")
    ap.add_argument("--pairs_file", default=None, help="override artifacts/train_pairs.parquet")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    log(f"device={device}  model={args.model}  suffix={args.suffix}  epochs={args.epochs}  seed={args.seed}")

    pairs_path = args.pairs_file or os.path.join(ART_DIR, "train_pairs.parquet")
    if args.pairs_file and not os.path.isabs(pairs_path):
        pairs_path = os.path.join(ART_DIR, args.pairs_file)
    pairs = pd.read_parquet(pairs_path)
    log(f"train_pairs={len(pairs)} ({os.path.basename(pairs_path)})  holdout_fold={HOLDOUT_FOLD}")

    log("Loading terms + items, building serialized docs...")
    terms_path = args.terms_file or os.path.join(DATA_DIR, "terms.csv")
    if args.terms_file and not os.path.isabs(terms_path):
        terms_path = os.path.join(DATA_DIR, args.terms_file)
    log(f"terms file: {terms_path}")
    terms = pd.read_csv(terms_path)
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    term2q = dict(zip(terms["term_id"], terms["query"].fillna("").astype(str)))
    item2doc = dict(zip(items["item_id"], build_docs(items, rich=not args.basic_docs)))

    pq = pairs["term_id"].map(term2q).fillna("").tolist()
    pdoc = pairs["item_id"].map(item2doc).fillna("").tolist()
    py = pairs["label"].to_numpy()
    pfold = pairs["fold"].to_numpy()

    tr_idx = np.where(pfold != HOLDOUT_FOLD)[0]
    ho_idx = np.where(pfold == HOLDOUT_FOLD)[0]
    if args.max_train and len(tr_idx) > args.max_train:
        rng = np.random.RandomState(args.seed)
        tr_idx = np.sort(rng.choice(tr_idx, args.max_train, replace=False))
        log(f"subsampled train rows -> {len(tr_idx)} (prevalence preserved in expectation)")
    log(f"train rows={len(tr_idx)}  holdout rows={len(ho_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, trust_remote_code=True).to(device)

    tr_q = [pq[i] for i in tr_idx]; tr_d = [pdoc[i] for i in tr_idx]
    tr_y = py[tr_idx].astype(np.int64)
    train_dl = DataLoader(PairDataset(tr_q, tr_d, tr_y), batch_size=args.train_batch, shuffle=True,
                          collate_fn=make_collate(tokenizer, True, args.max_len), num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    log(f"Training {args.epochs} epochs, {len(train_dl)} steps/epoch... (FGM={'on' if fgm else 'off'})")
    model.train()
    for ep in range(args.epochs):
        t_ep = time.time(); running = 0.0
        for step, enc in enumerate(train_dl):
            labels = enc.pop("labels").to(device)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(**enc).logits, labels)
            scaler.scale(loss).backward()
            if fgm is not None:                       # adversarial pass
                fgm.attack()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_adv = loss_fn(model(**enc).logits, labels)
                scaler.scale(loss_adv).backward()
                fgm.restore()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            if step % 200 == 0:
                log(f"  ep{ep+1} step {step}/{len(train_dl)} loss={running/max(1,step+1):.4f} ({time.time()-t_ep:.0f}s)")
        log(f"Epoch {ep+1} done in {time.time()-t_ep:.0f}s avg_loss={running/len(train_dl):.4f}")

    mdir = os.path.join(ART_DIR, f"ce_model_{args.suffix}")
    os.makedirs(mdir, exist_ok=True)
    model.save_pretrained(mdir); tokenizer.save_pretrained(mdir)
    log(f"Saved model -> {mdir}")

    # free optimizer/grad memory before inference (prevents VRAM thrashing)
    del opt, sched, scaler
    model.zero_grad(set_to_none=True)
    import gc; gc.collect(); torch.cuda.empty_cache()

    log("Scoring holdout fold...")
    ho_q = [pq[i] for i in ho_idx]; ho_d = [pdoc[i] for i in ho_idx]
    ce_ho = predict(model, tokenizer, ho_q, ho_d, args.max_len, args.infer_batch, tag="holdout")
    np.save(os.path.join(ART_DIR, f"ce_holdout_{args.suffix}.npy"), ce_ho)
    # shared meta (label, row index into train_pairs) — same for every run
    np.save(os.path.join(ART_DIR, "ce_holdout_meta.npy"),
            np.stack([py[ho_idx].astype(np.float32), ho_idx.astype(np.float32)], axis=1))
    from sklearn.metrics import f1_score
    bt, bs = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.01):
        s = f1_score(py[ho_idx], (ce_ho >= t).astype(int), average="macro")
        if s > bs:
            bs, bt = s, t
    log(f"CE[{args.suffix}]-only holdout macro-F1={bs:.5f} at thr={bt:.2f}")

    log("Scoring test...")
    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    te_q = sub["term_id"].map(term2q).fillna("").tolist()
    te_d = sub["item_id"].map(item2doc).fillna("").tolist()
    ce_te = predict(model, tokenizer, te_q, te_d, args.max_len, args.infer_batch, tag="test")
    np.save(os.path.join(ART_DIR, f"ce_test_{args.suffix}.npy"), ce_te)
    log(f"Saved ce_test_{args.suffix} ({len(ce_te)}).  DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
