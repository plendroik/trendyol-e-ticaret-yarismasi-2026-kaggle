"""
Clean hard-negative re-mining using the trained cross-encoder as a denoiser
(RocketQA-style). Attacks the false-negative ceiling.

For each training query:
  - embedding-ANN top-K candidate items (exclude the query's positives)
  - score (query, candidate) with the trained CE
  - DROP candidates with CE score > HI  (likely relevant -> false negatives)
  - KEEP as hard negatives candidates with CE score < LO  (CE-confident irrelevant
    but embedding-similar = genuinely hard, clean negatives)
  - sample HARD_PER_POS*P of them
  + RAND_PER_POS*P random easy negatives (stability)

Writes artifacts/train_pairs.parquet (OVERWRITES) at ~test prevalence.

Usage: python remine_negatives.py --ce_model artifacts/ce_model_berturk_hard
"""
import os, re, time, argparse, random, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import GroupKFold

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
EMB = os.path.join(DATA_DIR, "emb")
device = "cuda" if torch.cuda.is_available() else "cpu"

ANN_TOPK = 200
HARD_PER_POS = 6
RAND_PER_POS = 1
HI = 0.50        # CE score above this -> likely false negative -> drop
LO = 0.20        # keep candidates scored below this as clean hard negatives
N_FOLDS = 5
SEED = 42
random.seed(SEED); np.random.seed(SEED)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")
def build_docs(items):
    out = []
    for t, b, c, a in zip(items["title"].fillna("").astype(str),
                          items["brand"].fillna("").astype(str),
                          items["category"].fillna("").astype(str),
                          items["attributes"].fillna("").astype(str)):
        leaf = c.split("/")[-1] if c else ""
        al = a.lower(); cm = _C.search(al); mm = _M.search(al)
        out.append(f"{t} . marka {b} kategori {leaf} renk {cm.group(1).strip() if cm else ''} "
                   f"materyal {mm.group(1).strip() if mm else ''}")
    return out


@torch.no_grad()
def ce_score(model, tok, qs, ds, max_len=160, bs=512):
    model.eval(); out = np.zeros(len(qs), np.float32); p = 0
    for i in range(0, len(qs), bs):
        enc = tok(qs[i:i+bs], ds[i:i+bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**enc).logits
        pr = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
        out[p:p+len(pr)] = pr; p += len(pr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_model", default=os.path.join(ART, "ce_model_berturk_hard"))
    args = ap.parse_args()
    t0 = time.time()

    log("Loading embeddings...")
    q_emb = np.load(os.path.join(EMB, "query_emb.npy"))
    i_emb = np.load(os.path.join(EMB, "item_emb.npy"))
    q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
    term2row = {t: k for k, t in enumerate(q_ids)}
    item2row = {t: k for k, t in enumerate(i_ids)}
    n_items = len(i_ids)

    train = pd.read_csv(os.path.join(DATA_DIR, "training_pairs.csv"))
    pos_by_term = {}
    for t, it in zip(train.term_id.to_numpy(), train.item_id.to_numpy()):
        r = item2row.get(it)
        if r is not None and t in term2row:
            pos_by_term.setdefault(t, []).append(r)
    train_terms = list(pos_by_term.keys())
    log(f"train terms={len(train_terms)}")

    log("Loading terms/items + docs + CE model...")
    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    docs = build_docs(items)
    tok = AutoTokenizer.from_pretrained(args.ce_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ce_model).to(device)

    log("ANN top-K on GPU...")
    iemb_g = torch.tensor(i_emb, device=device, dtype=torch.float16)
    rows = np.array([term2row[t] for t in train_terms])
    qmat = torch.tensor(q_emb[rows], device=device, dtype=torch.float16)
    ann = np.zeros((len(rows), ANN_TOPK), dtype=np.int64)
    for b in range(0, len(rows), 256):
        sims = qmat[b:b+256] @ iemb_g.T
        ann[b:b+256] = torch.topk(sims, ANN_TOPK, dim=1).indices.cpu().numpy()
    del iemb_g, qmat; torch.cuda.empty_cache()

    log("Scoring ANN candidates with CE + selecting clean hard negatives...")
    rows_t, rows_i, rows_y = [], [], []
    n_drop_fn = 0; n_scored = 0
    BATCH_TERMS = 400
    for s in range(0, len(train_terms), BATCH_TERMS):
        bt = train_terms[s:s+BATCH_TERMS]
        # build candidate (query, doc) list for this batch (global index gi = s+ti)
        bq, bd, owner, cand_item = [], [], [], []
        for ti, t in enumerate(bt):
            gi = s + ti
            pset = set(pos_by_term[t]); qx = term2q.get(t, "")
            for r in ann[gi]:
                if r in pset:
                    continue
                bq.append(qx); bd.append(docs[r]); owner.append(t); cand_item.append(r)
        sc = ce_score(model, tok, bq, bd); n_scored += len(sc)
        # group back by term
        by_term = {}
        for q_t, r, s_ in zip(owner, cand_item, sc):
            by_term.setdefault(q_t, []).append((r, s_))
        for t in bt:
            P = len(pos_by_term[t])
            for r in pos_by_term[t]:
                rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(1)
            cands = by_term.get(t, [])
            clean = [r for r, sc_ in cands if sc_ < LO]
            n_drop_fn += sum(1 for r, sc_ in cands if sc_ > HI)
            need = HARD_PER_POS * P
            hard = random.sample(clean, need) if len(clean) > need else clean
            for r in hard:
                rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(0)
            pset = set(pos_by_term[t]); c = 0
            while c < RAND_PER_POS * P:
                r = random.randrange(n_items)
                if r not in pset:
                    rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(0); c += 1
        if s % (BATCH_TERMS * 5) == 0:
            log(f"  {s}/{len(train_terms)}  scored={n_scored}")

    df = pd.DataFrame({"term_id": rows_t, "item_id": rows_i, "label": np.array(rows_y, np.int8)})
    log(f"rows={len(df)}  pos={int((df.label==1).sum())}  neg={int((df.label==0).sum())}  "
        f"pos_rate={(df.label==1).mean():.3f}  FN-dropped(CE>{HI})={n_drop_fn}")

    gkf = GroupKFold(n_splits=N_FOLDS); fold = np.full(len(df), -1, np.int8)
    for f, (_, va) in enumerate(gkf.split(df, df.label, groups=df.term_id)):
        fold[va] = f
    df["fold"] = fold
    df.to_parquet(os.path.join(ART, "train_pairs.parquet"), index=False)
    log(f"Saved artifacts/train_pairs.parquet  DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
