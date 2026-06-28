"""
Generate TEST-LIKE training pairs: hard negatives mined by embedding-ANN, with
false-negative filtering, at the test's ~1:7 prevalence.

Why: the test is a reranking set (~104 retrieval candidates/query, ~13% relevant).
Our old negatives (random + TF-IDF mid-rank) were far too easy -> the model could
not discriminate hard distractors (proven: BERTurk 0.90 easy-holdout -> 0.53 on a
test-like hard set). This rebuilds the negative class to match the test.

For each training query:
  positives            -> label 1 (all kept)
  ANN top-K nearest    -> drop the query's positives; drop sim > FN_FILTER*max_pos_sim
                          (likely false negatives); take HARD_PER_POS*P from the top
  random items         -> RAND_PER_POS*P (easy, for stability)
Total negatives ~ (HARD+RAND)*P  -> prevalence ~= 1/(1+HARD+RAND).

Output: artifacts/train_pairs.parquet  (term_id, item_id, label, fold)  [OVERWRITES]
"""
import os, time, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
EMB = os.path.join(DATA_DIR, "emb")
os.makedirs(ART, exist_ok=True)

ANN_TOPK = 300
HARD_PER_POS = 5
RAND_PER_POS = 2
FN_FILTER = 0.95       # drop candidates more similar than 95% of the best positive
HARD_POOL = 80         # sample hard negs from the top-N filtered candidates
N_FOLDS = 5
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    log("Loading embeddings + ids...")
    q_emb = np.load(os.path.join(EMB, "query_emb.npy"))               # n_terms x D (fp16, L2)
    i_emb = np.load(os.path.join(EMB, "item_emb.npy"))                # n_items x D
    q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
    term2row = {t: k for k, t in enumerate(q_ids)}
    item2row = {t: k for k, t in enumerate(i_ids)}
    n_items = len(i_ids)

    log("Loading training positives...")
    train = pd.read_csv(os.path.join(DATA_DIR, "training_pairs.csv"))
    pos_by_term = {}
    for t, it in zip(train.term_id.to_numpy(), train.item_id.to_numpy()):
        r = item2row.get(it)
        if r is not None and t in term2row:
            pos_by_term.setdefault(t, []).append(r)
    train_terms = list(pos_by_term.keys())
    log(f"train terms={len(train_terms)}  positives={sum(len(v) for v in pos_by_term.values())}")

    # ANN on GPU
    log("ANN top-K on GPU...")
    iemb_g = torch.tensor(i_emb, device=device, dtype=torch.float16)
    term_rows = np.array([term2row[t] for t in train_terms])
    qmat = torch.tensor(q_emb[term_rows], device=device, dtype=torch.float16)
    ann_idx = np.zeros((len(train_terms), ANN_TOPK), dtype=np.int64)
    ann_sim = np.zeros((len(train_terms), ANN_TOPK), dtype=np.float32)
    B = 256
    for b in range(0, len(term_rows), B):
        sims = qmat[b:b+B] @ iemb_g.T
        vals, idx = torch.topk(sims, ANN_TOPK, dim=1)
        ann_idx[b:b+B] = idx.cpu().numpy()
        ann_sim[b:b+B] = vals.float().cpu().numpy()
        if b % (B*20) == 0:
            log(f"  ANN {b}/{len(term_rows)}")
    del iemb_g, qmat; torch.cuda.empty_cache()

    # per-query negative assembly
    log("Assembling hard + random negatives...")
    i_emb_f = i_emb.astype(np.float32)
    rows_t, rows_i, rows_y = [], [], []
    n_fn_dropped = 0
    for qi, t in enumerate(train_terms):
        pos_rows = pos_by_term[t]
        pos_set = set(pos_rows)
        P = len(pos_rows)
        # positives
        for r in pos_rows:
            rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(1)
        # false-negative threshold from best positive similarity
        qv = q_emb[term_rows[qi]].astype(np.float32)
        max_pos_sim = float(np.max(i_emb_f[pos_rows] @ qv)) if P else 1.0
        thr = FN_FILTER * max_pos_sim
        # filter ANN candidates: not a positive, not a false negative
        cand = []
        for r, s in zip(ann_idx[qi], ann_sim[qi]):
            if r in pos_set:
                continue
            if s > thr:
                n_fn_dropped += 1
                continue
            cand.append(r)
            if len(cand) >= HARD_POOL:
                break
        n_hard = HARD_PER_POS * P
        hard = random.sample(cand, n_hard) if len(cand) > n_hard else cand
        for r in hard:
            rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(0)
        # random easy negatives
        need = RAND_PER_POS * P
        c = 0
        while c < need:
            r = random.randrange(n_items)
            if r not in pos_set:
                rows_t.append(t); rows_i.append(i_ids[r]); rows_y.append(0); c += 1
        if qi % 4000 == 0:
            log(f"  {qi}/{len(train_terms)}")

    df = pd.DataFrame({"term_id": rows_t, "item_id": rows_i,
                       "label": np.array(rows_y, dtype=np.int8)})
    log(f"rows={len(df)}  pos={int((df.label==1).sum())}  neg={int((df.label==0).sum())}  "
        f"pos_rate={(df.label==1).mean():.3f}  FN-dropped={n_fn_dropped}")

    # group folds by term
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold = np.full(len(df), -1, dtype=np.int8)
    for f, (_, va) in enumerate(gkf.split(df, df.label, groups=df.term_id)):
        fold[va] = f
    df["fold"] = fold
    df.to_parquet(os.path.join(ART, "train_pairs.parquet"), index=False)
    log(f"Saved artifacts/train_pairs.parquet  DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
