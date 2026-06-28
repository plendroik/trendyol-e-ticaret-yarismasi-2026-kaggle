"""
Fast, battery-friendly GBDT submission for the Trendyol query-product relevance task.

Design goals (under a hard battery/time deadline):
  * No neural embeddings (would need GPU / hours of CPU). Pure lexical + structural + TF-IDF.
  * Expensive text work is done ONCE over ~966k items + ~50k terms, never per-pair.
  * Per-pair features are computed with precomputed token sets (tight loop) + batched
    sparse TF-IDF cosine. This keeps 3.36M test rows tractable in minutes.
  * Negatives are mostly HARD (TF-IDF mid-rank), because the test candidates are already
    retrieval candidates (~104 per query) -> the test "irrelevant" class is hard, not random.
  * Cold-start aware: train/test queries do NOT overlap, so GroupKFold by term_id.

Run:  python fast_submit.py
Output: <DATA_DIR>/submission.csv
"""
import os
import re
import sys
import time
import random
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------- config
DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
EMB_DIR = os.path.join(DATA_DIR, "emb")
ART_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)
SEED = 42
N_FOLDS = 5
TFIDF_MAX_FEATURES = 50000
RETRIEVE_TOPK = 200          # candidates retrieved per train term for hard negatives
HARD_SKIP_TOP = 5            # skip the very top (likely false negatives)
N_HARD_PER_POS = 2          # hard negatives per positive
N_RAND_PER_POS = 1          # random negatives per positive
RETRIEVE_BATCH = 128
COSINE_BATCH = 500_000

random.seed(SEED)
np.random.seed(SEED)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------- text normalization
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized Turkish-aware lowercase + punctuation strip."""
    s = s.fillna("").astype(str)
    s = s.str.replace("İ", "i", regex=False).str.replace("I", "ı", regex=False)
    s = s.str.lower()
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    s = s.str.replace(_SPACE_RE, " ", regex=True).str.strip()
    return s


# --------------------------------------------------------------------------------- load
def load_data():
    log("Loading CSVs...")
    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "training_pairs.csv"))
    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    log(f"  terms={len(terms)} items={len(items)} train={len(train)} sub={len(sub)}")
    return terms, items, train, sub


def build_structures(terms, items):
    log("Normalizing text (queries, titles, categories)...")
    terms = terms.copy()
    items = items.copy()
    terms["qn"] = normalize_series(terms["query"])
    items["tn"] = normalize_series(items["title"])
    items["cn"] = normalize_series(items["category"])
    items["brn"] = normalize_series(items["brand"])
    items["gender"] = items["gender"].fillna("").astype(str).str.lower().str.strip()
    items["age_group"] = items["age_group"].fillna("").astype(str).str.lower().str.strip()

    # index maps
    term_ids = terms["term_id"].to_numpy()
    item_ids = items["item_id"].to_numpy()
    term_idx = {t: i for i, t in enumerate(term_ids)}
    item_idx = {t: i for i, t in enumerate(item_ids)}

    log("Building token sets...")
    term_tok = [frozenset(q.split()) for q in terms["qn"].tolist()]
    item_tok = [frozenset(t.split()) for t in items["tn"].tolist()]
    item_cat_tok = [frozenset(c.split()) for c in items["cn"].tolist()]

    term_qn = terms["qn"].tolist()
    item_brn = items["brn"].tolist()
    item_gender = items["gender"].tolist()
    item_age = items["age_group"].tolist()

    term_qlen = np.array([max(1, len(s)) for s in term_tok], dtype=np.int32)
    item_tlen = np.array([len(s) for s in item_tok], dtype=np.int32)

    # query intent flags (vectorized)
    qn = terms["qn"]
    q_female = qn.str.contains(r"kadın|kız|bayan", regex=True).to_numpy()
    q_male = qn.str.contains(r"erkek|\bbay\b", regex=True).to_numpy()
    q_child = qn.str.contains(r"bebek|çocuk|baby|kid", regex=True).to_numpy()

    struct = dict(
        term_idx=term_idx, item_idx=item_idx,
        term_tok=term_tok, item_tok=item_tok, item_cat_tok=item_cat_tok,
        term_qn=term_qn, item_brn=item_brn, item_gender=item_gender, item_age=item_age,
        term_qlen=term_qlen, item_tlen=item_tlen,
        q_female=q_female, q_male=q_male, q_child=q_child,
        item_ids=item_ids,
    )
    return terms, items, struct


# --------------------------------------------------------------------------- tf-idf
def build_tfidf(terms, items):
    log(f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}) on item titles...")
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, token_pattern=r"\w+")
    X = vec.fit_transform(items["tn"].tolist())          # items x V
    Q = vec.transform(terms["qn"].tolist())              # terms x V
    Xn = sk_normalize(X, norm="l2", axis=1, copy=False)
    Qn = sk_normalize(Q, norm="l2", axis=1, copy=False)
    log(f"  X={Xn.shape} Q={Qn.shape}")
    return Xn, Qn


# --------------------------------------------------------------- embeddings (optional)
def load_embeddings(struct):
    """Load query/item embeddings (L2-normalized fp16) and align them to the
    terms/items ordering used in `struct`. Returns (q_emb, i_emb) or (None, None)."""
    qf = os.path.join(EMB_DIR, "query_emb.npy")
    itf = os.path.join(EMB_DIR, "item_emb.npy")
    if not (os.path.exists(qf) and os.path.exists(itf)):
        log("No embeddings found -> running lexical-only.")
        return None, None
    log("Loading + aligning embeddings...")
    q_raw = np.load(qf); q_ids = np.load(os.path.join(EMB_DIR, "query_ids.npy"), allow_pickle=True)
    i_raw = np.load(itf); i_ids = np.load(os.path.join(EMB_DIR, "item_ids.npy"), allow_pickle=True)
    term_idx = struct["term_idx"]; item_idx = struct["item_idx"]
    D = q_raw.shape[1]
    q_emb = np.zeros((len(term_idx), D), dtype=np.float16)
    pos = np.fromiter((term_idx[t] for t in q_ids), dtype=np.int64, count=len(q_ids))
    q_emb[pos] = q_raw
    i_emb = np.zeros((len(item_idx), D), dtype=np.float16)
    pos = np.fromiter((item_idx[t] for t in i_ids), dtype=np.int64, count=len(i_ids))
    i_emb[pos] = i_raw
    log(f"  q_emb={q_emb.shape} i_emb={i_emb.shape}")
    return q_emb, i_emb


def emb_cosine_pairs(t_idx, i_idx, q_emb, i_emb):
    out = np.zeros(len(t_idx), dtype=np.float32)
    if q_emb is None:
        return out
    B = 250_000
    for b in range(0, len(t_idx), B):
        e = min(b + B, len(t_idx))
        q = q_emb[t_idx[b:e]].astype(np.float32)
        it = i_emb[i_idx[b:e]].astype(np.float32)
        out[b:e] = np.einsum("ij,ij->i", q, it)  # both L2-normalized -> cosine
    return out


# --------------------------------------------------------------- negative sampling
def sample_negatives(train, struct, Xn, Qn):
    log("Building positives index + retrieving hard-negative candidates...")
    term_idx = struct["term_idx"]
    item_idx = struct["item_idx"]
    item_ids = struct["item_ids"]
    n_items = len(item_ids)

    # positives per term (as item-index sets)
    pos_by_term = {}
    for t, it in zip(train["term_id"].to_numpy(), train["item_id"].to_numpy()):
        ti = term_idx.get(t)
        ii = item_idx.get(it)
        if ti is None or ii is None:
            continue
        pos_by_term.setdefault(ti, []).append(ii)

    train_term_list = list(pos_by_term.keys())

    # Retrieve top-K candidates per train term via TF-IDF (batched)
    cand_by_term = {}
    qn_csr = Qn.tocsr()
    XnT = Xn.T.tocsr()
    tt = np.array(train_term_list)
    for b in range(0, len(tt), RETRIEVE_BATCH):
        batch = tt[b:b + RETRIEVE_BATCH]
        sims = (qn_csr[batch] @ XnT)            # |batch| x n_items (sparse-ish)
        sims = np.asarray(sims.todense())
        topk = min(RETRIEVE_TOPK, n_items)
        part = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]
        for r, ti in enumerate(batch):
            row = sims[r]
            order = part[r][np.argsort(-row[part[r]])]
            cand_by_term[ti] = order.astype(np.int64)
        if (b // RETRIEVE_BATCH) % 40 == 0:
            log(f"  retrieval {b}/{len(tt)}")

    log("Assembling training pairs (pos + hard + random negatives)...")
    rows_t, rows_i, rows_y = [], [], []
    for ti, pos_list in pos_by_term.items():
        pos_set = set(pos_list)
        p = len(pos_list)
        # positives
        for ii in pos_list:
            rows_t.append(ti); rows_i.append(ii); rows_y.append(1)
        # hard negatives from mid-rank candidates (skip very top -> false negatives)
        cands = [c for c in cand_by_term.get(ti, [])[HARD_SKIP_TOP:] if c not in pos_set]
        n_hard = N_HARD_PER_POS * p
        if len(cands) > n_hard:
            chosen = random.sample(cands, n_hard)
        else:
            chosen = cands
        for ii in chosen:
            rows_t.append(ti); rows_i.append(ii); rows_y.append(0)
        # random negatives
        n_rand = N_RAND_PER_POS * p
        cnt = 0
        while cnt < n_rand:
            ii = random.randrange(n_items)
            if ii not in pos_set:
                rows_t.append(ti); rows_i.append(ii); rows_y.append(0)
                cnt += 1

    t_idx = np.array(rows_t, dtype=np.int64)
    i_idx = np.array(rows_i, dtype=np.int64)
    y = np.array(rows_y, dtype=np.int8)
    log(f"  train rows={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}")
    return t_idx, i_idx, y


# --------------------------------------------------------------- feature engineering
FEATURE_COLS = [
    "coverage", "jaccard", "overlap", "q_len", "t_len", "len_ratio",
    "brand_in_q", "gender_contra", "age_contra", "cat_overlap", "tfidf_cos",
    "emb_cos",
]


def tfidf_cosine_pairs(t_idx, i_idx, Xn, Qn):
    out = np.zeros(len(t_idx), dtype=np.float32)
    Qc = Qn.tocsr()
    Xc = Xn.tocsr()
    for b in range(0, len(t_idx), COSINE_BATCH):
        e = min(b + COSINE_BATCH, len(t_idx))
        Qs = Qc[t_idx[b:e]]
        Xs = Xc[i_idx[b:e]]
        out[b:e] = np.asarray(Qs.multiply(Xs).sum(axis=1)).ravel()
    return out


def make_features(t_idx, i_idx, struct, Xn, Qn, tag=""):
    log(f"Feature engineering [{tag}] for {len(t_idx)} pairs...")
    term_tok = struct["term_tok"]; item_tok = struct["item_tok"]
    item_cat_tok = struct["item_cat_tok"]
    term_qn = struct["term_qn"]; item_brn = struct["item_brn"]
    item_gender = struct["item_gender"]; item_age = struct["item_age"]
    term_qlen = struct["term_qlen"]; item_tlen = struct["item_tlen"]
    q_female = struct["q_female"]; q_male = struct["q_male"]; q_child = struct["q_child"]

    n = len(t_idx)
    coverage = np.zeros(n, dtype=np.float32)
    jaccard = np.zeros(n, dtype=np.float32)
    overlap = np.zeros(n, dtype=np.float32)
    brand_in_q = np.zeros(n, dtype=np.float32)
    gender_contra = np.zeros(n, dtype=np.float32)
    age_contra = np.zeros(n, dtype=np.float32)
    cat_overlap = np.zeros(n, dtype=np.float32)

    for k in range(n):
        ti = t_idx[k]; ii = i_idx[k]
        qset = term_tok[ti]; tset = item_tok[ii]
        if qset:
            ov = 0
            for w in qset:
                if w in tset:
                    ov += 1
            overlap[k] = ov
            coverage[k] = ov / len(qset)
            uni = len(qset) + len(tset) - ov
            if uni > 0:
                jaccard[k] = ov / uni
            # category overlap
            cset = item_cat_tok[ii]
            cov = 0
            for w in qset:
                if w in cset:
                    cov += 1
            cat_overlap[k] = cov
        br = item_brn[ii]
        if br and br in term_qn[ti]:
            brand_in_q[k] = 1.0
        g = item_gender[ii]
        if (q_female[ti] and g == "erkek") or (q_male[ti] and g == "kadın"):
            gender_contra[k] = 1.0
        if q_child[ti] and g == "" and item_age[ii] == "yetişkin":
            pass
        if q_child[ti] and item_age[ii] == "yetişkin":
            age_contra[k] = 1.0
        if k and k % 1_000_000 == 0:
            log(f"  ...{k}/{n}")

    q_len = term_qlen[t_idx].astype(np.float32)
    t_len = item_tlen[i_idx].astype(np.float32)
    len_ratio = q_len / np.maximum(1.0, t_len)
    tfidf_cos = tfidf_cosine_pairs(t_idx, i_idx, Xn, Qn)
    emb_cos = emb_cosine_pairs(t_idx, i_idx, struct.get("q_emb"), struct.get("i_emb"))

    df = pd.DataFrame({
        "coverage": coverage, "jaccard": jaccard, "overlap": overlap,
        "q_len": q_len, "t_len": t_len, "len_ratio": len_ratio,
        "brand_in_q": brand_in_q, "gender_contra": gender_contra,
        "age_contra": age_contra, "cat_overlap": cat_overlap, "tfidf_cos": tfidf_cos,
        "emb_cos": emb_cos,
    })
    return df


# ----------------------------------------------------------------------- threshold
def best_threshold(y_true, probs):
    best_t, best_s = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        s = f1_score(y_true, (probs >= t).astype(int), average="macro")
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    terms, items, train, sub = load_data()
    terms, items, struct = build_structures(terms, items)
    Xn, Qn = build_tfidf(terms, items)
    q_emb, i_emb = load_embeddings(struct)
    struct["q_emb"] = q_emb
    struct["i_emb"] = i_emb

    # ---- training data
    use_existing = os.environ.get("USE_EXISTING_PAIRS") == "1"
    preset_fold = None
    if use_existing:
        pp = os.path.join(ART_DIR, "train_pairs.parquet")
        log(f"Loading existing pairs from {pp}")
        ppdf = pd.read_parquet(pp)
        term_idx = struct["term_idx"]; item_idx = struct["item_idx"]
        t_idx = ppdf["term_id"].map(term_idx).to_numpy().astype(np.int64)
        i_idx = ppdf["item_id"].map(item_idx).to_numpy().astype(np.int64)
        y = ppdf["label"].to_numpy().astype(np.int8)
        preset_fold = ppdf["fold"].to_numpy().astype(np.int8)
        log(f"  loaded {len(y)} pairs  pos_rate={y.mean():.3f}")
    else:
        t_idx, i_idx, y = sample_negatives(train, struct, Xn, Qn)
    groups = t_idx.copy()
    Xtr = make_features(t_idx, i_idx, struct, Xn, Qn, tag="train")

    # ---- CV train
    oof = np.zeros(len(y), dtype=np.float64)
    fold_arr = np.full(len(y), -1, dtype=np.int8)
    models = []
    if preset_fold is not None:
        splits = [(np.where(preset_fold != f)[0], np.where(preset_fold == f)[0]) for f in range(N_FOLDS)]
    else:
        splits = list(GroupKFold(n_splits=N_FOLDS).split(Xtr, y, groups))
    for fold, (tr, va) in enumerate(splits):
        fold_arr[va] = fold
        log(f"Fold {fold+1}/{N_FOLDS} train={len(tr)} val={len(va)}")
        lgbm = lgb.LGBMClassifier(objective="binary", n_estimators=400,
                                  learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
        lgbm.fit(Xtr.iloc[tr], y[tr],
                 eval_set=[(Xtr.iloc[va], y[va])],
                 callbacks=[lgb.early_stopping(40, verbose=False)])
        cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8,
                                loss_function="Logloss", random_seed=SEED,
                                verbose=False)
        cb.fit(Xtr.iloc[tr], y[tr], eval_set=(Xtr.iloc[va], y[va]),
               early_stopping_rounds=40, verbose=False)
        p = (lgbm.predict_proba(Xtr.iloc[va])[:, 1] + cb.predict_proba(Xtr.iloc[va])[:, 1]) / 2
        oof[va] = p
        models.append((lgbm, cb))

    thr, score = best_threshold(y, oof)
    log(f"OOF macro-F1={score:.5f} at threshold={thr:.2f}")
    pr, rc, f1, _ = precision_recall_fscore_support(y, (oof >= thr).astype(int),
                                                     labels=[0, 1], average=None)
    log(f"  class0 P={pr[0]:.3f} R={rc[0]:.3f} F1={f1[0]:.3f} | "
        f"class1 P={pr[1]:.3f} R={rc[1]:.3f} F1={f1[1]:.3f}")
    imp = sorted(zip(FEATURE_COLS, models[0][0].feature_importances_),
                 key=lambda x: -x[1])
    log(f"  LGBM importances: {imp}")

    # ---- save training artifacts (identical pairs/folds for the cross-encoder + blend)
    log("Saving training artifacts...")
    if not use_existing:
        term_id_arr = terms["term_id"].to_numpy()
        item_id_arr = items["item_id"].to_numpy()
        pairs_df = pd.DataFrame({
            "term_id": term_id_arr[t_idx],
            "item_id": item_id_arr[i_idx],
            "label": y.astype(np.int8),
            "fold": fold_arr,
        })
        pairs_df.to_parquet(os.path.join(ART_DIR, "train_pairs.parquet"), index=False)
    np.save(os.path.join(ART_DIR, "gbdt_oof.npy"), oof.astype(np.float32))
    log(f"  saved gbdt_oof to {ART_DIR} (use_existing={use_existing})")

    # ---- test
    log("Mapping test pairs to indices...")
    term_idx = struct["term_idx"]; item_idx = struct["item_idx"]
    sub_t = sub["term_id"].map(term_idx).to_numpy()
    sub_i = sub["item_id"].map(item_idx).to_numpy()
    miss = np.isnan(sub_t.astype(float)) | np.isnan(sub_i.astype(float))
    log(f"  unmapped test pairs: {int(miss.sum())}")
    sub_t = np.nan_to_num(sub_t, nan=0).astype(np.int64)
    sub_i = np.nan_to_num(sub_i, nan=0).astype(np.int64)

    Xte = make_features(sub_t, sub_i, struct, Xn, Qn, tag="test")
    probs = np.zeros(len(Xte), dtype=np.float64)
    for lgbm, cb in models:
        probs += lgbm.predict_proba(Xte)[:, 1] / (2 * len(models))
        probs += cb.predict_proba(Xte)[:, 1] / (2 * len(models))
    np.save(os.path.join(ART_DIR, "gbdt_test.npy"), probs.astype(np.float32))
    log(f"  saved gbdt_test ({len(probs)}) to {ART_DIR}")
    preds = (probs >= thr).astype(int)
    preds[miss] = 0  # unmappable -> irrelevant

    out = pd.DataFrame({"id": sub["id"].to_numpy(), "prediction": preds})
    out_path = os.path.join(DATA_DIR, "submission.csv")
    out.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  rows={len(out)}  pos_rate={preds.mean():.4f}")
    log(f"Prediction counts: {out['prediction'].value_counts().to_dict()}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
