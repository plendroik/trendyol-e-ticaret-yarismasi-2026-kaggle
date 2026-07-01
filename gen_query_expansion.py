"""
TF-IDF query expansion via embedding pseudo-relevance feedback (KDD Cup 2022
1st-place "day-day-up" style, adapted for cold-start).

Short/noisy queries ("kettle") are enriched with high-TF-IDF terms drawn from the
query's top-K embedding-nearest catalog products ("kettle su ısıtıcısı paslanmaz
çelik otomatik"). Fully consistent for train AND test queries (no labels used, so
no leak), pair-independent (same expanded query for all candidates of a term).

Output: expanded_terms.csv  (term_id, query=<original + top-M new terms>)
Use with:  train_cross_encoder.py --terms_file expanded_terms.csv
"""
import os, re, time, numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = r"C:\Users\ASUS\Desktop\trendyol"
EMB = os.path.join(DATA, "emb")
TOPK = 15          # nearest products used for expansion
ADD_M = 6          # new terms appended per query
device = "cuda" if torch.cuda.is_available() else "cpu"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    log("Loading embeddings + terms/items...")
    q_emb = np.load(os.path.join(EMB, "query_emb.npy"))
    i_emb = np.load(os.path.join(EMB, "item_emb.npy"))
    q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    title_by_row = items["title"].fillna("").astype(str).to_numpy()

    log("Fitting TF-IDF on titles...")
    vec = TfidfVectorizer(max_features=40000, token_pattern=r"[^\W\d_]{3,}", lowercase=True)
    X = vec.fit_transform(title_by_row)            # n_items x V (csr)
    vocab = np.array(vec.get_feature_names_out())

    log("ANN top-K on GPU...")
    iemb_g = torch.tensor(i_emb, device=device, dtype=torch.float16)
    term2row_emb = {t: k for k, t in enumerate(q_ids)}
    order_terms = terms["term_id"].to_numpy()
    rows = np.array([term2row_emb.get(t, 0) for t in order_terms])
    qmat = torch.tensor(q_emb[rows], device=device, dtype=torch.float16)
    ann = np.zeros((len(rows), TOPK), dtype=np.int64)
    for b in range(0, len(rows), 512):
        sims = qmat[b:b+512] @ iemb_g.T
        ann[b:b+512] = torch.topk(sims, TOPK, dim=1).indices.cpu().numpy()
    del iemb_g, qmat; torch.cuda.empty_cache()

    log("Building expanded queries...")
    Xc = X.tocsr()
    _word = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
    exp = []
    for qi, t in enumerate(order_terms):
        q = str(terms["query"].iloc[qi]) if not pd.isna(terms["query"].iloc[qi]) else ""
        qwords = set(w.lower() for w in _word.findall(q))
        agg = Xc[ann[qi]].sum(axis=0).A1                 # summed tf-idf over top-K titles
        top = np.argpartition(-agg, ADD_M * 4)[:ADD_M * 4]
        top = top[np.argsort(-agg[top])]
        add = []
        for idx in top:
            w = vocab[idx]
            if agg[idx] <= 0:
                break
            if w not in qwords:
                add.append(w)
            if len(add) >= ADD_M:
                break
        exp.append((q + " " + " ".join(add)).strip())
        if qi % 10000 == 0:
            log(f"  {qi}/{len(order_terms)}")

    out = pd.DataFrame({"term_id": order_terms, "query": exp})
    out.to_csv(os.path.join(DATA, "expanded_terms.csv"), index=False)
    log(f"Saved expanded_terms.csv ({len(out)}). e.g.: '{terms['query'].iloc[0]}' -> '{exp[0]}'")
    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
