"""
DIAGNOSTIC: does our model collapse on a TEST-LIKE (hard) candidate set?

Easy holdout (our constructed negatives) gives BERTurk macro-F1 ~0.90, but public
LB is 0.76. Hypothesis: the test is a reranking set (~104 retrieval candidates per
query, all semantically related), so its "irrelevant" class is hard — far harder
than our random/TF-IDF negatives.

This rebuilds a test-like validation for held-out (fold-4) terms:
  candidates(term) = positives(term)  ∪  embedding-ANN top-K nearest items
  label = 1 if candidate is a known positive, else 0
then scores with the trained BERTurk model and reports macro-F1.

If the score drops toward ~0.76 -> hypothesis confirmed, and we now have an
offline harness that correlates with the LB.
"""
import os, time, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import re

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
EMB = os.path.join(DATA_DIR, "emb")
KCAND = 100          # ANN candidates per term (mimic ~104 in test)
N_EVAL_TERMS = 3000  # fixed random held-out-ish terms for a model-independent compare
device = "cuda" if torch.cuda.is_available() else "cpu"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_ATTR_KEYS = ["renk", "materyal", "desen", "kumaş tipi", "ortam", "stil",
              "kol tipi", "yaka tipi", "boy", "kalıp"]
_ATTR_RE = {k: re.compile(re.escape(k) + r":\s*([^,]+)") for k in _ATTR_KEYS}
_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")


def build_docs(items, rich=True):
    """rich=True matches train_cross_encoder's enriched serialization (berturk_clean);
    rich=False is the basic one (berturk_hard / berturk)."""
    titles = items["title"].fillna("").astype(str).tolist()
    brands = items["brand"].fillna("").astype(str).tolist()
    cats = items["category"].fillna("").astype(str).tolist()
    genders = items["gender"].fillna("").astype(str).tolist()
    ages = items["age_group"].fillna("").astype(str).tolist()
    attrs = items["attributes"].fillna("").astype(str).tolist()
    out = []
    for t, b, c, g, ag, a in zip(titles, brands, cats, genders, ages, attrs):
        al = a.lower()
        if rich:
            cat_full = c.replace("/", " > ") if c else ""
            parts = [f"{t}", f"marka {b}", f"kategori {cat_full}", f"{g} {ag}"]
            for k in _ATTR_KEYS:
                m = _ATTR_RE[k].search(al)
                if m:
                    parts.append(f"{k} {m.group(1).strip()}")
            out.append(" . ".join(parts))
        else:
            leaf = c.split("/")[-1] if c else ""
            cm = _C.search(al); mm = _M.search(al)
            out.append(f"{t} . marka {b} kategori {leaf} renk {cm.group(1).strip() if cm else ''} "
                       f"materyal {mm.group(1).strip() if mm else ''}")
    return out


@torch.no_grad()
def score(model, tok, qs, ds, max_len=192, bs=384):
    model.eval(); out = np.zeros(len(qs), np.float32); pos = 0
    for i in range(0, len(qs), bs):
        enc = tok(qs[i:i+bs], ds[i:i+bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**enc).logits
        p = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
        out[pos:pos+len(p)] = p; pos += len(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_model", default=os.path.join(ART, "ce_model_berturk_hard"))
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--basic_docs", action="store_true", help="basic serialization")
    args = ap.parse_args()
    t0 = time.time()
    log(f"model={args.ce_model}  rich_docs={not args.basic_docs}")
    log("Loading embeddings + maps...")
    q_emb = np.load(os.path.join(EMB, "query_emb.npy"))
    i_emb = np.load(os.path.join(EMB, "item_emb.npy"))
    q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
    term2row = {t: k for k, t in enumerate(q_ids)}
    itemrow2id = i_ids
    item2row = {t: k for k, t in enumerate(i_ids)}

    train = pd.read_csv(os.path.join(DATA_DIR, "training_pairs.csv"))
    pos_by_term = train.groupby("term_id")["item_id"].apply(set).to_dict()
    all_terms = np.array([t for t in pos_by_term if t in term2row])
    rng = np.random.RandomState(123)
    ho_terms = rng.choice(all_terms, min(N_EVAL_TERMS, len(all_terms)), replace=False)
    log(f"eval terms={len(ho_terms)} (fixed seed=123, model-independent)")

    log("Loading terms/items + docs...")
    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    docs = build_docs(items, rich=not args.basic_docs)
    item2doc = dict(zip(items.item_id, docs))

    # ANN on GPU
    log("ANN top-K on GPU...")
    iemb_t = torch.tensor(i_emb, device=device, dtype=torch.float16)  # 962873 x 768
    rows = np.array([term2row[t] for t in ho_terms])
    qmat = torch.tensor(q_emb[rows], device=device, dtype=torch.float16)
    cand_rows = {}
    B = 256
    for b in range(0, len(rows), B):
        sims = qmat[b:b+B] @ iemb_t.T              # |b| x n_items
        topk = torch.topk(sims, KCAND, dim=1).indices.cpu().numpy()
        for j, t in enumerate(ho_terms[b:b+B]):
            cand_rows[t] = topk[j]
    del iemb_t, qmat; torch.cuda.empty_cache()

    # build candidate pairs (positives forced in + ANN distractors)
    log("Building test-like candidate pairs...")
    Q, D, Y = [], [], []
    for t in ho_terms:
        qtext = term2q.get(t, "")
        ptags = pos_by_term.get(t, set())
        cand = set(itemrow2id[r] for r in cand_rows[t]) | ptags
        for it in cand:
            Q.append(qtext); D.append(item2doc.get(it, "")); Y.append(1 if it in ptags else 0)
    Y = np.array(Y)
    log(f"candidate pairs={len(Y)}  pos={Y.sum()}  pos_rate={Y.mean():.3f}  "
        f"avg_cand/term={len(Y)/len(ho_terms):.1f}")

    log("Loading CE model + scoring...")
    tok = AutoTokenizer.from_pretrained(args.ce_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ce_model).to(device)
    s = score(model, tok, Q, D, max_len=args.max_len)

    # metrics
    for thr in [0.365, 0.44, 0.5]:
        f1 = f1_score(Y, (s >= thr).astype(int), average="macro")
        log(f"  thr={thr:.3f}  HARD-val macro-F1={f1:.5f}  pred_pos_rate={(s>=thr).mean():.3f}")
    bt, bs = 0.5, -1
    for t in np.arange(0.1, 0.95, 0.01):
        f = f1_score(Y, (s >= t).astype(int), average="macro")
        if f > bs: bs, bt = f, t
    log(f"  BEST thr={bt:.2f}  HARD-val macro-F1={bs:.5f}")
    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
