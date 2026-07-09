"""
LLM-as-judge labeling of test-like candidates (the "label war" lever).

Our synthetic negatives don't match the test -> LB stuck at 0.83. Here we build a
TRAINING set that mirrors the test: for each train query we pull embedding-ANN
candidates (like the test's ~104 retrieval candidates) and let gpt-4o-mini judge
each (query, product) pair relevant/irrelevant. The LLM-"irrelevant" ones become
CLEAN hard negatives; LLM-"relevant" ones are extra positives.

ANN runs on CPU (does NOT touch the GPU training in progress). Batched, resume-safe,
cost-tracked. Key from openai_key.txt or env OPENAI_API_KEY.

Usage:
  python gen_llm_labels.py --limit 500     # ~$0.05 quality check
  python gen_llm_labels.py                  # all train queries (~$1.5)
Output: llm_labels.csv  (term_id, item_id, label)   [in DATA dir]
"""
import os, sys, re, time, csv, argparse, numpy as np, pandas as pd
from openai import OpenAI

DATA = r"C:\Users\ASUS\Desktop\trendyol"
EMB = os.path.join(DATA, "emb")
OUT = os.path.join(DATA, "llm_labels.csv")
MODEL = "gpt-4o-mini"
IN_COST, OUT_COST = 0.15 / 1e6, 0.60 / 1e6
CAND_RANKS = [3, 10, 25, 45, 70, 110]       # span the relevance spectrum (test negs ~rank 30-100)
LLM_BATCH = 15
_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")

SYS = ("Sen Trendyol arama-alaka uzmanısın. Her (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURAL: ürün, sorgunun "
       "istediği ürün tipiyle/kategorisiyle AYNI ya da onu karşılıyorsa 1 (alakalı) — "
       "renk/model/marka farkı ÖNEMSİZ, aynı tip yeterli. Ürün TAMAMEN farklı bir "
       "tip/kategori ise 0. Kararsız kalırsan 1 ver. Sadece 'numara. 0' veya "
       "'numara. 1', her çift TEK satır, başka hiçbir şey yazma.")


def get_key():
    k = os.environ.get("OPENAI_API_KEY")
    kf = os.path.join(os.path.dirname(__file__), "openai_key.txt")
    if not k and os.path.exists(kf):
        k = open(kf, encoding="utf-8").read().strip()
    if not k:
        sys.exit("OPENAI_API_KEY / openai_key.txt yok.")
    return k


def prod_text(title, cat, attrs):
    leaf = cat.split("/")[-1] if isinstance(cat, str) and cat else ""
    al = attrs.lower() if isinstance(attrs, str) else ""
    cm = _C.search(al); mm = _M.search(al)
    t = (title or "")[:70]
    extra = f" renk:{cm.group(1).strip()}" if cm else ""
    extra += f" materyal:{mm.group(1).strip()}" if mm else ""
    return f"{t} [{leaf}]{extra}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max train queries (0=all)")
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())

    print("Loading embeddings + data...", flush=True)
    q_emb = np.load(os.path.join(EMB, "query_emb.npy")).astype(np.float32)
    i_emb = np.load(os.path.join(EMB, "item_emb.npy")).astype(np.float32)
    q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
    term2row = {t: k for k, t in enumerate(q_ids)}
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes)
            for r in items.itertuples(index=False)}
    train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
    pos_by_term = train.groupby("term_id")["item_id"].apply(set).to_dict()

    train_terms = [t for t in pos_by_term if t in term2row]
    if args.limit:
        train_terms = train_terms[:args.limit]

    done_terms = set()
    if os.path.exists(OUT):
        done_terms = set(pd.read_csv(OUT)["term_id"].astype(str))
    todo = [t for t in train_terms if t not in done_terms]
    print(f"train_terms={len(train_terms)} done={len(done_terms)} todo={len(todo)}", flush=True)

    # CPU ANN (batched) for the todo queries -> candidate item rows
    print("CPU ANN (top candidates)...", flush=True)
    rows = np.array([term2row[t] for t in todo])
    TOPN = max(CAND_RANKS) + 3
    cand_items = {}
    B = 256
    for b in range(0, len(rows), B):
        qs = q_emb[rows[b:b+B]]                      # |b| x D
        sims = qs @ i_emb.T                          # |b| x n_items
        part = np.argpartition(-sims, TOPN, axis=1)[:, :TOPN]
        for j, t in enumerate(todo[b:b+B]):
            r = part[j][np.argsort(-sims[j, part[j]])]
            cand_items[t] = i_ids[r]
        if b % (B*8) == 0:
            print(f"  ANN {b}/{len(rows)}", flush=True)

    fout = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fout)
    if not done_terms:
        w.writerow(["term_id", "item_id", "label"])

    # build (query, product) items to judge; batch to the LLM
    queue = []   # (term_id, item_id, query, prodtext)
    tin = tout = 0; t0 = time.time(); n_lab = 0

    def flush_batch(batch):
        nonlocal tin, tout, n_lab
        if not batch:
            return
        lines = "\n".join(f"{i+1}. {q} | {p}" for i, (_, _, q, p) in enumerate(batch))
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=len(batch) * 6,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": "Çiftler:\n" + lines}])
        except Exception as e:
            print(f"  API error: {e}; retry 10s", flush=True); time.sleep(10); flush_batch(batch); return
        tin += r.usage.prompt_tokens; tout += r.usage.completion_tokens
        labs = {}
        for l in r.choices[0].message.content.splitlines():
            m = re.match(r"\s*(\d+)\.\s*([01])", l.strip())
            if m:
                labs[int(m.group(1))] = int(m.group(2))
        for i, (tid, iid, _, _) in enumerate(batch):
            w.writerow([tid, iid, labs.get(i + 1, 0)])
        fout.flush(); n_lab += len(batch)

    for t in todo:
        q = term2q.get(t, ""); pset = pos_by_term.get(t, set())
        picks = [c for k, c in enumerate(cand_items[t]) if k in CAND_RANKS and c not in pset]
        for iid in picks:
            queue.append((t, iid, q, itxt.get(iid, "")))
            if len(queue) >= LLM_BATCH:
                flush_batch(queue); queue = []
        if n_lab and n_lab % 3000 < LLM_BATCH:
            cost = tin*IN_COST + tout*OUT_COST
            print(f"  labeled~{n_lab}  cost=${cost:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    flush_batch(queue)
    cost = tin*IN_COST + tout*OUT_COST
    print(f"DONE. labeled={n_lab}  tokens in={tin} out={tout}  cost=${cost:.3f} -> {OUT}", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
