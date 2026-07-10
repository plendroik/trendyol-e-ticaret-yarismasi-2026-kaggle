"""
Fine-tune the TY-ecomm bi-encoder on the judge labels (Baran's embedding recipe):
label-aware cosine fine-tune -> fresh query/item embeddings -> new stacker features
(cosine + within-term rank from a SECOND architecture family).

Train: 400k sampled judge-labeled pairs (arbiter-corrected), CosineSimilarityLoss.
Encode: all 50k queries + 963k item docs. Outputs to emb2/.

Usage: python train_biencoder.py
"""
import os, time, random
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from judge_local import prod_text

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
EMB2 = os.path.join(DATA, "emb2")
os.makedirs(EMB2, exist_ok=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log("Data...")
sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))
term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
        for r in items.itertuples(index=False)}

lab = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
id2lab = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))
# tahkim duzeltmeleri (en iyi hakem son soz: 4o > ajan > 41mini)
for f in ["disputed_gpt_41_mini.csv", "agent_labels.csv", "disputed_gpt_4o.csv"]:
    p = os.path.join(DATA, f)
    if os.path.exists(p):
        d = pd.read_csv(p)
        for i_, l_ in zip(d["id"].astype(str), d["label"].astype(int)):
            id2lab[i_] = l_

ids = sub["id"].astype(str).to_numpy()
tids = sub["term_id"].to_numpy(); iids = sub["item_id"].to_numpy()
rows = [(term2q.get(tids[k], ""), itxt.get(iids[k], ""), float(id2lab[ids[k]]))
        for k in range(len(sub)) if ids[k] in id2lab]
random.Random(42).shuffle(rows)
rows = rows[:400000]
log(f"train pairs: {len(rows)}  pos={np.mean([r[2] for r in rows]):.3f}")

model = SentenceTransformer("Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0",
                            trust_remote_code=True)
model.max_seq_length = 96
ex = [InputExample(texts=[q, p], label=l) for q, p, l in rows]
dl = DataLoader(ex, shuffle=True, batch_size=64)
loss = losses.CosineSimilarityLoss(model)
log("Fit 1 epoch...")
model.fit(train_objectives=[(dl, loss)], epochs=1, warmup_steps=400, show_progress_bar=False)
model.save(os.path.join(ART, "biencoder_ft"))
log("saved biencoder_ft")

log("Encode queries...")
q_ids = terms["term_id"].to_numpy()
q_emb = model.encode(terms["query"].fillna("").astype(str).tolist(),
                     batch_size=512, normalize_embeddings=True, show_progress_bar=False)
np.save(os.path.join(EMB2, "query_emb.npy"), q_emb.astype(np.float32))
np.save(os.path.join(EMB2, "query_ids.npy"), q_ids)
log("Encode items...")
i_ids = items["item_id"].to_numpy()
docs = [itxt[i] for i in i_ids]
i_emb = model.encode(docs, batch_size=512, normalize_embeddings=True, show_progress_bar=False)
np.save(os.path.join(EMB2, "item_emb.npy"), i_emb.astype(np.float32))
np.save(os.path.join(EMB2, "item_ids.npy"), i_ids)
log(f"DONE BIENCODER. emb2/ hazir (q={len(q_emb)}, i={len(i_emb)})")
