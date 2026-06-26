"""
Generate sentence embeddings for queries and item titles on GPU.

Battery-safe: item encoding is chunked and each chunk is checkpointed to disk,
so an interrupted run resumes where it left off. Embeddings are L2-normalized
and saved as float16, so downstream cosine == dot product.

Output (in EMB_DIR):
  query_emb.npy   (n_terms x D, float16, L2-normalized)
  query_ids.npy   (n_terms,)  term_id order
  item_emb.npy    (n_items x D, float16, L2-normalized)
  item_ids.npy    (n_items,)  item_id order
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
EMB_DIR = os.path.join(DATA_DIR, "emb")
os.makedirs(EMB_DIR, exist_ok=True)

# Domain-aligned Trendyol e-commerce embedding model (tarama.md recommendation),
# with a safe multilingual fallback if it cannot be loaded.
PRIMARY_MODEL = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
FALLBACK_MODEL = "intfloat/multilingual-e5-base"

BATCH = 256
ITEM_CHUNK = 200_000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            log(f"Loading model {name} on {device}...")
            m = SentenceTransformer(name, device=device, trust_remote_code=True)
            log(f"  loaded {name}  dim={m.get_sentence_embedding_dimension()} "
                f"max_seq={m.max_seq_length}")
            with open(os.path.join(EMB_DIR, "model_used.txt"), "w", encoding="utf-8") as f:
                f.write(name)
            # e5 family needs query:/passage: prefixes; Trendyol/gte does not.
            return m, ("e5" in name.lower())
        except Exception as e:
            log(f"  FAILED to load {name}: {e}")
    raise RuntimeError("No embedding model could be loaded.")


def encode_queries(model, needs_prefix):
    out_emb = os.path.join(EMB_DIR, "query_emb.npy")
    out_ids = os.path.join(EMB_DIR, "query_ids.npy")
    if os.path.exists(out_emb) and os.path.exists(out_ids):
        log("Query embeddings already exist. Skipping.")
        return
    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    ids = terms["term_id"].to_numpy()
    texts = terms["query"].fillna("").astype(str).tolist()
    if needs_prefix:
        texts = ["query: " + t for t in texts]
    log(f"Encoding {len(texts)} queries...")
    emb = model.encode(texts, batch_size=BATCH, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(out_emb, emb.astype(np.float16))
    np.save(out_ids, ids)
    log(f"  saved {out_emb}")


def encode_items(model, needs_prefix):
    out_emb = os.path.join(EMB_DIR, "item_emb.npy")
    out_ids = os.path.join(EMB_DIR, "item_ids.npy")
    if os.path.exists(out_emb) and os.path.exists(out_ids):
        log("Item embeddings already exist. Skipping.")
        return
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"), usecols=["item_id", "title"])
    ids = items["item_id"].to_numpy()
    texts = items["title"].fillna("").astype(str).tolist()
    if needs_prefix:
        texts = ["passage: " + t for t in texts]
    n = len(texts)
    n_chunks = (n + ITEM_CHUNK - 1) // ITEM_CHUNK
    log(f"Encoding {n} item titles in {n_chunks} chunks (checkpointed)...")
    parts = []
    for ci in range(n_chunks):
        cf = os.path.join(EMB_DIR, f"_item_chunk_{ci}.npy")
        if os.path.exists(cf):
            log(f"  chunk {ci+1}/{n_chunks} cached")
            parts.append(np.load(cf))
            continue
        s, e = ci * ITEM_CHUNK, min((ci + 1) * ITEM_CHUNK, n)
        t0 = time.time()
        emb = model.encode(texts[s:e], batch_size=BATCH, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
        emb = emb.astype(np.float16)
        np.save(cf, emb)
        parts.append(emb)
        log(f"  chunk {ci+1}/{n_chunks} done ({e-s} items, {time.time()-t0:.1f}s)")
    log("Concatenating chunks...")
    full = np.concatenate(parts, axis=0)
    np.save(out_emb, full)
    np.save(out_ids, ids)
    for ci in range(n_chunks):
        try:
            os.remove(os.path.join(EMB_DIR, f"_item_chunk_{ci}.npy"))
        except OSError:
            pass
    log(f"  saved {out_emb}  shape={full.shape}")


def main():
    t0 = time.time()
    model, needs_prefix = load_model()
    encode_queries(model, needs_prefix)
    encode_items(model, needs_prefix)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
