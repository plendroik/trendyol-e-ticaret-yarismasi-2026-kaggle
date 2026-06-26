import os
import sys

# Configure environment variables to bypass TensorFlow/Keras 3 issues in Hugging Face
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

import time
import pickle
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import configurations
from src.config import (
    ARTIFACTS_DIR,
    PROCESSED_TERMS_PATH,
    PROCESSED_ITEMS_PATH,
    SEED,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_SLEEP
)

def generate_embeddings():
    print("[Embeddings] Initializing embedding generation...")
    t_start = time.time()
    
    # Create artifacts directory
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Using device: {device}")
    
    # Define file paths
    query_emb_path = os.path.join(ARTIFACTS_DIR, "query_embeddings.npy")
    query_map_path = os.path.join(ARTIFACTS_DIR, "query_id_to_idx.pkl")
    item_emb_path = os.path.join(ARTIFACTS_DIR, "item_embeddings.npy")
    item_map_path = os.path.join(ARTIFACTS_DIR, "item_id_to_idx.pkl")
    
    # Load model if anything needs to be generated
    model = None
    
    # 1. Process terms/queries
    if os.path.exists(query_emb_path) and os.path.exists(query_map_path):
        print("  - Query embeddings and mapping already exist. Skipping query encoding.")
    else:
        print("  - Loading terms from processed parquet...")
        terms_df = pl.read_parquet(PROCESSED_TERMS_PATH)
        term_ids = terms_df["term_id"].to_list()
        queries = ["query: " + (q if q is not None else "") for q in terms_df["query"].to_list()]
        
        print("  - Loading intfloat/multilingual-e5-base model...")
        model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)
        
        print(f"  - Encoding {len(queries)} queries...")
        t0 = time.time()
        query_embeddings = model.encode(
            queries,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        query_embeddings = query_embeddings.astype(np.float16)
        print(f"  - Encoded queries in {time.time() - t0:.2f} seconds.")
        
        np.save(query_emb_path, query_embeddings)
        term_id_to_idx = {term_id: i for i, term_id in enumerate(term_ids)}
        with open(query_map_path, "wb") as f:
            pickle.dump(term_id_to_idx, f)
        print(f"  - Saved query embeddings to {query_emb_path}")
        print(f"  - Saved query mapping to {query_map_path}")
        
    # 2. Process items/products
    if os.path.exists(item_emb_path) and os.path.exists(item_map_path):
        print("  - Item embeddings and mapping already exist. Skipping item encoding.")
    else:
        print("  - Loading items from processed parquet...")
        items_df = pl.read_parquet(PROCESSED_ITEMS_PATH).select(["item_id", "title"])
        item_ids = items_df["item_id"].to_list()
        titles = ["passage: " + (t if t is not None else "") for t in items_df["title"].to_list()]
        
        if model is None:
            print("  - Loading intfloat/multilingual-e5-base model...")
            model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)
            
        print(f"  - Encoding {len(titles)} product titles with thermal-aware checkpointing...")
        t0 = time.time()
        
        # Checkpoint configuration
        chunk_size = 100000
        n_chunks = int(np.ceil(len(titles) / chunk_size))
        item_embeddings = []
        chunk_files = []
        
        for chunk_idx in range(n_chunks):
            chunk_file = os.path.join(ARTIFACTS_DIR, f"item_embeddings_chunk_{chunk_idx}.npy")
            chunk_files.append(chunk_file)
            c_start = chunk_idx * chunk_size
            c_end = min(c_start + chunk_size, len(titles))
            
            if os.path.exists(chunk_file):
                print(f"    * Chunk {chunk_idx + 1}/{n_chunks} found in cache. Loading...")
                chunk_emb = np.load(chunk_file)
                item_embeddings.append(chunk_emb)
            else:
                print(f"    * Encoding chunk {chunk_idx + 1}/{n_chunks} (indices {c_start} to {c_end})...")
                chunk_titles = titles[c_start:c_end]
                chunk_emb_list = []
                
                # Batch loop with thermal cooldown sleeps
                for b_start in tqdm(range(0, len(chunk_titles), EMBEDDING_BATCH_SIZE), desc=f"Chunk {chunk_idx+1}"):
                    b_end = min(b_start + EMBEDDING_BATCH_SIZE, len(chunk_titles))
                    batch_texts = chunk_titles[b_start:b_end]
                    
                    batch_emb = model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    chunk_emb_list.append(batch_emb.astype(np.float16))
                    
                    if EMBEDDING_SLEEP > 0:
                        time.sleep(EMBEDDING_SLEEP)
                        
                chunk_emb = np.concatenate(chunk_emb_list, axis=0)
                np.save(chunk_file, chunk_emb)
                item_embeddings.append(chunk_emb)
                
        print("  - Concatenating all chunks...")
        item_embeddings = np.concatenate(item_embeddings, axis=0)
        print(f"  - Encoded product titles in {time.time() - t0:.2f} seconds.")
        
        np.save(item_emb_path, item_embeddings)
        item_id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
        with open(item_map_path, "wb") as f:
            pickle.dump(item_id_to_idx, f)
            
        print(f"  - Saved item embeddings to {item_emb_path}")
        print(f"  - Saved item mapping to {item_map_path}")
        
        # Clean up chunk files
        print("  - Cleaning up chunk files...")
        for cf in chunk_files:
            try:
                os.remove(cf)
            except OSError:
                pass
                
    print(f"[Embeddings] Finished all embedding generations in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    generate_embeddings()
