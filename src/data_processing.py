import os
import sys
import time
import random
import numpy as np
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import configuration
from src.config import (
    TRAIN_PAIRS_PATH,
    ITEMS_PATH,
    TERMS_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_ITEMS_PATH,
    PROCESSED_TERMS_PATH,
    SEED,
    NEG_SAMPLING_RATIOS,
    TFIDF_MAX_FEATURES,
    TFIDF_BATCH_SIZE,
    LEXICAL_CANDIDATES_PER_QUERY
)
from src.text_normalization import normalize_text

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

def load_data():
    """Phase 1: High-Performance Data Processing with Polars.
    Loads training pairs, terms, and items datasets. Pre-normalizes text columns
    if processed versions do not exist on disk.
    """
    print("[Phase 1] Loading datasets with Polars...")
    t0 = time.time()
    
    # Load using Polars scan and collect
    pairs_df = pl.scan_csv(TRAIN_PAIRS_PATH).collect()
    
    # Check if pre-normalized datasets exist
    if os.path.exists(PROCESSED_TERMS_PATH) and os.path.exists(PROCESSED_ITEMS_PATH):
        print("  - Loading pre-normalized terms and items from cache...")
        terms_df = pl.read_parquet(PROCESSED_TERMS_PATH)
        items_df = pl.read_parquet(PROCESSED_ITEMS_PATH)
    else:
        import multiprocessing
        print("  - Loading raw terms and items...")
        terms_df = pl.scan_csv(TERMS_PATH).collect()
        items_df = pl.scan_csv(ITEMS_PATH).collect()
        
        # Limit worker count to 6 to prevent thermal shutdown on 32-core machine
        num_workers = min(6, os.cpu_count() or 1)
        print(f"  - Pre-normalizing queries in parallel (using {num_workers} processes)...")
        queries = terms_df["query"].fill_null("").to_list()
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            norm_queries = list(tqdm(pool.imap(normalize_text, queries, chunksize=500), total=len(queries), desc="Normalize Queries"))
            
        terms_df = terms_df.with_columns(pl.Series(name="normalized_query", values=norm_queries))
        
        print(f"  - Pre-normalizing item titles in parallel (using {num_workers} processes)...")
        titles = items_df["title"].fill_null("").to_list()
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            norm_titles = list(tqdm(pool.imap(normalize_text, titles, chunksize=1000), total=len(titles), desc="Normalize Titles"))
            
        items_df = items_df.with_columns(pl.Series(name="normalized_title", values=norm_titles))
        
        # Save to disk for future runs
        print("  - Saving pre-normalized datasets to disk...")
        os.makedirs(os.path.dirname(PROCESSED_TERMS_PATH), exist_ok=True)
        terms_df.write_parquet(PROCESSED_TERMS_PATH)
        items_df.write_parquet(PROCESSED_ITEMS_PATH)
        
    print(f"[Phase 1] Loaded datasets in {time.time() - t0:.2f} seconds.")
    print(f"  - Training Pairs: {pairs_df.height} rows")
    print(f"  - Terms: {terms_df.height} rows")
    print(f"  - Items: {items_df.height} rows")
    
    return pairs_df, terms_df, items_df

def build_lexical_candidates(terms_df, items_df, training_terms):
    """Generates lexical hard candidates for each training term using TF-IDF."""
    print("[Phase 2] Generating lexical hard candidates using TF-IDF...")
    t0 = time.time()
    
    # 1. Fit TF-IDF on all item titles
    print("  - Fitting TF-IDF Vectorizer on item titles...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer='word', token_pattern=r'\w+')
    
    # Make sure null titles are replaced by empty string
    item_titles = items_df["normalized_title"].to_list()
    X = vectorizer.fit_transform(item_titles)
    
    # 2. Get unique training queries
    # Join terms_df with training_terms to get query strings
    tr_queries_df = terms_df.filter(pl.col("term_id").is_in(training_terms))
    term_ids = tr_queries_df["term_id"].to_list()
    query_texts = tr_queries_df["normalized_query"].to_list()
    
    print(f"  - Vectorizing {len(query_texts)} training queries...")
    Q = vectorizer.transform(query_texts)
    
    # 3. Batch dot products and argpartition
    n_queries = len(term_ids)
    query_lexical_candidates = {}
    
    item_ids = items_df["item_id"].to_numpy()
    
    print("  - Querying TF-IDF matrix in batches...")
    for i in tqdm(range(0, n_queries, TFIDF_BATCH_SIZE), desc="TF-IDF Retrieval"):
        end_idx = min(i + TFIDF_BATCH_SIZE, n_queries)
        Q_batch = Q[i:end_idx]
        batch_term_ids = term_ids[i:end_idx]
        
        # Sparse matrix multiplication
        sim = Q_batch.dot(X.T)
        
        # Convert to dense array for argpartition
        dense_sim = sim.toarray()
        
        # Get top candidates
        top_k = min(LEXICAL_CANDIDATES_PER_QUERY, dense_sim.shape[1])
        partitioned = np.argpartition(dense_sim, -top_k, axis=1)[:, -top_k:]
        
        for idx_in_batch, term_id in enumerate(batch_term_ids):
            # Sort the partition to get true top elements
            row = dense_sim[idx_in_batch]
            row_partitioned = partitioned[idx_in_batch]
            sorted_indices = row_partitioned[np.argsort(-row[row_partitioned])]
            
            # Map index to item_id
            candidates = [item_ids[idx] for idx in sorted_indices]
            query_lexical_candidates[term_id] = candidates
            
    print(f"[Phase 2] Generated lexical candidates in {time.time() - t0:.2f} seconds.")
    return query_lexical_candidates

def sample_negatives(pairs_df, items_df, terms_df):
    """Phase 2: Advanced Heuristic Negative Sampling."""
    print("[Phase 2] Starting advanced heuristic negative sampling...")
    t0 = time.time()
    
    # Parse categories to get root and leaf categories
    # Root category is the first part, leaf category is the full path
    print("  - Parsing category hierarchy...")
    items_df = items_df.with_columns([
        pl.col("category").str.split("/").list.get(0).fill_null("unknown").alias("root_category"),
        pl.col("category").fill_null("unknown").alias("leaf_category")
    ])
    
    # Build fast lookups
    all_item_ids = items_df["item_id"].to_list()
    item_id_to_idx = {item_id: i for i, item_id in enumerate(all_item_ids)}
    
    # Store root and leaf categories for all items
    item_roots = items_df["root_category"].to_list()
    item_leaves = items_df["leaf_category"].to_list()
    
    # Group item IDs by root category
    root_to_items = {}
    for item_id, root in zip(all_item_ids, item_roots):
        if root not in root_to_items:
            root_to_items[root] = []
        root_to_items[root].append(item_id)
    
    # Query positives lookup
    # query_positives: term_id -> set(item_ids)
    print("  - Indexing positive query-product relationships...")
    query_positives = {}
    for row in pairs_df.iter_rows(named=True):
        term_id = row["term_id"]
        item_id = row["item_id"]
        if term_id not in query_positives:
            query_positives[term_id] = set()
        query_positives[term_id].add(item_id)
        
    # query_positive_leaves: term_id -> set(leaf_categories)
    # query_positive_roots: term_id -> list(root_categories)
    query_positive_leaves = {}
    query_positive_roots = {}
    for term_id, pos_items in query_positives.items():
        leaves = set()
        roots = set()
        for item_id in pos_items:
            idx = item_id_to_idx[item_id]
            leaves.add(item_leaves[idx])
            roots.add(item_roots[idx])
        query_positive_leaves[term_id] = leaves
        query_positive_roots[term_id] = list(roots)
        
    # query_used_items: term_id -> set(item_ids)
    print("  - Initializing tracking of used items per query...")
    query_used_items = {t_id: set(p_items) for t_id, p_items in query_positives.items()}
        
    # Get unique training term_ids
    training_terms = list(query_positives.keys())
    
    # Generate lexical candidates using TF-IDF
    query_lexical_candidates = build_lexical_candidates(terms_df, items_df, training_terms)
    
    # Configure negative sampling counts per positive sample
    random_ratio, cat_ratio, lexical_ratio = NEG_SAMPLING_RATIOS
    
    negative_pairs = []
    
    # Metrics tracking
    fallback_cat_to_rand = 0
    fallback_lex_to_cat = 0
    fallback_lex_to_rand = 0
    
    print("  - Running sampling loops...")
    # Iterate over positive pairs to sample negatives
    for row in tqdm(pairs_df.iter_rows(named=True), total=pairs_df.height, desc="Negative Sampling"):
        term_id = row["term_id"]
        pos_item_id = row["item_id"]
        
        used_set = query_used_items[term_id]
        pos_leaves = query_positive_leaves[term_id]
        pos_roots = query_positive_roots[term_id]
        
        pos_idx = item_id_to_idx[pos_item_id]
        pos_root = item_roots[pos_idx]
        
        # 1. Random Negatives (Coarse Decision Boundary)
        for _ in range(random_ratio):
            sampled_item = None
            for retry in range(10):
                candidate = random.choice(all_item_ids)
                if candidate not in used_set:
                    sampled_item = candidate
                    used_set.add(candidate)
                    break
            if sampled_item is None:
                # Absolute fallback
                for candidate in all_item_ids:
                    if candidate not in used_set:
                        sampled_item = candidate
                        used_set.add(candidate)
                        break
            negative_pairs.append((term_id, sampled_item, "random"))
            
        # 2. Category-Aware Negatives (Medium Decision Boundary)
        for _ in range(cat_ratio):
            sampled_item = None
            # Target root category: preference is the root category of the current positive item
            target_root = pos_root if pos_root != "unknown" and pos_root in root_to_items else None
            if target_root is None and len(pos_roots) > 0:
                target_root = random.choice(pos_roots)
                
            if target_root in root_to_items:
                candidates_list = root_to_items[target_root]
                for retry in range(15):
                    candidate = random.choice(candidates_list)
                    cand_idx = item_id_to_idx[candidate]
                    cand_leaf = item_leaves[cand_idx]
                    # Must be different leaf category and not used
                    if candidate not in used_set and cand_leaf not in pos_leaves:
                        sampled_item = candidate
                        used_set.add(candidate)
                        break
            
            # Fallback if category-aware negative sampling fails
            if sampled_item is None:
                fallback_cat_to_rand += 1
                for retry in range(10):
                    candidate = random.choice(all_item_ids)
                    if candidate not in used_set:
                        sampled_item = candidate
                        used_set.add(candidate)
                        break
                if sampled_item is None:
                    for candidate in all_item_ids:
                        if candidate not in used_set:
                            sampled_item = candidate
                            used_set.add(candidate)
                            break
            negative_pairs.append((term_id, sampled_item, "category_aware"))
            
        # 3. Lexical Hard Negatives (Fine Decision Boundary)
        for _ in range(lexical_ratio):
            sampled_item = None
            lex_candidates = query_lexical_candidates.get(term_id, [])
            
            # Find candidate that is not used
            for candidate in lex_candidates:
                if candidate not in used_set:
                    sampled_item = candidate
                    used_set.add(candidate)
                    break
            
            # Fallback 1: Category-aware negative
            if sampled_item is None:
                fallback_lex_to_cat += 1
                target_root = pos_root if pos_root != "unknown" and pos_root in root_to_items else None
                if target_root is None and len(pos_roots) > 0:
                    target_root = random.choice(pos_roots)
                if target_root in root_to_items:
                    candidates_list = root_to_items[target_root]
                    for retry in range(15):
                        candidate = random.choice(candidates_list)
                        cand_idx = item_id_to_idx[candidate]
                        cand_leaf = item_leaves[cand_idx]
                        if candidate not in used_set and cand_leaf not in pos_leaves:
                            sampled_item = candidate
                            used_set.add(candidate)
                            break
                            
            # Fallback 2: Random negative
            if sampled_item is None:
                fallback_lex_to_rand += 1
                for retry in range(10):
                    candidate = random.choice(all_item_ids)
                    if candidate not in used_set:
                        sampled_item = candidate
                        used_set.add(candidate)
                        break
                if sampled_item is None:
                    for candidate in all_item_ids:
                        if candidate not in used_set:
                            sampled_item = candidate
                            used_set.add(candidate)
                            break
            negative_pairs.append((term_id, sampled_item, "lexical_hard"))
            
    print(f"Negative sampling complete. Fallbacks:")
    print(f"  - Category-aware to Random: {fallback_cat_to_rand}")
    print(f"  - Lexical Hard to Category-aware: {fallback_lex_to_cat}")
    print(f"  - Lexical Hard to Random: {fallback_lex_to_rand}")
    
    # 4. Construct Final Sampled Dataset
    print("  - Building final dataset...")
    # Convert positives to list of tuples: (term_id, item_id, label, type)
    pos_tuples = [(row["term_id"], row["item_id"], 1, "positive") for row in pairs_df.iter_rows(named=True)]
    # Convert negatives to: (term_id, item_id, label, type)
    neg_tuples = [(t_id, i_id, 0, n_type) for t_id, i_id, n_type in negative_pairs]
    
    # Combine
    all_tuples = pos_tuples + neg_tuples
    
    # Shuffle dataset
    random.shuffle(all_tuples)
    
    # Convert to Polars DataFrame
    final_df = pl.DataFrame({
        "term_id": [t[0] for t in all_tuples],
        "item_id": [t[1] for t in all_tuples],
        "label": [t[2] for t in all_tuples],
        "sample_type": [t[3] for t in all_tuples]
    })
    
    # Generate unique IDs for all rows
    # In train.csv, the positive pairs have an 'id' column (e.g. TRN_c639ed31a5).
    # Since we shuffled and added negatives, we will generate new IDs like 'TRN_SAMPLED_xxxx' to avoid duplicate ids.
    n_rows = final_df.height
    ids = [f"TRN_SAMPLED_{i:07d}" for i in range(n_rows)]
    final_df = final_df.with_columns(pl.Series(name="id", values=ids))
    
    # Move 'id' to the first column
    final_df = final_df.select(["id", "term_id", "item_id", "label", "sample_type"])
    
    print(f"[Phase 2] Negative sampling finished in {time.time() - t0:.2f} seconds.")
    print(f"Total dataset size: {final_df.height} rows (1 positive : 3 negatives)")
    
    return final_df

def verify_dataset(df):
    """Verifies the integrity of the generated dataset."""
    print("[Verification] Running dataset validation checks...")
    
    # Check shape
    print(f"  - Rows: {df.height}, Columns: {df.width}")
    assert df.height == 1000000, f"Expected 1,000,000 rows, but got {df.height}"
    
    # Check class distribution
    counts = df["label"].value_counts()
    print("  - Label Distribution:")
    print(counts)
    
    pos_count = df.filter(pl.col("label") == 1).height
    neg_count = df.filter(pl.col("label") == 0).height
    assert pos_count == 250000, f"Expected 250,000 positives, got {pos_count}"
    assert neg_count == 750000, f"Expected 750,000 negatives, got {neg_count}"
    
    # Check sample types
    type_counts = df["sample_type"].value_counts()
    print("  - Sample Type Distribution:")
    print(type_counts)
    
    # Check for duplicate pairs
    dup_count = df.select(["term_id", "item_id"]).n_unique()
    print(f"  - Unique (term_id, item_id) pairs: {dup_count} / {df.height}")
    assert dup_count == df.height, "Warning: there are duplicate query-item pairs in the sampled dataset!"
    
    # Check for nulls
    nulls = df.null_count()
    print("  - Null counts:")
    print(nulls)
    for col in df.columns:
        assert df[col].null_count() == 0, f"Column {col} has nulls!"
        
    print("[Verification] All dataset integrity checks PASSED successfully!")

def main():
    pairs_df, terms_df, items_df = load_data()
    sampled_df = sample_negatives(pairs_df, items_df, terms_df)
    verify_dataset(sampled_df)
    
    # Save as parquet
    print(f"Saving sampled dataset to {PROCESSED_TRAIN_PATH}...")
    t0 = time.time()
    # Create artifacts directory if it doesn't exist
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    sampled_df.write_parquet(PROCESSED_TRAIN_PATH)
    print(f"Saved successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
