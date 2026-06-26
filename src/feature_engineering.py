import os
import sys
import time
import pickle
import re
import numpy as np
import polars as pl
from tqdm import tqdm

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import modules and configurations
from src.config import (
    TRAIN_PAIRS_PATH,
    SUBMISSION_PAIRS_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TERMS_PATH,
    PROCESSED_ITEMS_PATH,
    ARTIFACTS_DIR
)
from src.text_normalization import clean_punctuation, turkish_lower

# Define paths for precomputed embeddings
QUERY_EMB_PATH = os.path.join(ARTIFACTS_DIR, "query_embeddings.npy")
QUERY_MAP_PATH = os.path.join(ARTIFACTS_DIR, "query_id_to_idx.pkl")
ITEM_EMB_PATH = os.path.join(ARTIFACTS_DIR, "item_embeddings.npy")
ITEM_MAP_PATH = os.path.join(ARTIFACTS_DIR, "item_id_to_idx.pkl")

# Turkish colors list for color-matching feature
TURKISH_COLORS = {
    "siyah", "beyaz", "gri", "kırmızı", "mavi", "yeşil", "sarı", "pembe", "turuncu",
    "lacivert", "mor", "kahverengi", "bej", "ekru", "haki", "bordo", "altın", "gümüş"
}

def get_char_3grams(text):
    """Generates character 3-grams for a string."""
    if len(text) < 3:
        return {text}
    return {text[i:i+3] for i in range(len(text) - 2)}

def char_3gram_jaccard(s1, s2):
    """Computes character 3-gram Jaccard similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    g1 = get_char_3grams(s1)
    g2 = get_char_3grams(s2)
    intersection = len(g1.intersection(g2))
    union = len(g1.union(g2))
    return intersection / union if union > 0 else 0.0

def word_jaccard(w1_list, w2_list):
    """Computes word-level Jaccard similarity between two lists of words."""
    if not w1_list or not w2_list:
        return 0.0
    s1 = set(w1_list)
    s2 = set(w2_list)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0

def compute_token_coverage(q_words, t_words):
    """Computes the fraction of query tokens present in the title."""
    if not q_words:
        return 0.0
    q_set = set(q_words)
    t_set = set(t_words)
    overlap = len(q_set.intersection(t_set))
    return overlap / len(q_set)

def extract_features(pairs_df, terms_df, items_df, query_embeddings=None, term_id_to_idx=None, item_embeddings=None, item_id_to_idx=None):
    """Multi-Modal Feature Engineering."""
    t_start = time.time()
    print(f"[Features] Processing {pairs_df.height} pairs...")
    
    # 1. Joins using Polars
    print("  - Joining pairs with terms and items metadata...")
    joined_df = pairs_df.join(terms_df, on="term_id", how="left").join(items_df, on="item_id", how="left")
    
    # Extract fields as list for fast iterative processing
    queries = joined_df["query"].fill_null("").to_list()
    categories = joined_df["category"].fill_null("unknown").to_list()
    brands = joined_df["brand"].fill_null("unknown").to_list()
    genders = joined_df["gender"].fill_null("unknown").to_list()
    age_groups = joined_df["age_group"].fill_null("unknown").to_list()
    attributes_list = joined_df["attributes"].fill_null("").to_list()
    
    # Fetch pre-normalized fields directly from processed metadata
    norm_queries_list = joined_df["normalized_query"].fill_null("").to_list()
    norm_titles_list = joined_df["normalized_title"].fill_null("").to_list()
    
    # Feature lists
    f_char_len_query = []
    f_char_len_title = []
    f_word_len_query = []
    f_word_len_title = []
    f_word_count_ratio = []
    f_char_count_ratio = []
    f_jaccard_word = []
    f_jaccard_char_3gram = []
    f_token_coverage = []
    
    f_brand_in_query = []
    f_gender_contradiction = []
    f_age_group_contradiction = []
    f_color_match = []
    f_color_mismatch = []
    
    f_category_max_overlap_depth = []
    f_category_overlap_count = []
    
    print("  - Computing text, category, and attribute features...")
    # Loop over all pairs to extract features
    for idx in tqdm(range(len(queries)), desc="Feature Loop"):
        cat_path = categories[idx]
        brand = brands[idx].lower().strip()
        gender = genders[idx].lower().strip()
        age_group = age_groups[idx].lower().strip()
        attrs = attributes_list[idx].lower()
        
        # Use precomputed normalized texts
        q_norm = norm_queries_list[idx]
        t_norm = norm_titles_list[idx]
        
        # Word token lists
        q_words = q_norm.split()
        t_words = t_norm.split()
        
        # Length features
        char_len_q = len(q_norm)
        char_len_t = len(t_norm)
        word_len_q = len(q_words)
        word_len_t = len(t_words)
        
        f_char_len_query.append(char_len_q)
        f_char_len_title.append(char_len_t)
        f_word_len_query.append(word_len_q)
        f_word_len_title.append(word_len_t)
        f_word_count_ratio.append(word_len_q / max(1, word_len_t))
        f_char_count_ratio.append(char_len_q / max(1, char_len_t))
        
        # Overlaps
        f_jaccard_word.append(word_jaccard(q_words, t_words))
        f_jaccard_char_3gram.append(char_3gram_jaccard(q_norm, t_norm))
        f_token_coverage.append(compute_token_coverage(q_words, t_words))
        
        # Brand match
        # Brand in query check (e.g. brand is "nike", check if "nike" is in query)
        f_brand_in_query.append(1.0 if brand != "unknown" and brand in q_norm else 0.0)
        
        # Gender contradiction check
        # Query implies: kadın/kız (female), erkek/bay (male)
        q_female = "kadın" in q_norm or "kız" in q_norm or "bayan" in q_norm
        q_male = "erkek" in q_norm or "bay" in q_norm
        
        gender_contra = 0.0
        if q_female and gender == "erkek":
            gender_contra = 1.0
        elif q_male and gender == "kadın":
            gender_contra = 1.0
        f_gender_contradiction.append(gender_contra)
        
        # Age group contradiction check
        # Query implies: bebek/çocuk (child/baby)
        q_child = "bebek" in q_norm or "çocuk" in q_norm or "baby" in q_norm or "kid" in q_norm
        
        age_contra = 0.0
        if q_child and age_group == "yetişkin":
            age_contra = 1.0
        f_age_group_contradiction.append(age_contra)
        
        # Color match features
        q_colors = {c for c in TURKISH_COLORS if c in q_norm}
        
        color_match = 0.0
        color_mismatch = 0.0
        
        # Try to find renk: in attributes
        product_color = ""
        if "renk:" in attrs:
            # Extract color name from string (e.g., renk: gri, materyal: ...)
            match = re.search(r'renk:\s*([^,]+)', attrs)
            if match:
                product_color = match.group(1).strip()
                
        if product_color:
            # If query has colors
            if q_colors:
                # Check if product color matches any query color
                if any(qc in product_color for qc in q_colors) or product_color in q_colors:
                    color_match = 1.0
                else:
                    color_mismatch = 1.0
            # Check if color is in title
            elif any(qc in t_norm for qc in q_colors):
                color_match = 1.0
        elif q_colors:
            # If color is in query, check if it's in the title
            if any(qc in t_norm for qc in q_colors):
                color_match = 1.0
                
        f_color_match.append(color_match)
        f_color_mismatch.append(color_mismatch)
        
        # Category overlaps
        cat_levels = [turkish_lower(clean_punctuation(lvl)) for lvl in cat_path.split("/") if lvl]
        
        max_overlap_depth = -1.0
        overlap_count = 0.0
        for lvl_idx, lvl in enumerate(cat_levels):
            lvl_words = set(lvl.split())
            if lvl_words.intersection(q_words):
                max_overlap_depth = float(lvl_idx)
                overlap_count += 1.0
                
        f_category_max_overlap_depth.append(max_overlap_depth)
        f_category_overlap_count.append(overlap_count)
        
    # Construct features DataFrame
    features_dict = {
        "id": joined_df["id"],
        "term_id": joined_df["term_id"],
        "item_id": joined_df["item_id"],
        "char_len_query": f_char_len_query,
        "char_len_title": f_char_len_title,
        "word_len_query": f_word_len_query,
        "word_len_title": f_word_len_title,
        "word_count_ratio": f_word_count_ratio,
        "char_count_ratio": f_char_count_ratio,
        "jaccard_word": f_jaccard_word,
        "jaccard_char_3gram": f_jaccard_char_3gram,
        "token_coverage": f_token_coverage,
        "brand_in_query": f_brand_in_query,
        "gender_contradiction": f_gender_contradiction,
        "age_group_contradiction": f_age_group_contradiction,
        "color_match": f_color_match,
        "color_mismatch": f_color_mismatch,
        "category_max_overlap_depth": f_category_max_overlap_depth,
        "category_overlap_count": f_category_overlap_count,
    }
    
    # 4. Semantic similarity features from precomputed embeddings
    if query_embeddings is not None and item_embeddings is not None:
        print("  - Computing precomputed semantic similarities...")
        t_sem = time.time()
        
        # Retrieve indices for queries and items
        term_ids_list = joined_df["term_id"].to_list()
        item_ids_list = joined_df["item_id"].to_list()
        
        q_idxs = [term_id_to_idx.get(t_id, -1) for t_id in term_ids_list]
        i_idxs = [item_id_to_idx.get(i_id, -1) for i_id in item_ids_list]
        
        # Identify missing or invalid indexes
        valid_mask = (np.array(q_idxs) != -1) & (np.array(i_idxs) != -1)
        
        # Vectorized cosine similarity in batches to prevent high RAM spikes (OOM/Thrashing)
        cos_sims = np.zeros(len(term_ids_list), dtype=np.float32)
        
        cos_batch_size = 200000
        n_pairs = len(term_ids_list)
        
        print(f"  - Processing cosine similarities in batches of {cos_batch_size}...")
        for b_start in range(0, n_pairs, cos_batch_size):
            b_end = min(b_start + cos_batch_size, n_pairs)
            
            batch_mask = valid_mask[b_start:b_end]
            if not np.any(batch_mask):
                continue
                
            batch_q_idxs = np.array(q_idxs[b_start:b_end])[batch_mask]
            batch_i_idxs = np.array(i_idxs[b_start:b_end])[batch_mask]
            
            # Fetch batch embeddings in float32
            Q_batch = query_embeddings[batch_q_idxs].astype(np.float32)
            I_batch = item_embeddings[batch_i_idxs].astype(np.float32)
            
            # Normalize to unit length
            norms_q = np.linalg.norm(Q_batch, axis=1, keepdims=True)
            norms_i = np.linalg.norm(I_batch, axis=1, keepdims=True)
            norms_q[norms_q == 0] = 1.0
            norms_i[norms_i == 0] = 1.0
            Q_batch = Q_batch / norms_q
            I_batch = I_batch / norms_i
            
            # Row-wise dot product (cosine similarity)
            batch_sims = np.sum(Q_batch * I_batch, axis=1)
            
            # Map batch mask back to global indices
            indices_in_full = np.arange(b_start, b_end)[batch_mask]
            cos_sims[indices_in_full] = batch_sims
            
        features_dict["semantic_similarity"] = cos_sims.tolist()
        print(f"  - Computed semantic similarities in {time.time() - t_sem:.2f} seconds.")
    else:
        print("  - Warning: Precomputed embeddings not supplied. setting semantic_similarity to 0.0")
        features_dict["semantic_similarity"] = [0.0] * len(queries)
        
    # Combine with label if it exists (for training set)
    if "label" in joined_df.columns:
        features_dict["label"] = joined_df["label"]
        
    features_df = pl.DataFrame(features_dict)
    print(f"[Features] Finished feature extraction in {time.time() - t_start:.2f} seconds.")
    
    return features_df

def main():
    # Helper to load embeddings
    query_embeddings = None
    term_id_to_idx = None
    item_embeddings = None
    item_id_to_idx = None
    
    if os.path.exists(QUERY_EMB_PATH) and os.path.exists(ITEM_EMB_PATH):
        print("[Features] Loading precomputed embeddings...")
        t0 = time.time()
        query_embeddings = np.load(QUERY_EMB_PATH)
        item_embeddings = np.load(ITEM_EMB_PATH)
        
        with open(QUERY_MAP_PATH, "rb") as f:
            term_id_to_idx = pickle.load(f)
        with open(ITEM_MAP_PATH, "rb") as f:
            item_id_to_idx = pickle.load(f)
        print(f"[Features] Loaded embeddings in {time.time() - t0:.2f} seconds.")
    else:
        print("[Features] Warning: Precomputed embeddings files not found. Generate them first.")
        
    # Load metadata
    print("[Features] Loading metadata files...")
    terms_df = pl.read_parquet(PROCESSED_TERMS_PATH)
    items_df = pl.read_parquet(PROCESSED_ITEMS_PATH)
    
    # Process Train Features
    if os.path.exists(PROCESSED_TRAIN_PATH):
        print(f"[Features] Processing training features from {PROCESSED_TRAIN_PATH}...")
        train_df = pl.read_parquet(PROCESSED_TRAIN_PATH)
        train_features = extract_features(
            train_df, terms_df, items_df,
            query_embeddings, term_id_to_idx,
            item_embeddings, item_id_to_idx
        )
        out_train_path = os.path.join(ARTIFACTS_DIR, "train_features.parquet")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        train_features.write_parquet(out_train_path)
        print(f"[Features] Saved train features to {out_train_path}")
        
    # Process Test Features
    if os.path.exists(SUBMISSION_PAIRS_PATH):
        print(f"[Features] Processing test features from {SUBMISSION_PAIRS_PATH}...")
        test_df = pl.read_csv(SUBMISSION_PAIRS_PATH)
        test_features = extract_features(
            test_df, terms_df, items_df,
            query_embeddings, term_id_to_idx,
            item_embeddings, item_id_to_idx
        )
        out_test_path = os.path.join(ARTIFACTS_DIR, "test_features.parquet")
        test_features.write_parquet(out_test_path)
        print(f"[Features] Saved test features to {out_test_path}")

if __name__ == "__main__":
    main()
