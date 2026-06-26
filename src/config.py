import os

# Base paths
BASE_DIR = r"c:\Users\yiğit\Desktop\trendyol-e-ticaret-yarismasi-2026-kaggle"
DATA_DIR = BASE_DIR
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Input files
TRAIN_PAIRS_PATH = os.path.join(DATA_DIR, "training_pairs.csv")
SUBMISSION_PAIRS_PATH = os.path.join(DATA_DIR, "submission_pairs.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TERMS_PATH = os.path.join(DATA_DIR, "terms.csv")
ITEMS_PATH = os.path.join(DATA_DIR, "items.csv")

# Output/Intermediate files
PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, "train_sampled.parquet")
PROCESSED_ITEMS_PATH = os.path.join(ARTIFACTS_DIR, "items_processed.parquet")
PROCESSED_TERMS_PATH = os.path.join(ARTIFACTS_DIR, "terms_processed.parquet")

# Configuration
SEED = 42
N_FOLDS = 5

# Negative Sampling Ratios
# We want 1:3 positive-to-negative ratio.
# Ratio format is: (random_ratio, cat_aware_ratio, lexical_ratio) relative to positive count.
# E.g., (1, 1, 1) means for 250k positives, we sample 250k random, 250k category-aware, and 250k lexical hard negatives.
NEG_SAMPLING_RATIOS = (1, 1, 1)

# TF-IDF Configuration for Lexical Negatives
TFIDF_MAX_FEATURES = 30000
TFIDF_BATCH_SIZE = 250
LEXICAL_CANDIDATES_PER_QUERY = 30  # Retrieve top 30 to allow filtering out positives and sampling

# Thermal Mitigation Configuration
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_SLEEP = 0.2          # sleep delay in seconds between batches
MAX_THREADS = 1                # limit CPU cores for training/inference to prevent OpenMP thrashing

