import os
import sys
import time
import pickle
import numpy as np
import polars as pl

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import configuration
from src.config import (
    ARTIFACTS_DIR,
    BASE_DIR,
    N_FOLDS
)

# Paths
SUBMISSION_PAIRS_PATH = os.path.join(BASE_DIR, "submission_pairs.csv")
TEST_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "test_features.parquet")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "best_threshold.txt")
SUBMISSION_OUT_PATH = os.path.join(BASE_DIR, "submission.csv")

def predict_pipeline():
    print("[Predict] Starting prediction pipeline...")
    t_start = time.time()
    
    # 1. Load optimal threshold
    if not os.path.exists(THRESHOLD_PATH):
        print(f"  - Error: Optimal threshold file not found at {THRESHOLD_PATH}. Please run training first.")
        sys.exit(1)
        
    with open(THRESHOLD_PATH, "r") as f:
        optimal_threshold = float(f.read().strip())
    print(f"  - Loaded optimal threshold: {optimal_threshold:.4f}")
    
    # 2. Load test features
    print(f"  - Loading test features from {TEST_FEATURES_PATH}...")
    df = pl.read_parquet(TEST_FEATURES_PATH)
    
    id_cols = ["id", "term_id", "item_id"]
    feature_cols = [c for c in df.columns if c not in id_cols and c != "label"]
    
    X_test = df.select(feature_cols).to_pandas()
    pair_ids = df["id"].to_list()
    
    print(f"  - Number of test samples to predict: {len(X_test)}")
    
    # 3. Load ensemble models and predict probabilities
    # We will average the probabilities across all 5 folds and both LGBM and CatBoost models
    test_probs = np.zeros(len(X_test))
    
    total_models = N_FOLDS * 2
    print(f"  - Running ensemble inference across {total_models} models (5 folds x 2 models)...")
    
    for fold in range(N_FOLDS):
        # Load LGBM
        lgb_path = os.path.join(MODELS_DIR, f"lgb_fold_{fold}.pkl")
        print(f"    * Running Fold {fold+1} LightGBM...")
        with open(lgb_path, "rb") as f:
            lgb_model = pickle.load(f)
        test_probs += lgb_model.predict_proba(X_test)[:, 1] / total_models
        
        # Load CatBoost
        cb_path = os.path.join(MODELS_DIR, f"cb_fold_{fold}.pkl")
        print(f"    * Running Fold {fold+1} CatBoost...")
        with open(cb_path, "rb") as f:
            cb_model = pickle.load(f)
        test_probs += cb_model.predict_proba(X_test)[:, 1] / total_models
        
    # 4. Apply optimal threshold
    print(f"  - Applying decision threshold of {optimal_threshold:.4f}...")
    predictions = (test_probs >= optimal_threshold).astype(int)
    
    # 5. Output to submission.csv
    print(f"  - Writing predictions to {SUBMISSION_OUT_PATH}...")
    sub_df = pl.DataFrame({
        "id": pair_ids,
        "prediction": predictions
    })
    sub_df.write_csv(SUBMISSION_OUT_PATH)
    print(f"  - Submission saved in {time.time() - t_start:.2f} seconds.")
    
    # 6. Strict Sanity Checks
    print("[Predict] Running strict submission sanity checks...")
    
    # Check if submission file exists
    assert os.path.exists(SUBMISSION_OUT_PATH), "Sanity Check Failed: Submission file was not created!"
    
    # Check row count
    sub_pairs_df = pl.read_csv(SUBMISSION_PAIRS_PATH)
    expected_rows = sub_pairs_df.height
    actual_rows = sub_df.height
    print(f"  * Row Count: Expected: {expected_rows}, Got: {actual_rows}")
    assert expected_rows == actual_rows, f"Sanity Check Failed: Row count mismatch! Expected {expected_rows}, got {actual_rows}"
    
    # Check column names
    print(f"  * Column Headers: {sub_df.columns}")
    assert sub_df.columns == ["id", "prediction"], f"Sanity Check Failed: Column names must be ['id', 'prediction'], got {sub_df.columns}"
    
    # Check for missing values
    null_count = sub_df["prediction"].null_count()
    print(f"  * Null values: {null_count}")
    assert null_count == 0, f"Sanity Check Failed: Prediction column contains {null_count} nulls!"
    
    # Check for binary values
    unique_vals = sub_df["prediction"].unique().to_list()
    print(f"  * Unique prediction values: {unique_vals}")
    for val in unique_vals:
        assert val in [0, 1], f"Sanity Check Failed: Predictions must be binary 0 or 1, got unexpected value: {val}"
        
    # Check prediction distribution
    val_counts = sub_df["prediction"].value_counts()
    print("  * Prediction Distribution:")
    print(val_counts)
    
    # Ensure there is class variance
    assert len(unique_vals) > 1, "Sanity Check Failed: Prediction column contains only one class!"
    
    print("[Predict] All strict sanity checks PASSED successfully! Submission is ready.")

if __name__ == "__main__":
    predict_pipeline()
