import os
import sys
import time
import pickle
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
import lightgbm as lgb
from catboost import CatBoostClassifier

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import configuration
from src.config import (
    ARTIFACTS_DIR,
    SEED,
    N_FOLDS,
    MAX_THREADS
)

# Paths
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "train_features.parquet")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "best_threshold.txt")

def find_best_threshold(y_true, y_probs):
    """Sweeps thresholds from 0.10 to 0.90 with step 0.01 to find the optimal macro F1."""
    best_thresh = 0.5
    best_score = 0.0
    
    thresholds = np.arange(0.10, 0.91, 0.01)
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, y_pred, average='macro')
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

def print_detailed_metrics(y_true, y_pred):
    """Helper to compute and print precision/recall/F1 per class."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"  - Macro F1: {macro_f1:.5f}")
    print(f"  - Class 0 (Irrelevant) -> Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"  - Class 1 (Relevant)   -> Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")

def train_pipeline():
    print("[Train] Starting model training pipeline...")
    t_start = time.time()
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load feature dataset
    print(f"  - Loading features from {FEATURES_PATH}...")
    df = pl.read_parquet(FEATURES_PATH)
    
    # Separate identifiers and target
    id_cols = ["id", "term_id", "item_id"]
    target_col = "label"
    feature_cols = [c for c in df.columns if c not in id_cols and c != target_col]
    
    print(f"  - Feature count: {len(feature_cols)}")
    print(f"  - Features: {feature_cols}")
    
    # Convert to pandas/numpy for scikit-learn/models compatibility
    X = df.select(feature_cols).to_pandas()
    y = df[target_col].to_numpy()
    groups = df["term_id"].to_numpy() # Group by term_id/query to prevent query leakage!
    
    # Initialize OOF prediction arrays
    oof_lgb = np.zeros(len(df))
    oof_cb = np.zeros(len(df))
    oof_ensemble = np.zeros(len(df))
    
    # Setup GroupKFold cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Training Fold {fold+1} / {N_FOLDS} ---")
        t_fold = time.time()
        
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. Train LightGBM with thread limit
        print(f"  - Training LightGBM (threads limit = {MAX_THREADS})...")
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=SEED,
            n_estimators=300,
            learning_rate=0.05,
            n_jobs=MAX_THREADS,
            verbose=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Save LGBM model
        lgb_path = os.path.join(MODELS_DIR, f"lgb_fold_{fold}.pkl")
        with open(lgb_path, "wb") as f:
            pickle.dump(lgb_model, f)
            
        # Predict on validation fold
        val_lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = val_lgb_probs
        
        # 2. Train CatBoost with thread limit
        print(f"  - Training CatBoost (threads limit = {MAX_THREADS})...")
        cb_model = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            loss_function='Logloss',
            random_seed=SEED,
            thread_count=MAX_THREADS,
            verbose=100
        )
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save CatBoost model
        cb_path = os.path.join(MODELS_DIR, f"cb_fold_{fold}.pkl")
        with open(cb_path, "wb") as f:
            pickle.dump(cb_model, f)
            
        # Predict on validation fold
        val_cb_probs = cb_model.predict_proba(X_val)[:, 1]
        oof_cb[val_idx] = val_cb_probs
        
        # Ensemble prediction for the fold validation set (simple average)
        val_ensemble_probs = (val_lgb_probs + val_cb_probs) / 2.0
        oof_ensemble[val_idx] = val_ensemble_probs
        
        # Fold metrics
        best_fold_thresh, best_fold_score = find_best_threshold(y_val, val_ensemble_probs)
        print(f"  - Fold {fold+1} complete in {time.time() - t_fold:.2f} seconds.")
        print(f"  - Best Fold Threshold: {best_fold_thresh:.2f} | Macro F1: {best_fold_score:.5f}")
        
    print("\n==========================================")
    print("      CROSS-VALIDATION EVALUATION")
    print("==========================================")
    
    # 1. Evaluate LightGBM
    best_lgb_thresh, best_lgb_score = find_best_threshold(y, oof_lgb)
    print(f"\nLightGBM Out-Of-Fold Validation:")
    print(f"  - Optimal Threshold: {best_lgb_thresh:.2f}")
    print_detailed_metrics(y, (oof_lgb >= best_lgb_thresh).astype(int))
    
    # 2. Evaluate CatBoost
    best_cb_thresh, best_cb_score = find_best_threshold(y, oof_cb)
    print(f"\nCatBoost Out-Of-Fold Validation:")
    print(f"  - Optimal Threshold: {best_cb_thresh:.2f}")
    print_detailed_metrics(y, (oof_cb >= best_cb_thresh).astype(int))
    
    # 3. Evaluate Ensemble
    best_ens_thresh, best_ens_score = find_best_threshold(y, oof_ensemble)
    print(f"\nEnsemble (LGBM + CatBoost) Out-Of-Fold Validation:")
    print(f"  - Optimal Threshold: {best_ens_thresh:.2f}")
    print_detailed_metrics(y, (oof_ensemble >= best_ens_thresh).astype(int))
    
    # Save the best threshold for inference use
    print(f"\nSaving best threshold ({best_ens_thresh:.2f}) to {THRESHOLD_PATH}...")
    with open(THRESHOLD_PATH, "w") as f:
        f.write(f"{best_ens_thresh:.4f}")
        
    print(f"\n[Train] Finished training pipeline in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    train_pipeline()
