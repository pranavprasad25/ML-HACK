# Stage 3b: Model Training, GPU Acceleration (RTX 5060), F_0.5 Thresholding & Match Prediction
import os
import argparse
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score


def train_matching_model(X_train: np.ndarray, y_train: np.ndarray, use_gpu: bool = True):
    """
    Trains a Gradient Boosted Decision Tree classifier optimized for RTX 5060 (8GB VRAM) & 32GB RAM.
    """
    print(f"Training ML Model on {len(X_train):,} pairwise samples (features: {X_train.shape[1]})...")

    if XGB_AVAILABLE:
        print("  -> Using XGBoost Classifier with GPU Acceleration (device='cuda')...")
        tree_method = "hist"
        device = "cuda" if use_gpu else "cpu"
        try:
            model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method=tree_method,
                device=device,
                random_state=42,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"  [GPU Fallback] XGBoost GPU CUDA error: {e}. Falling back to CPU hist mode...")
            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.04,
                tree_method="hist",
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

    elif LGB_AVAILABLE:
        print("  -> Using LightGBM Classifier...")
        device = "gpu" if use_gpu else "cpu"
        try:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                num_leaves=63,
                learning_rate=0.03,
                device=device,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model
        except Exception:
            model = lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.04,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

    else:
        print("  -> Using Scikit-Learn HistGradientBoostingClassifier (CPU multi-threaded)...")
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=63,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model


def tune_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    """Finds the classification probability threshold that maximizes Macro F_0.5 score."""
    best_thresh, best_score = 0.50, 0.0
    for thresh in np.arange(0.10, 0.95, 0.02):
        y_pred = (y_probs >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    print(f"Optimal Decision Threshold: {best_thresh:.2f} | Validation Macro F_0.5 Score: {best_score:.4f}")
    return best_thresh, best_score


def filter_matches_with_subset_constraint(candidate_df: pd.DataFrame, predicted_matches_dict: dict) -> pd.DataFrame:
    """Generates matching_results.tsv enforcing the SUBSET CONSTRAINT:
    Every matched ID MUST be a subset of candidate_entity_ids.
    """
    results = []
    for _, row in candidate_df.iterrows():
        s1_id = row['source1_entity_id']
        candidates = set(row['candidate_entity_ids'].split(',')) if row['candidate_entity_ids'] else set()
        
        # Get raw predictions for this entity
        predicted = set(predicted_matches_dict.get(s1_id, []))
        
        # Strict Subset Constraint: matched IDs MUST exist in candidate set
        valid_matches = sorted(list(predicted & candidates))
        
        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ",".join(valid_matches)
        })
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Stage 3b Model Training module initialized with GPU & RTX 5060 optimizations.")
