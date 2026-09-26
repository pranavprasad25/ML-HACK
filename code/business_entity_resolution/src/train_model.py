"""
Stage 3B: Model Training Helper Functions (GPU XGBoost)
Amazon ML Challenge 2026: Business Entity Resolution
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

try:
    from .train_predict import get_gpu_xgboost_model, optimize_f05_threshold
except ImportError:
    from train_predict import get_gpu_xgboost_model, optimize_f05_threshold


def train_matching_model(X_train: np.ndarray, y_train: np.ndarray, use_gpu: bool = True):
    """Trains a GPU-accelerated XGBoost classifier."""
    print(f"Training XGBoost on {len(X_train):,} pairwise samples (GPU={use_gpu})...")
    model, backend = get_gpu_xgboost_model(use_gpu=use_gpu)
    model.fit(X_train, y_train, verbose=False)
    return model


def tune_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    """Finds decision threshold that maximizes Macro F_0.5 score."""
    return optimize_f05_threshold(y_true, y_probs)


def filter_matches_with_subset_constraint(candidate_df: pd.DataFrame, predicted_matches_dict: dict) -> pd.DataFrame:
    """
    Generates matching DataFrame enforcing the SUBSET CONSTRAINT:
    Every matched ID MUST be a subset of candidate_entity_ids.
    """
    results = []
    for row in candidate_df.itertuples():
        s1_id = str(getattr(row, 'source1_entity_id', '')).strip()
        cand_str = str(getattr(row, 'candidate_entity_ids', '') or '').strip()
        candidates = set(cand_str.split(',')) if cand_str else set()
        
        predicted = set(predicted_matches_dict.get(s1_id, []))
        valid_matches = sorted(list(predicted & candidates))
        
        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ",".join(valid_matches)
        })
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Stage 3B Model Training module initialized.")
