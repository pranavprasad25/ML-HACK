# Stage 3b: Model Training, F_0.5 Thresholding & Match Prediction
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score

def train_matching_model(X_train: np.ndarray, y_train: np.ndarray):
    """Trains a Gradient Boosted Decision Tree classifier."""
    print(f"Training HistGradientBoostingClassifier on {len(X_train)} pairwise samples...")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def tune_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    """Finds the decision threshold that maximizes Macro F_0.5 score."""
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    print(f"Optimal Decision Threshold: {best_thresh:.2f} | Validation F_0.5 Score: {best_score:.4f}")
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
    print("Stage 3b Model Training module initialized.")
