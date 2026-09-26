"""
Stage 3B: Model Training, F_0.5 Threshold Optimization & Submission Generation (Person 3B)
Amazon ML Challenge 2026: Business Entity Resolution

This module:
1. Constructs labeled training datasets from candidate pairs + ground truth
2. Trains a Gradient Boosted Decision Tree (LightGBM / XGBoost) classifier
3. Optimizes the decision threshold specifically for Macro F_0.5 (beta=0.5, Precision weighted 2x over Recall)
4. Generates the final output/matching_results.tsv with strict subset constraint enforcement

Usage:
  # Training mode (with synthetic features for independent dev):
  python -m code.business_entity_resolution.src.train_predict \
      --mode train \
      --candidate-file output/candidate_pairs.tsv \
      --normalized-dir data/normalized_train \
      --ground-truth dataset/train/train_ground_truth.tsv \
      --use-mock-features

  # Full pipeline (train + predict):
  python -m code.business_entity_resolution.src.train_predict \
      --mode both \
      --candidate-file output/candidate_pairs.tsv \
      --normalized-dir data/normalized_train \
      --ground-truth dataset/train/train_ground_truth.tsv \
      --test-candidate-file output/test_candidate_pairs.tsv \
      --test-normalized-dir data/normalized_test \
      --output output/matching_results.tsv

  # Predict-only mode (with saved model):
  python -m code.business_entity_resolution.src.train_predict \
      --mode predict \
      --model-path models/entity_resolution_model.pkl \
      --test-candidate-file output/candidate_pairs.tsv \
      --test-normalized-dir data/normalized_test \
      --output output/matching_results.tsv

Git Branch: feature/model-training
Owned by: Person 3B
"""

import os
import sys
import argparse
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Set, Callable, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CONSTANTS
# ==========================================

# Feature names matching Person 3A's FEATURE_NAMES contract in features.py
try:
    from features import FEATURE_NAMES as P3A_FEATURE_NAMES
    FEATURE_NAMES = list(P3A_FEATURE_NAMES)
except ImportError:
    try:
        from .features import FEATURE_NAMES as P3A_FEATURE_NAMES
        FEATURE_NAMES = list(P3A_FEATURE_NAMES)
    except ImportError:
        from code.business_entity_resolution.src.features import FEATURE_NAMES as P3A_FEATURE_NAMES
        FEATURE_NAMES = list(P3A_FEATURE_NAMES)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "output")

# Negative sampling ratio: for each positive pair, sample this many negatives
NEGATIVE_SAMPLE_RATIO = 5

# ==========================================
# 1. GLOBAL RULE 3: MOCK FEATURE EXTRACTOR
#    Person 3B uses this to test the full ML pipeline independently
#    without waiting for Person 3A's real features.py
# ==========================================

def mock_extract_features(s1_rec: Any, cand_rec: Any, *args, **kwargs) -> np.ndarray:
    """
    Mock feature extractor following Person 3A's (s1_rec, cand_rec) -> np.ndarray contract.
    Returns 9 features normalized between 0.0 and 1.0.
    """
    import random

    if isinstance(s1_rec, dict) and isinstance(cand_rec, dict):
        n1 = str(s1_rec.get('business_name_clean') or s1_rec.get('business_name') or '').lower().strip()
        n2 = str(cand_rec.get('business_name_clean') or cand_rec.get('business_name') or '').lower().strip()
        a1 = str(s1_rec.get('business_address_clean') or s1_rec.get('business_address') or '').lower().strip()
        a2 = str(cand_rec.get('business_address_clean') or cand_rec.get('business_address') or '').lower().strip()
        z1 = str(s1_rec.get('postal_code') or s1_rec.get('zip_code') or '').strip()
        z2 = str(cand_rec.get('postal_code') or cand_rec.get('zip_code') or '').strip()
    else:
        # Legacy positional compatibility
        n1 = str(s1_rec).lower().strip() if s1_rec else ""
        n2 = str(cand_rec).lower().strip() if cand_rec else ""
        a1 = str(args[2]).lower().strip() if len(args) > 2 else ""
        a2 = str(args[3]).lower().strip() if len(args) > 3 else ""
        z1 = str(args[4]).strip() if len(args) > 4 else ""
        z2 = str(args[5]).strip() if len(args) > 5 else ""

    common_prefix_len = 0
    for c1, c2 in zip(n1, n2):
        if c1 == c2:
            common_prefix_len += 1
        else:
            break

    prefix_ratio = common_prefix_len / max(len(n1), len(n2), 1)
    noise = lambda: random.uniform(-0.05, 0.05)

    name_lev = max(0.0, min(1.0, prefix_ratio + noise()))
    name_jaro = max(0.0, min(1.0, prefix_ratio * 0.85 + 0.15 + noise()))

    t1_set = set(n1.split()) if n1 else set()
    t2_set = set(n2.split()) if n2 else set()
    name_sort = len(t1_set & t2_set) / max(len(t1_set | t2_set), 1)
    name_set = name_sort

    addr_lev = max(0.0, min(1.0, random.uniform(0.1, 0.7) if (a1 and a2) else 0.0))
    addr_jaro = max(0.0, min(1.0, random.uniform(0.2, 0.8) if (a1 and a2) else 0.0))
    a1_set = set(a1.split()) if a1 else set()
    a2_set = set(a2.split()) if a2 else set()
    addr_sort = len(a1_set & a2_set) / max(len(a1_set | a2_set), 1)
    addr_set = addr_sort

    zip_match = 1.0 if (z1 and z2 and z1 == z2 and z1 != "0") else 0.0

    return np.array([
        name_lev, name_jaro, name_sort, name_set,
        addr_lev, addr_jaro, addr_sort, addr_set,
        zip_match
    ], dtype=np.float32)


def generate_synthetic_training_data(n_samples: int = 10000, n_features: int = len(FEATURE_NAMES), positive_ratio: float = 0.1):
    """
    Generates synthetic feature matrix + labels for pipeline development.
    Positive pairs (label=1): features biased toward high similarity (0.55-1.0)
    Negative pairs (label=0): features biased toward low similarity (0.0-0.45)
    """
    rng = np.random.RandomState(42)
    n_pos = int(n_samples * positive_ratio)
    n_neg = n_samples - n_pos

    # Positive pairs: high similarity scores
    X_pos = rng.uniform(0.55, 1.0, size=(n_pos, n_features))
    if n_features >= 9:
        X_pos[:, -1] = (rng.random(n_pos) > 0.15).astype(float)  # exact_zip_match
    y_pos = np.ones(n_pos, dtype=np.int32)

    # Negative pairs: low similarity scores
    X_neg = rng.uniform(0.0, 0.45, size=(n_neg, n_features))
    if n_features >= 9:
        X_neg[:, -1] = (rng.random(n_neg) > 0.95).astype(float)  # exact_zip_match rarely 1
    y_neg = np.zeros(n_neg, dtype=np.int32)

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([y_pos, y_neg]).astype(np.int32)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_synthetic_test_environment(
    n_entities: int = 50,
    save_to_dir: Optional[str] = "data/synthetic"
) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame, Dict[str, Set[str]]]:
    """
    Creates a realistic synthetic test environment with entity lookup, candidate pairs DataFrame,
    and ground truth map so that inference and TSV submission generation can be tested end-to-end.
    Optionally saves the synthetic files to disk for inspection.
    """
    lookup: Dict[str, Dict[str, str]] = {}
    candidates_list = []
    gt_map: Dict[str, Set[str]] = {}

    biz_templates = [
        ("Walmart Supercenter", "702 SW 8th St, Bentonville, AR", "72716"),
        ("Starbucks Coffee", "2401 Utah Ave S, Seattle, WA", "98134"),
        ("Target Store", "1000 Nicollet Mall, Minneapolis, MN", "55403"),
        ("Home Depot", "2455 Paces Ferry Rd, Atlanta, GA", "30339"),
        ("McDonald's", "110 N Carpenter St, Chicago, IL", "60607"),
        ("Best Buy", "7601 Penn Ave S, Richfield, MN", "55423"),
        ("Costco Wholesale", "999 Lake Dr, Issaquah, WA", "98027"),
        ("Apple Store", "1 Apple Park Way, Cupertino, CA", "95014"),
        ("CVS Pharmacy", "1 CVS Dr, Woonsocket, RI", "02895"),
        ("Walgreens", "108 Wilmot Rd, Deerfield, IL", "60015"),
    ]

    for i in range(1, n_entities + 1):
        s1_id = f"S1-{i:05d}"
        base_name, base_addr, base_zip = biz_templates[(i - 1) % len(biz_templates)]

        lookup[s1_id] = {
            'entity_id': s1_id,
            'business_name': f"{base_name} #{i}",
            'business_name_clean': f"{base_name} #{i}".lower(),
            'business_address': base_addr,
            'business_address_clean': base_addr.lower(),
            'country': 'US',
            'postal_code': base_zip,
            'name_tokens': f"{base_name.lower()} {i}",
        }

        # True match in S2 (close match)
        s2_id = f"S2-{10000 + i:05d}"
        lookup[s2_id] = {
            'entity_id': s2_id,
            'business_name': f"{base_name} Store {i}",
            'business_name_clean': f"{base_name} store {i}".lower(),
            'business_address': base_addr.replace("St,", "Street,"),
            'business_address_clean': base_addr.lower().replace("st,", "street,"),
            'country': 'US',
            'postal_code': base_zip,
            'name_tokens': f"{base_name.lower()} store {i}",
        }

        # Distractor candidate in S3 (different business)
        wrong_name, wrong_addr, wrong_zip = biz_templates[i % len(biz_templates)]
        s3_id = f"S3-{20000 + i:05d}"
        lookup[s3_id] = {
            'entity_id': s3_id,
            'business_name': f"{wrong_name} Branch {i}",
            'business_name_clean': f"{wrong_name} branch {i}".lower(),
            'business_address': wrong_addr,
            'business_address_clean': wrong_addr.lower(),
            'country': 'US',
            'postal_code': wrong_zip,
            'name_tokens': f"{wrong_name.lower()} branch {i}",
        }

        candidates_list.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': f"{s2_id},{s3_id}"
        })
        gt_map[s1_id] = {s2_id}

    cand_df = pd.DataFrame(candidates_list)

    # Optionally persist synthetic files to disk for inspection
    if save_to_dir:
        os.makedirs(save_to_dir, exist_ok=True)
        cand_path = os.path.join(save_to_dir, "synthetic_candidate_pairs.tsv")
        cand_df.to_csv(cand_path, sep='\t', index=False)

        gt_rows = [{'source1_entity_id': k, 'matched_entity_ids': ','.join(v)} for k, v in gt_map.items()]
        gt_path = os.path.join(save_to_dir, "synthetic_ground_truth.tsv")
        pd.DataFrame(gt_rows).to_csv(gt_path, sep='\t', index=False)

        records_df = pd.DataFrame(list(lookup.values()))
        rec_path = os.path.join(save_to_dir, "synthetic_normalized_records.tsv")
        records_df.to_csv(rec_path, sep='\t', index=False)

        print(f"  [OK] Saved synthetic dataset to: {os.path.abspath(save_to_dir)}")
        print(f"    - Candidate pairs:   {cand_path}")
        print(f"    - Ground truth:      {gt_path}")
        print(f"    - Normalized data:   {rec_path}")

    return lookup, cand_df, gt_map


# ==========================================
# 2. DATA LOADING & ENTITY LOOKUP
# ==========================================

def load_normalized_records(normalized_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Loads normalized TSV files from Stage 1 output into a lookup dictionary.
    Returns: {entity_id: {field_name: value, ...}}
    """
    entity_lookup: Dict[str, Dict[str, str]] = {}

    for filename in sorted(os.listdir(normalized_dir)):
        if not filename.endswith('.tsv') or 'ground_truth' in filename:
            continue

        filepath = os.path.join(normalized_dir, filename)
        print(f"  Loading normalized records from: {filepath}")
        df = pd.read_csv(filepath, sep='\t', dtype=str).fillna("")

        for _, row in df.iterrows():
            eid = str(row.get('entity_id', '')).strip()
            if eid:
                entity_lookup[eid] = {
                    'entity_id': eid,
                    'business_name_clean': str(row.get('business_name_clean', '')),
                    'business_address_clean': str(row.get('business_address_clean', '')),
                    'country': str(row.get('country', '')),
                    'postal_code': str(row.get('postal_code', '')),
                    'name_tokens': str(row.get('name_tokens', '')),
                }

    print(f"  Total entities loaded: {len(entity_lookup):,}")
    return entity_lookup


def load_candidate_pairs(candidate_file: str) -> pd.DataFrame:
    """Loads candidate_pairs.tsv produced by Person 2 (Stage 2)."""
    print(f"  Loading candidate pairs from: {candidate_file}")
    df = pd.read_csv(candidate_file, sep='\t', dtype=str).fillna("")
    print(f"  Loaded {len(df):,} S1 entities with candidates")
    return df


def load_ground_truth(ground_truth_file: str) -> Dict[str, Set[str]]:
    """
    Loads ground truth into a lookup: {s1_id: {matched_s2_id, matched_s3_id, ...}}
    """
    print(f"  Loading ground truth from: {ground_truth_file}")
    gt_map: Dict[str, Set[str]] = {}
    df = pd.read_csv(ground_truth_file, sep='\t', dtype=str).fillna("")

    for _, row in df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        matched_str = str(row.get('matched_entity_ids', '')).strip()
        if matched_str:
            gt_map[s1_id] = set(m.strip() for m in matched_str.split(',') if m.strip())
        else:
            gt_map[s1_id] = set()

    print(f"  Ground truth entries: {len(gt_map):,}")
    total_matches = sum(len(v) for v in gt_map.values())
    print(f"  Total true match pairs: {total_matches:,}")
    return gt_map


# ==========================================
# 3. DATASET CONSTRUCTION
# ==========================================

def build_training_dataset(
    candidate_df: pd.DataFrame,
    entity_lookup: Dict[str, Dict[str, str]],
    gt_map: Dict[str, Set[str]],
    feature_fn: Callable,
    neg_sample_ratio: int = NEGATIVE_SAMPLE_RATIO,
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds X_train, y_train from candidate pairs + ground truth labels.

    For each S1 entity:
    - All candidate IDs that ARE in ground truth -> label = 1 (positive)
    - Sample `neg_sample_ratio` candidates NOT in ground truth -> label = 0 (negative)

    Returns: (X, y, pair_ids) where pair_ids = [(s1_id, cand_id), ...]
    """
    print("\n=== Building Training Dataset ===")
    rng = np.random.RandomState(42)

    all_features: List[List[float]] = []
    all_labels: List[int] = []
    all_pair_ids: List[Tuple[str, str]] = []

    total_pos = 0
    total_neg = 0
    skipped_missing = 0
    processed_entities = 0

    for _, row in candidate_df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        cand_str = str(row.get('candidate_entity_ids', '')).strip()

        if not cand_str:
            continue

        cand_ids = [c.strip() for c in cand_str.split(',') if c.strip()]
        if not cand_ids:
            continue

        s1_rec = entity_lookup.get(s1_id)
        if not s1_rec:
            skipped_missing += 1
            continue

        true_matches = gt_map.get(s1_id, set())

        # Separate positives and negatives
        positive_cands = [cid for cid in cand_ids if cid in true_matches]
        negative_cands = [cid for cid in cand_ids if cid not in true_matches]

        # Add ALL positive pairs
        for cand_id in positive_cands:
            cand_rec = entity_lookup.get(cand_id)
            if not cand_rec:
                skipped_missing += 1
                continue

            feats = feature_fn(s1_rec, cand_rec)
            all_features.append(feats)
            all_labels.append(1)
            all_pair_ids.append((s1_id, cand_id))
            total_pos += 1

        # Sample negatives (up to neg_sample_ratio per positive, minimum 1)
        n_neg_to_sample = max(len(positive_cands) * neg_sample_ratio, 1)
        if len(negative_cands) > n_neg_to_sample:
            sampled_negs = rng.choice(negative_cands, size=n_neg_to_sample, replace=False).tolist()
        else:
            sampled_negs = negative_cands

        for cand_id in sampled_negs:
            cand_rec = entity_lookup.get(cand_id)
            if not cand_rec:
                skipped_missing += 1
                continue

            feats = feature_fn(s1_rec, cand_rec)
            all_features.append(feats)
            all_labels.append(0)
            all_pair_ids.append((s1_id, cand_id))
            total_neg += 1

        processed_entities += 1

        # Progress reporting
        if processed_entities % 50000 == 0:
            print(f"  Processed {processed_entities:,} S1 entities... "
                  f"(pos={total_pos:,}, neg={total_neg:,})")

        # Optional cap for development/debugging
        if max_pairs and (total_pos + total_neg) >= max_pairs:
            print(f"  Reached max_pairs cap ({max_pairs:,}). Stopping early.")
            break

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n  Dataset construction complete:")
    print(f"  Total pairs: {len(y):,} (Positive: {total_pos:,}, Negative: {total_neg:,})")
    print(f"  Positive ratio: {total_pos / max(len(y), 1):.4f}")
    print(f"  Feature matrix shape: {X.shape}")
    if skipped_missing > 0:
        print(f"  Skipped (entity not in lookup): {skipped_missing:,}")

    return X, y, all_pair_ids


# ==========================================
# 4. MODEL TRAINING
# ==========================================

def get_model(backend: str = "lightgbm", use_gpu: bool = False):
    """
    Initializes a Gradient Boosted Decision Tree classifier.

    Supports:
    - LightGBM (preferred, faster)
    - XGBoost (fallback)
    - sklearn HistGradientBoosting (no-dependency fallback)
    """
    if backend == "lightgbm":
        try:
            import lightgbm as lgb
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'num_leaves': 63,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'is_unbalance': True,  # Handle class imbalance
            }
            if use_gpu:
                params['device'] = 'gpu'
                params['gpu_use_dp'] = False
            model = lgb.LGBMClassifier(**params)
            print(f"  Using LightGBM {'(GPU)' if use_gpu else '(CPU)'}")
            return model, "lightgbm"
        except ImportError:
            print("  LightGBM not available, falling back to XGBoost...")
            backend = "xgboost"

    if backend == "xgboost":
        try:
            import xgboost as xgb
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                'scale_pos_weight': NEGATIVE_SAMPLE_RATIO,  # Handle class imbalance
                'eval_metric': 'logloss',
                'use_label_encoder': False,
            }
            if use_gpu:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
            else:
                params['tree_method'] = 'hist'
                params['device'] = 'cpu'
            model = xgb.XGBClassifier(**params)
            print(f"  Using XGBoost {'(GPU)' if use_gpu else '(CPU)'}")
            return model, "xgboost"
        except ImportError:
            print("  XGBoost not available, falling back to sklearn...")
            backend = "sklearn"

    # Fallback: sklearn HistGradientBoosting (always available, no GPU)
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        random_state=42,
    )
    print("  Using sklearn HistGradientBoostingClassifier (CPU only)")
    return model, "sklearn"


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "lightgbm",
    use_gpu: bool = False,
    n_folds: int = 5,
) -> Tuple[object, float, float]:
    """
    Trains a GBDT classifier with stratified K-fold cross-validation.

    Returns: (trained_model, best_threshold, best_f05_score)
    """
    print("\n=== Model Training ===")
    start_time = time.time()

    model, actual_backend = get_model(backend, use_gpu)

    # Stratified K-Fold for threshold optimization
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(y), dtype=np.float64)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        print(f"\n  Fold {fold_idx}/{n_folds}: "
              f"Train={len(y_tr):,} (pos={y_tr.sum():,}), "
              f"Val={len(y_val):,} (pos={y_val.sum():,})")

        # Fit the model
        if actual_backend == "lightgbm":
            import lightgbm as lgb
            model_fold, _ = get_model(backend, use_gpu)
            model_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
        elif actual_backend == "xgboost":
            model_fold, _ = get_model(backend, use_gpu)
            model_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model_fold, _ = get_model(backend, use_gpu)
            model_fold.fit(X_tr, y_tr)

        # Out-of-fold predictions
        val_probs = model_fold.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        # Quick F_0.5 at default 0.5 threshold
        val_pred = (val_probs >= 0.5).astype(int)
        fold_f05 = fbeta_score(y_val, val_pred, beta=0.5, zero_division=0)
        fold_scores.append(fold_f05)
        print(f"  Fold {fold_idx} F_0.5 @0.5: {fold_f05:.4f}")

    print(f"\n  Mean CV F_0.5 @0.5: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # Train final model on all data
    print("\n  Training final model on full dataset...")
    final_model, _ = get_model(backend, use_gpu)
    final_model.fit(X, y)

    # Optimize threshold using OOF predictions
    best_threshold, best_f05 = optimize_f05_threshold(y, oof_probs)

    elapsed = time.time() - start_time
    print(f"\n  Total training time: {elapsed:.1f}s")

    return final_model, best_threshold, best_f05


# ==========================================
# 5. F_0.5 THRESHOLD OPTIMIZATION
# ==========================================

def compute_macro_f05_per_entity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_ids: Optional[List[Tuple[str, str]]] = None,
) -> float:
    """
    Computes Macro F_0.5 score.
    If pair_ids provided, computes per-entity F_0.5 and averages (true macro).
    Otherwise, computes binary F_0.5 as a proxy.
    """
    if pair_ids is None:
        return fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

    # Group predictions by S1 entity for true macro calculation
    entity_scores: Dict[str, Dict[str, int]] = {}
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if s1_id not in entity_scores:
            entity_scores[s1_id] = {'tp': 0, 'fp': 0, 'fn': 0}

        if y_pred[i] == 1 and y_true[i] == 1:
            entity_scores[s1_id]['tp'] += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            entity_scores[s1_id]['fp'] += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            entity_scores[s1_id]['fn'] += 1

    f05_scores = []
    beta_sq = 0.5 ** 2  # 0.25

    for s1_id, counts in entity_scores.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        f05_scores.append(f05)

    return np.mean(f05_scores) if f05_scores else 0.0


def optimize_f05_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    pair_ids: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[float, float]:
    """
    Finds the decision threshold that maximizes F_0.5 score.
    Sweeps from 0.30 to 0.95 in steps of 0.01.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Precision is weighted 2x over Recall -> prefer higher thresholds.
    """
    print("\n=== F_0.5 Threshold Optimization ===")

    best_thresh = 0.5
    best_score = 0.0
    results = []

    for thresh in np.arange(0.30, 0.96, 0.01):
        y_pred = (y_probs >= thresh).astype(int)

        if pair_ids:
            score = compute_macro_f05_per_entity(y_true, y_pred, pair_ids)
        else:
            score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        n_predicted = int(y_pred.sum())

        results.append((thresh, score, prec, rec, n_predicted))

        if score > best_score:
            best_score = score
            best_thresh = thresh

    # Print top 10 thresholds around the best
    print(f"\n  {'Threshold':>10} {'F_0.5':>8} {'Precision':>10} {'Recall':>8} {'Predicted':>10}")
    print("  " + "-" * 52)
    results.sort(key=lambda x: x[1], reverse=True)
    for thresh, score, prec, rec, n_pred in results[:10]:
        marker = " <-- BEST" if abs(thresh - best_thresh) < 0.005 else ""
        print(f"  {thresh:>10.2f} {score:>8.4f} {prec:>10.4f} {rec:>8.4f} {n_pred:>10,}{marker}")

    print(f"\n  [OK] Optimal Threshold: {best_thresh:.2f}")
    print(f"  [OK] Best F_0.5 Score:  {best_score:.4f}")

    return best_thresh, best_score


# ==========================================
# 6. INFERENCE & SUBMISSION GENERATION
# ==========================================

def predict_matches(
    candidate_df: pd.DataFrame,
    entity_lookup: Dict[str, Dict[str, str]],
    model: object,
    threshold: float,
    feature_fn: Callable,
) -> Dict[str, List[str]]:
    """
    Runs inference on candidate pairs and returns predicted matches.
    Returns: {s1_id: [matched_cand_id, ...]}
    """
    print(f"\n=== Running Inference (threshold={threshold:.2f}) ===")

    predicted_matches: Dict[str, List[str]] = {}
    total_pairs = 0
    total_matches = 0
    skipped = 0

    for row_idx, row in enumerate(candidate_df.itertuples()):
        s1_id = str(row.source1_entity_id).strip()
        cand_str = str(getattr(row, 'candidate_entity_ids', '')).strip()

        if not cand_str:
            predicted_matches[s1_id] = []
            continue

        cand_ids = [c.strip() for c in cand_str.split(',') if c.strip()]
        if not cand_ids:
            predicted_matches[s1_id] = []
            continue

        s1_rec = entity_lookup.get(s1_id)
        if not s1_rec:
            predicted_matches[s1_id] = []
            skipped += 1
            continue

        # Extract features for all candidates of this S1 entity
        batch_features = []
        valid_cand_ids = []

        for cand_id in cand_ids:
            cand_rec = entity_lookup.get(cand_id)
            if not cand_rec:
                skipped += 1
                continue

            feats = feature_fn(s1_rec, cand_rec)
            batch_features.append(feats)
            valid_cand_ids.append(cand_id)

        if not batch_features:
            predicted_matches[s1_id] = []
            continue

        # Batch prediction
        X_batch = np.array(batch_features, dtype=np.float32)
        probs = model.predict_proba(X_batch)[:, 1]

        # Apply threshold
        matches = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                matches.append(valid_cand_ids[i])

        predicted_matches[s1_id] = matches
        total_pairs += len(valid_cand_ids)
        total_matches += len(matches)

        # Progress
        if (row_idx + 1) % 50000 == 0:
            print(f"  Processed {row_idx + 1:,} S1 entities... "
                  f"({total_matches:,} matches so far)")

    print(f"\n  Inference complete:")
    print(f"  Total pairs scored: {total_pairs:,}")
    print(f"  Total matches predicted: {total_matches:,}")
    print(f"  S1 entities with >=1 match: {sum(1 for v in predicted_matches.values() if v):,}")
    if skipped > 0:
        print(f"  Skipped (entity not in lookup): {skipped:,}")

    return predicted_matches


def generate_submission(
    candidate_df: pd.DataFrame,
    predicted_matches: Dict[str, List[str]],
    output_path: str,
) -> pd.DataFrame:
    """
    Generates matching_results.tsv enforcing the SUBSET CONSTRAINT:
    Every matched ID in matching_results MUST exist in candidate_pairs for that entity.

    Rules:
    - Every S1 entity gets exactly 1 row
    - Singletons have empty matched_entity_ids
    - Only S2/S3 IDs (no S1 self-matches)
    - No duplicate IDs in a row
    - Tab-separated output
    """
    print(f"\n=== Generating Submission: {output_path} ===")

    results = []

    for _, row in candidate_df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        cand_str = str(row.get('candidate_entity_ids', '')).strip()
        candidates = set(cand_str.split(',')) if cand_str else set()

        # Get raw model predictions for this entity
        raw_predicted = set(predicted_matches.get(s1_id, []))

        # STRICT SUBSET CONSTRAINT: matched IDs MUST exist in candidate set
        valid_matches = raw_predicted & candidates

        # Additional safety: exclude self-matches, ensure non-empty
        valid_matches = sorted([
            mid for mid in valid_matches
            if mid != s1_id and mid.startswith(('S2-', 'S3-')) and mid.strip()
        ])

        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ",".join(valid_matches)
        })

    result_df = pd.DataFrame(results, columns=['source1_entity_id', 'matched_entity_ids'])

    # Save
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_path, sep='\t', index=False)

    # Stats
    n_with_matches = sum(1 for _, r in result_df.iterrows() if r['matched_entity_ids'])
    n_singletons = len(result_df) - n_with_matches
    total_match_ids = sum(
        len(r['matched_entity_ids'].split(','))
        for _, r in result_df.iterrows()
        if r['matched_entity_ids']
    )

    print(f"  [OK] Saved {len(result_df):,} rows to: {os.path.abspath(output_path)}")
    print(f"  Entities with >=1 match: {n_with_matches:,}")
    print(f"  Singleton entities (no match): {n_singletons:,}")
    print(f"  Total matched pairs: {total_match_ids:,}")
    if len(result_df) > 0:
        print(f"\n  First 5 rows preview of {os.path.basename(output_path)}:")
        for _, r in result_df.head(5).iterrows():
            print(f"    {r['source1_entity_id']}\t{r['matched_entity_ids']}")

    return result_df


# ==========================================
# 7. MODEL PERSISTENCE
# ==========================================

def save_model(model, threshold: float, f05_score: float, model_dir: str, backend: str):
    """Saves the trained model, threshold, and metadata."""
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "entity_resolution_model.pkl")
    metadata = {
        'model': model,
        'threshold': threshold,
        'f05_score': f05_score,
        'backend': backend,
        'feature_names': FEATURE_NAMES,
        'n_features': len(FEATURE_NAMES),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n  [OK] Model saved to: {model_path}")
    print(f"    Backend: {backend}")
    print(f"    Threshold: {threshold:.2f}")
    print(f"    F_0.5 Score: {f05_score:.4f}")
    return model_path


def load_saved_model(model_path: str) -> Tuple[object, float, str]:
    """Loads a previously saved model + threshold."""
    print(f"  Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        metadata = pickle.load(f)

    model = metadata['model']
    threshold = metadata['threshold']
    backend = metadata.get('backend', 'unknown')

    print(f"  Backend: {backend}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Saved F_0.5: {metadata.get('f05_score', 'N/A')}")
    print(f"  Saved at: {metadata.get('timestamp', 'N/A')}")

    return model, threshold, backend


# ==========================================
# 8. FEATURE FUNCTION SELECTOR
# ==========================================

def get_feature_function(use_mock: bool = False) -> Callable:
    """
    Returns the feature extraction function.
    - If use_mock=True: uses mock_extract_features (Global Rule 3)
    - If use_mock=False: imports Person 3A's real extract_features from features.py
    """
    if use_mock:
        print("  Using MOCK feature extractor (Global Rule 3)")
        return mock_extract_features

    # 1. Try importing Person 3A's standard extract_features (s1_rec, cand_rec)
    try:
        from code.business_entity_resolution.src.features import extract_features
        print("  Using REAL feature extractor (features.py: extract_features)")
        return extract_features
    except ImportError:
        pass

    try:
        from .features import extract_features
        print("  Using REAL feature extractor (features.py via relative import)")
        return extract_features
    except (ImportError, SystemError):
        pass

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import features
        if hasattr(features, 'extract_features'):
            print("  Using REAL feature extractor (features.py via sys.path)")
            return features.extract_features
        elif hasattr(features, 'compute_pair_features'):
            print("  Using REAL feature extractor (features.py: compute_pair_features adapter)")
            def legacy_adapter(s1_rec, cand_rec):
                return features.compute_pair_features(
                    s1_rec.get('business_name_clean', ''), cand_rec.get('business_name_clean', ''),
                    s1_rec.get('name_tokens', ''), cand_rec.get('name_tokens', ''),
                    s1_rec.get('business_address_clean', ''), cand_rec.get('business_address_clean', ''),
                    s1_rec.get('postal_code', ''), cand_rec.get('postal_code', ''),
                )
            return legacy_adapter
    except ImportError:
        pass

    print("  WARNING: Could not import features.py. Falling back to MOCK features.")
    return mock_extract_features


# ==========================================
# 9. EVALUATION HELPERS
# ==========================================

def evaluate_predictions(
    predicted_matches: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluates predicted matches against ground truth using Macro F_0.5.
    """
    print("\n=== Evaluation Against Ground Truth ===")

    beta_sq = 0.25
    entity_f05_scores = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for s1_id, true_matches in gt_map.items():
        predicted = set(predicted_matches.get(s1_id, []))

        tp = len(predicted & true_matches)
        fp = len(predicted - true_matches)
        fn = len(true_matches - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        entity_f05_scores.append(f05)

    macro_f05 = np.mean(entity_f05_scores) if entity_f05_scores else 0.0
    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f05 = ((1 + beta_sq) * micro_prec * micro_rec / (beta_sq * micro_prec + micro_rec)
                 if (micro_prec + micro_rec) > 0 else 0.0)

    metrics = {
        'macro_f05': macro_f05,
        'micro_f05': micro_f05,
        'micro_precision': micro_prec,
        'micro_recall': micro_rec,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }

    print(f"  Macro F_0.5:      {macro_f05:.4f}")
    print(f"  Micro F_0.5:      {micro_f05:.4f}")
    print(f"  Micro Precision:  {micro_prec:.4f}")
    print(f"  Micro Recall:     {micro_rec:.4f}")
    print(f"  True Positives:   {total_tp:,}")
    print(f"  False Positives:  {total_fp:,}")
    print(f"  False Negatives:  {total_fn:,}")

    return metrics


# ==========================================
# 10. CLI ENTRY POINT
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3B: Model Training, F_0.5 Threshold Optimization & Submission (Person 3B)"
    )

    # Mode
    parser.add_argument(
        '--mode', type=str, choices=['train', 'predict', 'both', 'synthetic'],
        default='both',
        help="'train': train only, 'predict': inference only, 'both': train+predict, "
             "'synthetic': test pipeline with synthetic data"
    )

    # Training inputs
    parser.add_argument('--candidate-file', type=str, default='output/candidate_pairs.tsv',
                        help="Path to candidate_pairs.tsv from Stage 2")
    parser.add_argument('--normalized-dir', type=str, default='data/normalized_train',
                        help="Directory containing normalized TSVs from Stage 1")
    parser.add_argument('--ground-truth', type=str, default='dataset/train/train_ground_truth.tsv',
                        help="Path to training ground truth TSV")

    # Test/prediction inputs
    parser.add_argument('--test-candidate-file', type=str, default=None,
                        help="Candidate pairs for test inference (defaults to --candidate-file)")
    parser.add_argument('--test-normalized-dir', type=str, default=None,
                        help="Normalized test data dir (defaults to --normalized-dir)")

    # Output
    parser.add_argument('--output', type=str, default='matching_results.tsv',
                        help="Path to save matching_results.tsv (default: matching_results.tsv)")

    # Model
    parser.add_argument('--backend', type=str, choices=['lightgbm', 'xgboost', 'sklearn'],
                        default='lightgbm', help="GBDT backend")
    parser.add_argument('--use-gpu', action='store_true', help="Enable GPU acceleration")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to load a pre-trained model (predict mode)")
    parser.add_argument('--model-dir', type=str, default=None,
                        help="Directory to save trained models")

    # Feature extraction
    parser.add_argument('--use-mock-features', action='store_true',
                        help="Use mock feature extractor (Global Rule 3) for independent testing")

    # Development options
    parser.add_argument('--max-pairs', type=int, default=None,
                        help="Cap on training pairs (for development/debugging)")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument('--neg-ratio', type=int, default=NEGATIVE_SAMPLE_RATIO,
                        help="Negative samples per positive pair")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" Stage 3B: Model Training, F_0.5 Optimization & Submission")
    print(" Amazon ML Challenge 2026 - Business Entity Resolution")
    print(f" Mode: {args.mode.upper()}")
    print("=" * 70)

    model_dir = args.model_dir or DEFAULT_MODEL_DIR

    # ---- SYNTHETIC MODE: Test entire pipeline end-to-end with generated data ----
    if args.mode == 'synthetic':
        print("\n>>> SYNTHETIC MODE: Testing full pipeline end-to-end <<<")
        X_syn, y_syn = generate_synthetic_training_data(n_samples=5000, n_features=len(FEATURE_NAMES))
        print(f"\n[1/4] Synthetic training data: {X_syn.shape[0]:,} pairs, {X_syn.shape[1]} features")

        # Train model
        print("\n[2/4] Training model...")
        model, threshold, f05_score = train_model(
            X_syn, y_syn, backend=args.backend, use_gpu=args.use_gpu, n_folds=3
        )
        saved_model_path = save_model(model, threshold, f05_score, model_dir, args.backend)

        # Generate synthetic test set for inference
        print("\n[3/4] Generating synthetic test candidates for inference...")
        syn_lookup, syn_cand_df, syn_gt = generate_synthetic_test_environment(n_entities=50)

        # Run inference
        print("\n[4/4] Running inference & generating submission output...")
        feature_fn = get_feature_function(use_mock=args.use_mock_features)
        predicted_matches = predict_matches(syn_cand_df, syn_lookup, model, threshold, feature_fn)

        # Output path
        output_file = args.output
        generate_submission(syn_cand_df, predicted_matches, output_file)

        # Evaluate against synthetic ground truth
        evaluate_predictions(predicted_matches, syn_gt)

        print("\n" + "=" * 70)
        print(" [OK] Synthetic pipeline test PASSED! End-to-end run completed.")
        print(f"  Model saved to:      {os.path.abspath(saved_model_path)}")
        print(f"  Submission saved to: {os.path.abspath(output_file)}")
        print("=" * 70)
        return

    # ---- Select feature function ----
    feature_fn = get_feature_function(use_mock=args.use_mock_features)

    # ---- TRAIN MODE ----
    if args.mode in ('train', 'both'):
        print("\n--- Phase: Training ---")

        # Load data
        print("\n[1/4] Loading data...")
        entity_lookup = load_normalized_records(args.normalized_dir)
        candidate_df = load_candidate_pairs(args.candidate_file)
        gt_map = load_ground_truth(args.ground_truth)

        # Build dataset
        print("\n[2/4] Building training dataset...")
        X, y, pair_ids = build_training_dataset(
            candidate_df, entity_lookup, gt_map, feature_fn,
            neg_sample_ratio=args.neg_ratio, max_pairs=args.max_pairs,
        )

        if len(y) == 0:
            print("ERROR: No training pairs constructed. Check data paths and overlap.")
            sys.exit(1)

        # Train model
        print("\n[3/4] Training model...")
        model, threshold, f05_score = train_model(
            X, y, backend=args.backend, use_gpu=args.use_gpu, n_folds=args.n_folds
        )

        # Save model
        print("\n[4/4] Saving model...")
        model_path = save_model(model, threshold, f05_score, model_dir, args.backend)

        # Quick evaluation on training data
        print("\n--- Quick Training Evaluation ---")
        train_predicted = predict_matches(candidate_df, entity_lookup, model, threshold, feature_fn)
        evaluate_predictions(train_predicted, gt_map)

    # ---- PREDICT MODE ----
    if args.mode in ('predict', 'both'):
        print("\n--- Phase: Prediction & Submission ---")

        # Load model if not already trained
        if args.mode == 'predict':
            mp = args.model_path or os.path.join(model_dir, "entity_resolution_model.pkl")
            if not os.path.isfile(mp):
                print(f"ERROR: Model file not found: {mp}")
                print("Run in 'train' or 'both' mode first, or specify --model-path.")
                sys.exit(1)
            model, threshold, _ = load_saved_model(mp)

        # Resolve test paths
        test_cand_file = args.test_candidate_file or args.candidate_file
        test_norm_dir = args.test_normalized_dir or args.normalized_dir

        # Load test data
        print("\n[1/3] Loading test data...")
        test_entity_lookup = load_normalized_records(test_norm_dir)
        test_candidate_df = load_candidate_pairs(test_cand_file)

        # Run inference
        print("\n[2/3] Running inference...")
        predicted_matches = predict_matches(
            test_candidate_df, test_entity_lookup, model, threshold, feature_fn
        )

        # Generate submission
        print("\n[3/3] Generating submission...")
        generate_submission(test_candidate_df, predicted_matches, args.output)

        # Evaluate if ground truth available
        if args.ground_truth and os.path.isfile(args.ground_truth):
            gt_map_eval = load_ground_truth(args.ground_truth)
            evaluate_predictions(predicted_matches, gt_map_eval)

    print("\n" + "=" * 70)
    print(" [OK] Stage 3B Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
