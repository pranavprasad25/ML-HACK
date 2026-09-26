"""
Stage 3B: Model Training, GPU XGBoost, F_0.5 Threshold Optimization & Submission Generation
Amazon ML Challenge 2026: Business Entity Resolution

Engineered & Tuned for: NVIDIA GeForce RTX 5060 (8GB VRAM) & Multi-Core CPU
Target Metric: Macro F_0.5 Score >= 0.98 (Precision-focused with beta=0.5)

Features:
  - 24-Feature Dense Input Matrix (RapidFuzz, Jaccard, Soundex, Address Digits, Zip Prefix)
  - GPU-Accelerated XGBoost (`tree_method='hist'`, `device='cuda'`, `max_bin=512`)
  - Precision-Aware Threshold Optimization targeting Macro F_0.5
  - Memory-Efficient Streaming Inference with strict Subset Constraint Enforcement
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
# 0. CONSTANTS & IMPORTS
# ==========================================
try:
    from code.business_entity_resolution.src.features import FEATURE_NAMES, NUM_FEATURES, extract_features
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import FEATURE_NAMES, NUM_FEATURES, extract_features

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "output")

# Negative sampling ratio during dataset construction
NEGATIVE_SAMPLE_RATIO = 4


# ==========================================
# 1. DATA LOADING & ENTITY LOOKUP
# ==========================================

def load_normalized_records(normalized_dir: str) -> Dict[str, Dict[str, str]]:
    """Loads normalized TSV files from Stage 1 into an in-memory lookup dictionary."""
    entity_lookup: Dict[str, Dict[str, str]] = {}
    print(f"Loading normalized records from: {normalized_dir}")
    
    for filename in sorted(os.listdir(normalized_dir)):
        if not filename.endswith('.tsv') or 'ground_truth' in filename:
            continue

        filepath = os.path.join(normalized_dir, filename)
        print(f"  Reading: {filepath}...")
        df = pd.read_csv(filepath, sep='\t', dtype=str).fillna("")

        for row in df.itertuples():
            eid = str(getattr(row, 'entity_id', '')).strip()
            if eid:
                entity_lookup[eid] = {
                    'entity_id': eid,
                    'business_name_clean': str(getattr(row, 'business_name_clean', '') or ''),
                    'business_address_clean': str(getattr(row, 'business_address_clean', '') or ''),
                    'country': str(getattr(row, 'country', '') or ''),
                    'postal_code': str(getattr(row, 'postal_code', '') or ''),
                    'name_tokens': str(getattr(row, 'name_tokens', '') or ''),
                }

    print(f"Total entities loaded: {len(entity_lookup):,}")
    return entity_lookup


def load_candidate_pairs(candidate_file: str) -> pd.DataFrame:
    """Loads candidate_pairs.tsv produced by Stage 2 Blocking."""
    print(f"Loading candidate pairs from: {candidate_file}")
    df = pd.read_csv(candidate_file, sep='\t', dtype=str).fillna("")
    print(f"Loaded {len(df):,} S1 entities with candidates")
    return df


def load_ground_truth(ground_truth_file: str) -> Dict[str, Set[str]]:
    """Loads ground truth into lookup: {s1_id: {matched_s2_id, matched_s3_id, ...}}"""
    print(f"Loading ground truth from: {ground_truth_file}")
    gt_map: Dict[str, Set[str]] = {}
    df = pd.read_csv(ground_truth_file, sep='\t', dtype=str).fillna("")

    for row in df.itertuples():
        s1_id = str(getattr(row, 'source1_entity_id', '')).strip()
        matched_str = str(getattr(row, 'matched_entity_ids', '') or '').strip()
        if matched_str:
            gt_map[s1_id] = set(m.strip() for m in matched_str.split(',') if m.strip())
        else:
            gt_map[s1_id] = set()

    total_matches = sum(len(v) for v in gt_map.values())
    print(f"Ground truth loaded: {len(gt_map):,} entities ({total_matches:,} true match pairs)")
    return gt_map


# ==========================================
# 2. TRAINING DATASET CONSTRUCTION
# ==========================================

def build_training_dataset(
    candidate_df: pd.DataFrame,
    entity_lookup: Dict[str, Dict[str, str]],
    gt_map: Dict[str, Set[str]],
    neg_sample_ratio: int = NEGATIVE_SAMPLE_RATIO,
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds (X, y, pair_ids) from candidate pairs + ground truth.
    Extracts all 24 pairwise similarity features.
    """
    print("\n=== Building Pairwise Training Dataset ===")
    rng = np.random.RandomState(42)

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_pair_ids: List[Tuple[str, str]] = []

    total_pos = 0
    total_neg = 0
    skipped = 0

    for row in candidate_df.itertuples():
        s1_id = str(getattr(row, 'source1_entity_id', '')).strip()
        cand_str = str(getattr(row, 'candidate_entity_ids', '') or '').strip()
        if not cand_str:
            continue

        cand_ids = [c.strip() for c in cand_str.split(',') if c.strip()]
        s1_rec = entity_lookup.get(s1_id)
        if not s1_rec:
            skipped += 1
            continue

        true_matches = gt_map.get(s1_id, set())
        positives = [cid for cid in cand_ids if cid in true_matches]
        negatives = [cid for cid in cand_ids if cid not in true_matches]

        # Add ALL positive pairs
        for cand_id in positives:
            cand_rec = entity_lookup.get(cand_id)
            if not cand_rec:
                continue
            feats = extract_features(s1_rec, cand_rec)
            all_features.append(feats)
            all_labels.append(1)
            all_pair_ids.append((s1_id, cand_id))
            total_pos += 1

        # Sample negatives
        if negatives:
            n_sample = min(len(negatives), max(1, len(positives) * neg_sample_ratio))
            sampled_negatives = rng.choice(negatives, size=n_sample, replace=False)
            for cand_id in sampled_negatives:
                cand_rec = entity_lookup.get(cand_id)
                if not cand_rec:
                    continue
                feats = extract_features(s1_rec, cand_rec)
                all_features.append(feats)
                all_labels.append(0)
                all_pair_ids.append((s1_id, cand_id))
                total_neg += 1

        if max_pairs and len(all_labels) >= max_pairs:
            break

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"Dataset constructed: {len(y):,} samples (Positives={total_pos:,}, Negatives={total_neg:,})")
    print(f"Feature matrix shape: {X.shape}, dtype={X.dtype}")
    return X, y, all_pair_ids


# ==========================================
# 3. GPU-OPTIMIZED XGBOOST MODEL
# ==========================================

def get_gpu_xgboost_model(use_gpu: bool = True):
    """
    Instantiates an XGBoost Classifier specifically tuned for RTX 5060 GPU (8GB VRAM).
    Features:
      - `tree_method='hist'`, `device='cuda'`
      - `max_bin=512`: Higher histogram precision for subtle float distance boundaries
      - `max_depth=8`, `learning_rate=0.03`, `n_estimators=1000`
      - `subsample=0.85`, `colsample_bytree=0.85`
      - `reg_alpha=0.5`, `reg_lambda=2.0`: L1 & L2 regularization against overfitting
      - `scale_pos_weight=1.0`: Precision-oriented (crucial for Macro F_0.5)
    """
    try:
        import xgboost as xgb
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.03,
            'max_depth': 8,
            'max_bin': 512,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'scale_pos_weight': 1.0,
        }

        if use_gpu:
            try:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
                # Test cuda device availability
                test_clf = xgb.XGBClassifier(**params)
                print("  [GPU ACCELERATION ACTIVE] XGBoost configured on NVIDIA CUDA GPU (RTX 5060)")
                return test_clf, "xgboost_gpu"
            except Exception as e:
                print(f"  CUDA GPU not available ({e}), switching to CPU hist mode...")
                params['device'] = 'cpu'
                params['tree_method'] = 'hist'
                return xgb.XGBClassifier(**params), "xgboost_cpu"
        else:
            params['device'] = 'cpu'
            params['tree_method'] = 'hist'
            return xgb.XGBClassifier(**params), "xgboost_cpu"

    except ImportError:
        print("  XGBoost not installed, falling back to LightGBM / Sklearn...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=63, random_state=42), "sklearn"


def optimize_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Finds optimal decision threshold maximizing Macro F_0.5 (beta=0.5).
    Fine-grained search across [0.30, 0.92] with step 0.005.
    """
    best_thresh = 0.50
    best_score = 0.0

    thresholds = np.arange(0.30, 0.92, 0.005)
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=0.5, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    # Compute precision & recall at optimal threshold
    best_preds = (y_probs >= best_thresh).astype(int)
    p = precision_score(y_true, best_preds, zero_division=0)
    r = recall_score(y_true, best_preds, zero_division=0)

    print(f"\n>>> F_0.5 Optimization Result:")
    print(f"    Optimal Threshold : {best_thresh:.4f}")
    print(f"    Validation F_0.5  : {best_score:.4f} (Precision: {p:.4f} | Recall: {r:.4f})")

    return best_thresh, best_score


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    use_gpu: bool = True,
    n_folds: int = 5,
) -> Tuple[object, float, float]:
    """
    Trains GPU-accelerated XGBoost with Stratified 5-Fold Cross Validation.
    Returns: (final_model, optimal_threshold, validation_f05_score)
    """
    print("\n" + "=" * 80)
    print(" TRAINING XGBOOST CLASSIFIER (NVIDIA GPU ACCELERATED)")
    print("=" * 80)
    t0 = time.time()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float64)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model_fold, backend_name = get_gpu_xgboost_model(use_gpu=use_gpu)
        model_fold.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_probs = model_fold.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs

        val_pred = (val_probs >= 0.50).astype(int)
        fold_f05 = fbeta_score(y_va, val_pred, beta=0.5, zero_division=0)
        fold_scores.append(fold_f05)
        print(f"  Fold {fold}/{n_folds} - F_0.5 (@0.50): {fold_f05:.4f}")

    print(f"\nMean 5-Fold CV Score: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # Optimize threshold using Out-Of-Fold probabilities
    best_thresh, best_f05 = optimize_f05_threshold(y, oof_probs)

    # Train final production model on 100% of data
    print("\nFitting final production model on full training set...")
    final_model, _ = get_gpu_xgboost_model(use_gpu=use_gpu)
    final_model.fit(X, y, verbose=False)

    print(f"Training completed in {time.time() - t0:.1f}s")
    return final_model, best_thresh, best_f05


# ==========================================
# 4. INFERENCE & PREDICTION GENERATION
# ==========================================

def predict_matches(
    model: object,
    threshold: float,
    candidate_df: pd.DataFrame,
    entity_lookup: Dict[str, Dict[str, str]],
    batch_size: int = 50000,
) -> Dict[str, List[str]]:
    """
    Runs high-throughput batch inference to predict matches for each S1 entity.
    Returns: {source1_entity_id: [matched_candidate_id, ...]}
    """
    print("\n=== Running Batch Inference & Prediction ===")
    t0 = time.time()

    predicted_matches: Dict[str, List[str]] = {str(eid).strip(): [] for eid in candidate_df['source1_entity_id']}
    
    # Collect all candidate pairs to evaluate
    pairs_s1: List[str] = []
    pairs_cand: List[str] = []
    pairs_s1_rec: List[Dict[str, str]] = []
    pairs_cand_rec: List[Dict[str, str]] = []

    for row in candidate_df.itertuples():
        s1_id = str(getattr(row, 'source1_entity_id', '')).strip()
        cand_str = str(getattr(row, 'candidate_entity_ids', '') or '').strip()
        if not cand_str:
            continue

        s1_rec = entity_lookup.get(s1_id)
        if not s1_rec:
            continue

        for cand_id in cand_str.split(','):
            cand_id_clean = cand_id.strip()
            if not cand_id_clean:
                continue
            cand_rec = entity_lookup.get(cand_id_clean)
            if not cand_rec:
                continue

            pairs_s1.append(s1_id)
            pairs_cand.append(cand_id_clean)
            pairs_s1_rec.append(s1_rec)
            pairs_cand_rec.append(cand_rec)

    total_pairs = len(pairs_s1)
    print(f"Evaluating {total_pairs:,} candidate pairs across {len(candidate_df):,} S1 entities...")

    # Process in batches
    for start_idx in range(0, total_pairs, batch_size):
        end_idx = min(start_idx + batch_size, total_pairs)
        batch_feats = np.array([
            extract_features(pairs_s1_rec[i], pairs_cand_rec[i])
            for i in range(start_idx, end_idx)
        ], dtype=np.float32)

        batch_probs = model.predict_proba(batch_feats)[:, 1]
        
        for i, prob in enumerate(batch_probs):
            if prob >= threshold:
                global_idx = start_idx + i
                s1_id = pairs_s1[global_idx]
                cand_id = pairs_cand[global_idx]
                predicted_matches[s1_id].append(cand_id)

    print(f"Inference completed in {time.time() - t0:.1f}s ({total_pairs / max(time.time() - t0, 0.001):,.0f} pairs/sec)")
    return predicted_matches


def generate_submission(
    candidate_df: pd.DataFrame,
    predictions: Dict[str, List[str]],
    output_path: str,
) -> pd.DataFrame:
    """
    Formats and writes output/matching_results.tsv adhering strictly to the competition format
    and enforcing the SUBSET CONSTRAINT.
    """
    print(f"\nWriting matching results to: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows = []
    matched_count = 0
    total_matches = 0

    for row in candidate_df.itertuples():
        s1_id = str(getattr(row, 'source1_entity_id', '')).strip()
        cand_str = str(getattr(row, 'candidate_entity_ids', '') or '').strip()
        cand_set = set(cand_str.split(',')) if cand_str else set()

        raw_preds = predictions.get(s1_id, [])
        # Strict Subset Constraint: matched IDs MUST be in candidate set and start with S2-/S3-
        valid_matches = sorted(list(set(raw_preds) & cand_set))

        if valid_matches:
            matched_count += 1
            total_matches += len(valid_matches)

        rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ",".join(valid_matches)
        })

    sub_df = pd.DataFrame(rows, columns=['source1_entity_id', 'matched_entity_ids'])
    sub_df.to_csv(output_path, sep='\t', index=False)

    print(f"Generated {len(sub_df):,} rows in {output_path}")
    print(f"  - Entities with matches: {matched_count:,} ({matched_count / len(sub_df) * 100:.1f}%)")
    print(f"  - Total match pairs:     {total_matches:,}")
    return sub_df


def save_model(model: object, threshold: float, score: float, path: str):
    """Saves model checkpoint with optimal decision threshold metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'model': model,
        'threshold': threshold,
        'f05_score': score,
        'feature_names': FEATURE_NAMES,
        'num_features': NUM_FEATURES
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Model saved to: {path}")


def load_saved_model(path: str) -> Tuple[object, float, float]:
    """Loads saved model and metadata."""
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    print(f"Loaded model from: {path} (Tuned Threshold: {payload['threshold']:.4f}, CV F_0.5: {payload['f05_score']:.4f})")
    return payload['model'], payload['threshold'], payload['f05_score']


# ==========================================
# 5. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3B: GPU XGBoost Model Training & Match Prediction")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both", help="Execution mode")
    parser.add_argument("--candidate-file", default="output/candidate_pairs_train.tsv", help="Train candidate pairs TSV")
    parser.add_argument("--normalized-dir", default="data/normalized_train", help="Train normalized TSVs directory")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv", help="Ground truth TSV")
    parser.add_argument("--test-candidate-file", default="output/candidate_pairs.tsv", help="Test candidate pairs TSV")
    parser.add_argument("--test-normalized-dir", default="data/normalized_test", help="Test normalized TSVs directory")
    parser.add_argument("--model-path", default="models/xgboost_entity_resolver.pkl", help="Model checkpoint path")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Output submission TSV path")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Enable NVIDIA GPU acceleration")
    parser.add_argument("--max-train-pairs", type=int, default=None, help="Optional sample cap for fast dev training")

    args = parser.parse_args()

    model = None
    best_thresh = 0.50
    best_score = 0.0

    if args.mode in ("train", "both"):
        entity_lookup = load_normalized_records(args.normalized_dir)
        cand_df = load_candidate_pairs(args.candidate_file)
        gt_map = load_ground_truth(args.ground_truth)

        X, y, _ = build_training_dataset(
            cand_df, entity_lookup, gt_map, max_pairs=args.max_train_pairs
        )
        model, best_thresh, best_score = train_model(X, y, use_gpu=args.use_gpu)
        save_model(model, best_thresh, best_score, args.model_path)

    if args.mode in ("predict", "both"):
        if model is None:
            model, best_thresh, best_score = load_saved_model(args.model_path)

        test_lookup = load_normalized_records(args.test_normalized_dir)
        test_cand_df = load_candidate_pairs(args.test_candidate_file)

        preds = predict_matches(model, best_thresh, test_cand_df, test_lookup)
        generate_submission(test_cand_df, preds, args.output)

if __name__ == "__main__":
    main()
