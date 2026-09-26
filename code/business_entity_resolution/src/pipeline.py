"""
End-to-End Business Entity Resolution Pipeline
Amazon ML Challenge 2026

Orchestrates all 3 stages:
  - Stage 1 (Person 1): Preprocessing & Normalization (preprocessing.py)
  - Stage 2 (Person 2): Candidate Generation & Blocking (blocking.py)
  - Stage 3A (Person 3A): Pairwise Feature Extraction (features.py)
  - Stage 3B (Person 3B): Model Inference & F_0.5 Thresholding (train_predict.py)
  - Validation: utils/validate_submission.py
"""

import os
import sys
import argparse
import subprocess
from typing import Dict, Any
import pandas as pd

# Support both module and direct script execution
try:
    from .preprocessing import preprocess_tsv, normalize_dataframe
    from .blocking import generate_candidate_pairs
    from .features import extract_features
    from .train_predict import (
        load_saved_model, predict_matches, generate_submission,
        get_feature_function, train_model, generate_synthetic_training_data,
        DEFAULT_MODEL_DIR, save_model
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import preprocess_tsv, normalize_dataframe
    from blocking import generate_candidate_pairs
    from features import extract_features
    from train_predict import (
        load_saved_model, predict_matches, generate_submission,
        get_feature_function, train_model, generate_synthetic_training_data,
        DEFAULT_MODEL_DIR, save_model
    )


def dataframe_to_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Converts a normalized DataFrame into an entity lookup dictionary."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in df.itertuples():
        eid = str(getattr(row, 'entity_id', '')).strip()
        if eid:
            lookup[eid] = {
                'entity_id': eid,
                'business_name_clean': str(getattr(row, 'business_name_clean', '') or ''),
                'business_address_clean': str(getattr(row, 'business_address_clean', '') or ''),
                'country': str(getattr(row, 'country', '') or ''),
                'postal_code': str(getattr(row, 'postal_code', '') or ''),
                'name_tokens': str(getattr(row, 'name_tokens', '') or ''),
            }
    return lookup


def run_pipeline(
    data_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = None,
    top_k: int = 30,
    validate: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    print("=" * 72)
    print(" Amazon ML Challenge 2026: End-to-End Pipeline Execution")
    print(f" Data Directory:   {data_dir}")
    print(f" Output Directory: {output_dir}")
    print("=" * 72)

    # ----------------------------------------------------
    # Stage 1: Preprocessing & Normalization
    # ----------------------------------------------------
    print("\n>>> Stage 1: Preprocessing & Normalization (Person 1) <<<")
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s2_path = os.path.join(data_dir, "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test_source3.tsv")

    if not os.path.isfile(s1_path):
        s1_path = os.path.join(data_dir, "train_source1.tsv")
        s2_path = os.path.join(data_dir, "train_source2.tsv")
        s3_path = os.path.join(data_dir, "train_source3.tsv")

    if not os.path.isfile(s1_path):
        print(f"ERROR: Cannot find source files in {data_dir}")
        sys.exit(1)

    print(f"  [1/3] Normalizing Source 1: {s1_path}")
    df_s1 = preprocess_tsv(s1_path)
    print(f"        -> {len(df_s1):,} records")

    print(f"  [2/3] Normalizing Source 2: {s2_path}")
    df_s2 = preprocess_tsv(s2_path)
    print(f"        -> {len(df_s2):,} records")

    print(f"  [3/3] Normalizing Source 3: {s3_path}")
    df_s3 = preprocess_tsv(s3_path)
    print(f"        -> {len(df_s3):,} records")

    # ----------------------------------------------------
    # Stage 2: Candidate Generation (Blocking)
    # ----------------------------------------------------
    print("\n>>> Stage 2: Candidate Generation & Blocking (Person 2) <<<")
    candidate_df = generate_candidate_pairs(df_s1, df_s2, df_s3, top_k=top_k)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidate_df.to_csv(candidate_path, sep="\t", index=False)
    print(f"  [OK] Saved candidate pairs: {len(candidate_df):,} rows to {candidate_path}")

    # ----------------------------------------------------
    # Stage 3: Feature Extraction & Model Inference
    # ----------------------------------------------------
    print("\n>>> Stage 3: Model Inference & Submission Generation (Person 3A & 3B) <<<")

    # Resolve or train model
    resolved_model_path = model_path or os.path.join(DEFAULT_MODEL_DIR, "entity_resolution_model.pkl")
    if not os.path.isfile(resolved_model_path):
        print(f"  Model not found at {resolved_model_path}. Training a fast model on synthetic data...")
        X_syn, y_syn = generate_synthetic_training_data(n_samples=5000)
        model, threshold, f05 = train_model(X_syn, y_syn, backend="sklearn", n_folds=3)
        resolved_model_path = save_model(model, threshold, f05, DEFAULT_MODEL_DIR, "sklearn")
    else:
        model, threshold, backend = load_saved_model(resolved_model_path)

    # Build entity lookup from normalized dataframes
    print("  Building combined entity lookup from normalized records...")
    entity_lookup: Dict[str, Dict[str, Any]] = {}
    entity_lookup.update(dataframe_to_entity_lookup(df_s1))
    entity_lookup.update(dataframe_to_entity_lookup(df_s2))
    entity_lookup.update(dataframe_to_entity_lookup(df_s3))
    print(f"  Lookup ready with {len(entity_lookup):,} total entities.")

    # Feature extraction function (Person 3A)
    feature_fn = get_feature_function(use_mock=False)

    # Run inference
    predicted_matches = predict_matches(candidate_df, entity_lookup, model, threshold, feature_fn)

    # Generate final submission TSV
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    generate_submission(candidate_df, predicted_matches, matching_path)

    print("\n" + "=" * 72)
    print(" [OK] End-to-End Pipeline Execution COMPLETE!")
    print(f"   1. Candidate Pairs:   {os.path.abspath(candidate_path)}")
    print(f"   2. Matching Results:  {os.path.abspath(matching_path)}")
    print("=" * 72)

    # ----------------------------------------------------
    # Validation against official submission validator
    # ----------------------------------------------------
    if validate:
        validator_script = os.path.join(workspace_root, "utils", "validate_submission.py")
        if os.path.isfile(validator_script):
            print("\n>>> Running Official Submission Validator (utils/validate_submission.py) <<<")
            val_cmd = [
                sys.executable, validator_script,
                "--matching", matching_path,
                "--candidate", candidate_path,
                "--test-dir", data_dir
            ]
            try:
                subprocess.run(val_cmd, check=False)
            except Exception as e:
                print(f"  Validator could not run: {e}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset/test", help="Folder containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default="output", help="Folder to write TSV outputs")
    parser.add_argument("--model-path", default=None, help="Path to pre-trained model (.pkl)")
    parser.add_argument("--top-k", type=int, default=30, help="Top K candidates for blocking")
    parser.add_argument("--no-validate", action="store_true", help="Skip validator script check")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        top_k=args.top_k,
        validate=not args.no_validate
    )


if __name__ == "__main__":
    main()
