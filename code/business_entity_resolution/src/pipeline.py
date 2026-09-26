# End-to-End Pipeline Execution
import os
import argparse
import pandas as pd

try:
    from .preprocessing import preprocess_tsv
    from .blocking import generate_candidates_by_country
    from .train_model import filter_matches_with_subset_constraint
except ImportError:
    from preprocessing import preprocess_tsv
    from blocking import generate_candidates_by_country
    from train_model import filter_matches_with_subset_constraint

def run_pipeline(data_dir: str = "dataset/test", output_dir: str = "output", top_k: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Stage 1: Preprocessing Input Datasets ===")
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    if not os.path.isfile(s1_path):
        s1_path = os.path.join(data_dir, "train_source1.tsv")
        s2_path = os.path.join(data_dir, "train_source2.tsv")
        s3_path = os.path.join(data_dir, "train_source3.tsv")
    else:
        s2_path = os.path.join(data_dir, "test_source2.tsv")
        s3_path = os.path.join(data_dir, "test_source3.tsv")

    df_s1 = preprocess_tsv(s1_path)
    df_s2 = preprocess_tsv(s2_path)
    df_s3 = preprocess_tsv(s3_path)
    
    print("=== Stage 2: Candidate Generation (Blocking) ===")
    candidate_df = generate_candidates_by_country(df_s1, df_s2, df_s3, top_k=top_k)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidate_df.to_csv(candidate_path, sep="\t", index=False)
    print(f"Saved candidate pairs to {candidate_path}")
    
    print("=== Stage 3: Matching Results Generation (Subset Constraint Enforced) ===")
    matching_df = filter_matches_with_subset_constraint(candidate_df, {})
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    matching_df.to_csv(matching_path, sep="\t", index=False)
    print(f"Saved matching results to {matching_path}")
    print("=== Pipeline Complete! Ready for Submission Validation ===")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset/test", help="Folder with test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default="output", help="Folder to write TSV outputs")
    parser.add_argument("--top-k", type=int, default=30, help="Top K candidates for blocking")
    args = parser.parse_args()
    
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, top_k=args.top_k)

if __name__ == "__main__":
    main()
