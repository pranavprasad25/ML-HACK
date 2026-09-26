# Stage 2: Inverted Indexing & Candidate Generation (Blocking)
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def block_country_subset(df_s1: pd.DataFrame, df_targets: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    """Generates candidate pairs for entities within the same country."""
    if len(df_s1) == 0 or len(df_targets) == 0:
        return pd.DataFrame({'source1_entity_id': df_s1['entity_id'], 'candidate_entity_ids': ""})

    # TF-IDF Vectorizer on core_name using character 3-4 n-grams
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
    
    # Fit vectorizer on Target entities (S2 and S3)
    target_names = df_targets['core_name'].fillna("")
    s1_names = df_s1['core_name'].fillna("")
    
    X_target = vectorizer.fit_transform(target_names)
    X_s1 = vectorizer.transform(s1_names)
    
    # Cap k if target dataset is smaller than top_k
    k = min(top_k, X_target.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(X_target)
    distances, indices = nn.kneighbors(X_s1)
    
    target_ids = df_targets['entity_id'].values
    candidate_records = []
    
    for i, s1_id in enumerate(df_s1['entity_id']):
        matched_target_ids = [target_ids[idx] for idx in indices[i]]
        # Filter out any accidental S1 self-matches and deduplicate while keeping order
        filtered_ids = []
        seen = set()
        for tid in matched_target_ids:
            if tid.startswith(('S2-', 'S3-')) and tid not in seen:
                seen.add(tid)
                filtered_ids.append(tid)
                
        candidate_records.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ",".join(filtered_ids)
        })
        
    return pd.DataFrame(candidate_records)

def generate_candidates_by_country(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    """Partition by country and run blocking to enforce country boundary rules."""
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # Ensure country column exists
    if 'country' not in df_s1.columns:
        df_s1['country'] = 'UNKNOWN'
    if 'country' not in df_targets.columns:
        df_targets['country'] = 'UNKNOWN'
        
    countries = df_s1['country'].str.upper().unique()
    results = []
    
    for c in countries:
        sub_s1 = df_s1[df_s1['country'].str.upper() == c]
        sub_target = df_targets[df_targets['country'].str.upper() == c]
        
        if len(sub_target) == 0:
            # Fallback to full target set if country is unseen or unmatched
            sub_target = df_targets
            
        print(f"Blocking country '{c}': {len(sub_s1)} S1 entities against {len(sub_target)} target entities...")
        res_df = block_country_subset(sub_s1, sub_target, top_k=top_k)
        results.append(res_df)
        
    final_candidates = pd.concat(results, ignore_index=True)
    
    # Guarantee that every single S1 entity is present in exact original order
    s1_order_df = pd.DataFrame({'source1_entity_id': df_s1['entity_id']})
    final_candidates = s1_order_df.merge(final_candidates, on='source1_entity_id', how='left').fillna({'candidate_entity_ids': ""})
    return final_candidates

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Candidate Generation & Blocking")
    parser.add_argument("--normalized-dir", required=True, help="Folder containing normalized TSVs")
    parser.add_argument("--output", required=True, help="Output path for candidate_pairs.tsv")
    parser.add_argument("--top-k", type=int, default=30, help="Top K candidates per S1 entity")
    args = parser.parse_args()

    # Load normalized TSVs
    s1_path = os.path.join(args.normalized_dir, "test_source1.tsv")
    if not os.path.isfile(s1_path):
        s1_path = os.path.join(args.normalized_dir, "train_source1.tsv")
        s2_path = os.path.join(args.normalized_dir, "train_source2.tsv")
        s3_path = os.path.join(args.normalized_dir, "train_source3.tsv")
    else:
        s2_path = os.path.join(args.normalized_dir, "test_source2.tsv")
        s3_path = os.path.join(args.normalized_dir, "test_source3.tsv")

    print(f"Reading normalized data from {args.normalized_dir}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")

    candidates_df = generate_candidates_by_country(df_s1, df_s2, df_s3, top_k=args.top_k)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    candidates_df.to_csv(args.output, sep="\t", index=False)
    print(f"Saved candidate pairs to {args.output} successfully.")

if __name__ == "__main__":
    main()
