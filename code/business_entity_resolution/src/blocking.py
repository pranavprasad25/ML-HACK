"""
Stage 2: Candidate Generation & Blocking Module (Person 2)
Amazon ML Challenge 2026: Business Entity Resolution

This module filters millions of potential comparisons down to a small, high-recall
candidate pool (top 15-30 candidates per S1 entity) using multi-key inverted indexing,
country partitioning, and TF-IDF character n-gram Nearest Neighbors.
"""

import os
import re
import argparse
from typing import Generator, Tuple, Dict, Set, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. PHONETIC INDEXING HELPERS (Soundex)
# ==========================================

def get_soundex(word: str) -> str:
    """Computes standard Soundex phonetic code for a string."""
    if not word or not isinstance(word, str):
        return ""
    word_clean = re.sub(r'[^a-zA-Z]', '', word).upper()
    if not word_clean:
        return ""
    
    first_letter = word_clean[0]
    char_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    codes = [first_letter]
    last_code = char_map.get(first_letter, '')
    
    for char in word_clean[1:]:
        code = char_map.get(char, '')
        if code and code != last_code:
            codes.append(code)
            last_code = code
        elif not code:
            last_code = ''
            
    return "".join(codes).ljust(4, '0')[:4]

# ==========================================
# 2. MULTI-KEY BLOCKING ENGINE
# ==========================================

def generate_candidates_for_country(
    df_s1: pd.DataFrame, 
    df_targets: pd.DataFrame, 
    top_k: int = 30
) -> Dict[str, Set[str]]:
    """
    Generates high-recall candidate target IDs (S2/S3) for each S1 entity in a country partition.
    Combines:
    1. Exact Blocking Rule: Country + Postal Code
    2. Exact Blocking Rule: Country + First Word of name_tokens
    3. Phonetic Soundex Indexing
    4. TF-IDF Character 3-5 Gram Cosine Nearest Neighbors (Name + Address)
    """
    s1_candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in df_s1['entity_id']}
    
    if len(df_s1) == 0 or len(df_targets) == 0:
        return s1_candidates

    target_ids = df_targets['entity_id'].values

    # ----------------------------------------------------
    # Inverted Index 1: Postal Code & First Word Blocking
    # ----------------------------------------------------
    postal_index: Dict[str, List[int]] = {}
    first_word_index: Dict[str, List[int]] = {}
    soundex_index: Dict[str, List[int]] = {}

    for idx, row in enumerate(df_targets.itertuples()):
        postal = getattr(row, 'postal_code', '')
        tokens = str(getattr(row, 'name_tokens', '')).split()
        first_word = tokens[0] if tokens else ""

        if postal:
            postal_index.setdefault(postal, []).append(idx)
        if first_word:
            first_word_index.setdefault(first_word, []).append(idx)
            soundex_key = get_soundex(first_word)
            if soundex_key:
                soundex_index.setdefault(soundex_key, []).append(idx)

    # Fast Inverted Index Lookup for S1 entities
    for s1_row in df_s1.itertuples():
        s1_id = s1_row.entity_id
        postal = getattr(s1_row, 'postal_code', '')
        tokens = str(getattr(s1_row, 'name_tokens', '')).split()
        first_word = tokens[0] if tokens else ""

        # Exact Rule A: Postal Code match
        if postal and postal in postal_index:
            for target_idx in postal_index[postal][:15]:
                tid = target_ids[target_idx]
                if tid.startswith(('S2-', 'S3-')):
                    s1_candidates[s1_id].add(tid)

        # Exact Rule B: First Word match
        if first_word and first_word in first_word_index:
            for target_idx in first_word_index[first_word][:15]:
                tid = target_ids[target_idx]
                if tid.startswith(('S2-', 'S3-')):
                    s1_candidates[s1_id].add(tid)

        # Exact Rule C: Phonetic Soundex match
        if first_word and len(s1_candidates[s1_id]) < 10:
            soundex_key = get_soundex(first_word)
            if soundex_key in soundex_index:
                for target_idx in soundex_index[soundex_key][:10]:
                    tid = target_ids[target_idx]
                    if tid.startswith(('S2-', 'S3-')):
                        s1_candidates[s1_id].add(tid)

    # ----------------------------------------------------
    # Vector Search: TF-IDF Character 3-5 Grams on (Name + Address)
    # ----------------------------------------------------
    s1_text = (df_s1['business_name_clean'].fillna("") + " " + df_s1['business_address_clean'].fillna("")).values
    target_text = (df_targets['business_name_clean'].fillna("") + " " + df_targets['business_address_clean'].fillna("")).values

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    X_target = vectorizer.fit_transform(target_text)
    X_s1 = vectorizer.transform(s1_text)

    k = min(top_k, X_target.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(X_target)
    distances, indices = nn.kneighbors(X_s1)

    s1_id_list = df_s1['entity_id'].values
    for i, s1_id in enumerate(s1_id_list):
        for idx in indices[i]:
            tid = target_ids[idx]
            if tid.startswith(('S2-', 'S3-')):
                s1_candidates[s1_id].add(tid)

    return s1_candidates


def generate_candidate_pairs(
    df_s1: pd.DataFrame, 
    df_s2: pd.DataFrame, 
    df_s3: pd.DataFrame, 
    top_k: int = 30
) -> pd.DataFrame:
    """
    Executes country-partitioned multi-key blocking across S1, S2, and S3.
    Ensures strict formatting rules enforced by validate_submission.py.
    """
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    if 'country' not in df_s1.columns:
        df_s1['country'] = 'UNKNOWN'
    if 'country' not in df_targets.columns:
        df_targets['country'] = 'UNKNOWN'

    countries = df_s1['country'].str.upper().unique()
    all_candidate_map: Dict[str, Set[str]] = {}

    for c in countries:
        sub_s1 = df_s1[df_s1['country'].str.upper() == c]
        sub_target = df_targets[df_targets['country'].str.upper() == c]

        if len(sub_target) == 0:
            sub_target = df_targets

        print(f"Blocking country '{c}': {len(sub_s1):,} S1 entities against {len(sub_target):,} target entities...")
        country_candidates = generate_candidates_for_country(sub_s1, sub_target, top_k=top_k)
        all_candidate_map.update(country_candidates)

    rows = []
    for s1_id in df_s1['entity_id']:
        cand_set = all_candidate_map.get(s1_id, set())
        valid_cands = [cid for cid in sorted(cand_set) if cid.startswith(('S2-', 'S3-'))]
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ",".join(valid_cands)
        })

    candidate_df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_ids'])
    return candidate_df


# ==========================================
# 3. CANDIDATE RECALL EVALUATION METRIC
# ==========================================

def evaluate_blocking_recall(candidate_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Evaluates blocking recall against ground truth labels:
    Recall = (True Matches Captured in Candidate Pairs) / (Total True Ground Truth Matches)
    """
    print("Evaluating Candidate Blocking Recall against Ground Truth...")
    cand_map = {}
    for _, row in candidate_df.iterrows():
        s1 = row['source1_entity_id']
        cands = set(row['candidate_entity_ids'].split(',')) if row['candidate_entity_ids'] else set()
        cand_map[s1] = cands

    total_true_matches = 0
    captured_true_matches = 0

    for _, row in ground_truth_df.iterrows():
        s1 = row['source1_entity_id']
        true_mids = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
        if not true_mids:
            continue
            
        cands = cand_map.get(s1, set())
        for t_id in true_mids:
            total_true_matches += 1
            if t_id in cands:
                captured_true_matches += 1

    recall = (captured_true_matches / max(total_true_matches, 1)) * 100.0
    print(f"Captured {captured_true_matches:,} / {total_true_matches:,} True Matches")
    print(f"Candidate Blocking Recall: {recall:.2f}%")
    return recall


# ==========================================
# 4. STREAMING ITERATOR HELPER
# ==========================================

def iterate_candidate_pairs(candidates_path: str) -> Generator[Tuple[str, str], None, None]:
    """Streaming generator helper that yields individual (source1_entity_id, candidate_entity_id) tuples."""
    with open(candidates_path, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split('\t')
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                for cand_id in cand_str.split(','):
                    if cand_id:
                        yield (s1_id, cand_id)


# ==========================================
# 5. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Candidate Generation & Blocking")
    parser.add_argument("--normalized-dir", required=True, help="Directory containing normalized TSVs from Stage 1")
    parser.add_argument("--output", required=True, help="Output path for candidate_pairs.tsv")
    parser.add_argument("--top-k", type=int, default=30, help="Top K candidates per S1 entity (default: 30)")
    parser.add_argument("--ground-truth", default=None, help="Optional ground truth TSV to evaluate blocking recall")
    args = parser.parse_args()

    s1_path = os.path.join(args.normalized_dir, "normalized_source1.tsv")
    s2_path = os.path.join(args.normalized_dir, "normalized_source2.tsv")
    s3_path = os.path.join(args.normalized_dir, "normalized_source3.tsv")

    if not os.path.isfile(s1_path):
        s1_path = os.path.join(args.normalized_dir, "test_source1.tsv")
        s2_path = os.path.join(args.normalized_dir, "test_source2.tsv")
        s3_path = os.path.join(args.normalized_dir, "test_source3.tsv")
        
    if not os.path.isfile(s1_path):
        s1_path = os.path.join(args.normalized_dir, "train_source1.tsv")
        s2_path = os.path.join(args.normalized_dir, "train_source2.tsv")
        s3_path = os.path.join(args.normalized_dir, "train_source3.tsv")

    print(f"Reading Stage 1 normalized data from {args.normalized_dir}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")

    candidates_df = generate_candidate_pairs(df_s1, df_s2, df_s3, top_k=args.top_k)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    candidates_df.to_csv(args.output, sep="\t", index=False)
    print(f"Successfully generated candidate pairs: {len(candidates_df):,} rows saved to {args.output}")

    if args.ground_truth and os.path.isfile(args.ground_truth):
        gt_df = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna("")
        evaluate_blocking_recall(candidates_df, gt_df)

if __name__ == "__main__":
    main()
