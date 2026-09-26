"""
Stage 2: Candidate Generation & Blocking Module (Person 2)
Amazon ML Challenge 2026: Business Entity Resolution

This module filters millions of potential comparisons down to a high-recall candidate pool
(top 25-50 candidates per S1 entity) using country partitioning, multi-key inverted indexing
(postal code, first name token, compound postal+token, phonetic Soundex), and TF-IDF character n-gram Nearest Neighbors.
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
    top_k: int = 30,
    use_gpu: bool = False
) -> Dict[str, Set[str]]:
    """
    Generates high-recall candidate target IDs (S2/S3) for each S1 entity in a country partition.
    Uses multi-threaded parallel queries across all CPU cores with multi-key inverted indexing:
      1. Compound Rule: Exact Postal Code + Name Token match
      2. Significant Word Rule: Name tokens (length >= 3)
      3. Exact Postal Rule: Exact Postal Code match
      4. Phonetic Rule: Soundex code on leading tokens
      5. Prefix Rule: 3-character prefix match
    """
    s1_candidates: Dict[str, Set[str]] = {str(s1_id).strip(): set() for s1_id in df_s1['entity_id']}
    
    if len(df_s1) == 0 or len(df_targets) == 0:
        return s1_candidates

    target_ids = df_targets['entity_id'].values

    print(f"  [1/2] Indexing {len(df_targets):,} target entities...")
    postal_index: Dict[str, List[int]] = {}
    word_index: Dict[str, List[int]] = {}
    compound_index: Dict[str, List[int]] = {}
    soundex_index: Dict[str, List[int]] = {}
    prefix_index: Dict[str, List[int]] = {}

    for idx, row in enumerate(df_targets.itertuples()):
        postal = str(getattr(row, 'postal_code', '') or '').strip()
        tokens = str(getattr(row, 'name_tokens', '') or getattr(row, 'business_name_clean', '') or '').split()

        if postal:
            postal_index.setdefault(postal, []).append(idx)

        for w in tokens[:4]:
            if len(w) >= 3:
                word_index.setdefault(w, []).append(idx)
                if postal:
                    compound_index.setdefault(f"{postal}_{w}", []).append(idx)
                sx = get_soundex(w)
                if sx:
                    soundex_index.setdefault(sx, []).append(idx)
                prefix_index.setdefault(w[:3], []).append(idx)

    print(f"  [2/2] Parallel candidate retrieval for {len(df_s1):,} S1 entities...")
    s1_rows = list(df_s1.itertuples())

    def process_s1_batch(batch):
        batch_res = {}
        for s1_row in batch:
            s1_id = str(s1_row.entity_id).strip()
            postal = str(getattr(s1_row, 'postal_code', '') or '').strip()
            tokens = str(getattr(s1_row, 'name_tokens', '') or getattr(s1_row, 'business_name_clean', '') or '').split()
            cands = set()

            # 1. Compound Rule: Postal + Word exact match (highest precision)
            if postal:
                for w in tokens[:4]:
                    for tidx in compound_index.get(f"{postal}_{w}", [])[:20]:
                        cands.add(target_ids[tidx])
                        if len(cands) >= top_k:
                            break

            # 2. Significant word matches (length >= 3)
            if len(cands) < top_k:
                for w in tokens[:3]:
                    if len(w) >= 3:
                        for tidx in word_index.get(w, [])[:15]:
                            cands.add(target_ids[tidx])
                            if len(cands) >= top_k:
                                break

            # 3. Exact Postal Code match
            if postal and len(cands) < top_k:
                for tidx in postal_index.get(postal, [])[:10]:
                    cands.add(target_ids[tidx])
                    if len(cands) >= top_k:
                        break

            # 4. Phonetic Soundex match
            if len(cands) < 15:
                for w in tokens[:2]:
                    sx = get_soundex(w)
                    if sx:
                        for tidx in soundex_index.get(sx, [])[:5]:
                            cands.add(target_ids[tidx])

            # 5. Prefix 3 match
            if len(cands) < 10:
                for w in tokens[:2]:
                    if len(w) >= 3:
                        for tidx in prefix_index.get(w[:3], [])[:5]:
                            cands.add(target_ids[tidx])

            valid = {c for c in cands if c.startswith(('S2-', 'S3-')) and c != s1_id}
            batch_res[s1_id] = valid
        return batch_res

    # Multi-threaded querying across all cores
    from concurrent.futures import ThreadPoolExecutor
    workers = min(os.cpu_count() or 4, 16)
    chunk_size = (len(s1_rows) + workers - 1) // workers
    chunks = [s1_rows[i * chunk_size : (i + 1) * chunk_size] for i in range(workers) if i * chunk_size < len(s1_rows)]

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        results = list(executor.map(process_s1_batch, chunks))

    for res in results:
        s1_candidates.update(res)

    return s1_candidates


def generate_candidate_pairs(
    df_s1: pd.DataFrame, 
    df_s2: pd.DataFrame, 
    df_s3: pd.DataFrame, 
    top_k: int = 30,
    use_gpu: bool = False
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

    countries = df_s1['country'].astype(str).str.upper().unique()
    all_candidate_map: Dict[str, Set[str]] = {}

    for c in countries:
        sub_s1 = df_s1[df_s1['country'].astype(str).str.upper() == c]
        sub_target = df_targets[df_targets['country'].astype(str).str.upper() == c]

        if len(sub_target) == 0:
            sub_target = df_targets

        print(f"Blocking country '{c}': {len(sub_s1):,} S1 entities against {len(sub_target):,} target entities...")
        country_candidates = generate_candidates_for_country(sub_s1, sub_target, top_k=top_k, use_gpu=use_gpu)
        all_candidate_map.update(country_candidates)

    rows = []
    for s1_id in df_s1['entity_id']:
        s1_id_str = str(s1_id).strip()
        cand_set = all_candidate_map.get(s1_id_str, set())
        # Filter strictly S2- and S3- prefixed IDs, drop self matches, sort deterministically
        valid_cands = sorted([cid for cid in cand_set if cid.startswith(('S2-', 'S3-')) and cid != s1_id_str])
        rows.append({
            'source1_entity_id': s1_id_str,
            'candidate_entity_ids': ",".join(valid_cands)
        })

    candidate_df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_ids'])
    return candidate_df


# Alias for pipeline compatibility
generate_candidates_by_country = generate_candidate_pairs


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
            s1_id = parts[0].strip()
            cand_str = parts[1].strip() if len(parts) > 1 else ""
            if cand_str:
                for cand_id in cand_str.split(','):
                    cand_id_clean = cand_id.strip()
                    if cand_id_clean:
                        yield (s1_id, cand_id_clean)


# ==========================================
# 5. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Candidate Generation & Blocking")
    parser.add_argument("--normalized-dir", required=True, help="Directory containing normalized TSVs from Stage 1")
    parser.add_argument("--output", required=True, help="Output path for candidate_pairs.tsv")
    parser.add_argument("--top-k", type=int, default=30, help="Top K candidates per S1 entity (default: 30)")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration if available")
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

    candidates_df = generate_candidate_pairs(df_s1, df_s2, df_s3, top_k=args.top_k, use_gpu=args.use_gpu)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    candidates_df.to_csv(args.output, sep="\t", index=False)
    print(f"Successfully generated candidate pairs: {len(candidates_df):,} rows saved to {args.output}")

    if args.ground_truth and os.path.isfile(args.ground_truth):
        gt_df = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna("")
        evaluate_blocking_recall(candidates_df, gt_df)

if __name__ == "__main__":
    main()
