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
from typing import Generator, Tuple, Dict, Set, List, Optional, Any
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
# 2. MULTI-KEY BLOCKING ENGINE (BALANCED S2 & S3)
# ==========================================

TLD_PATTERN = re.compile(r'\b(com|org|net|co\s*in|in|biz|info|io|gov)\b', re.IGNORECASE)
STREET_NUM_PATTERN = re.compile(r'\b\d{1,6}\b')


def extract_street_number(addr: str) -> str:
    """Extracts first numerical street number from clean address."""
    if not addr or not isinstance(addr, str):
        return ""
    m = STREET_NUM_PATTERN.search(addr)
    return m.group(0) if m else ""


def get_compact_name(name: str) -> str:
    """Strips common domain extensions and spaces to match domain-style company names."""
    if not name or not isinstance(name, str):
        return ""
    stripped = TLD_PATTERN.sub('', name.lower())
    compact = re.sub(r'[^a-z0-9]', '', stripped)
    return compact[:10] if len(compact) >= 5 else ""


def build_source_index(df_target: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds high-recall inverted index structures for a specific target source (S2 or S3).
    Includes:
      1. Exact Clean Name
      2. Compact Domain Name Prefix (squashed spaces & TLD stripped)
      3. Joint 2-Token Key (order invariant)
      4. Compound Postal + Token
      5. Street Number + Token
      6. Significant Name Tokens (length >= 3)
      7. Phonetic Soundex
    """
    target_ids = df_target['entity_id'].values

    exact_name_idx: Dict[str, List[int]] = {}
    compact_name_idx: Dict[str, List[int]] = {}
    pair_tok_idx: Dict[str, List[int]] = {}
    compound_postal_idx: Dict[str, List[int]] = {}
    street_num_idx: Dict[str, List[int]] = {}
    word_idx: Dict[str, List[int]] = {}
    soundex_idx: Dict[str, List[int]] = {}

    for idx, row in enumerate(df_target.itertuples()):
        name = str(getattr(row, 'business_name_clean', '') or '').strip()
        addr = str(getattr(row, 'business_address_clean', '') or '').strip()
        postal = str(getattr(row, 'postal_code', '') or '').strip()
        tokens = str(getattr(row, 'name_tokens', '') or name).split()
        sig_tokens = sorted([w for w in set(tokens) if len(w) >= 3])
        s_num = extract_street_number(addr)

        if name:
            exact_name_idx.setdefault(name, []).append(idx)

        comp = get_compact_name(name)
        if comp:
            compact_name_idx.setdefault(comp, []).append(idx)

        # 2-token joint key (sorted so word order doesn't matter)
        for i in range(len(sig_tokens)):
            for j in range(i + 1, min(i + 4, len(sig_tokens))):
                pair_tok_idx.setdefault(f"{sig_tokens[i]}_{sig_tokens[j]}", []).append(idx)

        for w in sig_tokens:
            word_idx.setdefault(w, []).append(idx)
            if postal:
                compound_postal_idx.setdefault(f"{postal}_{w}", []).append(idx)
            if s_num:
                street_num_idx.setdefault(f"{s_num}_{w}", []).append(idx)
            sx = get_soundex(w)
            if sx:
                soundex_idx.setdefault(sx, []).append(idx)

    index_bundle = {
        'exact': exact_name_idx,
        'compact': compact_name_idx,
        'pairs': pair_tok_idx,
        'compound': compound_postal_idx,
        'street': street_num_idx,
        'words': word_idx,
        'soundex': soundex_idx,
    }
    return target_ids, index_bundle


def retrieve_candidates_from_source(
    s1_row: Any,
    target_ids: np.ndarray,
    index: Dict[str, Any],
    max_cands: int = 20,
) -> Set[str]:
    """Retrieves top candidates from a single target source using multi-key matching."""
    s1_id = str(s1_row.entity_id).strip()
    name = str(getattr(s1_row, 'business_name_clean', '') or '').strip()
    addr = str(getattr(s1_row, 'business_address_clean', '') or '').strip()
    postal = str(getattr(s1_row, 'postal_code', '') or '').strip()
    tokens = str(getattr(s1_row, 'name_tokens', '') or name).split()
    sig_tokens = sorted([w for w in set(tokens) if len(w) >= 3])
    s_num = extract_street_number(addr)
    comp = get_compact_name(name)

    cands: Set[str] = set()

    # 1. Exact clean name (highest signal)
    if name:
        for tidx in index['exact'].get(name, [])[:15]:
            cands.add(target_ids[tidx])

    # 2. Compact domain name prefix
    if comp:
        for tidx in index['compact'].get(comp, [])[:15]:
            cands.add(target_ids[tidx])

    # 3. Two-token joint pair match (catches word order swaps)
    for i in range(len(sig_tokens)):
        for j in range(i + 1, min(i + 4, len(sig_tokens))):
            for tidx in index['pairs'].get(f"{sig_tokens[i]}_{sig_tokens[j]}", [])[:10]:
                cands.add(target_ids[tidx])
                if len(cands) >= max_cands:
                    break

    # 4. Compound postal + token
    if postal and len(cands) < max_cands:
        for w in sig_tokens:
            for tidx in index['compound'].get(f"{postal}_{w}", [])[:10]:
                cands.add(target_ids[tidx])

    # 5. Street number + token
    if s_num and len(cands) < max_cands:
        for w in sig_tokens:
            for tidx in index['street'].get(f"{s_num}_{w}", [])[:10]:
                cands.add(target_ids[tidx])

    # 6. Rare words (sorted by target frequency ascending)
    if len(cands) < max_cands:
        sorted_tokens = sorted(sig_tokens, key=lambda w: len(index['words'].get(w, [])))
        for w in sorted_tokens:
            hits = index['words'].get(w, [])
            if hits:
                for tidx in hits[:15]:
                    cands.add(target_ids[tidx])
                if len(cands) >= max_cands:
                    break

    # 7. Phonetic Soundex fallback
    if len(cands) < 10:
        for w in sig_tokens[:2]:
            sx = get_soundex(w)
            if sx:
                for tidx in index['soundex'].get(sx, [])[:5]:
                    cands.add(target_ids[tidx])

    return {c for c in cands if c != s1_id}


def generate_candidates_for_country_balanced(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = 40,
) -> Dict[str, Set[str]]:
    """
    Generates balanced high-recall candidates for each S1 entity in a country partition.
    Allocates candidates evenly between Source 2 and Source 3 (e.g. top 20 from S2, top 20 from S3).
    """
    s1_candidates: Dict[str, Set[str]] = {str(s1_id).strip(): set() for s1_id in df_s1['entity_id']}

    if len(df_s1) == 0:
        return s1_candidates

    per_source_k = max(top_k // 2, 15)

    print(f"  [1/3] Indexing Source 2 ({len(df_s2):,} records)...")
    s2_ids, s2_index = build_source_index(df_s2) if len(df_s2) > 0 else (np.array([]), {})

    print(f"  [2/3] Indexing Source 3 ({len(df_s3):,} records)...")
    s3_ids, s3_index = build_source_index(df_s3) if len(df_s3) > 0 else (np.array([]), {})

    print(f"  [3/3] Parallel candidate retrieval for {len(df_s1):,} S1 entities (S2 & S3 balanced)...")
    s1_rows = list(df_s1.itertuples())

    def process_s1_batch(batch):
        batch_res = {}
        for s1_row in batch:
            s1_id = str(s1_row.entity_id).strip()
            cands_s2 = retrieve_candidates_from_source(s1_row, s2_ids, s2_index, max_cands=per_source_k) if len(s2_ids) > 0 else set()
            cands_s3 = retrieve_candidates_from_source(s1_row, s3_ids, s3_index, max_cands=per_source_k) if len(s3_ids) > 0 else set()
            batch_res[s1_id] = cands_s2 | cands_s3
        return batch_res

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
    top_k: int = 40,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Executes country-partitioned balanced multi-key blocking across S1, S2, and S3.
    Guarantees that neither Source 2 nor Source 3 is starved out.
    """
    for df in (df_s1, df_s2, df_s3):
        if 'country' not in df.columns:
            df['country'] = 'UNKNOWN'

    countries = df_s1['country'].astype(str).str.upper().unique()
    all_candidate_map: Dict[str, Set[str]] = {}

    for c in countries:
        sub_s1 = df_s1[df_s1['country'].astype(str).str.upper() == c]
        sub_s2 = df_s2[df_s2['country'].astype(str).str.upper() == c]
        sub_s3 = df_s3[df_s3['country'].astype(str).str.upper() == c]

        if len(sub_s2) == 0:
            sub_s2 = df_s2
        if len(sub_s3) == 0:
            sub_s3 = df_s3

        print(f"\nBlocking country '{c}': {len(sub_s1):,} S1 vs ({len(sub_s2):,} S2, {len(sub_s3):,} S3)...")
        country_candidates = generate_candidates_for_country_balanced(sub_s1, sub_s2, sub_s3, top_k=top_k)
        all_candidate_map.update(country_candidates)

    rows = []
    for s1_id in df_s1['entity_id']:
        s1_id_str = str(s1_id).strip()
        cand_set = all_candidate_map.get(s1_id_str, set())
        valid_cands = sorted([cid for cid in cand_set if cid.startswith(('S2-', 'S3-')) and cid != s1_id_str])
        rows.append({
            'source1_entity_id': s1_id_str,
            'candidate_entity_ids': ",".join(valid_cands)
        })

    candidate_df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_ids'])
    return candidate_df


# Alias for pipeline compatibility
generate_candidates_by_country = generate_candidate_pairs
generate_candidates_for_country = generate_candidate_pairs


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
