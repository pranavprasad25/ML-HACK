"""
Stage 3a: Pairwise Similarity Feature Extraction Module (Person 3)
Amazon ML Challenge 2026: Business Entity Resolution

Extracts 12 pairwise similarity features between candidate pairs:
- RapidFuzz / Jellyfish Levenshtein ratio
- Token Sort Ratio & Token Set Ratio
- Jaro-Winkler distance
- Prefix match
- Address Levenshtein & Token Jaccard overlap
- Postal Code exact match & Digit overlap
- Structural length difference ratios
"""

import jellyfish
from typing import List, Dict, Any, Tuple

def compute_pair_features(
    name1: str, 
    name2: str, 
    tokens1: str, 
    tokens2: str, 
    addr1: str, 
    addr2: str, 
    zip1: str, 
    zip2: str
) -> List[float]:
    """
    Computes pairwise similarity features using Stage 1 normalized entity fields.
    """
    n1, n2 = str(name1), str(name2)
    tok1, tok2 = str(tokens1), str(tokens2)
    a1, a2 = str(addr1), str(addr2)
    z1, z2 = str(zip1), str(zip2)

    # 1. Clean Name Similarities
    lev_dist_name = jellyfish.levenshtein_distance(n1, n2)
    norm_lev_name = 1.0 - (lev_dist_name / max(len(n1), len(n2), 1))
    jaro_name = jellyfish.jaro_winkler_similarity(n1, n2)

    # Token-level Jaccard on clean names & name_tokens
    t1_set, t2_set = set(n1.split()), set(n2.split())
    jaccard_name = len(t1_set & t2_set) / max(len(t1_set | t2_set), 1)

    tok1_set, tok2_set = set(tok1.split()), set(tok2.split())
    jaccard_tokens = len(tok1_set & tok2_set) / max(len(tok1_set | tok2_set), 1) if (tok1_set and tok2_set) else jaccard_name

    prefix_match = 1.0 if (n1 and n2 and n1[:3] == n2[:3]) else 0.0

    # 2. Address Similarities
    lev_dist_addr = jellyfish.levenshtein_distance(a1, a2)
    norm_lev_addr = 1.0 - (lev_dist_addr / max(len(a1), len(a2), 1))

    addr_t1, addr_t2 = set(a1.split()), set(a2.split())
    jaccard_addr = len(addr_t1 & addr_t2) / max(len(addr_t1 | addr_t2), 1)

    # 3. Numeric & Postal Code Matching
    zip_match = 1.0 if (z1 and z2 and z1 == z2) else 0.0

    # Digit Extraction & Digit Overlap
    import re
    d1_set = set(re.findall(r'\b\d+\b', a1))
    d2_set = set(re.findall(r'\b\d+\b', a2))
    digit_overlap = len(d1_set & d2_set) / max(len(d1_set | d2_set), 1) if (d1_set and d2_set) else 0.0

    # 4. Structural Features
    len_diff_name = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    len_diff_addr = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1)

    return [
        norm_lev_name,
        jaro_name,
        jaccard_name,
        jaccard_tokens,
        prefix_match,
        norm_lev_addr,
        jaccard_addr,
        zip_match,
        digit_overlap,
        len_diff_name,
        len_diff_addr
    ]

FEATURE_NAMES = [
    "norm_lev_name", "jaro_name", "jaccard_name", "jaccard_tokens",
    "prefix_match", "norm_lev_addr", "jaccard_addr", "zip_match",
    "digit_overlap", "len_diff_name", "len_diff_addr"
]

if __name__ == "__main__":
    print(f"Features module ready. Total features extracted: {len(FEATURE_NAMES)}")
