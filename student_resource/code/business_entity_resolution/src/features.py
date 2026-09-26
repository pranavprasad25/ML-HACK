# Stage 3a: Pairwise Similarity Feature Extraction
# pyrefly: ignore [missing-import]
import jellyfish

def compute_pair_features(name1: str, name2: str, addr1: str, addr2: str, zip1: str, zip2: str, digits1: str = "", digits2: str = ""):
    """Computes comprehensive pairwise similarity features between two entity records."""
    n1, n2 = str(name1), str(name2)
    a1, a2 = str(addr1), str(addr2)
    
    # 1. Name Similarities
    lev_dist_name = jellyfish.levenshtein_distance(n1, n2)
    norm_lev_name = 1.0 - (lev_dist_name / max(len(n1), len(n2), 1))
    jaro_name = jellyfish.jaro_winkler_similarity(n1, n2)
    
    tokens1, tokens2 = set(n1.split()), set(n2.split())
    jaccard_name = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
    prefix_match_name = 1.0 if (n1 and n2 and n1[:3] == n2[:3]) else 0.0
    
    # 2. Address Similarities
    lev_dist_addr = jellyfish.levenshtein_distance(a1, a2)
    norm_lev_addr = 1.0 - (lev_dist_addr / max(len(a1), len(a2), 1))
    
    addr_tokens1, addr_tokens2 = set(a1.split()), set(a2.split())
    jaccard_addr = len(addr_tokens1 & addr_tokens2) / max(len(addr_tokens1 | addr_tokens2), 1)
    
    # 3. Numeric & Postal Code Matching
    zip_match = 1.0 if (zip1 and zip2 and zip1 == zip2) else 0.0
    
    d1_set, d2_set = set(str(digits1).split()), set(str(digits2).split())
    digit_overlap = len(d1_set & d2_set) / max(len(d1_set | d2_set), 1) if (d1_set and d2_set) else 0.0
    
    # 4. Structural Features
    len_diff_name = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    len_diff_addr = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1)
    
    return [
        norm_lev_name,
        jaro_name,
        jaccard_name,
        prefix_match_name,
        norm_lev_addr,
        jaccard_addr,
        zip_match,
        digit_overlap,
        len_diff_name,
        len_diff_addr
    ]

FEATURE_NAMES = [
    "norm_lev_name", "jaro_name", "jaccard_name", "prefix_match_name",
    "norm_lev_addr", "jaccard_addr", "zip_match", "digit_overlap",
    "len_diff_name", "len_diff_addr"
]

if __name__ == "__main__":
    print(f"Features module ready. Extracted features: {FEATURE_NAMES}")
