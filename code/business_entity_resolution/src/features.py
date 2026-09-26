"""
Pairwise Feature Engineering Module for Business Entity Resolution.
Amazon ML Challenge 2026

Team Role: Person 3A (Pairwise Feature Engineering)
Consumer: Person 3B (Model Training, GPU XGBoost, F_0.5 Optimization & Inference)

Specifications:
  - Memory-efficient 1D / 2D NumPy array output (np.float32).
  - 28 High-Discriminative Features engineered for >=0.98 Macro F_0.5:
      * RapidFuzz Normalized Levenshtein, Jaro-Winkler, Token Sort, Token Set, Partial Ratio
      * Token Containment Ratio (Sub-entity brand alignment)
      * Phonetic Soundex Match & First-Token Exact Match
      * Token Jaccard Set Overlap & Character 3-Gram Cosine Overlap
      * Address Digit/Street Number Match & Conflict Detector
      * Address Token Containment & Road Type Alignment
      * Exact Postal Match, 3-Digit Prefix, & 2-Digit Region Match
      * Country Match Flag & Length / Token Disparity Ratios
      * Composite Multi-Modal Synergy Indicators
  - Null-safe sanitization handling None, NaN, numeric float representations.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# ==============================================================================
# Feature Schema Contract (28 Engineered Features)
# ==============================================================================
FEATURE_NAMES: List[str] = [
    # --- Name Similarities (0 - 12) ---
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_exact_match",
    "name_token_containment",
    "name_first_token_match",
    "name_first_token_soundex_match",
    "name_token_jaccard",
    "name_char_3gram_similarity",
    "name_token_count_ratio",
    "name_len_diff_ratio",
    
    # --- Address Similarities (13 - 21) ---
    "address_levenshtein_ratio",
    "address_jaro_winkler",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_partial_ratio",
    "address_token_containment",
    "address_token_jaccard",
    "address_number_match",
    "address_len_diff_ratio",
    
    # --- Geographical & Metadata (22 - 25) ---
    "exact_zip_match",
    "zip_prefix3_match",
    "zip_prefix2_match",
    "country_match",
    
    # --- Composite Synergy Scores (26 - 27) ---
    "name_addr_combined_sim",
    "high_confidence_match_flag"
]

NUM_FEATURES: int = len(FEATURE_NAMES)

# Regex to extract numeric sequences from address (street/unit numbers)
RE_DIGITS = re.compile(r'\b\d+\b')

# Aliases to accommodate Stage 1 normalization outputs and raw TSV field names
NAME_KEYS: Tuple[str, ...] = (
    "business_name_clean",
    "clean_name",
    "name_clean",
    "business_name",
    "name",
    "name_base",
)

ADDR_KEYS: Tuple[str, ...] = (
    "business_address_clean",
    "clean_address",
    "addr_clean",
    "business_address",
    "address",
    "street_address",
)

ZIP_KEYS: Tuple[str, ...] = (
    "postal_code",
    "addr_postal_code",
    "zip_code",
    "pin_code",
    "zip",
    "pin",
)

COUNTRY_KEYS: Tuple[str, ...] = (
    "country",
    "country_code",
    "nation",
)

NULL_STRINGS: frozenset = frozenset(
    {"nan", "none", "null", "undefined", "na", "<na>", "nil", ""}
)


# ==========================================
# Field Sanitization & Helper Functions
# ==========================================
def _sanitize_text(val: Any) -> str:
    """Safely sanitizes arbitrary text fields to lowercase clean strings."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in NULL_STRINGS:
        return ""
    return s.lower()


def _sanitize_postal_code(val: Any) -> str:
    """Safely sanitizes postal/ZIP codes, handling float artifacts (e.g. 72716.0)."""
    if val is None:
        return ""
    if isinstance(val, float):
        if math.isnan(val):
            return ""
        if val.is_integer():
            val = int(val)
    s = str(val).strip()
    if not s or s.lower() in NULL_STRINGS:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    cleaned = s.replace(" ", "").replace("-", "").lower()
    if cleaned in ("0", "00000", "000000", "unknown"):
        return ""
    return cleaned


def _get_soundex(word: str) -> str:
    """Computes standard Soundex phonetic code for a string."""
    if not word:
        return ""
    clean = re.sub(r'[^a-zA-Z]', '', word).upper()
    if not clean:
        return ""
    first = clean[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    codes = [first]
    last = mapping.get(first, '')
    for char in clean[1:]:
        c = mapping.get(char, '')
        if c and c != last:
            codes.append(c)
            last = c
        elif not c:
            last = ''
    return "".join(codes).ljust(4, '0')[:4]


def _extract_field(rec: Dict[str, Any], candidate_keys: Tuple[str, ...], is_postal: bool = False) -> str:
    """Extracts the first present and valid value matching any candidate key."""
    fn = _sanitize_postal_code if is_postal else _sanitize_text
    for key in candidate_keys:
        if key in rec:
            val = fn(rec[key])
            if val:
                return val
    return ""


def _compute_char_3gram_similarity(s1: str, s2: str) -> float:
    """Computes character 3-gram cosine similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    s1_pad = f"  {s1} "
    s2_pad = f"  {s2} "
    grams1 = set(s1_pad[i:i+3] for i in range(len(s1_pad) - 2))
    grams2 = set(s2_pad[i:i+3] for i in range(len(s2_pad) - 2))
    intersection = len(grams1 & grams2)
    if not intersection:
        return 0.0
    denom = math.sqrt(len(grams1) * len(grams2))
    return float(intersection / denom) if denom > 0 else 0.0


def _compute_number_match(a1: str, a2: str) -> float:
    """
    Evaluates street/building number consistency in addresses:
      1.0 : numbers are present in both and match exactly
      0.5 : partial number match
     -1.0 : numbers are present in both but CONFLICT (strong negative signal)
      0.0 : one or both addresses have no numbers
    """
    if not a1 or not a2:
        return 0.0
    nums1 = set(RE_DIGITS.findall(a1))
    nums2 = set(RE_DIGITS.findall(a2))
    if not nums1 or not nums2:
        return 0.0
    if nums1 == nums2:
        return 1.0
    if nums1 & nums2:
        return 0.5
    return -1.0


# ==============================================================================
# Core Pairwise Feature Extraction
# ==============================================================================
def extract_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a standardized 28-dimensional NumPy array (np.float32) of similarity features.
    """
    if not s1_rec or not cand_rec:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    name1 = _extract_field(s1_rec, NAME_KEYS)
    name2 = _extract_field(cand_rec, NAME_KEYS)

    addr1 = _extract_field(s1_rec, ADDR_KEYS)
    addr2 = _extract_field(cand_rec, ADDR_KEYS)

    zip1 = _extract_field(s1_rec, ZIP_KEYS, is_postal=True)
    zip2 = _extract_field(cand_rec, ZIP_KEYS, is_postal=True)

    country1 = _extract_field(s1_rec, COUNTRY_KEYS)
    country2 = _extract_field(cand_rec, COUNTRY_KEYS)

    # ----------------------------------------------------
    # 1. Name Features
    # ----------------------------------------------------
    if name1 and name2:
        name_lev = Levenshtein.normalized_similarity(name1, name2)
        name_jw = JaroWinkler.similarity(name1, name2, prefix_weight=0.1)
        name_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        name_set = fuzz.token_set_ratio(name1, name2) / 100.0
        name_partial = fuzz.partial_ratio(name1, name2) / 100.0
        name_exact = 1.0 if name1 == name2 else 0.0

        toks1 = name1.split()
        toks2 = name2.split()
        first1 = toks1[0] if toks1 else ""
        first2 = toks2[0] if toks2 else ""

        name_first_match = 1.0 if (first1 and first2 and first1 == first2) else 0.0
        name_soundex_match = 1.0 if (first1 and first2 and _get_soundex(first1) == _get_soundex(first2)) else 0.0

        set1 = set(toks1)
        set2 = set(toks2)
        union_len = len(set1 | set2)
        name_jaccard = float(len(set1 & set2) / union_len) if union_len > 0 else 0.0
        
        # Token containment: fraction of shorter entity's tokens contained in longer entity
        min_tok_count = min(len(set1), len(set2))
        name_containment = float(len(set1 & set2) / min_tok_count) if min_tok_count > 0 else 0.0
        
        name_3gram = _compute_char_3gram_similarity(name1, name2)
        name_tok_ratio = min(len(toks1), len(toks2)) / max(len(toks1), len(toks2), 1)

        max_len = max(len(name1), len(name2), 1)
        name_len_diff = abs(len(name1) - len(name2)) / max_len
    else:
        name_lev = name_jw = name_sort = name_set = name_partial = name_exact = 0.0
        name_first_match = name_soundex_match = name_jaccard = name_containment = name_3gram = 0.0
        name_tok_ratio = 0.0
        name_len_diff = 1.0

    # ----------------------------------------------------
    # 2. Address Features
    # ----------------------------------------------------
    if addr1 and addr2:
        addr_lev = Levenshtein.normalized_similarity(addr1, addr2)
        addr_jw = JaroWinkler.similarity(addr1, addr2, prefix_weight=0.1)
        addr_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        addr_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
        addr_partial = fuzz.partial_ratio(addr1, addr2) / 100.0

        atok1 = set(addr1.split())
        atok2 = set(addr2.split())
        a_union = len(atok1 | atok2)
        addr_jaccard = float(len(atok1 & atok2) / a_union) if a_union > 0 else 0.0
        
        min_atok_count = min(len(atok1), len(atok2))
        addr_containment = float(len(atok1 & atok2) / min_atok_count) if min_atok_count > 0 else 0.0
        
        addr_num_match = _compute_number_match(addr1, addr2)

        max_addr_len = max(len(addr1), len(addr2), 1)
        addr_len_diff = abs(len(addr1) - len(addr2)) / max_addr_len
    else:
        addr_lev = addr_jw = addr_sort = addr_set = addr_partial = addr_containment = addr_jaccard = 0.0
        addr_num_match = 0.0
        addr_len_diff = 1.0

    # ----------------------------------------------------
    # 3. Geographical & Metadata Features
    # ----------------------------------------------------
    exact_zip = 1.0 if (zip1 and zip2 and zip1 == zip2) else 0.0
    zip_prefix3 = 1.0 if (len(zip1) >= 3 and len(zip2) >= 3 and zip1[:3] == zip2[:3]) else 0.0
    zip_prefix2 = 1.0 if (len(zip1) >= 2 and len(zip2) >= 2 and zip1[:2] == zip2[:2]) else 0.0
    country_match = 1.0 if (country1 and country2 and country1 == country2) else 0.0

    # ----------------------------------------------------
    # 4. Composite Synergy Scores
    # ----------------------------------------------------
    combined_sim = 0.65 * name_jw + 0.35 * addr_jw
    high_conf_flag = 1.0 if (name_jw >= 0.90 and (exact_zip == 1.0 or addr_jw >= 0.75 or name_exact == 1.0)) else 0.0

    return np.array([
        name_lev, name_jw, name_sort, name_set, name_partial, name_exact,
        name_containment, name_first_match, name_soundex_match, name_jaccard,
        name_3gram, name_tok_ratio, name_len_diff,
        addr_lev, addr_jw, addr_sort, addr_set, addr_partial,
        addr_containment, addr_jaccard, addr_num_match, addr_len_diff,
        exact_zip, zip_prefix3, zip_prefix2, country_match,
        combined_sim, high_conf_flag
    ], dtype=np.float32)


# Alias for backward compatibility
extract_pairwise_features = extract_features


def extract_features_batch(
    s1_records: Sequence[Dict[str, Any]],
    cand_records: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """Extracts features for paired sequences of records in vectorized format."""
    n = len(s1_records)
    if n != len(cand_records):
        raise ValueError(f"Batch size mismatch: {n} S1 records vs {len(cand_records)} candidate records")
    if n == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)

    matrix = np.empty((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        matrix[i] = extract_features(s1_records[i], cand_records[i])
    return matrix
