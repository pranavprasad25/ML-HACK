"""
Pairwise Feature Engineering Module for Business Entity Resolution.

Team Role: Person 3A (Pairwise Feature Engineering - CPU Intensive)
Consumer: Person 3B (Model Training, F_0.5 Optimization & Inference)
Reference Docs: COLLABORATION_GUIDE.md, 3_step_divide.txt

Specifications:
  - Memory-efficient 1D NumPy array output (np.float32).
  - RapidFuzz text similarity metrics for business names and addresses:
      * Normalized Levenshtein ratio [0.0, 1.0]
      * Jaro-Winkler similarity [0.0, 1.0]
      * Token Sort Ratio [0.0, 1.0]
      * Token Set Ratio [0.0, 1.0]
  - Geographical Feature: Binary exact ZIP/PIN code match flag (1.0 or 0.0).
  - Robust edge-case sanitization: Gracefully handles None, NaN, float representations
    (e.g., 72716.0), empty strings, and missing dictionary keys without throwing exceptions.
  - Multi-schema key resolution: Transparently resolves aliases used by Person 1 and raw TSVs:
      * Names: 'business_name', 'business_name_clean', 'clean_name', 'name_clean', 'name', 'name_base'
      * Addresses: 'business_address', 'business_address_clean', 'clean_address', 'addr_clean', 'address'
      * Postal/ZIP: 'postal_code', 'addr_postal_code', 'zip_code', 'pin_code', 'zip', 'pin'
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# ==============================================================================
# Feature Schema Contract (Strictly ordered for Person 3B consumption)
# ==============================================================================
# ==============================================================================
# Feature Schema Contract (Strictly ordered for Person 3B consumption)
# ==============================================================================
FEATURE_NAMES: List[str] = [
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_prefix_3_match",
    "name_length_diff_ratio",
    "address_levenshtein_ratio",
    "address_jaro_winkler",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_digit_overlap",
    "address_length_diff_ratio",
    "exact_zip_match",
]

NUM_FEATURES: int = len(FEATURE_NAMES)

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

NULL_STRINGS: frozenset = frozenset(
    {"nan", "none", "null", "undefined", "na", "<na>", "nil", ""}
)


# ==============================================================================
# Field Sanitization & Resolution Helpers
# ==============================================================================
def _sanitize_text(val: Any) -> str:
    """
    Safely sanitizes arbitrary text fields.
    Returns an empty string for None, NaN, null tokens, or empty/whitespace strings.
    """
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""

    s = str(val).strip()
    if not s or s.lower() in NULL_STRINGS:
        return ""
    return s.lower()


def _sanitize_postal_code(val: Any) -> str:
    """
    Safely sanitizes postal/ZIP/PIN codes.
    Handles numeric float artifacts (e.g. 72716.0 -> '72716'), removes hyphens
    and spaces, and filters out null-like or dummy values (e.g. '0', '00000').
    """
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

    # Normalize string float artifact like "560001.0" -> "560001"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    # Clean punctuation and spacing for robust comparison
    cleaned = s.replace(" ", "").replace("-", "").lower()
    if cleaned in ("0", "00000", "000000", "unknown"):
        return ""
    return cleaned


def _extract_field(rec: Dict[str, Any], candidate_keys: Tuple[str, ...], is_postal: bool = False) -> str:
    """Extracts the first present and valid value matching any candidate key."""
    sanitize_fn = _sanitize_postal_code if is_postal else _sanitize_text
    for key in candidate_keys:
        if key in rec:
            val = sanitize_fn(rec[key])
            if val:
                return val
    return ""


# ==============================================================================
# Core Pairwise Feature Extraction
# ==============================================================================
def extract_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a standardized 1D NumPy array of 16 numerical similarity features
    between a Source 1 record and a candidate record.
    """
    feats = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. Resolve and sanitize input attributes
    name1 = _extract_field(s1_rec, NAME_KEYS)
    name2 = _extract_field(cand_rec, NAME_KEYS)

    addr1 = _extract_field(s1_rec, ADDR_KEYS)
    addr2 = _extract_field(cand_rec, ADDR_KEYS)

    zip1 = _extract_field(s1_rec, ZIP_KEYS, is_postal=True)
    zip2 = _extract_field(cand_rec, ZIP_KEYS, is_postal=True)

    # 2. Text Similarities: Business Name
    if name1 and name2:
        feats[0] = Levenshtein.normalized_similarity(name1, name2)
        feats[1] = JaroWinkler.similarity(name1, name2)
        feats[2] = fuzz.token_sort_ratio(name1, name2) * 0.01
        feats[3] = fuzz.token_set_ratio(name1, name2) * 0.01
        feats[4] = fuzz.partial_ratio(name1, name2) * 0.01
        
        t1, t2 = set(name1.split()), set(name2.split())
        feats[5] = len(t1 & t2) / max(len(t1 | t2), 1)
        feats[6] = 1.0 if (len(name1) >= 3 and len(name2) >= 3 and name1[:3] == name2[:3]) else 0.0
        feats[7] = abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)

    # 3. Text Similarities: Business Address
    if addr1 and addr2:
        feats[8] = Levenshtein.normalized_similarity(addr1, addr2)
        feats[9] = JaroWinkler.similarity(addr1, addr2)
        feats[10] = fuzz.token_sort_ratio(addr1, addr2) * 0.01
        feats[11] = fuzz.token_set_ratio(addr1, addr2) * 0.01
        
        a1, a2 = set(addr1.split()), set(addr2.split())
        feats[12] = len(a1 & a2) / max(len(a1 | a2), 1)
        
        # Digit Overlap (House / Unit numbers)
        import re
        d1 = set(re.findall(r'\b\d+\b', addr1))
        d2 = set(re.findall(r'\b\d+\b', addr2))
        feats[13] = len(d1 & d2) / max(len(d1 | d2), 1) if (d1 and d2) else 0.0
        feats[14] = abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2), 1)

    # 4. Geographical Feature: Exact ZIP/PIN match
    if zip1 and zip2 and zip1 == zip2:
        feats[15] = 1.0

    return feats


def extract_features_dict(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience wrapper returning a dictionary mapping feature name -> float value.
    Fulfills the interface contract specified in 3_step_divide.txt (Global Rule 1).
    """
    vector = extract_features(s1_rec, cand_rec)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vector)}


def extract_features_batch(
    s1_records: Sequence[Dict[str, Any]],
    cand_records: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """
    Vectorized batch feature extractor for pairs of records.
    Designed for memory-efficient iteration over millions of candidate pairs.

    Args:
        s1_records: Sequence of N Source 1 record dictionaries.
        cand_records: Sequence of N candidate record dictionaries.

    Returns:
        np.ndarray: 2D array of shape (N, 9) with dtype np.float32.
    """
    n = len(s1_records)
    if n != len(cand_records):
        raise ValueError(
            f"Record count mismatch: received {n} S1 records vs {len(cand_records)} candidate records."
        )

    out = np.empty((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        out[i] = extract_features(s1_records[i], cand_records[i])
    return out


# ==============================================================================
# Standalone Unit & Integration Tests
# ==============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Amazon ML Challenge 2026: Feature Engineering (Person 3A)")
    print("=" * 72)
    print(f"Standardized Output Contract ({NUM_FEATURES} features):")
    for idx, f_name in enumerate(FEATURE_NAMES):
        print(f"  [{idx}] {f_name}")
    print("-" * 72)

    # Test Case 1: Standard match with typical real-world variations
    s1_standard = {
        "entity_id": "S1-00100",
        "business_name_clean": "walmart supercenter store 100",
        "business_address_clean": "702 sw 8th street bentonville ar",
        "postal_code": "72716",
        "country": "US",
    }
    cand_standard = {
        "entity_id": "S2-00451",
        "business_name": "Wal-Mart Supercenter #100",
        "address": "702 South West 8th St, Bentonville",
        "zip_code": 72716,  # integer zip representation
        "country": "US",
    }

    feat_std = extract_features(s1_standard, cand_standard)
    print("\nTest Case 1 (Standard High-Confidence Pair):")
    for name, val in zip(FEATURE_NAMES, feat_std):
        print(f"  {name:<28}: {val:.4f}")

    assert feat_std.dtype == np.float32
    assert feat_std.shape == (NUM_FEATURES,)
    assert feat_std[8] == 1.0, "Expected exact ZIP match = 1.0"
    assert feat_std[0] > 0.6, "Expected high name Levenshtein similarity"

    # Test Case 2: Messy / Missing Data / NaN values
    s1_messy = {
        "entity_id": "S1-00200",
        "business_name": "Target Store",
        "business_address": None,
        "zip_code": float("nan"),
    }
    cand_messy = {
        "entity_id": "S3-00999",
        "business_name": "Target Corporation",
        "address": "1000 Nicollet Mall Minneapolis MN",
        "postal_code": "55403",
    }

    feat_messy = extract_features(s1_messy, cand_messy)
    print("\nTest Case 2 (Missing Address & NaN ZIP):")
    for name, val in zip(FEATURE_NAMES, feat_messy):
        print(f"  {name:<28}: {val:.4f}")

    assert feat_messy[4] == 0.0, "Missing address must default to 0.0"
    assert feat_messy[5] == 0.0
    assert feat_messy[8] == 0.0, "Missing ZIP must default to 0.0"

    # Test Case 3: Empty Records (Global Rule 2: Never throw an exception)
    feat_empty = extract_features({}, {})
    print("\nTest Case 3 (Completely Empty Records):")
    print(f"  Features: {feat_empty}")
    assert np.all(feat_empty == 0.0), "All features must default to 0.0 on empty input"

    # Test Case 4: Dictionary Interface Contract (3_step_divide.txt Global Rule 1)
    dict_out = extract_features_dict(s1_standard, cand_standard)
    print("\nTest Case 4 (Dictionary Contract Verification):")
    print(f"  Keys returned: {list(dict_out.keys())}")
    assert isinstance(dict_out, dict)
    assert set(dict_out.keys()) == set(FEATURE_NAMES)

    # Test Case 5: Batch Processing Performance & Shape Verification
    batch_s1 = [s1_standard, s1_messy, {}]
    batch_cand = [cand_standard, cand_messy, {}]
    batch_matrix = extract_features_batch(batch_s1, batch_cand)
    print("\nTest Case 5 (Batch Matrix Shape):")
    print(f"  Matrix shape: {batch_matrix.shape}, dtype: {batch_matrix.dtype}")
    assert batch_matrix.shape == (3, NUM_FEATURES)
    assert batch_matrix.dtype == np.float32

    print("\n" + "=" * 72)
    print("SUCCESS: All Person 3A contracts and unit assertions passed!")
    print("=" * 72)

