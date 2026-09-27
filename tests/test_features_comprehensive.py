import math
import numpy as np
import sys
import os

# Add the project root to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features, FEATURE_NAMES, NUM_FEATURES

def run_20_tests():
    test_cases = [
        {
            "desc": "1. Exact match across all fields",
            "s1": {"name": "Apple Inc", "address": "123 Main St", "zip": "90210"},
            "cand": {"name": "Apple Inc", "address": "123 Main St", "zip": "90210"},
            "expected_zip": 1.0,
            "check_names_1": True,
        },
        {
            "desc": "2. Case insensitivity check",
            "s1": {"name": "APPLE INC", "address": "123 MAIN ST", "zip": "90210"},
            "cand": {"name": "apple inc", "address": "123 main st", "zip": "90210"},
            "expected_zip": 1.0,
            "check_names_1": True,
        },
        {
            "desc": "3. Mismatched ZIP codes",
            "s1": {"name": "Target", "address": "456 Oak Rd", "zip": "10001"},
            "cand": {"name": "Target", "address": "456 Oak Rd", "zip": "10002"},
            "expected_zip": 0.0,
        },
        {
            "desc": "4. Missing ZIP in Source 1",
            "s1": {"name": "Target", "address": "456 Oak Rd"},
            "cand": {"name": "Target", "address": "456 Oak Rd", "zip": "10002"},
            "expected_zip": 0.0,
        },
        {
            "desc": "5. Missing ZIP in Candidate",
            "s1": {"name": "Target", "address": "456 Oak Rd", "zip": "10001"},
            "cand": {"name": "Target", "address": "456 Oak Rd"},
            "expected_zip": 0.0,
        },
        {
            "desc": "6. ZIP codes with float representation (e.g. 90210.0)",
            "s1": {"name": "Best Buy", "address": "789 Pine Ln", "zip": 90210.0},
            "cand": {"name": "Best Buy", "address": "789 Pine Ln", "zip": "90210"},
            "expected_zip": 1.0,
        },
        {
            "desc": "7. ZIP codes with hyphens and spaces",
            "s1": {"name": "Best Buy", "zip": "90210-1234"},
            "cand": {"name": "Best Buy", "zip": "90210 1234"},
            "expected_zip": 1.0,
        },
        {
            "desc": "8. Dummy ZIP code (00000) treated as missing",
            "s1": {"name": "Kmart", "zip": "00000"},
            "cand": {"name": "Kmart", "zip": "00000"},
            "expected_zip": 0.0, # Treated as missing, so 0.0
        },
        {
            "desc": "9. Different name aliases used (business_name_clean vs name)",
            "s1": {"business_name_clean": "Walmart", "zip": "72716"},
            "cand": {"name": "Walmart", "zip": "72716"},
            "expected_zip": 1.0,
            "check_names_1": True,
        },
        {
            "desc": "10. Different address aliases used",
            "s1": {"business_address_clean": "123 Market St", "name": "A"},
            "cand": {"addr_clean": "123 Market St", "name": "A"},
            "expected_addr_1": True,
        },
        {
            "desc": "11. NaN values as float('nan')",
            "s1": {"name": "Starbucks", "address": float("nan"), "zip": float("nan")},
            "cand": {"name": "Starbucks", "address": "123 Coffee Dr", "zip": "98101"},
            "expected_zip": 0.0,
            "expected_addr_0": True,
        },
        {
            "desc": "12. NaN values as string 'nan' or 'null'",
            "s1": {"name": "nan", "address": "null", "zip": "na"},
            "cand": {"name": "Starbucks", "address": "123 Coffee Dr", "zip": "98101"},
            "expected_zip": 0.0,
            "expected_name_0": True,
            "expected_addr_0": True,
        },
        {
            "desc": "13. None values",
            "s1": {"name": None, "address": None, "zip": None},
            "cand": {"name": "Starbucks", "address": "123 Coffee Dr", "zip": "98101"},
            "expected_zip": 0.0,
            "expected_name_0": True,
            "expected_addr_0": True,
        },
        {
            "desc": "14. Token Sort Ratio advantage (out of order words)",
            "s1": {"name": "Pizza Hut LLC"},
            "cand": {"name": "LLC Pizza Hut"},
            # sort ratio should be 1.0 (100%), while pure levenshtein will be lower
            "check_token_sort": True, 
        },
        {
            "desc": "15. Token Set Ratio advantage (extra words)",
            "s1": {"name": "McDonalds"},
            "cand": {"name": "McDonalds Corporation Restaurant"},
            # set ratio should be very high compared to standard ratio
            "check_token_set": True,
        },
        {
            "desc": "16. Whitespace only strings",
            "s1": {"name": "   ", "address": "\t", "zip": "\n"},
            "cand": {"name": "Starbucks", "address": "123 Coffee Dr", "zip": "98101"},
            "expected_zip": 0.0,
            "expected_name_0": True,
            "expected_addr_0": True,
        },
        {
            "desc": "17. Similar but distinct ZIP codes",
            "s1": {"zip": "12345"},
            "cand": {"zip": "12346"},
            "expected_zip": 0.0,
        },
        {
            "desc": "18. Extremely long strings",
            "s1": {"name": "A" * 500, "address": "B" * 500},
            "cand": {"name": "A" * 500, "address": "B" * 500},
            "expected_name_1": True,
            "expected_addr_1": True,
        },
        {
            "desc": "19. Completely empty dictionaries",
            "s1": {},
            "cand": {},
            "expected_zip": 0.0,
            "expected_name_0": True,
            "expected_addr_0": True,
        },
        {
            "desc": "20. Partial address match Jaro-Winkler",
            "s1": {"address": "123 Main Street"},
            "cand": {"address": "123 Main St"},
            # Should have high jaro winkler due to prefix match
            "check_jaro_winkler": True,
        }
    ]

    passed = 0
    for i, tc in enumerate(test_cases):
        try:
            feats = extract_features(tc["s1"], tc["cand"])
            
            # Map for easy reading
            fmap = {k: v for k, v in zip(FEATURE_NAMES, feats)}
            
            # Checks
            if "expected_zip" in tc:
                assert fmap["exact_zip_match"] == tc["expected_zip"], f"Zip mismatch: expected {tc['expected_zip']}, got {fmap['exact_zip_match']}"
                
            if tc.get("check_names_1"):
                assert fmap["name_levenshtein_ratio"] == 1.0, "Name levenshtein should be 1.0"
                assert fmap["name_jaro_winkler"] == 1.0, "Name jaro should be 1.0"
                
            if tc.get("expected_name_0"):
                assert fmap["name_levenshtein_ratio"] == 0.0, "Name levenshtein should be 0.0"
                
            if tc.get("expected_addr_1"):
                assert fmap["address_levenshtein_ratio"] == 1.0, "Address levenshtein should be 1.0"
                
            if tc.get("expected_addr_0"):
                assert fmap["address_levenshtein_ratio"] == 0.0, "Address levenshtein should be 0.0"

            if tc.get("check_token_sort"):
                assert fmap["name_token_sort_ratio"] == 1.0, f"Expected sort ratio 1.0, got {fmap['name_token_sort_ratio']}"
                assert fmap["name_levenshtein_ratio"] < 1.0, "Levenshtein should be < 1.0"
                
            if tc.get("check_token_set"):
                assert fmap["name_token_set_ratio"] >= 0.9, f"Expected set ratio >= 0.9, got {fmap['name_token_set_ratio']}"
                assert fmap["name_levenshtein_ratio"] < 0.8, "Levenshtein should be lower"

            if tc.get("check_jaro_winkler"):
                assert fmap["address_jaro_winkler"] >= 0.85, f"Expected Jaro-Winkler >= 0.85, got {fmap['address_jaro_winkler']}"

            print(f"[PASS] {tc['desc']}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {tc['desc']} - FAILED: {str(e)}")
            print(f"   Features Output: {feats}")

    print("-" * 60)
    print(f"Total Tests Run: {len(test_cases)}")
    print(f"Total Passed: {passed}")
    print(f"Total Failed: {len(test_cases) - passed}")

if __name__ == "__main__":
    run_20_tests()
