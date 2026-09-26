# Amazon ML Challenge 2026: Business Entity Resolution
## Multi-Agent & Team Collaboration Guide

---

## 1. Challenge Overview & Problem Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers. 

The task is to build an end-to-end Machine Learning pipeline that links matching records across **3 independent data sources** (`Source 1`, `Source 2`, and `Source 3`):
- **Source 1** is the deduplicated reference source.
- Find all matching records from **Source 2** and **Source 3** for each Source 1 entity.
- A Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

### ⚠️ Critical Competition Rules

1. **Tab-Separated Files (`.tsv`):** All datasets and submission files MUST use tab separators (`\t`). Address and ID lists contain commas; reading without `sep="\t"` will break column parsing.
2. **Open-Set Country Rule:** 
   - Training set: `US` and `India`.
   - Test set: `US`, `India`, and **`France`** (which does NOT appear in training data).
   - **Never** hardcode filters to only `{US, India}`. All test entities across all countries must be processed.
3. **Evaluation Metric (Macro $F_{0.5}$):**
   $$\beta = 0.5 \implies F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
   - **Precision is weighted $2\times$ higher than Recall.** 
   - False positives (wrong merges) are penalized heavily. Prefer outputting high-confidence matches.
4. **Candidate Subset Rule:** Every match in `output/matching_results.tsv` **MUST** be present in `output/candidate_pairs.tsv` for that entity.

---

## 2. Standard Codebase Structure

All teammates must strictly adhere to the following directory and file structure:

```
ml-challenge-2026/
├── .gitignore
├── Documentation_template.md
├── COLLABORATION_GUIDE.md
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv         # Deduplicated reference records (~2.2M rows)
│   │   ├── train_source2.tsv         # Source 2 training records
│   │   ├── train_source3.tsv         # Source 3 training records
│   │   └── train_ground_truth.tsv    # Matching ground truth
│   └── test/
│       ├── test_source1.tsv          # Source 1 test records (generate matches for ALL)
│       ├── test_source2.tsv          # Source 2 test records
│       └── test_source3.tsv          # Source 3 test records (includes France)
├── output/
│   ├── candidate_pairs.tsv           # Candidate pairs produced by Stage 2 (Blocking)
│   └── matching_results.tsv          # Final matched entities for Leaderboard (Stage 3)
├── utils/
│   └── validate_submission.py        # Official format and integrity validation tool
└── code/
    └── business_entity_resolution/
        ├── requirements.txt          # Shared dependencies
        ├── README.md                 # Code execution instructions
        └── src/
            ├── preprocess.py         # <-- Person 1 (Preprocessing & Normalization)
            ├── blocking.py           # <-- Person 2 (Candidate Generation & Blocking)
            ├── features.py           # <-- Person 3 (Pairwise Feature Extraction)
            └── train_predict.py      # <-- Person 3 (Model Training, F_0.5 Tuning & Inference)
```

---

## 3. Team Responsibilities & Git Branches

| Role | Member | Focus Area | Git Branch | Working Files |
| :--- | :--- | :--- | :--- | :--- |
| **Person 1** | Team Member 1 | Preprocessing & Normalization | `feature/preprocessing` | `code/business_entity_resolution/src/preprocess.py` |
| **Person 2** | Team Member 2 | Blocking & Candidate Generation | `feature/blocking` | `code/business_entity_resolution/src/blocking.py` |
| **Person 3** | Team Member 3 | Feature Engineering, ML Model & Submission | `feature/matching` | `code/business_entity_resolution/src/features.py`<br>`code/business_entity_resolution/src/train_predict.py` |

---

## 4. Antigravity IDE Prompt Templates

Copy-paste the exact prompt template below into your respective **Google Antigravity IDE** chat session.

---

### 🔹 Person 1: Preprocessing & Normalization Prompt

```markdown
Hello Antigravity. I am Person 1 on a 3-person team competing in the Amazon ML Challenge 2026 for Business Entity Resolution.

My sole responsibility is Stage 1: Preprocessing & Data Normalization.
My working file is: code/business_entity_resolution/src/preprocess.py

Dataset Context & Paths:
- Raw training data: dataset/train/train_source1.tsv, dataset/train/train_source2.tsv, dataset/train/train_source3.tsv
- Raw test data: dataset/test/test_source1.tsv, dataset/test/test_source2.tsv, dataset/test/test_source3.tsv
- Columns: entity_id, business_name, business_address, country

Key Requirements:
1. Normalization Logic (code/business_entity_resolution/src/preprocess.py):
   - Clean Business Names: Lowercase, remove accents (NFKD), strip honorifics (M/s, Dr, Shri), expand legal entity abbreviations (pvt ltd -> private limited, inc -> incorporated, corp -> corporation, llc -> limited liability company).
   - Clean Addresses: Lowercase, standardize street types (rd -> road, st -> street, ave -> avenue, blvd -> boulevard, hwy -> highway), standardize sub-units (apt -> apartment, ste -> suite, fl -> floor, bldg -> building), strip landmark noise (near, opp, b/h, behind).
   - Numeric Extraction: Extract house numbers, unit numbers, US ZIP codes (5 or 9 digit), Indian PIN codes (6-digit), French postal codes (5-digit).
   - Core Name Extraction: Produce a stripped "core_name" with legal suffixes removed for Person 2's blocking indexing.
   - Tokenization: Produce clean token sets for name and address.
2. Open Set Rule: Never hardcode or drop records based on country; must seamlessly support US, India, and France.
3. Performance: The dataset contains over 2 million rows. Use compiled regexes (re.compile), vectorized processing, or parallel mappings.
4. CLI Interface: Support batch processing of TSVs with input and output directories:
   python -m code.business_entity_resolution.src.preprocess --input-dir dataset/train --output-dir data/normalized_train
```

---

### 🔹 Person 2: Blocking & Candidate Generation Prompt

```markdown
Hello Antigravity. I am Person 2 on a 3-person team competing in the Amazon ML Challenge 2026 for Business Entity Resolution.

My sole responsibility is Stage 2: Candidate Generation & Blocking.
My working file is: code/business_entity_resolution/src/blocking.py

Input Context:
- Preprocessed records from Person 1 (preprocess.py) containing: entity_id, clean_name, core_name, clean_address, country, extracted_numbers, name_tokens.
- Target: Generate candidate pairs for every Source 1 entity against Source 2 and Source 3 records.

Key Requirements:
1. Blocking Strategy (code/business_entity_resolution/src/blocking.py):
   - Country Partitioning: Only compare entities within the same country (US to US, India to India, France to France).
   - Multi-Key Inverted Indexing:
     * Exact Postal/PIN code + first significant name token
     * Core name character 3-gram / token overlap TF-IDF
     * Phonetic key indexing (Double Metaphone / Soundex)
   - Aim for high recall (>95% true matches captured) with a manageable candidate pool (top K = 25 to 50 candidates per S1 entity).
2. Output Specification:
   - Output Path: output/candidate_pairs.tsv
   - Format: Tab-separated, exactly two columns: source1_entity_id\tcandidate_entity_ids
   - Every single Source 1 entity in the evaluated set MUST have a row.
   - candidate_entity_ids must be comma-separated with NO spaces (e.g. S2-00047,S3-00812).
   - Singletons must have an empty candidate_entity_ids column.
   - Must only contain S2- or S3- IDs (no S1- self pairs, no duplicate IDs in a row).
3. CLI Interface:
   python -m code.business_entity_resolution.src.blocking --normalized-dir data/normalized_train --output output/candidate_pairs.tsv
4. Validation:
   Verify formatting using utils/validate_submission.py.
```

---

### 🔹 Person 3: ML Model, Threshold Tuning & Submission Prompt

```markdown
Hello Antigravity. I am Person 3 on a 3-person team competing in the Amazon ML Challenge 2026 for Business Entity Resolution.

My sole responsibility is Stage 3: Feature Engineering, Model Training, F_0.5 Threshold Optimization & Submission Generation.
My working files are: 
- code/business_entity_resolution/src/features.py
- code/business_entity_resolution/src/train_predict.py

Input Context:
- Candidate pairs from Person 2: output/candidate_pairs.tsv
- Normalized entity records from Person 1: data/normalized_*
- Training ground truth: dataset/train/train_ground_truth.tsv

Key Requirements:
1. Pairwise Feature Engineering (code/business_entity_resolution/src/features.py):
   - Name Similarity: RapidFuzz Levenshtein ratio, Token Sort Ratio, Token Set Ratio, Jaro-Winkler distance, Prefix match.
   - Address Similarity: Token Jaccard overlap, Levenshtein ratio, numerical digit set overlap (PIN/ZIP/building number match flag).
   - Structural Features: Length difference ratios, missing field indicators.
2. Classifier & Metric Optimization (code/business_entity_resolution/src/train_predict.py):
   - Train a fast Gradient Boosted Decision Tree (LightGBM / XGBoost / CatBoost) on ground-truth positive pairs and sampled negative candidate pairs.
   - Decision Threshold Tuning: Optimize the classification probability threshold specifically for Macro F_0.5 (beta = 0.5, where Precision is weighted 2x over Recall).
   - Penalty Awareness: False positives (wrong merges) severely degrade F_0.5 score. Maintain high precision.
3. Final Output (output/matching_results.tsv):
   - Format: Tab-separated: source1_entity_id\tmatched_entity_ids
   - Every S1 entity must have exactly 1 row; singletons leave matched_entity_ids empty.
   - SUBSET CONSTRAINT: Every matched ID in matching_results.tsv MUST exist in output/candidate_pairs.tsv.
4. Validation & Submission:
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 5. Official Submission Validation Checklist

Before opening a pull request or submitting to the challenge portal, run:

```bash
# Verify submission integrity and compliance
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Quality Checklist:
- [ ] Output files are tab-separated (`.tsv`) with exact headers: `source1_entity_id` and `matched_entity_ids` / `candidate_entity_ids`.
- [ ] Every entity from `test_source1.tsv` is present in both output files.
- [ ] All matched IDs strictly belong to `test_source2.tsv` or `test_source3.tsv`.
- [ ] No self-matches (`S1-` IDs in matched list) and no duplicate IDs in any row.
- [ ] Subset Rule holds: `matched_entity_ids` $\subseteq$ `candidate_entity_ids` for every row.
- [ ] France records in the test set are properly processed and not dropped.
