# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA CUDA](https://img.shields.io/badge/CUDA-13.2_|_RTX_5060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![XGBoost GPU](https://img.shields.io/badge/XGBoost-GPU_Accelerated-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Metric F0.5](https://img.shields.io/badge/Metric-Macro_F0.5-8B5CF6?style=for-the-badge)](#-evaluation-metric)

A high-performance, GPU-accelerated Machine Learning solution to perform **Business Entity Resolution (ER)** across noisy, multi-source commercial data.

---

## 📌 Project Overview

In commercial platforms, business identity data arrives from 3 independent sources (`Source 1`, `Source 2`, `Source 3`) without shared unique global identifiers.

- **Goal**: Match deduplicated reference entities in `Source 1` to matching records in `Source 2` and `Source 3`.
- **Challenges**: Legal suffix variations (`Corp` vs `Corporation`, `Pvt Ltd` vs `Private Limited`), trade names, missing address components, typos, and open-set countries (`US`, `India`, `France`).
- **Optimization Metric**: **Macro $F_{0.5}$ Score** — weights Precision $2\times$ over Recall to penalize erroneous business merges.

---

## 🛠️ ML Architecture & Pipeline Stages

```
Raw TSVs ──► Stage 1: Preprocessing ──► Stage 2: Blocking ──► Stage 3: Feature Eng. + GPU Model ──► Output TSVs
```

### 1️⃣ Stage 1: Preprocessing & Data Normalization
* **File**: [`preprocessing.py`](file:///c:/Users/prana/ML-HACK/code/business_entity_resolution/src/preprocessing.py)
* NFKD unicode cleaning, lowercasing, legal/street term expansion (`pvt ltd` $\rightarrow$ `private limited`, `rd` $\rightarrow$ `road`).
* Regex extracts country-specific postal codes (India 6-digit PIN, US 5/9-digit ZIP, France 5-digit postal code).

### 2️⃣ Stage 2: Multi-Key Candidate Blocking
* **File**: [`blocking.py`](file:///c:/Users/prana/ML-HACK/code/business_entity_resolution/src/blocking.py)
* Replaces 8.2 Trillion $O(N \times M)$ comparisons with 5 $O(1)$ inverted hash indices (Compound Key, Significant Tokens, Exact Postal, Phonetic Soundex, 3-char Prefix).
* Parallel lookup across **28 CPU cores** at ~10M lookups/sec generates `candidate_pairs.tsv` (~30 candidates/entity).

### 3️⃣ Stage 3A: Pairwise Feature Engineering (16 Features)
* **File**: [`features.py`](file:///c:/Users/prana/ML-HACK/code/business_entity_resolution/src/features.py)
* Extracts 16 continuous pairwise similarity signals using C++ `RapidFuzz`:
  - **Name Signals**: Levenshtein Ratio, Jaro-Winkler, Token Sort/Set Ratio, Partial Ratio, Token Jaccard, 3-char Prefix Match, Length Diff Ratio.
  - **Address Signals**: Levenshtein Ratio, Jaro-Winkler, Token Sort/Set Ratio, Token Jaccard, Digit Overlap Ratio, Length Diff Ratio.
  - **Geographical Signal**: Exact Postal/ZIP Code match flag (`1.0` or `0.0`).

### 4️⃣ Stage 3B: Model Training & Hyperparameters
* **File**: [`train_predict.py`](file:///c:/Users/prana/ML-HACK/code/business_entity_resolution/src/train_predict.py)
* **Classifier**: XGBoost / LightGBM GBDT trained via 5-Fold Stratified Cross-Validation on ~200,000 sampled candidate pairs (5:1 negative sampling).

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_estimators` | `500` | Tree capacity with early stopping |
| `max_depth` | `6` | Captures non-linear feature interactions |
| `learning_rate` | `0.05` | Smooth gradient convergence |
| `subsample` / `colsample_bytree` | `0.8` / `0.8` | Regularization and feature diversity |
| `tree_method` / `device` | `hist` / `cuda` | Native NVIDIA CUDA GPU acceleration |

* **Threshold Tuning**: Sweeps decision thresholds (`0.30`–`0.95`) on OOF validation scores to maximize Macro $F_{0.5}$.

---

## ⚡ Performance Benchmarks

Execution on **NVIDIA RTX 5060 GPU** & **28 CPU Cores** (16 GB RAM):

| Pipeline Stage | Operation | Data Volume | Execution Time |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Text Preprocessing & Normalization | 17.6M total rows | **~3.7 min** |
| **Stage 2** | Multi-Key Candidate Blocking | 5.1M test rows | **~1.5 min** |
| **Stage 3A & 3B** | Feature Extraction & GPU Inference | ~40M candidate pairs | **~3.0 min** |
| **Total** | **End-to-End Pipeline Run** | Full Dataset | **~7 to 9 MIN** |

---

## 🚀 Quickstart & Usage

### 1️⃣ Install Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2️⃣ Run Full Pipeline
```bash
python code/business_entity_resolution/src/pipeline.py --data-dir dataset/test --output-dir output
```

### 3️⃣ Output
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
