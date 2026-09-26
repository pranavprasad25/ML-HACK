# Business Entity Resolution Pipeline

## Overview
This repository contains the Machine Learning pipeline for ML Challenge 2026: Business Entity Resolution.

## Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline End-to-End

1. **Step 1: Preprocessing** (Person 1)
   ```bash
   python src/preprocess.py
   ```
2. **Step 2: Candidate Generation / Blocking** (Person 2)
   ```bash
   python src/blocking.py
   ```
3. **Step 3: Feature Extraction & Model Training** (Person 3)
   ```bash
   python src/train_predict.py
   ```

Outputs will be saved to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Validate formatting before submission:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
