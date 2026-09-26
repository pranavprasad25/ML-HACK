# Business Entity Resolution Pipeline

## Overview
This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026: Business Entity Resolution.

## Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline End-to-End

You can run the complete end-to-end pipeline with a single command:
```bash
python src/pipeline.py --data-dir dataset/test --output-dir output
```

### Or run individual stage modules:

1. **Stage 1: Preprocessing & Data Normalization** (Person 1)
   ```bash
   python -m code.business_entity_resolution.src.preprocessing --input-dir dataset/train --output-dir data/normalized_train
   ```

2. **Stage 2: Candidate Generation & Blocking** (Person 2)
   ```bash
   python -m code.business_entity_resolution.src.blocking --normalized-dir data/normalized_train --output output/candidate_pairs.tsv --top-k 30
   ```

3. **Stage 3: Feature Extraction, Model Training & Prediction** (Person 3)
   ```bash
   python -m code.business_entity_resolution.src.train_model
   ```

## Validation
Before submitting to the portal, validate submission format integrity:
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
