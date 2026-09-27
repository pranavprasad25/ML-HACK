"""
Stage 2 (Enhanced): Semantic Blocking via Ollama Embeddings
Amazon ML Challenge 2026: Business Entity Resolution

Replaces TF-IDF blocking (11% recall) with dense vector embeddings from an
8B-class model served locally via Ollama (nomic-embed-text or llama3.1:8b).

Architecture:
  1. For every S2/S3 entity  -> embed(business_name_clean + postal_code)
  2. Build FAISS ANN index over all target embeddings
  3. For every S1 entity     -> embed(business_name_clean + postal_code)
  4. Query top-K nearest neighbors from FAISS index
  5. Apply country-partition filter (US vs INDIA) as hard constraint
  6. Output candidate_pairs.tsv with same schema as blocking.py

Expected recall improvement: 11% -> 60-80%+ (semantic similarity captures
abbreviations, misspellings, legal suffix variants that TF-IDF misses).

Requirements:
  - Ollama running locally:  ollama serve
  - Embedding model pulled:  ollama pull nomic-embed-text
  - pip install faiss-cpu requests numpy pandas tqdm

Usage (CLI):
  python -m code.business_entity_resolution.src.semantic_blocking \
      --normalized-dir data/normalized_train \
      --output data/semantic_candidate_pairs_train.tsv \
      --ground-truth dataset/train/train_ground_truth.tsv \
      --model nomic-embed-text \
      --top-k 30 \
      --batch-size 64

  # To use the full 8B LLaMA model for embeddings (slower, more VRAM):
  python -m code.business_entity_resolution.src.semantic_blocking \
      --model llama3.1:8b \
      --top-k 30
"""

import os
import sys
import time
import argparse
import warnings
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"  # 137M params, 768-dim, very fast
FALLBACK_EMBED_MODEL = "llama3.1:8b"     # 8B params, richer semantics


def build_entity_text(name: str, postal: str, country: str = "") -> str:
    """Builds a compact text representation of a business entity for embedding."""
    parts = [name.strip()]
    if postal and postal not in ("0", "nan", ""):
        parts.append(postal.strip())
    if country and country not in ("nan", ""):
        parts.append(country.strip())
    return " | ".join(parts)


# ==========================================
# 2. OLLAMA EMBEDDING CLIENT
# ==========================================

class OllamaEmbedder:
    """
    Thin client for Ollama's /api/embed endpoint.
    Handles batching, retries, and connection errors gracefully.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embed"
        self._dim: Optional[int] = None

        print(f"  [Ollama] Connecting to {self.base_url} with model '{self.model}'")
        self._check_connection()

    def _check_connection(self):
        """Verifies Ollama is running and model is available."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            models_clean = [m.split(":")[0] for m in models]
            model_base = self.model.split(":")[0]

            if model_base not in models_clean and self.model not in models:
                print(f"  [Ollama] Model '{self.model}' not found locally.")
                print(f"  [Ollama] Pulling '{self.model}'... (this may take a few minutes)")
                self._pull_model()
            else:
                print(f"  [Ollama] Model '{self.model}' ready checkmark")
        except requests.exceptions.ConnectionError:
            print(f"\n  [ERROR] Cannot connect to Ollama at {self.base_url}")
            print("  Please start Ollama with:  ollama serve")
            print("  Then pull the model with:  ollama pull nomic-embed-text")
            sys.exit(1)

    def _pull_model(self):
        """Pulls a model from Ollama registry."""
        try:
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=600
            )
            r.raise_for_status()
            print(f"  [Ollama] Model '{self.model}' pulled successfully")
        except Exception as e:
            print(f"  [ERROR] Failed to pull model: {e}")
            sys.exit(1)

    def embed_batch(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """Embeds a batch of texts. Returns (N, dim) float32 array."""
        for attempt in range(max_retries):
            try:
                payload = {"model": self.model, "input": texts}
                r = requests.post(self.embed_url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                embeddings = data.get("embeddings", [])
                if not embeddings:
                    raise ValueError(f"Empty embeddings response: {data}")
                arr = np.array(embeddings, dtype=np.float32)
                if self._dim is None:
                    self._dim = arr.shape[1]
                    print(f"  [Ollama] Embedding dimension: {self._dim}")
                return arr
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [ERROR] Embedding failed after {max_retries} attempts: {e}")
                raise

    def embed_all(self, texts: List[str], batch_size: int = 64, desc: str = "Embedding") -> np.ndarray:
        """Embeds all texts in batches with a progress bar."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", unit="batch"):
            batch = texts[i: i + batch_size]
            embs = self.embed_batch(batch)
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)

    @property
    def dim(self) -> int:
        return self._dim or 768


# ==========================================
# 3. FAISS ANN INDEX
# ==========================================

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False):
    """
    Builds a FAISS IndexFlatIP (cosine similarity) index.
    Embeddings are L2-normalized before indexing so IP == cosine similarity.
    """
    try:
        import faiss
    except ImportError:
        print("  [ERROR] faiss not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_norm = (embeddings / norms).astype(np.float32)

    dim = embeddings_norm.shape[1]
    n = embeddings_norm.shape[0]
    print(f"  [FAISS] Building index: {n:,} vectors x {dim}d")

    if n < 100_000:
        index = faiss.IndexFlatIP(dim)
        index_type = "FlatIP (exact)"
    else:
        nlist = min(4096, max(256, int(np.sqrt(n))))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index_type = f"IVFFlat (nlist={nlist})"

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"  [FAISS] Using GPU acceleration")
        except Exception:
            print(f"  [FAISS] GPU not available, using CPU")

    t0 = time.time()
    if hasattr(index, 'train'):
        index.train(embeddings_norm)
    index.add(embeddings_norm)
    print(f"  [FAISS] Index built ({index_type}) in {time.time()-t0:.1f}s")
    return index, embeddings_norm


def search_faiss(index, query_embeddings: np.ndarray, top_k: int = 30, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Searches FAISS index for top_k nearest neighbors for each query."""
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    queries_norm = (query_embeddings / norms).astype(np.float32)

    n_queries = queries_norm.shape[0]
    all_distances = np.zeros((n_queries, top_k), dtype=np.float32)
    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)

    for i in tqdm(range(0, n_queries, batch_size), desc="  FAISS search", unit="batch"):
        batch = queries_norm[i: i + batch_size]
        D, I = index.search(batch, top_k)
        all_distances[i: i + len(batch)] = D
        all_indices[i: i + len(batch)] = I

    return all_distances, all_indices


# ==========================================
# 4. COUNTRY-PARTITIONED SEMANTIC BLOCKING
# ==========================================

def semantic_generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    embedder: OllamaEmbedder,
    top_k: int = 30,
    batch_size: int = 64,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Generates candidate pairs using semantic embeddings + FAISS ANN.
    Returns DataFrame with columns: [source1_entity_id, candidate_entity_ids]
    """
    print("\n=== Semantic Blocking: Embedding-Based Candidate Generation ===")

    for df in (df_s1, df_s2, df_s3):
        if 'country' not in df.columns:
            df['country'] = 'UNKNOWN'
        df['country'] = df['country'].fillna('UNKNOWN').str.upper().str.strip()

    all_results: Dict[str, Set[str]] = {}
    countries = df_s1['country'].unique().tolist()
    print(f"  Country partitions: {countries}")

    for country in countries:
        s1_part = df_s1[df_s1['country'] == country].copy().reset_index(drop=True)
        if country == 'UNKNOWN':
            t2_part = df_s2.copy()
            t3_part = df_s3.copy()
        else:
            t2_part = df_s2[df_s2['country'] == country].copy()
            t3_part = df_s3[df_s3['country'] == country].copy()

        targets = pd.concat([t2_part, t3_part], ignore_index=True)

        if len(s1_part) == 0 or len(targets) == 0:
            print(f"  [{country}] Skipping (S1={len(s1_part):,}, targets={len(targets):,})")
            continue

        print(f"\n  [{country}] {len(s1_part):,} S1 x {len(targets):,} targets")

        s1_texts = [
            build_entity_text(row.get('business_name_clean', ''), row.get('postal_code', ''), country)
            for _, row in s1_part.iterrows()
        ]
        target_texts = [
            build_entity_text(row.get('business_name_clean', ''), row.get('postal_code', ''), country)
            for _, row in targets.iterrows()
        ]

        target_ids = targets['entity_id'].tolist()
        s1_ids = s1_part['entity_id'].tolist()

        print(f"  [{country}] Embedding {len(target_texts):,} target entities...")
        target_embeddings = embedder.embed_all(target_texts, batch_size=batch_size, desc=f"[{country}] Targets")

        index, _ = build_faiss_index(target_embeddings, use_gpu=use_gpu)

        print(f"  [{country}] Embedding {len(s1_texts):,} S1 entities...")
        s1_embeddings = embedder.embed_all(s1_texts, batch_size=batch_size, desc=f"[{country}] S1 queries")

        print(f"  [{country}] Searching top-{top_k} candidates...")
        _, nn_indices = search_faiss(index, s1_embeddings, top_k=top_k)

        target_ids_arr = np.array(target_ids)
        for i, s1_id in enumerate(s1_ids):
            cand_idx = nn_indices[i]
            valid = cand_idx[cand_idx >= 0]
            cands = set(target_ids_arr[valid].tolist())
            cands.discard(s1_id)
            if s1_id in all_results:
                all_results[s1_id].update(cands)
            else:
                all_results[s1_id] = cands

        print(f"  [{country}] Done")

    rows = [
        {'source1_entity_id': s1_id, 'candidate_entity_ids': ','.join(sorted(cands))}
        for s1_id, cands in all_results.items()
    ]
    result_df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_ids'])
    total_cands = sum(len(v) for v in all_results.values())
    print(f"\n  Generated {len(result_df):,} S1 rows, {total_cands:,} total candidates "
          f"(avg {total_cands/max(len(result_df),1):.1f}/entity)")
    return result_df


# ==========================================
# 5. BLOCKING RECALL EVALUATION
# ==========================================

def evaluate_blocking_recall(candidates_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    print("\n=== Evaluating Semantic Blocking Recall ===")
    cand_map: Dict[str, Set[str]] = {}
    for _, row in candidates_df.iterrows():
        s1 = str(row['source1_entity_id']).strip()
        cands_str = str(row.get('candidate_entity_ids', '')).strip()
        cand_map[s1] = set(c.strip() for c in cands_str.split(',') if c.strip())

    total_true = 0
    captured = 0
    for _, row in gt_df.iterrows():
        s1 = str(row['source1_entity_id']).strip()
        mids_str = str(row.get('matched_entity_ids', '')).strip()
        if not mids_str:
            continue
        true_mids = set(m.strip() for m in mids_str.split(',') if m.strip())
        cands = cand_map.get(s1, set())
        for t in true_mids:
            total_true += 1
            if t in cands:
                captured += 1

    recall = (captured / max(total_true, 1)) * 100.0
    print(f"  Captured {captured:,} / {total_true:,} true matches")
    print(f"  Semantic Blocking Recall: {recall:.2f}%  (vs TF-IDF baseline: 11.03%)")
    return recall


# ==========================================
# 6. CLI ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 (Enhanced): Semantic Blocking via Ollama Embeddings"
    )
    parser.add_argument("--normalized-dir", required=True,
                        help="Directory with normalized_source1/2/3.tsv from Stage 1")
    parser.add_argument("--output", required=True,
                        help="Output path for semantic candidate_pairs.tsv")
    parser.add_argument("--ground-truth", default=None,
                        help="Optional ground truth TSV to evaluate blocking recall")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                        help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL}). "
                             "Use 'llama3.1:8b' for the full 8B model.")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL,
                        help=f"Ollama server URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top K candidates per S1 entity (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU-accelerated FAISS index")
    args = parser.parse_args()

    print("=" * 70)
    print("  Stage 2 (Enhanced): Semantic Blocking via Ollama Embeddings")
    print(f"  Model: {args.model}")
    print(f"  Top-K: {args.top_k}")
    print("=" * 70)

    def load_tsv(path: str) -> pd.DataFrame:
        print(f"  Loading: {path}")
        df = pd.read_csv(path, sep='\t', dtype=str).fillna("")
        print(f"    -> {len(df):,} records")
        return df

    s1_path = os.path.join(args.normalized_dir, "normalized_source1.tsv")
    s2_path = os.path.join(args.normalized_dir, "normalized_source2.tsv")
    s3_path = os.path.join(args.normalized_dir, "normalized_source3.tsv")

    if not os.path.isfile(s1_path):
        s1_path = os.path.join(args.normalized_dir, "train_source1.tsv")
        s2_path = os.path.join(args.normalized_dir, "train_source2.tsv")
        s3_path = os.path.join(args.normalized_dir, "train_source3.tsv")

    print("\n[1/4] Loading normalized data...")
    df_s1 = load_tsv(s1_path)
    df_s2 = load_tsv(s2_path)
    df_s3 = load_tsv(s3_path)

    print("\n[2/4] Initializing Ollama embedder...")
    embedder = OllamaEmbedder(model=args.model, base_url=args.ollama_url)

    print("\n[3/4] Generating semantic candidate pairs...")
    t0 = time.time()
    candidates_df = semantic_generate_candidate_pairs(
        df_s1, df_s2, df_s3,
        embedder=embedder,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
    )
    print(f"\n  Semantic blocking completed in {time.time()-t0:.1f}s")

    print("\n[4/4] Saving candidate pairs...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    candidates_df.to_csv(args.output, sep='\t', index=False)
    print(f"  Saved {len(candidates_df):,} rows to {args.output}")

    if args.ground_truth and os.path.isfile(args.ground_truth):
        gt_df = pd.read_csv(args.ground_truth, sep='\t', dtype=str).fillna("")
        evaluate_blocking_recall(candidates_df, gt_df)

    print("\n" + "=" * 70)
    print("  Semantic Blocking Complete!")
    print(f"  Output: {os.path.abspath(args.output)}")
    print("\n  Next -- retrain the model on semantic candidates:")
    print(f"    python -m code.business_entity_resolution.src.train_predict \\")
    print(f"        --mode train \\")
    print(f"        --candidate-file {args.output} \\")
    print(f"        --normalized-dir {args.normalized_dir} \\")
    print(f"        --ground-truth {args.ground_truth or 'dataset/train/train_ground_truth.tsv'} \\")
    print(f"        --backend lightgbm --max-pairs 200000 --skip-train-eval")
    print("=" * 70)


if __name__ == "__main__":
    main()
