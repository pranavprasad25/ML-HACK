"""
Dataset Batch Preprocessor & Normalizer.

Streams large TSV files chunk-by-chunk and generates normalized fields:
- name_clean, name_base, name_tokens_sorted, name_first_token
- addr_clean, addr_tokens_sorted, addr_postal_code, addr_has_val
- country_norm

Can process millions of records with minimal memory footprint.
"""

import os
import csv
import sys
import time
from typing import Optional, Iterator, Dict, Any

from src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country
)

def process_record(row: Dict[str, str]) -> Dict[str, Any]:
    """Applies end-to-end normalization on a single data record."""
    entity_id = row.get("entity_id", "").strip()
    raw_name = row.get("business_name", "")
    raw_addr = row.get("business_address", "")
    raw_country = row.get("country", "")

    # Normalize components
    norm_name = normalize_business_name(raw_name)
    norm_addr = normalize_address(raw_addr, raw_country)
    norm_country = normalize_country(raw_country)

    return {
        "entity_id": entity_id,
        "name_clean": norm_name["name_clean"],
        "name_base": norm_name["name_base"],
        "name_tokens_sorted": norm_name["name_tokens_sorted"],
        "name_first_token": norm_name["name_first_token"],
        "addr_clean": norm_addr["addr_clean"],
        "addr_tokens_sorted": norm_addr["addr_tokens_sorted"],
        "addr_postal_code": norm_addr["addr_postal_code"],
        "addr_has_val": norm_addr["addr_has_val"],
        "country": norm_country
    }

def stream_and_normalize(
    input_tsv_path: str,
    output_tsv_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    log_interval: int = 100000
) -> int:
    """
    Streams a TSV file and normalizes every record to output_tsv_path.
    Returns the total count of processed records.
    """
    if not os.path.exists(input_tsv_path):
        raise FileNotFoundError(f"Input file not found: {input_tsv_path}")

    fieldnames = [
        "entity_id", "name_clean", "name_base", "name_tokens_sorted",
        "name_first_token", "addr_clean", "addr_tokens_sorted",
        "addr_postal_code", "addr_has_val", "country"
    ]

    if not output_tsv_path:
        raise ValueError("output_tsv_path must be provided for batch processing.")

    os.makedirs(os.path.dirname(os.path.abspath(output_tsv_path)), exist_ok=True)
    out_file = open(output_tsv_path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    start_time = time.time()
    count = 0

    try:
        with open(input_tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                norm_row = process_record(row)
                writer.writerow(norm_row)

                count += 1
                if count % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"Processed {count:,} records in {elapsed:.1f}s ({rate:,.0f} records/sec)")

                if max_rows and count >= max_rows:
                    break
    finally:
        out_file.close()

    total_time = time.time() - start_time
    print(f"Completed: {count:,} records written to {output_tsv_path} in {total_time:.2f}s.")
    return count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess and normalize entity dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input .tsv file")
    parser.add_argument("--output", type=str, default=None, help="Optional output .tsv path")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum rows to process (for testing)")
    args = parser.parse_args()

    stream_and_normalize(args.input, args.output, args.max_rows)
