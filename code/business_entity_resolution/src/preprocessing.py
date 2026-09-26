"""
Stage 1: Preprocessing & Data Normalization Module (Person 1)
Amazon ML Challenge 2026: Business Entity Resolution

This module transforms noisy, unstandardized business records from Source 1, 2, and 3
into clean, normalized datasets adhering strictly to the Stage 1 contract schema:
[entity_id, business_name_clean, business_address_clean, country, postal_code, name_tokens]
"""

import os
import re
import argparse
import unicodedata
from typing import Set, List, Optional, Tuple, Any
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. DICTIONARIES & REGEX DEFINITIONS
# ==========================================

# Legal and Corporate entity terms expansion mapping
LEGAL_MAPPINGS = [
    (re.compile(r'\bpvt\.?\s*ltd\.?\b', re.IGNORECASE), 'private limited'),
    (re.compile(r'\bpty\.?\s*ltd\.?\b', re.IGNORECASE), 'proprietary limited'),
    (re.compile(r'\bltd\.?\b', re.IGNORECASE), 'limited'),
    (re.compile(r'\binc\.?\b', re.IGNORECASE), 'incorporated'),
    (re.compile(r'\bcorp\.?\b', re.IGNORECASE), 'corporation'),
    (re.compile(r'\bllc\.?\b', re.IGNORECASE), 'limited liability company'),
    (re.compile(r'\bllp\.?\b', re.IGNORECASE), 'limited liability partnership'),
    (re.compile(r'\bplc\.?\b', re.IGNORECASE), 'public limited company'),
    (re.compile(r'\bco\.?\b', re.IGNORECASE), 'company'),
    (re.compile(r'\benterprises?\b', re.IGNORECASE), 'enterprise'),
    (re.compile(r'\bservices?\b', re.IGNORECASE), 'service'),
    (re.compile(r'\bgmbh\b', re.IGNORECASE), 'gmbh'),
    (re.compile(r'\bsarl\b', re.IGNORECASE), 'sarl'),
    (re.compile(r'\bsas\b', re.IGNORECASE), 'sas'),
    (re.compile(r'\bsa\b', re.IGNORECASE), 'sa'),
]

# Legal & Generic stop-words to exclude from `name_tokens`
LEGAL_WORDS = {
    'private', 'pvt', 'limited', 'ltd', 'incorporated', 'inc',
    'corporation', 'corp', 'company', 'co', 'enterprise', 'enterprises',
    'service', 'services', 'gmbh', 'sarl', 'sas', 'sa', 'llc', 'llp', 'plc',
    'pty', 'group', 'holdings', 'holding', 'associates', 'consulting',
    'solutions', 'technologies', 'technology', 'international', 'global',
    'liability', 'partners', 'partnership', 'industries', 'industry', 'ventures'
}

STOP_WORDS = {
    'the', 'and', 'of', 'in', 'for', 'at', 'by', 'to', 'a', 'an', 'on',
    'with', 'from', 'as', 'is', 'or', 'm/s', 'dr', 'mr', 'mrs', 'ms',
    'shri', 'shree', 'prof'
}

# Road and address abbreviations mapping
ADDRESS_MAPPINGS = [
    (re.compile(r'\brd\.?\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bst\.?\b', re.IGNORECASE), 'street'),
    (re.compile(r'\bave\.?\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bblvd\.?\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bdr\.?\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\bln\.?\b', re.IGNORECASE), 'lane'),
    (re.compile(r'\bct\.?\b', re.IGNORECASE), 'court'),
    (re.compile(r'\bpkwy\.?\b', re.IGNORECASE), 'parkway'),
    (re.compile(r'\bhwy\.?\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\bapt\.?\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\bste\.?\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\bfl\.?\b|\bflr\.?\b', re.IGNORECASE), 'floor'),
    (re.compile(r'\bbldg\.?\b', re.IGNORECASE), 'building'),
    (re.compile(r'\bhn\.?\b|\bh\.no\.?\b|\bhouse\s*no\.?\b', re.IGNORECASE), 'house'),
    (re.compile(r'\bflat\s*no\.?\b', re.IGNORECASE), 'flat'),
    (re.compile(r'\bplot\s*no\.?\b', re.IGNORECASE), 'plot'),
    (re.compile(r'\bopp\.?\b|\bopposite\b', re.IGNORECASE), 'opposite'),
    (re.compile(r'\bb/h\b|\bbehind\b', re.IGNORECASE), 'behind'),
    (re.compile(r'\bnear\b', re.IGNORECASE), 'near'),
    (re.compile(r'\bno\.?\b', re.IGNORECASE), 'number'),
]

# Compiled Regexes for sanitization
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_NULL_STRINGS = re.compile(r'^(null|none|nan|undefined|\s*)$', re.IGNORECASE)
RE_PREFIX_HONORIFICS = re.compile(r'^(m\s*s|dr|mr|mrs|ms|shree|shri|prof)\s+', re.IGNORECASE)
RE_LEADING_ZEROS = re.compile(r'\b0+([1-9]\d*)\b')
RE_REPEATED_WORDS = re.compile(r'\b(\w+)(?:\s+\1\b)+', re.IGNORECASE)

# Country Postal Code Regexes (Precision-oriented)
RE_POSTAL_INDIA = re.compile(r'\b([1-9][0-9]{2}\s?[0-9]{3})\b')                       # 6 digits (e.g. 110001, 500 034)
RE_POSTAL_US_CONTEXT = re.compile(r'(?:[A-Z]{2}[,\s]+|,\s*|\bstate\s+)([0-9]{5}(?:-[0-9]{4})?)\b', re.IGNORECASE) # US ZIP with state/comma context
RE_POSTAL_US_END = re.compile(r'\b([0-9]{5}(?:-[0-9]{4})?)$')                          # US ZIP at the end of address
RE_POSTAL_FRANCE = re.compile(r'\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b')                 # 5 digits (e.g. 75001, 13008)


# ==========================================
# 2. CORE TRANSFORMATION FUNCTIONS
# ==========================================

def sanitize_text(text: Optional[str]) -> str:
    """Base text cleaning: lowercase, unicode normalization (NFKD), special char removal."""
    if text is None:
        return ""
    text_str = str(text).strip()
    if RE_NULL_STRINGS.match(text_str):
        return ""
    
    # Unicode decomposition (e.g., 'café' -> 'cafe', 'পশ্চিমবঙ্গ' -> decomposed/safe ascii)
    text_str = unicodedata.normalize('NFKD', text_str).encode('ASCII', 'ignore').decode('utf-8')
    text_str = text_str.lower()
    
    # Standardize common symbols
    text_str = text_str.replace('&', ' and ')
    text_str = text_str.replace('@', ' at ')
    text_str = text_str.replace('/', ' ')
    text_str = text_str.replace('-', ' ')
    text_str = text_str.replace('_', ' ')
    
    # Strip non-alphanumeric chars
    text_str = RE_NON_ALPHANUM.sub(' ', text_str)
    # Collapse multiple spaces
    text_str = RE_MULTIPLE_SPACES.sub(' ', text_str).strip()
    return text_str


def clean_business_name(name: Optional[str]) -> str:
    """Normalizes business names, expands legal abbreviations, strips honorifics."""
    norm = sanitize_text(name)
    if not norm:
        return ""
    
    # Strip honorifics from the beginning of the name
    norm = RE_PREFIX_HONORIFICS.sub('', norm)
    
    # Standardize legal suffixes
    for pattern, replacement in LEGAL_MAPPINGS:
        norm = pattern.sub(replacement, norm)
        
    # Collapse accidental duplicated adjacent words
    norm = RE_REPEATED_WORDS.sub(r'\1', norm)
    norm = RE_MULTIPLE_SPACES.sub(' ', norm).strip()
    return norm


def clean_address(address: Optional[str]) -> str:
    """Normalizes address strings, expands road/unit types, standardizes numbers."""
    norm = sanitize_text(address)
    if not norm:
        return ""
    
    # Remove leading zeros on stand-alone numbers (e.g., 0017560 -> 17560)
    norm = RE_LEADING_ZEROS.sub(r'\1', norm)
    
    # Expand address abbreviations
    for pattern, replacement in ADDRESS_MAPPINGS:
        norm = pattern.sub(replacement, norm)
        
    # Collapse accidental duplicated adjacent words (e.g., road road -> road)
    norm = RE_REPEATED_WORDS.sub(r'\1', norm)
    norm = RE_MULTIPLE_SPACES.sub(' ', norm).strip()
    return norm


def extract_postal_code(address: Optional[str], country: Optional[str] = None) -> str:
    """
    Extracts the postal code based on the record's country (US, India, France, or generic).
    Avoids false positives from house numbers at the start of addresses.
    """
    if not address:
        return ""
    addr_str = str(address).strip()
    if not addr_str or RE_NULL_STRINGS.match(addr_str):
        return ""
    
    country_clean = str(country).strip().upper() if country else ""
    
    if country_clean == 'INDIA':
        m = RE_POSTAL_INDIA.search(addr_str)
        if m:
            return m.group(1).replace(' ', '')
    elif country_clean == 'US':
        # Look for ZIP preceded by state/comma, or at the end of the address
        m = RE_POSTAL_US_CONTEXT.search(addr_str)
        if m:
            return m.group(1).split('-')[0]
        m_end = RE_POSTAL_US_END.search(addr_str)
        if m_end:
            return m_end.group(1).split('-')[0]
    elif country_clean == 'FRANCE':
        m = RE_POSTAL_FRANCE.search(addr_str)
        if m:
            return m.group(1)
            
    return ""


def extract_name_tokens(clean_name: str) -> str:
    """
    Extracts informative name tokens excluding stop words and generic legal terms.
    Provides a guaranteed fallback to original tokens if all are filtered out.
    """
    if not clean_name:
        return ""
    
    tokens = clean_name.split()
    filtered = []
    seen = set()
    
    for tok in tokens:
        if tok in STOP_WORDS or tok in LEGAL_WORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            filtered.append(tok)
            
    longer_tokens = [t for t in filtered if len(t) > 1]
    if longer_tokens:
        return " ".join(longer_tokens)
    if filtered:
        return " ".join(filtered)
        
    # Safe Fallback: If all tokens were legal words (e.g. 'Global Solutions LLC'), keep original unique tokens
    fallback = []
    seen_fb = set()
    for tok in tokens:
        if tok not in seen_fb:
            seen_fb.add(tok)
            fallback.append(tok)
    return " ".join(fallback)


# ==========================================
# 3. DATAFRAME BATCH NORMALIZATION
# ==========================================

def _normalize_chunk(records: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str, str, str]]:
    """Worker function for parallel processing across multiple CPU cores."""
    result = []
    for eid, name, addr, country in records:
        cn = clean_business_name(name)
        ca = clean_address(addr)
        pc = extract_postal_code(addr, country)
        toks = extract_name_tokens(cn)
        result.append((eid, cn, ca, country, pc, toks))
    return result


def normalize_dataframe(df: pd.DataFrame, n_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Transforms a raw input DataFrame into the standardized Stage 1 schema:
    [entity_id, business_name_clean, business_address_clean, country, postal_code, name_tokens]
    Uses multi-core parallel processing for large datasets.
    """
    required_cols = {'entity_id', 'business_name', 'business_address', 'country'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    entity_ids = [str(x).strip() if pd.notna(x) else "" for x in df['entity_id']]
    names = [str(x) if pd.notna(x) else "" for x in df['business_name']]
    addresses = [str(x) if pd.notna(x) else "" for x in df['business_address']]
    countries = [str(x).strip() if pd.notna(x) else "" for x in df['country']]

    n_rows = len(df)
    # Sequential for small inputs
    if n_rows < 20000:
        clean_names = [clean_business_name(n) for n in names]
        clean_addrs = [clean_address(a) for a in addresses]
        postal_codes = [extract_postal_code(a, c) for a, c in zip(addresses, countries)]
        name_tokens = [extract_name_tokens(cn) for cn in clean_names]
        return pd.DataFrame({
            'entity_id': entity_ids,
            'business_name_clean': clean_names,
            'business_address_clean': clean_addrs,
            'country': countries,
            'postal_code': postal_codes,
            'name_tokens': name_tokens
        }, dtype=str)

    # Multi-core parallel execution for large datasets (e.g. 2M - 5M rows)
    from concurrent.futures import ProcessPoolExecutor

    workers = n_jobs or min(os.cpu_count() or 4, 16)
    chunk_size = (n_rows + workers - 1) // workers
    records = list(zip(entity_ids, names, addresses, countries))
    chunks = [records[i * chunk_size : (i + 1) * chunk_size] for i in range(workers) if i * chunk_size < n_rows]

    print(f"  Parallel normalization across {len(chunks)} CPU worker processes...")
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        chunk_results = list(executor.map(_normalize_chunk, chunks))

    all_normalized = [row for chunk in chunk_results for row in chunk]

    return pd.DataFrame(
        all_normalized,
        columns=['entity_id', 'business_name_clean', 'business_address_clean', 'country', 'postal_code', 'name_tokens'],
        dtype=str
    )


def preprocess_tsv(input_filepath: str) -> pd.DataFrame:
    """Reads a raw TSV, normalizes it, and returns the cleaned DataFrame."""
    print(f"Loading raw data from: {input_filepath}")
    df = pd.read_csv(input_filepath, sep='\t', dtype=str)
    return normalize_dataframe(df)


def process_file(input_filepath: str, output_filepath: str) -> None:
    """Reads a raw TSV, normalizes it, and saves it as a clean TSV."""
    if os.path.isfile(output_filepath) and os.path.getsize(output_filepath) > 1024:
        print(f"File already normalized: {output_filepath} (skipping re-run)\n")
        return

    print(f"Loading raw data from: {input_filepath}")
    df = pd.read_csv(input_filepath, sep='\t', dtype=str)

    print(f"Normalizing {len(df):,} records...")
    norm_df = normalize_dataframe(df)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    norm_df.to_csv(output_filepath, sep='\t', index=False)
    print(f"Successfully saved normalized dataset to: {output_filepath} ({len(norm_df):,} rows)\n")


def process_directory(input_dir: str, output_dir: str) -> None:
    """Processes all source1, source2, and source3 TSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.tsv') or 'ground_truth' in filename:
            continue
        
        if 'source1' in filename:
            out_name = 'normalized_source1.tsv'
        elif 'source2' in filename:
            out_name = 'normalized_source2.tsv'
        elif 'source3' in filename:
            out_name = 'normalized_source3.tsv'
        else:
            out_name = f"normalized_{filename}"
            
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, out_name)
        process_file(in_path, out_path)


# ==========================================
# 4. CLI ENTRY POINT
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1: Preprocessing & Data Normalization")
    parser.add_argument('--input-dir', type=str, help="Directory containing raw TSV files (e.g., dataset/train)")
    parser.add_argument('--output-dir', type=str, help="Directory to save normalized TSV files (e.g., data/normalized)")
    parser.add_argument('--input-file', type=str, default=None, help="Path to a single raw TSV file")
    parser.add_argument('--output-file', type=str, default=None, help="Path to save the single normalized TSV")

    args = parser.parse_args()

    if args.input_file and args.output_file:
        process_file(args.input_file, args.output_file)
    elif args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir)
    else:
        parser.print_help()

