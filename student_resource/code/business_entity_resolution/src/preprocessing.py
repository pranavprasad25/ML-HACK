# Stage 1: Text Cleaning & Normalization
import re
import unicodedata
import os
import argparse
import pandas as pd

# Honorifics & Salutations to strip
HONORIFICS = [
    r'\bm/s\b', r'\bdr\b', r'\bshri\b', r'\bmr\b', r'\bmrs\b', r'\bms\b',
    r'\bprof\b', r'\bmme\b', r'\bmlle\b', r'\bmonsieur\b'
]
HONORIFIC_PATTERN = re.compile(r'|'.join(HONORIFICS), flags=re.IGNORECASE)

# Legal Suffixes to expand & normalize / strip
LEGAL_SUFFIXES = [
    r'\bpvt\s+ltd\b', r'\bprivate\s+limited\b', r'\binc\b', r'\bincorporated\b',
    r'\bllc\b', r'\bllp\b', r'\bltd\b', r'\blimited\b', r'\bcorp\b', r'\bcorporation\b',
    r'\bco\b', r'\bcompany\b', r'\bpc\b', r'\bpa\b', r'\bsarl\b', r'\bsa\b', r'\bgmbh\b'
]
SUFFIX_PATTERN = re.compile(r'|'.join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Address Abbreviations & Landmark noise
ADDRESS_ABBR = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bdr\b': 'drive', r'\bblvd\b': 'boulevard', r'\bhwy\b': 'highway',
    r'\bln\b': 'lane', r'\bpkwy\b': 'parkway', r'\bste\b': 'suite',
    r'\bapt\b': 'apartment', r'\bfl\b': 'floor', r'\bbldg\b': 'building',
    r'\bno\b': 'number', r'\bopp\b': 'opposite', r'\bnr\b': 'near', r'\bb/h\b': 'behind'
}

def clean_text(text: str) -> str:
    """Removes accents, lowercases, handles symbols, and normalizes whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    text = text.replace('&', ' and ').replace('+', ' plus ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_business_name(name: str):
    """Strips honorifics, cleans text, and generates clean_name and stripped core_name."""
    cleaned = clean_text(name)
    cleaned = HONORIFIC_PATTERN.sub('', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    core = SUFFIX_PATTERN.sub('', cleaned)
    core = re.sub(r'\s+', ' ', core).strip()
    return cleaned, core if core else cleaned

def normalize_address(address: str) -> str:
    """Standardizes street types, sub-units, and strips landmark noise."""
    addr = clean_text(address)
    for abbr, full in ADDRESS_ABBR.items():
        addr = re.sub(abbr, full, addr)
    return re.sub(r'\s+', ' ', addr).strip()

def extract_postal_code(address: str, country: str) -> str:
    """Extracts country-specific postal code (US, India, France)."""
    if not isinstance(address, str):
        return ""
    country_upper = str(country).upper().strip()
    if country_upper == 'US':
        match = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
        return match.group(1) if match else ""
    elif country_upper == 'INDIA':
        match = re.search(r'\b([1-9]\d{5})\b', address)
        return match.group(1) if match else ""
    elif country_upper == 'FRANCE':
        match = re.search(r'\b(0[1-9]|[1-8]\d|9[0-8])\d{3}\b', address)
        return match.group(1) if match else ""
    match = re.search(r'\b\d{5,6}\b', address)
    return match.group(0) if match else ""

def extract_digits(text: str) -> str:
    """Extracts all numerical digits from text for digit overlap matching."""
    return " ".join(re.findall(r'\b\d+\b', text))

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a DataFrame containing raw business entity records."""
    names = df['business_name'].apply(process_business_name)
    df['clean_name'] = [n[0] for n in names]
    df['core_name'] = [n[1] for n in names]
    df['clean_address'] = df['business_address'].apply(normalize_address)
    df['postal_code'] = df.apply(lambda r: extract_postal_code(r['business_address'], r['country']), axis=1)
    df['extracted_numbers'] = df['clean_address'].apply(extract_digits)
    df['first_word'] = df['core_name'].apply(lambda x: x.split()[0] if x else "")
    return df

def preprocess_tsv(file_path: str) -> pd.DataFrame:
    print(f"Preprocessing {file_path}...")
    df = pd.read_csv(file_path, sep="\t", dtype=str).fillna("")
    return preprocess_df(df)

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocessing & Data Normalization")
    parser.add_argument("--input-dir", required=True, help="Input directory containing TSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for normalized TSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if fname.endswith(".tsv") and "ground_truth" not in fname:
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(args.output_dir, fname)
            print(f"Processing {in_path} -> {out_path}")
            df = preprocess_tsv(in_path)
            df.to_csv(out_path, sep="\t", index=False)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
