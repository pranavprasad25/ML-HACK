"""
Data Normalization & Cleaning Module for Business Entity Resolution.

Handles:
1. Unicode & accent removal (diacritics like é -> e).
2. Business name normalization, dotted acronym compaction (S.A.R.L. -> sarl), legal suffix extraction & removal.
3. Symbol expansion (& -> and, + -> plus).
4. Street and locality abbreviation expansions (Rd -> road, St -> street, etc.).
5. State code standardizations (US & India).
6. Number and postal/PIN code extraction.
7. Token-sorted canonical representation (resilient to word-order flips).
"""

import re
import unicodedata
from typing import Dict, Any, List

# ==============================================================================
# 1. LEGAL SUFFIXES (US, India, France, International)
# ==============================================================================

LEGAL_SUFFIX_PATTERNS = [
    # Multi-word combinations first
    r'\b(pvt|private)\s+(ltd|limited)\b',
    r'\b(pte|private)\s+ltd\b',
    r'\bproprietorship\b',
    r'\bproprietor\b',
    r'\bprop\b',
    r'\bpartnership\b',
    r'\b(corp|corporation)\b',
    r'\b(inc|incorporated)\b',
    r'\bllc\b',
    r'\bllp\b',
    r'\b(ltd|limited)\b',
    r'\b(co|company)\b',
    r'\bcompanies\b',
    r'\bpllc\b',
    r'\bpc\b',
    # European / French suffixes
    r'\bsarl\b',
    r'\bsas\b',
    r'\bsasu\b',
    r'\bsa\b',
    r'\beurl\b',
    r'\bsnc\b',
    r'\bgmbh\b',
    r'\bag\b',
    # Common associations / organizations
    r'\bassociation\b',
    r'\bassoc\b',
    r'\bfoundation\b',
    r'\btrust\b',
    r'\bsociety\b',
    r'\bholding(s)?\b',
    r'\bgroup\b'
]

COMPILED_LEGAL_SUFFIXES = [re.compile(p, re.IGNORECASE) for p in LEGAL_SUFFIX_PATTERNS]

# ==============================================================================
# 2. STREET & LOCALITY ABBREVIATIONS
# ==============================================================================

STREET_ABBREVIATIONS = [
    # Indian / Building designations with optional dot and numbers
    (r'\bh\.?\s*no\.?\s*', 'house number '),
    (r'\bd\.?\s*no\.?\s*', 'door number '),
    (r'\bflat\.?\s*no\.?\s*', 'flat number '),
    (r'\bplot\.?\s*no\.?\s*', 'plot number '),
    (r'\bopp\.?\s*', 'opposite '),
    (r'\bnr\.?\s*', 'near '),
    # Street suffixes
    (r'\brd\b', 'road'),
    (r'\bst\b', 'street'),
    (r'\bave\b', 'avenue'),
    (r'\bav\b', 'avenue'),
    (r'\bdr\b', 'drive'),
    (r'\bln\b', 'lane'),
    (r'\bblvd\b', 'boulevard'),
    (r'\bbvd\b', 'boulevard'),
    (r'\bpkwy\b', 'parkway'),
    (r'\bhwy\b', 'highway'),
    (r'\bct\b', 'court'),
    (r'\bpl\b', 'place'),
    (r'\bsq\b', 'square'),
    (r'\bcir\b', 'circle'),
    (r'\bter\b', 'terrace'),
    (r'\bwy\b', 'way'),
    (r'\btrl\b', 'trail'),
    (r'\baly\b', 'alley'),
    (r'\bapt\b', 'apartment'),
    (r'\bste\b', 'suite'),
    (r'\bfl\b', 'floor'),
    (r'\bbldg\b', 'building'),
    (r'\bdept\b', 'department'),
    (r'\brm\b', 'room'),
    (r'\bbsmt\b', 'basement'),
    (r'\bunit\b', 'unit'),
    # Directions
    (r'\bn\b', 'north'),
    (r'\bs\b', 'south'),
    (r'\be\b', 'east'),
    (r'\bw\b', 'west'),
    (r'\bne\b', 'northeast'),
    (r'\bnw\b', 'northwest'),
    (r'\bse\b', 'southeast'),
    (r'\bsw\b', 'southwest'),
    # French specific address prefixes
    (r'\brue\b', 'rue'),
    (r'\ball\b', 'allee'),
    (r'\ballee\b', 'allee'),
    (r'\bimp\b', 'impasse'),
    (r'\bimpasse\b', 'impasse')
]

COMPILED_STREET_ABBRS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in STREET_ABBREVIATIONS]

# ==============================================================================
# 3. STATE CODE MAPPINGS (US & India)
# ==============================================================================

US_STATE_CODES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
}

INDIA_STATE_CODES = {
    "mh": "maharashtra", "dl": "delhi", "ka": "karnataka", "tn": "tamil nadu",
    "wb": "west bengal", "up": "uttar pradesh", "ap": "andhra pradesh", "tg": "telangana",
    "ts": "telangana", "gj": "gujarat", "rj": "rajasthan", "kl": "kerala", "mp": "madhya pradesh",
    "pb": "punjab", "hr": "haryana", "br": "bihar", "or": "odisha", "od": "odisha",
    "as": "assam", "jh": "jharkhand", "uk": "uttarakhand", "ua": "uttarakhand",
    "hp": "himachal pradesh", "ga": "goa", "ct": "chhattisgarh", "cg": "chhattisgarh"
}


# ==============================================================================
# 4. NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_unicode(text: str) -> str:
    """Strip diacritics/accents and convert to clean lowercase ASCII."""
    if not text or not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize('NFKD', text)
    ascii_bytes = decomposed.encode('ASCII', 'ignore')
    return ascii_bytes.decode('utf-8').lower().strip()


def compact_dotted_acronyms(text: str) -> str:
    """
    Transforms dotted acronyms like 'S.A.R.L.' -> 'sarl', 'U.S.A.' -> 'usa', 'L.L.C.' -> 'llc'.
    """
    # Matches patterns like a.b.c. or a.b.
    def replace_dots(m):
        return m.group(0).replace('.', '').lower()
    return re.sub(r'\b([A-Za-z]\.){2,}[A-Za-z]?\.?\b', replace_dots, text)


def normalize_business_name(name: str) -> Dict[str, Any]:
    """
    Normalizes a business name:
    - Unicode / diacritics stripped
    - Dotted acronyms compacted (e.g. S.A.R.L. -> sarl, L.L.C. -> llc)
    - Symbols expanded (& -> and, + -> plus)
    - Punctuation removed
    - Legal suffixes identified and removed to produce 'name_base'
    - Sorted tokens created to handle word-order inversions
    """
    if not name or not isinstance(name, str):
        return {
            "name_clean": "",
            "name_base": "",
            "name_legal_suffix": "",
            "name_tokens_sorted": "",
            "name_first_token": "",
            "name_char_len": 0,
            "name_word_count": 0
        }

    # 1. Unicode & lower
    s = normalize_unicode(name)

    # 2. Compact dotted acronyms before stripping dots
    s = compact_dotted_acronyms(s)

    # 3. Expand common symbols
    s = re.sub(r'&', ' and ', s)
    s = re.sub(r'\+', ' plus ', s)
    s = re.sub(r'@', ' at ', s)
    s = re.sub(r'%', ' percent ', s)

    # 4. Strip apostrophes completely (Orelee's -> orelees)
    s = re.sub(r"['’`]", '', s)

    # 5. Remove remaining non-alphanumeric characters
    clean = re.sub(r'[^a-z0-9\s]', ' ', s)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # 6. Extract and strip legal suffixes to obtain base name
    base = clean
    detected_suffixes = []
    for pattern in COMPILED_LEGAL_SUFFIXES:
        match = pattern.search(base)
        if match:
            detected_suffixes.append(match.group(0))
            base = pattern.sub(' ', base)

    base = re.sub(r'\s+', ' ', base).strip()
    if not base:
        # Avoid empty base if entire name was a legal keyword
        base = clean

    # 7. Word tokens & sorted canonical representation
    tokens = [t for t in base.split() if len(t) > 0]
    sorted_tokens = " ".join(sorted(tokens))
    first_token = tokens[0] if tokens else ""

    return {
        "name_clean": clean,
        "name_base": base,
        "name_legal_suffix": " ".join(detected_suffixes),
        "name_tokens_sorted": sorted_tokens,
        "name_first_token": first_token,
        "name_char_len": len(clean),
        "name_word_count": len(tokens)
    }


def normalize_address(address: str, country: str = "") -> Dict[str, Any]:
    """
    Normalizes a business address:
    - Expands street, unit, direction abbreviations
    - Maps 2-letter state codes to full names (US & India)
    - Extracts numbers (building numbers, house numbers)
    - Extracts country-specific postal / PIN codes
    - Produces sorted tokens for order-invariant address matching
    """
    if not address or not isinstance(address, str):
        return {
            "addr_clean": "",
            "addr_tokens_sorted": "",
            "addr_numbers": [],
            "addr_postal_code": "",
            "addr_has_val": 0,
            "addr_char_len": 0
        }

    s = normalize_unicode(address)

    # 1. Expand abbreviations & landmark markers
    for pattern, repl in COMPILED_STREET_ABBRS:
        s = pattern.sub(repl, s)

    # 2. Extract postal / PIN codes before stripping punctuation
    postal_code = ""
    country_upper = str(country).strip().upper()
    if country_upper in ["INDIA", "IN"]:
        pin_match = re.search(r'\b[1-9][0-9]{5}\b', s)
        if pin_match:
            postal_code = pin_match.group(0)
    elif country_upper in ["US", "USA", "FRANCE", "FR"]:
        zip_match = re.search(r'\b[0-9]{5}\b', s)
        if zip_match:
            postal_code = zip_match.group(0)
    if not postal_code:
        generic_match = re.search(r'\b[0-9]{5,6}\b', s)
        if generic_match:
            postal_code = generic_match.group(0)

    # 3. Extract all numbers (house/building numbers, floor, etc.)
    numbers = re.findall(r'\b\d+\b', s)

    # 4. Clean punctuation
    clean_addr = re.sub(r'[^a-z0-9\s]', ' ', s)

    # 5. Expand state abbreviations if token matches
    tokens = clean_addr.split()
    expanded_tokens = []
    state_dict = US_STATE_CODES if country_upper in ["US", "USA"] else (INDIA_STATE_CODES if country_upper in ["INDIA", "IN"] else {})

    for t in tokens:
        if t in state_dict:
            expanded_tokens.extend(state_dict[t].split())
        else:
            expanded_tokens.append(t)

    clean_addr = " ".join(expanded_tokens)
    clean_addr = re.sub(r'\s+', ' ', clean_addr).strip()

    # 6. Sorted tokens
    addr_tokens = [t for t in clean_addr.split() if len(t) > 1]
    sorted_addr_tokens = " ".join(sorted(addr_tokens))

    return {
        "addr_clean": clean_addr,
        "addr_tokens_sorted": sorted_addr_tokens,
        "addr_numbers": numbers,
        "addr_postal_code": postal_code,
        "addr_has_val": 1 if len(clean_addr) > 0 else 0,
        "addr_char_len": len(clean_addr)
    }


def normalize_country(country: str) -> str:
    """Standardizes country label into canonical uppercase string."""
    if not country or not isinstance(country, str):
        return "UNKNOWN"
    c = country.strip().upper()
    if c in ["US", "USA", "UNITED STATES"]:
        return "US"
    if c in ["INDIA", "IND", "IN"]:
        return "INDIA"
    if c in ["FRANCE", "FR", "FRA"]:
        return "FRANCE"
    return c
