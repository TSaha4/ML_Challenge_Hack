"""
ER Core Utilities & Normalization Module
Handles multilingual representation, legal suffixes, normalization, and token statistics.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any
from unidecode import unidecode

COMMON_LEGAL_SUFFIXES = [
    # English / US / International
    "inc", "incorporated", "corp", "corporation", "llc", "l.l.c", "l.l.c.",
    "ltd", "limited", "pvt ltd", "pvt. ltd.", "private limited", "co", "company",
    "plc", "llp", "gmbh", "sa", "sarl", "sas", "bv", "ag", "nv", "spa", "srl",
    "pty ltd", "pty", "enterprises", "enterprise", "holdings", "holding", "group",
    "services", "solutions", "technologies", "tech", "international", "intl"
]

_LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in sorted(COMMON_LEGAL_SUFFIXES, key=len, reverse=True)) + r')\b',
    flags=re.IGNORECASE
)

PUNCT_REGEX = re.compile(r'[^\w\s]', flags=re.UNICODE)
MULTI_SPACE_REGEX = re.compile(r'\s+')


def normalize_text(text: Optional[str]) -> Dict[str, str]:
    """
    Creates multiple representations of an input text string without destroying Unicode:
    - raw: original string or empty string
    - lowercase: standard lower
    - unicode_normalized: NFKC decomposition + recomposition
    - punctuation_normalized: punctuation replaced by space
    - alphanumeric_normalized: only alphanumeric + space
    - core_name: legal suffixes removed (for business names)
    - sorted_tokens: alphabetically sorted tokens
    - initials: first letters of each token
    - transliterated: ASCII transliteration via Unidecode (preserving original in raw/other fields)
    """
    if text is None:
        return {
            "raw": "",
            "lowercase": "",
            "unicode_normalized": "",
            "punctuation_normalized": "",
            "alphanumeric_normalized": "",
            "core_name": "",
            "sorted_tokens": "",
            "initials": "",
            "transliterated": "",
        }

    raw = str(text).strip()
    lowered = raw.lower()
    
    # Unicode normalized (NFKC preserves semantics while unifying compatibility forms)
    uni_norm = unicodedata.normalize("NFKC", lowered)
    
    # Punctuation normalized: replace non-word/space with single space
    punct_norm = PUNCT_REGEX.sub(" ", uni_norm)
    punct_norm = MULTI_SPACE_REGEX.sub(" ", punct_norm).strip()
    
    tokens = punct_norm.split()
    
    # Legal suffix extraction & core name
    core_name = _LEGAL_SUFFIX_PATTERN.sub(" ", punct_norm)
    core_name = MULTI_SPACE_REGEX.sub(" ", core_name).strip()
    if not core_name:
        core_name = punct_norm  # Avoid empty core name if entity name is only suffix
        
    sorted_tokens = " ".join(sorted(tokens))
    initials = "".join([t[0] for t in tokens if t])
    translit = unidecode(raw).lower().strip()
    translit = PUNCT_REGEX.sub(" ", translit)
    translit = MULTI_SPACE_REGEX.sub(" ", translit).strip()

    return {
        "raw": raw,
        "lowercase": lowered,
        "unicode_normalized": uni_norm,
        "punctuation_normalized": punct_norm,
        "alphanumeric_normalized": punct_norm,
        "core_name": core_name,
        "sorted_tokens": sorted_tokens,
        "initials": initials,
        "transliterated": translit,
    }


def normalize_address(address: Optional[str]) -> Dict[str, str]:
    """
    Normalizes business address components and extracts PIN/postal codes.
    Handles US (5-digit ZIP / 5+4), India (6-digit PIN), France (5-digit code).
    """
    if address is None:
        return {"raw": "", "normalized": "", "postal_code": "", "tokens": ""}

    raw = str(address).strip()
    norm = unicodedata.normalize("NFKC", raw.lower())
    norm = PUNCT_REGEX.sub(" ", norm)
    norm = MULTI_SPACE_REGEX.sub(" ", norm).strip()

    # Postal code extraction regex: 6 digits (India) or 5 digits (US/France)
    postal_code = ""
    # Try 6-digit PIN first
    m6 = re.search(r'\b[1-9]\d{5}\b', raw)
    if m6:
        postal_code = m6.group(0)
    else:
        # Try 5-digit ZIP
        m5 = re.search(r'\b\d{5}\b', raw)
        if m5:
            postal_code = m5.group(0)

    return {
        "raw": raw,
        "normalized": norm,
        "postal_code": postal_code,
        "tokens": " ".join(sorted(set(norm.split()))),
    }
