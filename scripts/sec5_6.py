import nbformat as nbf

def get_p5_p6_cells():
    c = []
    # Phase 5
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 5 — Normalization & Multi-Representation Engine
Demonstrates Unicode-preserving normalization that produces multiple views:
- `raw`: Original untouched string
- `unicode_normalized`: NFKC canonical equivalence
- `core_name`: Business name stripped of jurisdictional legal suffixes (*Corp*, *LLC*, *Pvt Ltd*)
- `sorted_tokens`: Word tokens sorted alphabetically to conquer token order permutation
- `transliterated`: ASCII phonetic conversion via Unidecode for non-Latin / romanized names
- `postal_code`: Extracted 5-digit (US/FR) or 6-digit (IN) postal code
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Demonstrate multi-representation normalization on real sample pairs
test_names = [
    "Orelee's Barbershop Inc.",
    "B+ Retail Corporation",
    "Tata Consultancy Services Pvt. Ltd.",
    "Société Générale S.A.S.",
    "M/S Sharma Traders & Solutions"
]

print("Multi-Representation Normalization Demo:")
norm_demo = []
for name in test_names:
    rep = normalize_text(name)
    norm_demo.append({
        "Original": rep["raw"],
        "Unicode Norm": rep["unicode_normalized"],
        "Core Name": rep["core_name"],
        "Sorted Tokens": rep["sorted_tokens"],
        "Initials": rep["initials"],
        "Transliterated": rep["transliterated"]
    })

display(pd.DataFrame(norm_demo))
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Address Normalization & Postal Code Extraction Demo
test_addrs = [
    "1795 Westchester Drive, High Point, NC 27262",
    "Plot No 44, Okhla Phase 3, New Delhi, 110020",
    "15 Rue de Rivoli, 75001 Paris",
    "Near SBI ATM, MG Road, Bangalore",
    "PO Box 1234, Seattle WA 98101-1234"
]

addr_demo = []
for addr in test_addrs:
    rep = normalize_address(addr)
    addr_demo.append({
        "Original": rep["raw"],
        "Normalized": rep["normalized"],
        "Extracted Postal": rep["postal_code"],
        "Token Set": rep["tokens"][:30] + "..." if len(rep["tokens"]) > 30 else rep["tokens"]
    })

display(pd.DataFrame(addr_demo))
"""
    ))

    # Phase 6
    c.append(nbf.v4.new_markdown_cell(
"""# Phase 6 — Adaptive Multi-Channel Blocking
Candidate generation reduces the $2.2M \\times 10M = 2.2 \\times 10^{13}$ all-pairs search space down to high-probability candidates.

### Adaptive Blocking Logic:
1. **Primary Channels**: Exact Core Name, Exact Normalized Name, Country + First Word, Transliterated Name.
2. **Locality Channels**: Country + Postal Code + Name Prefix.
3. **Adaptive Thresholds**: 
   - Blocks with $< 500$ records are retained completely.
   - Massive blocks (e.g., common names like *Starbucks* or generic addresses) are capped and prioritized by channel intersection to preserve recall without explosion.
"""
    ))

    c.append(nbf.v4.new_code_cell(
"""# Demonstrate blocking key extraction on sample records
sample_rec = {
    "entity_id": "S1-925783039",
    "business_name": "Orelee's Barbershop Inc.",
    "business_address": "1795 Westchester Drive, High Point, NC 27262",
    "country": "US",
    "name_norm": "orelee s barbershop inc",
    "name_core": "orelee s barbershop",
    "name_translit": "orelee s barbershop inc",
    "addr_postal": "27262"
}

keys = extract_blocking_keys(sample_rec)
print("Generated Blocking Keys for Sample S1 Record:")
for ch, val in keys:
    print(f" - [{ch}]: '{val}'")
"""
    ))
    return c
