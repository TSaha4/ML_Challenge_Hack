"""Polars-vectorised name and address normalisation.

Everything is expressed as polars string/list expressions so ~10M records
normalise in seconds.  Non-ASCII strings (accents, Devanagari/Gujarati/Tamil/
Telugu names and states) are transliterated once per *unique* value, first via
the Indic token dictionary learned from the training pairs (``translit.py``) and
otherwise via Unidecode.  No external data or services are used.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import polars as pl
from unidecode import unidecode

NON_ASCII = r"[^\x00-\x7F]"
_PUNCT = ',.;:"\'()[]{}#-'

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
#: word-level canonicalisation applied to name tokens
NAME_CANON: Dict[str, str] = {
    "private": "pvt", "pvt": "pvt", "prvt": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cos": "co",
    "and": "&", "et": "&", "n": "n",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services", "svc": "services",
    "assoc": "associates", "assocs": "associates",
    "bros": "brothers",
    "ctr": "center", "centre": "center",
    "tech": "technologies", "technology": "technologies",
    "ets": "etablissements",
}

#: legal-form tokens removed from the "core" name (after canonicalisation)
LEGAL = {
    "inc", "corp", "co", "llc", "ltd", "pvt", "llp", "lp", "plc", "pc", "pllc",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "selarl", "gmbh",
    "&", "the", "of", "de", "des", "du", "la", "le", "les", "d", "l",
}

#: honorifics / salutations prepended by some sources
HONORIFIC = {"dr", "mr", "mrs", "ms", "smt", "shri", "sri", "shree", "messrs", "kumari", "km", "mx"}

#: address token canonicalisation (US + India + France)
ADDR_CANON: Dict[str, str] = {
    # street types
    "street": "st", "str": "st", "st": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "av": "ave", "ave": "ave", "aven": "ave",
    "drive": "dr", "drv": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd", "boul": "blvd",
    "place": "pl", "pl": "pl",
    "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq",
    "expressway": "expy", "freeway": "fwy",
    "mount": "mt", "saint": "st", "sainte": "ste",
    "fort": "ft",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # France
    "r": "rue", "rue": "rue",
    "all": "allee", "allee": "allee",
    "imp": "impasse", "impasse": "impasse",
    "che": "chemin", "chem": "chemin", "chemin": "chemin",
    "qu": "quai", "quai": "quai",
    "rte": "route", "route": "route",
    "fbg": "faubourg", "faubourg": "faubourg",
    "sq.": "sq",
    # India
    "nr": "near", "near": "near", "opp": "opposite", "opposite": "opposite",
    "colony": "colony", "coly": "colony",
    "sector": "sector", "sec": "sector",
    "marg": "marg",
    "bombay": "mumbai", "poona": "pune", "gurugram": "gurgaon", "calcutta": "kolkata",
    "madras": "chennai", "bangalore": "bengaluru",
}

#: unit / number marker tokens that carry no identity by themselves
ADDR_DROP = {
    "unit", "apt", "apartment", "suite", "ste", "fl", "floor", "pmb", "bldg",
    "building", "room", "rm", "no", "door", "h", "hno", "hn", "flat", "plot",
    "shop", "house", "null", "none", "deg", "appt", "appartement", "bat",
    "batiment", "num", "number", "po", "box",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld", "ladakh": "la",
}

_ALIAS_RE = r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|f\s*/\s*k\s*/\s*a|f\.k\.a\.?|fka|a\s*/\s*k\s*/\s*a|a\.k\.a\.?|aka|nee|formerly|trading as|t\s*/\s*a)\b\s*:?"
_DOMAIN_RE = r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|biz|info|us)$"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _phrase_map_expr(expr: pl.Expr, mapping: Dict[str, str]) -> pl.Expr:
    """Replace multi-word phrases (longest first) on word boundaries."""
    multi = sorted((k for k in mapping if " " in k), key=len, reverse=True)
    for phrase in multi:
        expr = expr.str.replace_all(r"\b" + phrase + r"\b", mapping[phrase])
    return expr


def transliterate(s: pl.Series, token_map: Optional[Dict[str, str]] = None) -> pl.Series:
    """ASCII-fold a string series, touching only the unique non-ASCII values."""
    mask = s.str.contains(NON_ASCII).fill_null(False)
    uniq = s.filter(mask).unique()
    if uniq.len() == 0:
        return s
    tm = token_map or {}

    def fold(text: str) -> str:
        out = []
        for tok in text.split():
            if tok.isascii():
                out.append(tok)
                continue
            core = tok.strip(_PUNCT)
            hit = tm.get(core)
            out.append(tok.replace(core, hit) if hit else unidecode(tok))
        return " ".join(out)

    mapped = pl.Series([fold(x) for x in uniq.to_list()], dtype=pl.Utf8)
    return s.replace(uniq, mapped)


def _tokens_canon(expr: pl.Expr, canon: Dict[str, str]) -> pl.Expr:
    return expr.str.split(" ").list.eval(
        pl.element().replace(list(canon.keys()), list(canon.values()))
    )


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def normalize_names(df: pl.DataFrame, col: str = "business_name",
                    token_map: Optional[Dict[str, str]] = None) -> pl.DataFrame:
    """Add name representations.

    Columns added:
      ``nm``     canonical token string (honorifics dropped, legal words canonical, deduped)
      ``core``   ``nm`` without legal forms / stop-words
      ``sq``     ``core`` with spaces removed (matches domain-style names)
      ``nm_pre`` text before a DBA / f/k/a / née alias marker ('' if none)
      ``n_alias``/``n_domain``/``n_indic`` flags
    """
    raw = df.get_column(col).fill_null("")
    indic = raw.str.contains(r"[ऀ-෿]")
    folded = transliterate(raw, token_map).str.to_lowercase()

    tmp = pl.DataFrame({"_f": folded, "_indic": indic})
    tmp = tmp.with_columns(
        pl.col("_f")
        .str.replace_all(r"\bm\s*/\s*s\b\.?", " ")
        .str.replace_all(r"\bl\.\s*l\.\s*c\.?", "llc")
        .str.replace_all(r"\bl\.\s*l\.\s*p\.?", "llp")
        .str.replace_all(r"\bp\.\s*c\.", "pc")
        .str.replace_all(r"\s-\s*\d{7,}\s*$", " ")
        # digit-for-letter typos inside words: "c0astal", "hea1th", "federati0n"
        .str.replace_all(r"([a-z])0([a-z])", "${1}o${2}")
        .str.replace_all(r"([a-z])1([a-z])", "${1}l${2}")
        .str.replace_all(r"([a-z]{2})0\b", "${1}o")
        .str.strip_chars()
        .alias("_f")
    )
    tmp = tmp.with_columns(
        pl.col("_f").str.contains(_ALIAS_RE).alias("n_alias"),
        pl.col("_f").str.replace_all(r"^[^a-z0-9]+", "").str.contains(_DOMAIN_RE).alias("n_domain"),
    )
    # the real name follows the alias marker ("Fake DBA: Real Name")
    tmp = tmp.with_columns(
        pl.when(pl.col("n_alias"))
        .then(pl.col("_f").str.replace(r"^.*" + _ALIAS_RE, ""))
        .otherwise(pl.col("_f")).alias("_main"),
        pl.when(pl.col("n_alias"))
        .then(pl.col("_f").str.extract(r"^(.*?)" + _ALIAS_RE, 1))
        .otherwise(pl.lit("")).alias("nm_pre"),
    )
    tmp = tmp.with_columns(
        pl.when(pl.col("n_domain"))
        .then(pl.col("_main").str.replace_all(r"^[^a-z0-9]+", "").str.extract(_DOMAIN_RE, 1))
        .otherwise(pl.col("_main")).alias("_main")
    )

    def clean(e: pl.Expr) -> pl.Expr:
        return (
            e.str.replace_all(r"['`’]", "")
            .str.replace_all(r"&", " & ")
            .str.replace_all(r"[^a-z0-9&]+", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    tmp = tmp.with_columns(clean(pl.col("_main")).alias("_c"), clean(pl.col("nm_pre")).alias("nm_pre"))
    toks = (
        _tokens_canon(pl.col("_c"), NAME_CANON)
        .list.eval(pl.element().filter(~pl.element().is_in(list(HONORIFIC)) & (pl.element() != "")))
        .list.unique(maintain_order=True)
    )
    tmp = tmp.with_columns(toks.alias("_t"))
    tmp = tmp.with_columns(
        pl.col("_t").list.join(" ").alias("nm"),
        pl.col("_t").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))).alias("_ct"),
    )
    tmp = tmp.with_columns(
        pl.when(pl.col("_ct").list.len() > 0).then(pl.col("_ct")).otherwise(pl.col("_t")).alias("_ct")
    )
    tmp = tmp.with_columns(
        pl.col("_ct").list.join(" ").alias("core"),
        pl.col("_ct").list.join("").alias("sq"),
    )
    return df.with_columns(
        tmp.get_column("nm"),
        tmp.get_column("core"),
        tmp.get_column("sq"),
        tmp.get_column("nm_pre").fill_null(""),
        tmp.get_column("n_alias").cast(pl.Int8),
        tmp.get_column("n_domain").cast(pl.Int8),
        tmp.get_column("_indic").cast(pl.Int8).alias("n_indic"),
    )


# ---------------------------------------------------------------------------
# addresses
# ---------------------------------------------------------------------------
def normalize_addresses(df: pl.DataFrame, col: str = "business_address",
                        token_map: Optional[Dict[str, str]] = None) -> pl.DataFrame:
    """Add address representations.

    Columns added:
      ``ad``      canonical token string (abbreviations unified, leading zeros removed,
                  unit/number markers dropped, deduped)
      ``ad_nums`` space-joined numeric tokens in order
      ``ad_num``  first numeric token ('' if none)
      ``ad_null`` 1 when the address is missing
    """
    raw = df.get_column(col).fill_null("")
    raw = pl.Series(raw).str.replace(r"^(?i:none|null|nan)$", "")
    folded = transliterate(raw, token_map).str.to_lowercase()
    tmp = pl.DataFrame({"_f": folded})
    e = (
        pl.col("_f")
        .str.replace_all(r"n\s*deg\s*", " ")
        .str.replace_all(r"['`’]", "")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.replace_all(r"\b0+(\d)", "$1")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    e = _phrase_map_expr(e, US_STATES)
    e = _phrase_map_expr(e, IN_STATES)
    tmp = tmp.with_columns(e.alias("_c"))
    single_states = {k: v for k, v in {**US_STATES, **IN_STATES}.items() if " " not in k}
    canon = {**ADDR_CANON, **single_states}
    toks = (
        _tokens_canon(pl.col("_c"), canon)
        .list.eval(pl.element().filter(~pl.element().is_in(list(ADDR_DROP)) & (pl.element() != "")))
        .list.unique(maintain_order=True)
    )
    tmp = tmp.with_columns(toks.alias("_t"))
    tmp = tmp.with_columns(
        pl.col("_t").list.join(" ").alias("ad"),
        pl.col("_t").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+[a-z]?$"))).alias("_n"),
    )
    return df.with_columns(
        tmp.get_column("ad"),
        tmp.get_column("_n").list.join(" ").alias("ad_nums"),
        tmp.get_column("_n").list.first().fill_null("").alias("ad_num"),
        (tmp.get_column("ad") == "").cast(pl.Int8).alias("ad_null"),
    )


def normalize_frame(df: pl.DataFrame, name_map: Optional[Dict[str, str]] = None,
                    addr_map: Optional[Dict[str, str]] = None) -> pl.DataFrame:
    """Normalise names + addresses; keep only compact columns needed downstream."""
    df = normalize_names(df, token_map=name_map)
    df = normalize_addresses(df, token_map=addr_map)
    return df.select(
        "id", "entity_id", pl.col("country").fill_null("").str.strip_chars().str.to_lowercase().alias("cty"),
        "nm", "core", "sq", "nm_pre", "n_alias", "n_domain", "n_indic",
        "ad", "ad_nums", "ad_num", "ad_null",
    )
