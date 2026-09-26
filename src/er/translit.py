"""Indic-script -> Latin token dictionary learned from the training ground truth.

About a quarter of Indian Source-2/3 names (and some state components of
addresses) are written in Devanagari / Gujarati / Tamil / Telugu / Kannada /
Bengali ... script, while Source-1 is always Latin.  Unidecode's phonetic output
("iNddo bilddrs praaivett") is a poor match for the English spelling ("indo
builders private"), so we align tokens position-by-position on labelled pairs
with equal token counts and keep the dominant Latin spelling per native token.
Only the provided training data is used.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import polars as pl

from src.er.text import IN_STATES

_INDIC = re.compile(r"[ऀ-෿]")
_PUNCT = ',.;:"\'()[]{}#-'


def _latin_tokens(text: str):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def _learn(pairs: Iterable[Tuple[str, str]], min_count: int, min_share: float) -> Dict[str, str]:
    counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for native, latin in pairs:
        nt = [t.strip(_PUNCT) for t in native.split()]
        nt = [t for t in nt if t]
        lt = _latin_tokens(latin)
        if len(nt) != len(lt):
            continue
        for a, b in zip(nt, lt):
            if _INDIC.search(a):
                counts[a][b] += 1
    out: Dict[str, str] = {}
    for tok, ctr in counts.items():
        best, n = ctr.most_common(1)[0]
        total = sum(ctr.values())
        if n >= min_count and n / total >= min_share:
            out[tok] = best
    return out


def learn_name_map(pairs: pl.DataFrame, min_count: int = 2, min_share: float = 0.5) -> Dict[str, str]:
    """``pairs`` has columns ``tname`` (target raw name) and ``sname`` (Source-1 raw name)."""
    sub = pairs.filter(pl.col("tname").str.contains(r"[ऀ-෿]"))
    return _learn(zip(sub["tname"].to_list(), sub["sname"].to_list()), min_count, min_share)


def learn_addr_map(pairs: pl.DataFrame, min_count: int = 5, min_share: float = 0.6) -> Dict[str, str]:
    """Align native-script address components with the Source-1 state component."""
    sub = pairs.filter(pl.col("taddr").str.contains(r"[ऀ-෿]"))
    states = set(IN_STATES)

    def gen():
        for ta, sa in zip(sub["taddr"].to_list(), sub["saddr"].to_list()):
            if not ta or not sa:
                continue
            s_state = next((c.strip() for c in sa.split(",") if c.strip().lower() in states), None)
            if s_state is None:
                continue
            for comp in ta.split(","):
                if _INDIC.search(comp):
                    yield comp, s_state

    return _learn(gen(), min_count, min_share)


def save_maps(path: str | Path, name_map: Dict[str, str], addr_map: Dict[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"name": name_map, "addr": addr_map}, fh, ensure_ascii=False)


def load_maps(path: str | Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return d["name"], d["addr"]
