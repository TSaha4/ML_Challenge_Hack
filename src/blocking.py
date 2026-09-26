"""
Blocking, Candidate Generation, and Indexing Strategies.
Implements Multi-Channel Adaptive Blocking with safe limits to prevent combinatorial explosion.
"""

import collections
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


def extract_blocking_keys(record: Dict[str, Any], idf_dict: Optional[Dict[str, float]] = None) -> List[Tuple[str, str]]:
    """
    Extracts multi-channel blocking keys for a record:
    Keys generated:
    1. exact normalized name: ('norm_name', str)
    2. exact core name: ('core_name', str)
    3. country + first 3 letters of core name: ('c_prefix3', str)
    4. country + postal code (if available): ('c_postal', str)
    5. country + rarest token (if idf_dict provided): ('c_rare_tok', str)
    6. transliterated name: ('translit_name', str)
    """
    keys = []
    country = str(record.get('country', '')).strip().upper()
    norm_name = str(record.get('name_norm', '')).strip()
    core_name = str(record.get('name_core', '')).strip()
    postal = str(record.get('addr_postal', '')).strip()
    translit = str(record.get('name_translit', '')).strip()

    if norm_name:
        keys.append(('norm_name', norm_name))
    if core_name and core_name != norm_name:
        keys.append(('core_name', core_name))
    if translit and translit != norm_name and translit != core_name:
        keys.append(('translit_name', translit))

    # Locality / Sub-blocking channels
    if country and core_name:
        tokens = core_name.split()
        if tokens:
            # First token + country
            keys.append(('c_first_tok', f"{country}::{tokens[0]}"))
            
            # Country + rare token
            if idf_dict:
                rarest_token = max(tokens, key=lambda t: idf_dict.get(t, 0.0))
                if idf_dict.get(rarest_token, 0.0) > 3.0:  # Only if meaningfully rare
                    keys.append(('c_rare_tok', f"{country}::{rarest_token}"))
            elif len(tokens[0]) >= 4:
                keys.append(('c_prefix4', f"{country}::{tokens[0][:4]}"))

    if country and postal and len(postal) >= 4:
        # Postal + first token of core name
        tokens = core_name.split()
        first_tok = tokens[0] if tokens else ""
        if first_tok:
            keys.append(('c_postal_tok', f"{country}::{postal}::{first_tok[:3]}"))
        else:
            keys.append(('c_postal', f"{country}::{postal}"))

    return keys


class AdaptiveBlocker:
    """
    Adaptive Inverted Index Blocker.
    Builds inverted index over Source 2 and Source 3 records.
    Implements adaptive thresholds:
    - Normal blocks (< max_block_size): all candidates retrieved.
    - Large blocks (>= max_block_size): subdivide by secondary key or cap candidate fan-out.
    """
    def __init__(self, max_block_size: int = 500, max_candidates_per_entity: int = 100):
        self.max_block_size = max_block_size
        self.max_candidates_per_entity = max_candidates_per_entity
        self.index: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)

    def fit(self, records: List[Dict[str, Any]], idf_dict: Optional[Dict[str, float]] = None):
        """Index target records (Source 2 and 3)."""
        for r in records:
            e_id = r['entity_id']
            keys = extract_blocking_keys(r, idf_dict)
            for k in keys:
                self.index[k].append(e_id)

    def query_batch(
        self,
        s1_records: List[Dict[str, Any]],
        idf_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch API mirroring :meth:`DuckDBBlocker.query_batch`."""
        return {rec["entity_id"]: self.query(rec, idf_dict=idf_dict) for rec in s1_records}

    def query(self, s1_record: Dict[str, Any], idf_dict: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Query inverted index for Source 1 record candidates.
        Returns deduplicated candidates with channel metadata.
        """
        s1_id = s1_record['entity_id']
        keys = extract_blocking_keys(s1_record, idf_dict)
        
        cand_map: Dict[str, Dict[str, Any]] = {}
        
        for ch, val in keys:
            matched_ids = self.index.get((ch, val), [])
            n_matches = len(matched_ids)
            
            # Adaptive guard: if block is overly large, prioritize top matches or secondary check
            if n_matches > self.max_block_size:
                # Down-sample or prioritize
                matched_ids = matched_ids[:self.max_block_size]
                
            for rank, target_id in enumerate(matched_ids):
                if target_id not in cand_map:
                    cand_map[target_id] = {
                        "source1_id": s1_id,
                        "candidate_id": target_id,
                        "channels": [ch],
                        "channel_count": 1,
                        "min_rank": rank,
                    }
                else:
                    cand_map[target_id]["channels"].append(ch)
                    cand_map[target_id]["channel_count"] += 1
                    cand_map[target_id]["min_rank"] = min(cand_map[target_id]["min_rank"], rank)

        # Sort candidates by channel_count descending, then min_rank ascending
        candidates = sorted(cand_map.values(), key=lambda c: (-c["channel_count"], c["min_rank"]))
        return candidates[:self.max_candidates_per_entity]


# ---------------------------------------------------------------------------
# Disk-backed blocker
# ---------------------------------------------------------------------------
#: Step 1 - materialise the matched block rows for the current query batch.
#:
#: The query used to be a single statement whose ``row_number()`` window could be
#: legally pushed *below* the join, i.e. evaluated over the whole
#: ``blocking_keys`` table (millions of hash partitions).  At >=2.5M indexed
#: targets that exceeded DuckDB's memory limit and crashed the scale benchmark
#: with an OOM error.  Materialising the join output into a bounded temp table
#: first makes the subsequent window cost proportional to the batch, never to
#: the index size.
#:
#: Matching on ``key`` alone (instead of ``channel`` + ``key``) is a deliberate
#: superset - it also picks up the occasional key shared across channels -
#: which costs ~4% extra temp rows, runs ~3x faster than the two-column join,
#: and is harmless because step 2 joins back on ``(channel, key)`` so only
#: query-key pairs survive to the output; the extra pairs' partitions never
#: reach it.  Output equality is verified by the equivalence stage of
#: ``scripts/bench_train_scale.py``.
_BLOCK_MATCH_SQL = """
CREATE OR REPLACE TEMP TABLE _block_matched AS
SELECT DISTINCT k.channel AS channel,
                k.key AS key,
                k.seq AS seq,
                k.entity_id AS entity_id
FROM blocking_keys k
WHERE k.key IN (SELECT key FROM query_keys)
"""

#: Step 2 - reproduce AdaptiveBlocker candidate semantics on the materialised
#: match set (same rank / truncation / ordering semantics as before).
#:
#: ``channel_count`` is deliberately *not* computed here: a second
#: ``count(DISTINCT ...)`` aggregate state roughly doubles the per-group memory
#: of the final ``GROUP BY`` and was the last straw in the scale-benchmark OOM
#: (976 MiB used of a 976.5 MiB limit).  Python derives it as
#: ``len(channels.split('|'))``, which is identical for the fixed, pipe-free
#: channel labels emitted by :func:`extract_blocking_keys`.
_BLOCK_BATCH_SQL = """
WITH ranked AS (
    SELECT channel,
           key,
           seq,
           entity_id,
           row_number() OVER (PARTITION BY channel, key ORDER BY seq) AS rk
    FROM _block_matched
), kept AS (
    SELECT * FROM ranked WHERE rk <= ?
)
SELECT q.s1_id AS s1_id,
       k.entity_id AS cand_id,
       min(k.rk) - 1                       AS min_rank,
       min(q.chan_order * 1000000 + k.rk)  AS order_key,
       string_agg(DISTINCT k.channel, '|') AS channels
FROM query_keys q
JOIN kept k ON k.channel = q.channel AND k.key = q.key
GROUP BY q.s1_id, k.entity_id
"""


class DuckDBBlocker:
    """
    Disk-backed replacement for :class:`AdaptiveBlocker`.

    The in-memory inverted index costs ~800 bytes per indexed target (~8 GB for the
    ~10M-record competition sources), which does not fit on a 14 GB machine. This
    class keeps the same inverted index in an on-disk DuckDB table and evaluates the
    identical candidate-selection logic in SQL:

    * ``rank`` - position of a target inside a ``(channel, key)`` block in insertion
      (file) order, i.e. the order ``AdaptiveBlocker`` enumerates;
    * ``rk <= max_block_size`` - the same per-block truncation as
      ``matched_ids[:max_block_size]``;
    * ``channel_count`` = number of distinct channels that retrieved the candidate,
      ``min_rank`` = best (smallest) rank across them;
    * candidates ordered by ``channel_count`` desc, ``min_rank`` asc and, for exact
      ties, the first (channel order, rank) appearance - byte-identical ordering to
      :class:`AdaptiveBlocker` (verified by ``scripts/bench_train_scale.py``).

    ``query_batch`` answers a whole batch of Source-1 records in a pair of SQL
    statements per sub-batch (materialise matches, then rank + aggregate), so
    per-entity Python overhead stays negligible while memory stays bounded by
    ``query_chunk_size`` rather than by the size of the index.
    """

    def __init__(
        self,
        db_path,
        max_block_size: int = 500,
        max_candidates_per_entity: int = 100,
        memory_limit: str = "3GB",
        threads: int = 4,
        temp_directory=None,
        query_chunk_size: int = 10_000,
    ):
        import duckdb  # lazy: keeps this module importable without DuckDB

        self.db_path = str(db_path)
        self.max_block_size = int(max_block_size)
        self.max_candidates_per_entity = int(max_candidates_per_entity)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.con = duckdb.connect(self.db_path)
        self.con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        self.con.execute(f"PRAGMA threads={int(threads)}")
        if temp_directory:
            self.con.execute(f"PRAGMA temp_directory='{temp_directory}'")
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS blocking_keys (
                   channel VARCHAR,
                   key VARCHAR,
                   seq BIGINT,
                   entity_id VARCHAR
               )"""
        )
        self._seq = 0
        self.n_records = 0

    # -- indexing ---------------------------------------------------------
    def fit(self, records: List[Dict[str, Any]], idf_dict: Optional[Dict[str, float]] = None):
        """Index target records (Source 2 and 3). Appends, never resets the index."""
        import pandas as pd

        rows: List[Tuple[str, str, int, str]] = []
        for r in records:
            e_id = r["entity_id"]
            for channel, value in extract_blocking_keys(r, idf_dict):
                rows.append((channel, value, self._seq, e_id))
                self._seq += 1
            self.n_records += 1
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=["channel", "key", "seq", "entity_id"])
        self.con.register("blocking_chunk", frame)
        try:
            self.con.execute(
                "INSERT INTO blocking_keys (channel, key, seq, entity_id) "
                "SELECT channel, key, seq, entity_id FROM blocking_chunk"
            )
        finally:
            self.con.unregister("blocking_chunk")
        del rows, frame

    def finalize(self):
        """Build the lookup index once indexing is complete."""
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocking_keys ON blocking_keys(channel, key)"
        )
        self.con.execute("ANALYZE blocking_keys")

    def load_existing(self) -> Tuple[int, int]:
        """Re-open a previously built index; returns ``(n_records, n_keys)``."""
        rows, max_seq = self.con.execute(
            "SELECT count(DISTINCT entity_id), max(seq) FROM blocking_keys"
        ).fetchone()
        self.n_records = int(rows or 0)
        self._seq = int(max_seq or -1) + 1
        n_keys = self.con.execute("SELECT count(*) FROM blocking_keys").fetchone()[0]
        return self.n_records, int(n_keys)

    # -- querying ---------------------------------------------------------
    def _chunk_step(self) -> int:
        """Sub-batch size for ``query_batch``, bounded by the index size.

        The final ``GROUP BY (s1_id, cand_id)`` produces candidate groups per
        Source-1 record that grow roughly proportionally to the index size
        (535 groups/S1 at 2.5M targets => ~2.14e-4 per indexed record).  The
        aggregate hash table is the dominant query-time memory term, so keep
        ``step * n_records`` <= ~5M, i.e. never more than ~1.1M groups
        (~250 MiB of aggregate state) in flight - at any index size, from the
        200k equivalence pool up to the full 10.3M-target run.  The floor of
        400 keeps tiny indexes at one sub-batch per call.
        """
        if self.n_records <= 0:
            # Self-protect: a caller that queries a freshly opened index
            # without load_existing()/fit() must not disable the bound.
            (only,) = self.con.execute(
                "SELECT count(DISTINCT entity_id) FROM blocking_keys"
            ).fetchone()
            self.n_records = int(only or 0)
        by_index = 5_000_000 // max(self.n_records, 1)
        return max(400, min(self.query_chunk_size, by_index))

    def query_batch(
        self,
        s1_records: List[Dict[str, Any]],
        idf_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Blocking candidates for a batch of Source-1 records.

        The batch is processed in sub-batches sized by :meth:`_chunk_step`
        (``query_chunk_size`` capped by the index size).  Each sub-batch runs
        two statements: materialise the matched block rows
        (``_BLOCK_MATCH_SQL``), then rank + aggregate them
        (``_BLOCK_BATCH_SQL``).  Splitting the statements keeps DuckDB from
        pushing the ``row_number()`` window below the join - which would
        recompute it over the *entire* key table and exhaust the connection's
        memory limit on a 2.5M+ target index (the failure mode that crashed
        the scale benchmark).  Sizing the sub-batch by index size in turn
        bounds the final ``GROUP BY`` (the aggregate hash table over
        ``(s1_id, cand_id)`` groups was the *second* OOM discovered there).
        """
        import pandas as pd

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        step = self._chunk_step()
        for start in range(0, len(s1_records), step):
            batch = s1_records[start:start + step]
            key_rows: List[Tuple[str, int, str, str]] = []
            for s1_rec in batch:
                s1_id = s1_rec["entity_id"]
                for chan_order, (channel, value) in enumerate(extract_blocking_keys(s1_rec, idf_dict)):
                    key_rows.append((s1_id, chan_order, channel, value))
            if not key_rows:
                continue

            frame = pd.DataFrame(key_rows, columns=["s1_id", "chan_order", "channel", "key"])
            frame = frame.drop_duplicates(subset=["s1_id", "channel", "key"], keep="first")
            self.con.register("query_keys", frame)
            try:
                self.con.execute(_BLOCK_MATCH_SQL)
                result = self.con.execute(_BLOCK_BATCH_SQL, [self.max_block_size]).fetchall()
            finally:
                self.con.unregister("query_keys")
                self.con.execute("DROP TABLE IF EXISTS _block_matched")
            del key_rows, frame

            touched: Set[str] = set()
            for s1_id, cand_id, min_rank, order_key, channels in result:
                channels = str(channels).split("|")
                grouped.setdefault(s1_id, []).append({
                    "source1_id": s1_id,
                    "candidate_id": cand_id,
                    "channels": channels,
                    # identical to the SQL count(DISTINCT channel) removed
                    # from _BLOCK_BATCH_SQL: string_agg(DISTINCT ...) lists
                    # every channel exactly once
                    "channel_count": len(channels),
                    "min_rank": int(min_rank),
                    "_order_key": int(order_key),
                })
                touched.add(s1_id)
            del result

            # Trim every touched S1 list to the candidate cap right away.
            # Keeping all pre-truncation groups (~535/S1 at 2.5M targets =>
            # ~10.7M dicts per 20k batch) ballooned the benchmark process to
            # ~8 GB RSS; per-sub-batch trimming is exact under top-k merge
            # semantics: top_k(A u B) == top_k(top_k(A) u B).
            for s1_id in touched:
                lst = grouped[s1_id]
                if len(lst) > self.max_candidates_per_entity:
                    lst.sort(key=lambda c: (-c["channel_count"], c["min_rank"], c["_order_key"]))
                    del lst[self.max_candidates_per_entity:]

        hits: Dict[str, List[Dict[str, Any]]] = {}
        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            cands = grouped.get(s1_id, [])
            cands.sort(key=lambda c: (-c["channel_count"], c["min_rank"], c["_order_key"]))
            hits[s1_id] = cands[: self.max_candidates_per_entity]
        return hits

    def query(
        self, s1_record: Dict[str, Any], idf_dict: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Single-record wrapper with the same contract as AdaptiveBlocker.query."""
        return self.query_batch([s1_record], idf_dict=idf_dict)[s1_record["entity_id"]]

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass
