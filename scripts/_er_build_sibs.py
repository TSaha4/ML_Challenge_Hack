"""Build sibling tables for existing candidate parts (without re-running blocking)."""
import sys, time
from pathlib import Path
from src.er.resources import lower_priority, rss_gb; lower_priority()
from src.er.siblings import build_sibling_tables
ART = Path("artifacts/er")
for split in sys.argv[1:]:
    t = time.time()
    build_sibling_tables(str(ART / f"{split}_cands" / "part-*.parquet"), str(ART / f"{split}_tg.parquet"),
                         str(ART / f"{split}_s1.parquet"), ART / f"{split}_sibs")
    print(f"{split}: sibling tables built in {time.time()-t:.0f}s  rss {rss_gb():.1f}GB", flush=True)
