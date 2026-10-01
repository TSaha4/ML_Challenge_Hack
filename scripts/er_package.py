"""Build the final submission zip in the layout required by the challenge.

    python scripts/er_package.py --team TEAMNAME

<TEAM>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_INIT = '"""Business entity resolution - see src/er and README.md."""\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TEAM")
    ap.add_argument("--out-dir", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    zpath = out / f"{args.team}_submission.zip"
    code = "code/business_entity_resolution"

    files = {
        "output/matching_results.tsv": ROOT / "output" / "matching_results.tsv",
        "output/candidate_pairs.tsv": ROOT / "output" / "candidate_pairs.tsv",
        f"{code}/README.md": ROOT / "README_er.md",
        f"{code}/requirements.txt": ROOT / "requirements-er.txt",
        "Documentation_template.md": ROOT / "Documentation_template.md",
    }
    for py in sorted((ROOT / "src" / "er").rglob("*.py")):
        files[f"{code}/src/er/{py.relative_to(ROOT / 'src' / 'er').as_posix()}"] = py
    for arc, path in files.items():
        if not path.exists():
            raise SystemExit(f"missing: {path}")

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{code}/src/__init__.py", SRC_INIT)
        for arc, path in files.items():
            z.write(path, arc)
            print(f"  + {arc}")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
