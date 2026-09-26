"""Execute notebooks/entity_resolution.ipynb end-to-end with nbclient and store the outputs.

Usage:
    python scripts/execute_notebook.py                      # in-place execution
    python scripts/execute_notebook.py --output <path>      # write the executed copy elsewhere
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        default=str(PROJECT_ROOT / "notebooks" / "entity_resolution.ipynb"),
    )
    parser.add_argument("--output", default=None, help="Write the executed notebook here (default: in place).")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name to use.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds.")
    args = parser.parse_args()

    path = Path(args.notebook)
    notebook = nbformat.read(str(path), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    started = time.time()
    print(f"Executing {path} with kernel '{args.kernel}' ...", flush=True)
    client.execute()
    output_path = Path(args.output) if args.output else path
    nbformat.write(notebook, str(output_path))

    executed = sum(
        1 for cell in notebook.cells
        if cell["cell_type"] == "code" and cell.get("execution_count") is not None
    )
    total_code = sum(1 for cell in notebook.cells if cell["cell_type"] == "code")
    print(
        f"Executed {executed}/{total_code} code cells in {time.time() - started:.1f}s "
        f"-> {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
