import sys
from pathlib import Path

# Allow both `python scripts/build_notebook.py` and `python -m scripts.build_notebook`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import platform

import nbformat as nbf
from pathlib import Path

from scripts.sec0 import get_p0_cells
from scripts.sec1_2 import get_p1_p2_cells
from scripts.sec3 import get_p3_cells
from scripts.sec4 import get_p4_cells
from scripts.sec5_6 import get_p5_p6_cells
from scripts.sec7 import get_p7_cells
from scripts.sec8_10 import get_p8_p10_cells
from scripts.sec11_13 import get_p11_p13_cells
from scripts.sec14 import get_p14_cells
from scripts.sec15_17 import get_p15_p17_cells
from scripts.sec18_19 import get_p18_p19_cells
from scripts.sec20 import get_p20_cells
from scripts.sec21_22 import get_p21_p22_cells

def build_full_notebook():
    nb = nbf.v4.new_notebook()
    all_cells = []
    
    all_cells.extend(get_p0_cells())
    all_cells.extend(get_p1_p2_cells())
    all_cells.extend(get_p3_cells())
    all_cells.extend(get_p4_cells())
    all_cells.extend(get_p5_p6_cells())
    all_cells.extend(get_p7_cells())
    all_cells.extend(get_p8_p10_cells())
    all_cells.extend(get_p11_p13_cells())
    all_cells.extend(get_p14_cells())
    all_cells.extend(get_p15_p17_cells())
    all_cells.extend(get_p18_p19_cells())
    all_cells.extend(get_p20_cells())
    all_cells.extend(get_p21_p22_cells())

    nb.cells = all_cells

    # Notebook metadata (kept here so every rebuild is reproducible).
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": platform.python_version(),
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
        "colab": {"provenance": [], "toc_visible": True},
    }
    
    out_path = Path("notebooks/entity_resolution.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated notebook: {out_path} with {len(all_cells)} cells.")

if __name__ == "__main__":
    build_full_notebook()
