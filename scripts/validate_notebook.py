"""Execute the checked-in Colab notebook in a real Jupyter kernel.

Install the package first: python -m pip install -e '.[dev]'. Outputs stay in
memory; the reusable notebook remains cleared in git.
"""

from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / "notebooks/colab_demo.ipynb", as_version=4)
nbformat.validate(notebook)
NotebookClient(
    notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
).execute(env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
print(
    f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} notebook code cells successfully"
)
