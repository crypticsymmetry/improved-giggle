"""Execute repository notebooks in real Jupyter kernels; keep outputs out of git.

Install the package first: python -m pip install -e '.[dev]'. By default all
repository notebooks execute. Optional arguments select paths relative to the repo.
"""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    args = parser.parse_args()
    paths = (
        [ROOT / path for path in args.notebooks]
        if args.notebooks
        else sorted((ROOT / "notebooks").glob("*.ipynb"))
    )
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        with TemporaryDirectory() as workdir:
            NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": workdir}},
            ).execute(env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        count = sum(c.cell_type == "code" for c in notebook.cells)
        print(f"Executed {path.name}: {count} code cells successfully")


if __name__ == "__main__":
    main()
