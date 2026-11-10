"""Execute the tutorial notebooks in docs/tutorials and store them with their outputs.

    python scripts/build_tutorials.py                        # all notebooks
    python scripts/build_tutorials.py 01_reference_and_query # one notebook

Notebooks run with the repository root as working directory (they read
data/pancreas_sparse.h5ad and write to results/tutorials/). stderr output (progress bars,
warnings) is dropped from the stored notebooks.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"


ANSI = re.compile(r"\x1b\[[0-9;]*m")
PROGRESS = re.compile(r"\d+%\|")


def clean_outputs(nb) -> None:
    """Drop stderr, keep only the last progress-bar line, strip ANSI codes and empty outputs."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        out, last_progress = [], None
        for o in cell.outputs:
            if o.output_type == "stream":
                if o.name == "stderr":
                    continue
                lines = [ln for ln in re.split(r"[\r\n]", ANSI.sub("", o.text)) if ln.strip()]
                keep = []
                for ln in lines:
                    if PROGRESS.search(ln):
                        last_progress = ln.strip()
                    else:
                        keep.append(ln)
                if keep:
                    o.text = "\n".join(keep) + "\n"
                    out.append(o)
            elif o.output_type in ("execute_result", "display_data"):
                text = o.data.get("text/plain")
                if text is not None:
                    o.data["text/plain"] = ANSI.sub("", text)
                if set(o.data) == {"text/plain"} and not o.data["text/plain"].strip("'\" \n"):
                    continue
                out.append(o)
            else:
                out.append(o)
        if last_progress:
            out.insert(0, nbformat.v4.new_output("stream", name="stdout", text=last_progress + "\n"))
        cell.outputs = out


def build(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    clean_outputs(nb)
    nb.metadata.pop("widgets", None)
    nbformat.write(nb, path)
    print(f"{path.name}: executed in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--clean-only"]:
        for p in sorted(TUTORIALS.glob("*.ipynb")):
            nb = nbformat.read(p, as_version=4)
            clean_outputs(nb)
            nbformat.write(nb, p)
        sys.exit()
    names = sys.argv[1:]
    paths = sorted(TUTORIALS.glob("*.ipynb"))
    if names:
        paths = [p for p in paths if p.stem in names]
    for p in paths:
        build(p)
