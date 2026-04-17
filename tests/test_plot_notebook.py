"""Test plot functions that require an IPython/Jupyter kernel.

Runs a notebook programmatically via nbclient — no browser needed.
Coverage is collected inside the kernel via a startup/teardown cell pair
that writes a .coverage file, which pytest-cov's parallel mode can combine.
"""

import os
import pytest
import nbformat
from nbclient import NotebookClient

_COV_DIR = os.path.join(os.path.dirname(__file__), '..', '.coverage_combined')


def _make_notebook(cells, collect_coverage=True):
    """Create an in-memory notebook from a list of code strings."""
    all_cells = []
    if collect_coverage:
        all_cells.append(nbformat.v4.new_code_cell(
            "import coverage as _cov_mod, os; "
            "_cov = _cov_mod.Coverage(source=['ocdkit'], "
            f"data_file={_COV_DIR!r} + '/.coverage.notebook', "
            "data_suffix=True); "
            "_cov.start()"
        ))
    all_cells.extend(nbformat.v4.new_code_cell(c) for c in cells)
    if collect_coverage:
        all_cells.append(nbformat.v4.new_code_cell(
            "_cov.stop(); _cov.save()"
        ))
    nb = nbformat.v4.new_notebook()
    nb.cells = all_cells
    return nb


def _run_notebook(nb, timeout=30):
    """Execute all cells in a notebook, raising on any error."""
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    client.execute()
    return nb


class TestSetup:
    def test_setup_runs(self):
        nb = _make_notebook([
            "from ocdkit.plot.defaults import setup, apply_mpl_defaults",
            "apply_mpl_defaults()",
            "setup()",
            "import matplotlib as mpl; assert mpl.rcParams['figure.dpi'] == 300",
            "assert 'plt' in dir()",
            "assert 'widgets' in dir()",
            "assert 'display' in dir()",
        ])
        _run_notebook(nb)


class TestImshow:
    def test_single_image(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.display import imshow",
            "img = np.random.rand(8, 8, 3).astype(np.float32)",
            "imshow(img)",
        ])
        _run_notebook(nb)

    def test_image_list(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.display import imshow",
            "imgs = [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(3)]",
            "imshow(imgs, titles=['a', 'b', 'c'])",
        ])
        _run_notebook(nb)

    def test_hold_returns_fig(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.display import imshow",
            "from matplotlib.figure import Figure",
            "img = np.random.rand(8, 8, 3).astype(np.float32)",
            "fig = imshow(img, hold=True)",
            "assert isinstance(fig, Figure)",
        ])
        _run_notebook(nb)

    def test_existing_ax(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.figure import figure",
            "from ocdkit.plot.display import imshow",
            "fig, ax = figure()",
            "img = np.random.rand(8, 8, 3).astype(np.float32)",
            "result = imshow(img, ax=ax)",
        ])
        _run_notebook(nb)

    def test_outline(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.display import imshow",
            "img = np.random.rand(8, 8, 3).astype(np.float32)",
            "imshow(img, outline_color='red', outline_width=2)",
        ])
        _run_notebook(nb)

    def test_figsize_tuple(self):
        nb = _make_notebook([
            "import numpy as np",
            "from ocdkit.plot.display import imshow",
            "imgs = [np.random.rand(8, 8, 3).astype(np.float32) for _ in range(2)]",
            "imshow(imgs, figsize=(6, 3))",
        ])
        _run_notebook(nb)
