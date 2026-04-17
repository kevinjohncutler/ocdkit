"""Matplotlib rcParams and Jupyter environment setup helpers."""

import matplotlib as mpl


def apply_mpl_defaults():
    """Set baseline mpl rcParams (SVG-friendly text, no LaTeX)."""
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['text.usetex'] = False


def setup():
    """Configure a Jupyter notebook for inline plots with transparent backgrounds.

    Sets matplotlib rcParams suitable for retina displays, injects CSS to
    center plots and make widget backgrounds transparent (works in JupyterLab
    and VS Code), patches ``tqdm.notebook`` progress bars to a neutral grey,
    and exposes ``mpl``, ``plt``, ``widgets``, ``display``, ``tqdm`` in the
    notebook's global namespace.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, HTML
    from tqdm.notebook import tqdm as notebook_tqdm

    apply_mpl_defaults()

    display(HTML("""
    <style>
        .jp-OutputArea-output img {
            display: block;
            margin: 0 auto;
        }
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        .jp-OutputArea,
        .jp-OutputArea-child,
        .jp-OutputArea-output,
        .jp-Cell-outputWrapper,
        .jp-Cell-outputArea {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground, currentColor);
            --jp-widgets-font-size: var(--vscode-editor-font-size, inherit);
        }

        .widget-hprogress {
            background-color: transparent !important;
            border: none !important;
            display: inline-flex !important;
            justify-content: center;
            align-items: center;
        }
        .widget-hprogress .p-ProgressBar,
        .widget-hprogress .p-ProgressBar-track,
        .widget-hprogress .widget-progress .progress,
        .widget-hprogress .progress {
            background-color: rgba(128, 128, 128, 0.5) !important;
            border-radius: 6px !important;
            border: none !important;
            box-sizing: border-box;
            overflow: hidden;
        }
        .widget-hprogress .progress-bar,
        .widget-hprogress .p-ProgressBar-fill,
        .widget-hprogress [role="progressbar"]::part(value) {
            background-color: #8a8a8a !important;
            border-radius: 6px !important;
        }
    </style>
    """))

    def _patch_tqdm_progress():
        if getattr(notebook_tqdm, "_ocdkit_bar_styled", False):
            return

        default_fill = "#8a8a8a"
        original_status_printer = notebook_tqdm.status_printer

        def _status_printer(*args, **kwargs):
            container = original_status_printer(*args, **kwargs)
            try:
                _, pbar, _ = container.children
            except Exception:
                return container

            style = getattr(pbar, "style", None)
            if style is not None:
                style.bar_color = default_fill

            return container

        notebook_tqdm.status_printer = staticmethod(_status_printer)
        notebook_tqdm._ocdkit_bar_styled = True

    _patch_tqdm_progress()

    ipython = get_ipython()  # noqa: F821 — provided by IPython runtime
    ipython.user_global_ns['mpl'] = mpl
    ipython.user_global_ns['plt'] = plt
    ipython.user_global_ns['widgets'] = widgets
    ipython.user_global_ns['display'] = display
    ipython.user_global_ns['tqdm'] = notebook_tqdm

    ipython.run_line_magic('matplotlib', 'inline')

    rc_params = {
        'figure.dpi': 300,
        'figure.figsize': (2, 2),
        'image.cmap': 'gray',
        'image.interpolation': 'nearest',
        'figure.frameon': False,
        'axes.grid': False,
        'axes.facecolor': 'none',
        'figure.facecolor': 'none',
        'savefig.facecolor': 'none',
        'text.color': 'gray',
        'axes.labelcolor': 'gray',
        'xtick.color': 'gray',
        'ytick.color': 'gray',
        'axes.edgecolor': 'gray',
        'legend.loc': 'center left',
        'legend.frameon': False,
        'legend.framealpha': 0,
        'legend.borderaxespad': 0.0,
        'lines.scale_dashes': False,
    }

    mpl.rcParams.update(rc_params)

    from matplotlib.axes import Axes as _Axes
    _orig_legend = _Axes.legend

    def _legend(self, *args, **kwargs):
        kwargs.setdefault('loc', 'center left')
        kwargs.setdefault('bbox_to_anchor', (1.02, 0.5))
        kwargs.setdefault('frameon', False)
        kwargs.setdefault('framealpha', 0)
        kwargs.setdefault('borderaxespad', 0.0)
        return _orig_legend(self, *args, **kwargs)

    _Axes.legend = _legend
