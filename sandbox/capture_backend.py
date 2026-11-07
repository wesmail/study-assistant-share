"""A matplotlib backend for a machine with no screen: showing a figure saves it.

Code pasted from a notebook usually ends in plt.show(), which on a normal headless
backend does nothing, and the plot would vanish.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib import _pylab_helpers
from matplotlib.backend_bases import _Backend
from matplotlib.backends.backend_agg import _BackendAgg
from matplotlib.figure import Figure

DPI = 120
saved = 0
_savefig = Figure.savefig


def _noting_savefig(self, *args, **kwargs):
    """A figure the code saved itself needs no second copy."""
    self._saved_by_code = True
    return _savefig(self, *args, **kwargs)


Figure.savefig = _noting_savefig


def save_open_figures() -> None:
    global saved
    for manager in _pylab_helpers.Gcf.get_all_fig_managers():
        figure = manager.canvas.figure
        if getattr(figure, "_saved_by_code", False):
            continue
        saved += 1
        _savefig(figure, Path.cwd() / f"figure-{saved}.png", dpi=DPI, bbox_inches="tight")
    _pylab_helpers.Gcf.destroy_all()


@_Backend.export
class _BackendCapture(_BackendAgg):
    @classmethod
    def show(cls, *args, **kwargs) -> None:
        save_open_figures()
