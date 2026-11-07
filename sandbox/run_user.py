"""Runs /work/main.py with /work/out as its folder. Runs only inside the sandbox."""
from __future__ import annotations

import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT = "/work/main.py"
FONT_CACHE = Path("/opt/mplconfig")
WRITABLE_CACHE = Path("/tmp/mplconfig")     # matplotlib insists on a writable folder


def set_up() -> None:
    if FONT_CACHE.is_dir():
        shutil.copytree(FONT_CACHE, WRITABLE_CACHE, dirs_exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(WRITABLE_CACHE)
    os.environ["MPLBACKEND"] = "module://capture_backend"
    os.makedirs("/work/out", exist_ok=True)
    os.chdir("/work/out")


def save_figures() -> None:
    """Whatever was drawn and never shown still counts as the plot."""
    if "matplotlib.pyplot" in sys.modules:
        import capture_backend

        capture_backend.save_open_figures()


def report(error: BaseException) -> None:
    """The traceback from the reader's own code, without this file's frames."""
    frames = error.__traceback__
    while frames and frames.tb_frame.f_code.co_filename != SCRIPT:
        frames = frames.tb_next
    traceback.print_exception(type(error), error, frames)


def main() -> int:
    set_up()
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    except BaseException as error:
        report(error)
        code = 1
    else:
        code = 0
    try:
        save_figures()
    except Exception as error:
        print(f"Couldn't save the figures: {error}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
