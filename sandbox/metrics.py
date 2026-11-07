"""How much room text takes in the deck's fonts, measured and not guessed.

Carlito and Caladea have the same character widths as Calibri and Cambria, so wrapping
measured here is the wrapping PowerPoint does. Without them, as in the app's own tests,
it falls back on an average character width.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

FOLDER = Path("/usr/share/fonts/truetype/crosextra")
FILES = {"Calibri": "Carlito-Regular.ttf", "Calibri bold": "Carlito-Bold.ttf", "Cambria": "Caladea-Regular.ttf",
         "Cambria italic": "Caladea-Italic.ttf"}
LINE_HEIGHT = {"Calibri": 1.22, "Calibri bold": 1.22, "Cambria": 1.17, "Cambria italic": 1.17}
AVERAGE_WIDTH = 0.5                 # of the size, when the font isn't there
POINTS = 72
SCALE = 10                          # measured larger, then scaled, for finer widths


@lru_cache(maxsize=32)
def _font(name: str, points: int):
    from PIL import ImageFont

    path = FOLDER / FILES[name]
    return ImageFont.truetype(str(path), points * SCALE) if path.exists() else None


def width_of(text: str, font: str, points: int) -> float:
    """Width in inches."""
    measured = _font(font, points)
    if measured is None:
        return len(text) * AVERAGE_WIDTH * points / POINTS
    return measured.getlength(text) / SCALE / POINTS


def lines_of(text: str, font: str, points: int, width: float) -> int:
    """How many lines a paragraph wraps to, breaking at spaces as PowerPoint does."""
    lines, current = 1, ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if width_of(trial, font, points) > width and current:
            lines += 1
            current = word
        else:
            current = trial
    return lines


def height_of(paragraphs: list[str], font: str, points: int, width: float, gap: float = 0.0) -> float:
    """Height in inches of paragraphs at `points`, `gap` inches apart."""
    line = LINE_HEIGHT[font] * points / POINTS
    return sum(lines_of(text, font, points, width) * line for text in paragraphs) + gap * max(len(paragraphs) - 1, 0)
