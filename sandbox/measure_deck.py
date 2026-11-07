"""Does the text in a finished deck fit its boxes? Runs in the sandbox, for the real fonts.

    python /opt/sandbox/measure_deck.py /work/deck.pptx     prints one line of JSON

It reads the file, not what the builder meant to do, so it can disagree with the builder.
"""
from __future__ import annotations

import json
import sys

import metrics

EMU = 914400
TOLERANCE = 0.03                # inches
INDENT = 0.3                    # the bullet itself
GAP = 0.14
DEFAULT_POINTS = 18
INSET_HEIGHT = 0.1              # PowerPoint's default top and bottom margins, together


def size_of(paragraph) -> int:
    sizes = [run.font.size.pt for run in paragraph.runs if run.font.size]
    return int(max(sizes)) if sizes else DEFAULT_POINTS


def overflowing(path: str) -> list[str]:
    from pptx import Presentation

    problems = []
    for number, slide in enumerate(Presentation(path).slides, 1):
        title = slide.shapes.title
        for shape in slide.shapes:
            if not shape.has_text_frame or not shape.text_frame.text.strip() or shape.name == "slide number":
                continue
            paragraphs = [item for item in shape.text_frame.paragraphs if item.text.strip()]
            points = size_of(paragraphs[0])
            is_title = title is not None and shape.shape_id == title.shape_id
            bulleted = shape.is_placeholder and not is_title and "Content" in shape.name
            italic = any(run.font.italic for item in paragraphs for run in item.runs)
            font = "Calibri bold" if is_title else "Cambria italic" if italic else "Cambria"
            width = shape.width / EMU - (INDENT if bulleted else 0)
            needed = metrics.height_of([item.text for item in paragraphs], font, points, width, GAP if bulleted else 0)
            room = shape.height / EMU
            if needed + INSET_HEIGHT > room + TOLERANCE:
                problems.append(f"slide {number} {shape.name}: needs {needed:.2f} in, has {room:.2f}")
    return problems


if __name__ == "__main__":
    print(json.dumps({"overflowing": overflowing(sys.argv[1])}, ensure_ascii=False))
