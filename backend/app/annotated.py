"""Annotated copies of a PDF (roadmap item 4, step 5).

The original is opened read-only and the copy saved to outputs/annotated/, never under
library/ (which is ingested) and never over the original. SQLite stays the source of truth:
the copy is made again from it each time, so it can always be regenerated.

What goes in: every accepted highlight (the reader's, and AI suggestions they accepted) as a
PDF highlight annotation over its boxes, in its colour, its note as the comment (an AI
suggestion's note, or the reader's comment), titled by who made it; and every note anchored
to a passage, as an underline whose comment is the note. Suggestions not yet accepted, and
anchors whose passage was lost when the file changed, are left out and counted.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import fitz

from app.annotations import NotFound, document_path, rects_of
from app.config import Settings

COLORS = {"yellow": (1.0, 0.9, 0.2), "green": (0.55, 0.85, 0.3), "blue": (0.45, 0.72, 0.95), "pink": (0.97, 0.55, 0.72)}
NOTE_COLOR = (0.06, 0.46, 0.43)
FOLDER = "annotated"


def copy_path(settings: Settings, doc_id: int) -> Path:
    """One copy per document, made again (overwritten) each time from SQLite."""
    return settings.outputs_dir / FOLDER / f"{doc_id}-annotated.pdf"


def download_name(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", title).strip("-")[:80] or "document"
    return f"{slug}-annotated.pdf"


def write_copy(conn: sqlite3.Connection, settings: Settings, doc_id: int) -> dict:
    original = document_path(conn, settings, doc_id)
    doc = conn.execute("SELECT title, path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    out = copy_path(settings, doc_id)
    library = settings.library_dir.resolve()
    if out.resolve().is_relative_to(library) or out.resolve() == original:
        raise NotFound("The annotated copy can't go into the library")
    highlights = conn.execute("SELECT * FROM annotations WHERE doc_id = ? ORDER BY page, id", (doc_id,)).fetchall()
    notes = conn.execute("SELECT * FROM notes WHERE doc_id = ? AND page IS NOT NULL ORDER BY page, id", (doc_id,)).fetchall()
    counts = {"highlights": 0, "notes": 0, "suggested_left_out": 0, "lost_left_out": 0}
    with fitz.open(original) as pdf:
        for row in highlights:
            if row["status"] != "accepted":
                counts["suggested_left_out"] += 1
                continue
            boxes = rects_of(row)       # a NULL rects broke the export (review C9)
            if row["anchor_status"] == "unanchored" or not boxes or not 1 <= row["page"] <= pdf.page_count:
                counts["lost_left_out"] += 1
                continue
            page = pdf[row["page"] - 1]
            annot = page.add_highlight_annot(quads=[fitz.Rect(r).quad for r in boxes])
            annot.set_colors(stroke=COLORS.get(row["color"], COLORS["yellow"]))
            title = "AI suggestion, accepted" if row["origin"] == "ai" else "Highlight"
            annot.set_info(content=row["body"] or "", title=title, subject=row["instruction"] or "")
            annot.update()
            counts["highlights"] += 1
        for row in notes:
            boxes = rects_of(row)
            if row["anchor_status"] == "unanchored" or not boxes or not 1 <= row["page"] <= pdf.page_count:
                counts["lost_left_out"] += 1
                continue
            page = pdf[row["page"] - 1]
            annot = page.add_underline_annot(quads=[fitz.Rect(r).quad for r in boxes])
            annot.set_colors(stroke=NOTE_COLOR)
            tags = " ".join(f"#{tag}" for tag in json.loads(row["tags"] or "[]"))
            annot.set_info(content=row["body_md"] + (f"\n\n{tags}" if tags else ""), title="Your note")
            annot.update()
            counts["notes"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(out, garbage=3, deflate=True)
    return {"name": download_name(doc["title"] or Path(doc["path"]).stem), "url": f"/api/annotated/{doc_id}", **counts}
