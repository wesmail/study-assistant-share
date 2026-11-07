"""The one design every generated file shares: colours, fonts, limits and layout names.

Decks, Word files and PDFs all read these, so a change here changes all three.

Office gets Calibri headings and Cambria text, a pair made to go together, and Cambria
Math is what Word sets equations in. TeX can't use those. Carlito has Calibri's metrics,
so PDF headings match; Caladea would match Cambria but has no Greek, and "λ" in running
text would vanish, so PDF body text stays in Libertinus Serif beside its math font.
"""
from __future__ import annotations

ACCENT = "0F766E"           # headings, rules, table heads
ACCENT_DARK = "134E4A"      # the title slide
HIGHLIGHT = "B45309"        # the one warm colour, sparingly
INK = "1C1917"              # text and equations
MUTED = "57534E"            # footers and captions
RULE = "D6D3D1"             # thin lines
SURFACE = "F0F5F4"          # panels behind equations
WHITE = "FFFFFF"

HEADING_FONT = "Calibri"
BODY_FONT = "Cambria"
MONO_FONT = "Consolas"

LAYOUTS = ("title", "bullets_equation", "big_equation", "figure_text", "two_columns", "summary")
MAX_BULLETS = 4             # on a slide, across both columns of two_columns
MAX_BULLET_WORDS = 12
MAX_SLIDE_WORDS = 55        # everything visible, footer included
MAX_EQUATIONS = 2           # with bullets; a big_equation slide takes MAX_DERIVATION_LINES
MAX_DERIVATION_LINES = 5
