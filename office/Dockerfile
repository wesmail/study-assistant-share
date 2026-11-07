# LibreOffice, for rendering Word and PowerPoint files to PDF and reading decks (roadmap item 7).
#
#     scripts/build_office.sh
#
# Every file it opens is untrusted: a deck from the library or a document a model wrote. So
# it runs like the sandbox (app/tools/sandbox.py): no network, a read-only root, only one
# temporary folder mounted, no capabilities. The host never parses a deck itself; this image
# renders it and reads its slides and speaker notes (render.py).
#
# The reader's Windows fonts are mounted read-only at run time (office_fonts_dir) so decks
# render with the fonts they were made with. They are never copied into this image.
FROM python:3.14-slim-trixie

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

# The -nogui packages: no X11, no Java. Math is needed for Word's equations (OMML): without
# it they vanish from the render (found Nov 6, 2026 against Word's own PDF); decks didn't
# show it because PowerPoint stores a picture of each equation as a fallback. Carlito and
# Caladea stand in for Calibri and Cambria when the Windows fonts aren't mounted; DejaVu and
# Liberation cover the rest.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-core-nogui libreoffice-impress-nogui libreoffice-writer-nogui libreoffice-math-nogui \
        fontconfig fonts-crosextra-caladea fonts-crosextra-carlito fonts-dejavu-core \
        fonts-liberation fonts-opensymbol \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-compile python-pptx==1.0.2 python-docx==1.2.0

COPY render.py registrymodifications.xcu /opt/office/
COPY fonts.conf /etc/fonts/conf.d/99-study-assistant.conf

RUN fc-cache -f && chmod -R a+rX /opt/office
