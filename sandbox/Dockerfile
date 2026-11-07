# The one place that runs model-written code, TeX and pandoc.
#
#     scripts/build_sandbox.sh
#
# The build context is this folder and nothing else, so no project file can reach the
# image. The build needs the network, to fetch these pinned files and TeX's packages.
# Running never does: every container starts with --network none (app/tools/sandbox.py),
# so whatever TeX needs has to be baked in here.
FROM python:3.14-slim-trixie

ENV TECTONIC_CACHE_DIR=/opt/tectonic-cache \
    MPLCONFIGDIR=/opt/mplconfig \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl fontconfig fonts-crosextra-caladea fonts-crosextra-carlito poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Checksums come from each project's release metadata, and the build stops on a mismatch.
RUN curl -fsSL -o /tmp/pandoc.tgz \
        https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-linux-amd64.tar.gz \
    && echo "c7edd535941c48be6a362081a748272837de81ae11777202d9c341d3d8261c9a  /tmp/pandoc.tgz" | sha256sum -c - \
    && tar -xzf /tmp/pandoc.tgz -C /tmp \
    && install -m 755 /tmp/pandoc-3.10.2/bin/pandoc /usr/local/bin/pandoc \
    && rm -rf /tmp/pandoc*

RUN curl -fsSL -o /tmp/tectonic.tgz \
        "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz" \
    && echo "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7  /tmp/tectonic.tgz" | sha256sum -c - \
    && tar -xzf /tmp/tectonic.tgz -C /usr/local/bin tectonic \
    && chmod 755 /usr/local/bin/tectonic \
    && rm -f /tmp/tectonic.tgz

RUN pip install --no-compile \
        numpy==2.5.4 scipy==1.18.1 sympy==1.14.0 matplotlib==3.11.2 \
        python-pptx==1.0.2 python-docx==1.2.0 pillow==12.3.0

COPY *.py *.lua /opt/sandbox/

# Fill TeX's cache and matplotlib's font list while there's a network and a writable disk.
RUN fc-cache -f \
    && mkdir -p "$TECTONIC_CACHE_DIR" "$MPLCONFIGDIR" \
    && python /opt/sandbox/make_reference.py /opt/sandbox/reference.docx \
    && python /opt/sandbox/bake.py \
    && python -c "import matplotlib.font_manager" \
    && chmod -R a+rX "$TECTONIC_CACHE_DIR" "$MPLCONFIGDIR" /opt/sandbox

# Nothing to fetch with at run time, even if a network were ever attached.
RUN apt-get purge -y --auto-remove curl
