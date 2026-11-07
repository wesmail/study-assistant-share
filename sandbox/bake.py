"""Fills TeX's cache while the image is built, because containers have no network.

Tectonic fetches each package the first time it's used. Running this sample through the
same pipeline as real files means real files find what they need with --only-cached.
Anything used here is available later, so it's deliberately broad.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

SAMPLE = r"""# Heading

## Subheading

Text with *emphasis*, **bold**, `code`, a [link](https://example.org), Greek λ and umlauts äöüß.

| Column | Other |
|:-------|------:|
| $a$    | $b$   |
| **c**  | *d*   |

1. One
2. Two

- Bullet

> A quotation with $x$ in it.

| A long heading that wraps across the width of the page in its column | Another long heading that also has to wrap in its own column on the page |
|:-----------------------------------------------------------|:----------------------------------------------------------|
| A cell | Another |

```python
def f(x):
    return x ** 2  # a comment with a very long line that has to wrap to fit on the page at all costs
```

$$
\boldsymbol{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}, \quad
\begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad
\begin{cases} x & x > 0 \\ 0 & \text{else} \end{cases}, \quad
\mathbf{b} \mathsf{s} \mathtt{t} \mathfrak{f} \mathscr{c} \mathit{i} \text{\textbf{bold}}
$$

$$
\begin{aligned} \int_0^\infty e^{-x^2}\,dx &= \frac{\sqrt{\pi}}{2} \\
\sum_{i=1}^n \prod_{j} \mathbb{E}[X] \mathcal{L} \mathrm{d}x \operatorname{tr} \nabla \partial \hat{x} \bar{y} \tilde{z}
&\xrightarrow{a} \underbrace{b}_{c}
\end{aligned} \tag{1}
$$

![A picture](images/sample.png)

[^1]: A footnote.
"""
# TeX loads the legacy math fonts for every size that any math appears at, even under
# unicode-math, so a formula in a heading or footnote needs them too.
SIZES = ("tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge")
EVERY_SIZE = "\n".join(
    f"{{\\{size} $x^2_i \\sum \\frac{{a}}{{b}}$ {{\\bfseries b $x$ \\boldmath $x$}} \\textit{{i}} \\texttt{{t}}}}\\par"
    for size in SIZES)
DECK = {"slides": [
    {"layout": "title", "title": "A title", "subtitle": "A subtitle"},
    {"layout": "bullets_equation", "title": "Slide", "bullets": ["One"], "notes": "Notes",
     "footer": "Sources: Book, p. 1", "animate": True, "equations": [
         r"\boldsymbol{A} = \begin{bmatrix} a & b \end{bmatrix} \mathbb{E}[X] \mathcal{L}",
         r"\begin{align} a &= b \\ c &= d \end{align}"]},
    {"layout": "big_equation", "title": "Slide", "caption": "A caption", "animate": True, "equations": [
        r"\text{words } x_{\text{sub}} \textbf{bold} \mathrm{r} \mathbf{b} \mathsf{s} \mathtt{t} "
        r"\mathfrak{f} \mathscr{c} \mathit{i}",
        r"\begin{aligned} a &= b \\ c &= d \end{aligned}"]},
]}


def png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))


def build(kind: str, work: Path) -> None:
    environment = {**os.environ, "SANDBOX_WORK": str(work), "SANDBOX_BAKING": "1"}
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("build.py")), kind],
                          env=environment, capture_output=True, text=True)
    print(done.stdout, done.stderr[-2000:])
    if done.returncode != 0 or not json.loads(done.stdout.strip().splitlines()[-1])["ok"]:
        raise SystemExit(f"Baking {kind} failed")


def every_size(work: Path) -> None:
    """The same preamble as a real PDF, with math at each size added before the end."""
    tex = (work / "doc.tex").read_text(encoding="utf-8")
    (work / "sizes.tex").write_text(tex.replace("\\end{document}", f"{EVERY_SIZE}\n\\end{{document}}"),
                                    encoding="utf-8")
    done = subprocess.run(["tectonic", "--untrusted", "--keep-logs", "--chatter", "minimal", "sizes.tex"],
                          cwd=work, capture_output=True, text=True)
    print(done.stderr[-2000:])
    if done.returncode != 0:
        raise SystemExit("Baking the font sizes failed")


def main() -> None:
    work = Path(tempfile.mkdtemp())
    (work / "images").mkdir()
    png(work / "images" / "sample.png")
    (work / "doc.md").write_text(SAMPLE, encoding="utf-8")
    (work / "deck.json").write_text(json.dumps(DECK))
    build("pdf", work)
    build("deck", work)
    every_size(work)
    # Offline this time, so a package the sample missed stops the image build, not a user.
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("build.py")), "pdf"],
                          env={**os.environ, "SANDBOX_WORK": str(work)}, capture_output=True, text=True)
    print(done.stdout, done.stderr[-2000:])
    if not json.loads(done.stdout.strip().splitlines()[-1])["ok"]:
        raise SystemExit("The baked cache doesn't work offline")
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
