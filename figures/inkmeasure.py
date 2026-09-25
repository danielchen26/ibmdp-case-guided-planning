#!/usr/bin/env python3
"""Measure on-page type size from a figure PDF by measuring glyph ink.

GR converts all text to paths, so `pdffonts` reports nothing and the requested
font size cannot be read out of the file.  The only way to know what size of type
actually reaches the page is to rasterise and measure ink.

The measurement is capital-height: the tallest run of contiguous dark rows inside
a caller-supplied crop that contains capitals and no ascender-only lowercase.
Effective body size is reported as cap_height / 0.70, the usual cap-height ratio
for a text face -- so a run measuring 4.2 pt of capital ink is ~6 pt type.

  python3 inkmeasure.py <pdf> --dpi 1200 [--scale 0.8496]
      [--crop x0,y0,x1,y1 ...]           # fractions of the page, repeatable

With no --crop it reports the connected dark-row bands over the whole page, which
is enough to find the smallest type present.
"""
import argparse
import shutil
import subprocess
import tempfile
import os

import numpy as np
from PIL import Image

# Ghostscript, not poppler: GR writes text as paths, and gs rasterises those
# faithfully at arbitrary dpi.  Found on PATH so this works wherever gs is
# installed (/usr/local/bin on Intel Homebrew, /opt/homebrew/bin on Apple
# silicon, /usr/bin on most Linux); override with GS=/path/to/gs.
GS = os.environ.get("GS") or shutil.which("gs") or "gs"
CAP_RATIO = 0.70   # capital height as a fraction of em for a normal text face


def raster(pdf, dpi):
    out = os.path.join(tempfile.mkdtemp(), "p.png")
    subprocess.run([GS, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray",
                    f"-r{dpi}", f"-sOutputFile={out}", pdf], check=True)
    return np.asarray(Image.open(out).convert("L"))


def bands(mask):
    """Contiguous row bands where any pixel is dark, as (row0, row1) pairs."""
    rows = mask.any(axis=1)
    out, start = [], None
    for i, r in enumerate(rows):
        if r and start is None:
            start = i
        elif not r and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(rows)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=int, default=1200)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="\\includegraphics scale: include_width / native_width")
    ap.add_argument("--crop", action="append", default=[],
                    help="x0,y0,x1,y1 as page fractions")
    ap.add_argument("--thresh", type=int, default=160)
    a = ap.parse_args()

    img = raster(a.pdf, a.dpi)
    H, W = img.shape
    pt_per_px = 72.0 / a.dpi * a.scale
    print(f"{a.pdf}: {W}x{H} px at {a.dpi} dpi, scale {a.scale:.4f}")
    print(f"  1 px = {pt_per_px:.5f} pt on the page")

    crops = a.crop or ["0,0,1,1"]
    for c in crops:
        x0, y0, x1, y1 = (float(v) for v in c.split(","))
        sub = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        m = sub < a.thresh
        bs = bands(m)
        print(f"  crop {c}: {len(bs)} row bands")
        hs = sorted(((r1 - r0) * pt_per_px, r0, r1) for r0, r1 in bs)
        for h, r0, r1 in hs[:14]:
            print(f"     band rows {r0:5d}-{r1:5d}  ink {h:6.2f} pt"
                  f"  -> ~{h / CAP_RATIO:5.2f} pt type if capitals")


if __name__ == "__main__":
    main()
