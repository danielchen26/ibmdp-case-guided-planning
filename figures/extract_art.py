#!/usr/bin/env python3
r"""Extract the embedded line-art panels from figs/TOCScheme.svg.

figs/TOCScheme.svg is the author's original workflow drawing -- the same art that
figs/IBMDP_scheme_hires.png is a flattened raster of.  It is not an editable
vector source: it has zero <text> elements and 230 <path> elements, and all of
its lettering lives inside five base64-embedded PNGs.  That is why the schematic
had to be re-laid-out (scheme_fig1.py) rather than rescaled.

Four of those five panels are pure line art with no lettering, so they are reused
verbatim in the re-laid-out figure; only the fifth (the data table) carries text
and is redrawn as vector.  Extraction order is the SVG's document order:

  index  native px   content            reused as
  0      715 x 396   the data table     no -- redrawn as vector (its text is raster)
  1      521 x 416   caffeine skeleton  art/molecule.png
  2      416 x 387   petri dish + pipette  art/petri.png
  3      189 x 143   mouse              art/mouse.png
  4      292 x 289   monkey             art/monkey.png

Usage:
  python3 figures/extract_art.py --svg figures/art/TOCScheme.svg --out figures/art
"""
import argparse
import base64
import os
import re

NAMES = {1: "molecule.png", 2: "petri.png", 3: "mouse.png", 4: "monkey.png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    svg = open(a.svg).read()
    tags = re.findall(r"<image[^>]*?>", svg, re.S)
    print(f"{a.svg}: {len(tags)} embedded images, 0 <text> elements "
          f"({len(re.findall('<text', svg))} found)")
    for i, tag in enumerate(tags):
        m = re.search(r'href="data:image/png;base64,([^"]+)"', tag)
        if m is None:
            print(f"  {i}: not a base64 PNG, skipped")
            continue
        raw = base64.b64decode(m.group(1))
        if i not in NAMES:
            print(f"  {i}: {len(raw)} B, carries lettering -- not reused")
            continue
        path = os.path.join(a.out, NAMES[i])
        with open(path, "wb") as f:
            f.write(raw)
        print(f"  {i}: {len(raw)} B -> {path}")


if __name__ == "__main__":
    main()
