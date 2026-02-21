"""
generate_grid_cells.py

Takes 6 view images and produces a single composed grid image:
  6 rows (BACK/FRONT/LEFT/RIGHT/TOP/BOTTOM) x 6 columns (0/60/120/180/240/300 deg)

Usage:
    python generate_grid_cells.py \
        --back   BACK.png  \
        --front  FRONT.png \
        --left   LEFT.png  \
        --right  RIGHT.png \
        --top    TOP.png   \
        --bottom BOTTOM.png \
        --out    docs/Figure4_dataset.png
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageFont

VIEWS  = ["BACK", "FRONT", "LEFT", "RIGHT", "TOP", "BOTTOM"]
ANGLES = [0, 60, 120, 180, 240, 300]

# Layout
CELL       = 140          # px per cell (content area)
PADDING    = 8            # inner padding inside each cell
ROW_LABEL  = 110          # width of the row-label column
COL_LABEL  = 52           # height of the column-label row
BORDER     = 1            # cell border thickness
GAP        = 2            # gap between cells

# Colours (dark theme)
BG         = (16,  22,  34)
CELL_BG    = (26,  35,  50)
BORDER_COL = (42,  54,  73)
LABEL_COL  = (200, 210, 230)
ACCENT     = (19,  91, 236)


def load_font(size):
    for name in ("DejaVuSansMono.ttf", "Menlo.ttc", "Courier New.ttf",
                 "courbd.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_cell(img, angle):
    """Rotate image, white-fill corners, fit into CELL x CELL square."""
    rgba = img.convert("RGBA")
    rot  = rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
    bg   = Image.new("RGBA", rot.size, (255, 255, 255, 255))
    bg.paste(rot, mask=rot.split()[3])
    rgb  = bg.convert("RGB")
    rgb.thumbnail((CELL - PADDING * 2, CELL - PADDING * 2), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), (255, 255, 255))
    x = (CELL - rgb.width)  // 2
    y = (CELL - rgb.height) // 2
    canvas.paste(rgb, (x, y))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    for v in VIEWS:
        parser.add_argument(f"--{v.lower()}", required=True)
    parser.add_argument("--out", default="docs/Figure4_dataset.png")
    args = parser.parse_args()

    sources = {v: Image.open(getattr(args, v.lower())) for v in VIEWS}

    n_rows  = len(VIEWS)
    n_cols  = len(ANGLES)
    total_w = ROW_LABEL + n_cols * (CELL + GAP) + GAP
    total_h = COL_LABEL + n_rows * (CELL + GAP) + GAP + 48

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    font_label   = load_font(20)
    font_small   = load_font(13)
    font_caption = load_font(15)

    # Column headers
    for ci, angle in enumerate(ANGLES):
        x = ROW_LABEL + GAP + ci * (CELL + GAP) + CELL // 2
        y = COL_LABEL // 2 - 7
        draw.text((x, y), f"{angle}°", fill=ACCENT,
                  font=font_label, anchor="mm")

    # Rows
    for ri, view in enumerate(VIEWS):
        y0 = COL_LABEL + GAP + ri * (CELL + GAP)

        # Row label
        draw.text((ROW_LABEL // 2, y0 + CELL // 2),
                  view, fill=LABEL_COL, font=font_label, anchor="mm")

        for ci, angle in enumerate(ANGLES):
            x0 = ROW_LABEL + GAP + ci * (CELL + GAP)

            # Border then cell background
            draw.rectangle(
                [x0 - BORDER, y0 - BORDER,
                 x0 + CELL + BORDER, y0 + CELL + BORDER],
                fill=BORDER_COL
            )
            draw.rectangle(
                [x0, y0, x0 + CELL, y0 + CELL],
                fill=CELL_BG
            )

            # Rotated product image
            canvas.paste(make_cell(sources[view], angle), (x0, y0))

            # Small angle label bottom-right of cell
            draw.text((x0 + CELL - 4, y0 + CELL - 4),
                      f"{angle}°", fill=(120, 140, 170),
                      font=font_small, anchor="rb")

    # Caption
    caption = "156 images  .  5 products  .  6 views x 6 rotations  (0 - 300 deg in 60 deg steps)"
    cap_y   = COL_LABEL + GAP + n_rows * (CELL + GAP) + 16
    draw.text((total_w // 2, cap_y), caption,
              fill=LABEL_COL, font=font_caption, anchor="mm")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"Saved -> {args.out}  ({total_w}x{total_h}px)")


if __name__ == "__main__":
    main()
