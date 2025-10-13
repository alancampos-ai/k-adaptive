#!/usr/bin/env python3
from pathlib import Path
import json
import argparse
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def load_font(pt: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", pt)
    except Exception:
        return ImageFont.load_default()

def read_config(cfg_path: Path, images_root: Path | None) -> Tuple[List[str], List[List[str]]]:
    cfg = json.loads(cfg_path.read_text())
    titles = cfg.get("titles", ["", "", "", ""])
    panels_raw = cfg["panels"]
    if len(panels_raw) != 4 or any(len(p) != 9 for p in panels_raw):
        raise ValueError("config.panels must have 4 entries, each with 9 paths.")
    panels: List[List[str]] = []
    for nine in panels_raw:
        row = []
        for p in nine:
            pth = Path(p)
            if pth.is_absolute():
                q = pth
            else:
                if images_root is not None:
                    q = (images_root / pth.name).resolve()
                else:
                    q = (cfg_path.parent / pth).resolve()
            row.append(str(q))
        panels.append(row)
    missing = [q for row in panels for q in row if not Path(q).exists()]
    if missing:
        raise FileNotFoundError(f"Missing file: {missing[0]}")
    return titles, panels

def pick_axial_size(panels: List[List[str]]) -> Tuple[int, int]:
    for nine in panels:
        for p in nine:
            if "axial" in Path(p).name.lower():
                w, h = Image.open(p).convert("RGB").size
                return w, h
    raise RuntimeError("No file containing 'axial' was found in the config.")

def compose(
    titles: List[str],
    panels: List[List[str]],
    outfile: Path,
    tile_w: int,
    tile_h: int,
    gap: int = 18,
    panel_gap_x: int = 80,
    panel_gap_y: int = 80,
    margin: int = 60,
    title_pt: int = 44,
    bg=(255, 255, 255),
):
    font = load_font(title_pt)
    panel_w = 3 * tile_w + 2 * gap
    panel_h = 3 * tile_h + 2 * gap
    title_h = font.getbbox("Hy")[3]
    W = margin + panel_w + panel_gap_x + panel_w + margin
    H = margin + (title_h + panel_h) + panel_gap_y + (title_h + panel_h) + margin
    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)
    bases = [
        (margin, margin),
        (margin + panel_w + panel_gap_x, margin),
        (margin, margin + title_h + panel_h + panel_gap_y),
        (margin + panel_w + panel_gap_x, margin + title_h + panel_h + panel_gap_y),
    ]
    for pidx, (nine, title) in enumerate(zip(panels, titles)):
        base_x, base_y = bases[pidx]
        if title:
            tw, th = draw.textbbox((0, 0), title, font=font)[2:]
            tx = base_x + (panel_w - tw) // 2
            draw.text((tx, base_y), title, fill=(0, 0, 0), font=font)
        grid_y0 = base_y + title_h
        for i, path in enumerate(nine):
            img = Image.open(path).convert("RGB")
            img = img.resize((tile_w, tile_h), Image.NEAREST)
            r, c = divmod(i, 3)
            x = base_x + c * (tile_w + gap)
            y = grid_y0 + r * (tile_h + gap)
            canvas.paste(img, (x, y))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outfile)
    print(f"ok: {outfile} [{W}x{H}px]")

def main():
    here = Path(__file__).resolve()
    root = here.parents[1]
    default_config = root / "dataset" /"json"/ "panel.json"
    default_outdir = root / "results" / "figs" / "panel"
    default_outfile = default_outdir / "panel.png"
    default_images_root = root / "results" / "img" / "color"

    ap = argparse.ArgumentParser(description="2x2 poster with four 3x3 rigid grids.")
    ap.add_argument("--config", type=Path, default=default_config, help="JSON with 'titles' and 'panels' (4x9 paths).")
    ap.add_argument("-o", "--outfile", type=Path, default=default_outfile, help="Output image path.")
    ap.add_argument("--images-root", type=Path, default=default_images_root, help="Root directory for image files.")
    ap.add_argument("--tile-w", type=int, default=0, help="Exact tile width in pixels.")
    ap.add_argument("--tile-h", type=int, default=0, help="Exact tile height in pixels.")
    ap.add_argument("--use-axial-size", action="store_true", help="Infer tile size from the first file containing 'axial'.")
    ap.add_argument("--gap", type=int, default=18)
    ap.add_argument("--panel-gap-x", type=int, default=80)
    ap.add_argument("--panel-gap-y", type=int, default=80)
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument("--title-pt", type=int, default=44)
    args = ap.parse_args()

    titles, panels = read_config(args.config, args.images_root)
    if args.use_axial_size:
        tw, th = pick_axial_size(panels)
    else:
        if args.tile_w <= 0 or args.tile_h <= 0:
            raise SystemExit("Set --tile-w and --tile-h or use --use-axial-size.")
        tw, th = args.tile_w, args.tile_h

    compose(
        titles, panels, args.outfile,
        tile_w=tw, tile_h=th,
        gap=args.gap, panel_gap_x=args.panel_gap_x, panel_gap_y=args.panel_gap_y,
        margin=args.margin, title_pt=args.title_pt,
    )

if __name__ == "__main__":
    main()

