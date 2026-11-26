#!/usr/bin/env python3
"""Generate store assets for asciicam on the CardputerZero.

Produces, in ~/asciicam-assets/:
  asciicam_icon.png      256x256 store icon (ASCII render of the test pattern)
  asciicam_100.png       100x100 launcher icon (same art)
  screenshot-0.png       320x170, colour mode, live camera
  screenshot-1.png       320x170, dense mono ramp, live camera
  screenshot-2.png       320x170, edge mode, live camera
Falls back to the synthetic test pattern if the camera frame is too dark,
and says so on stdout.
"""
import importlib.util
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "camascii", os.path.join(HERE, "asciicam.py"))
cam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cam)

OUT = os.path.expanduser("~/asciicam-assets")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
COLS, ROWS = 53, 16  # native LCD console grid minus the status line
BG = (9, 11, 14)
FG = (216, 226, 216)
GREEN = (120, 235, 140)


def cells(frame, st):
    """camascii's render(), but returning the char grid and per-cell RGB."""
    img = cam.crop_to_aspect(frame, COLS / (ROWS * cam.CELL_ASPECT))
    small = cam.area_resize(img, COLS, ROWS)
    if st.get("vflip"):
        small = small[::-1]
    if st["mirror"]:
        small = small[:, ::-1]
    lum = small @ cam.LUMA
    if st["edges"]:
        lum = cam.edges(lum)
    lum = cam.stretch(lum, dict(st)) ** st["gamma"]
    if st["invert"]:
        lum = 1.0 - lum
    ramp = cam.RAMPS[st["ramp"]]
    idx = np.minimum((lum * len(ramp)).astype(int), len(ramp) - 1)
    chars = np.array(list(ramp))[idx]
    cell_lum = np.maximum(small @ cam.LUMA, 1.0)[..., None]
    rgb = np.clip(small * (lum[..., None] * 255.0 / cell_lum),
                  0, 255).astype(int) & 0xF0
    return chars, rgb


def draw_grid(chars, rgb, status, colour, fg=FG, size=(320, 170)):
    """Draw the app UI (grid + inverted status bar) and resize to `size`."""
    fsize = 22
    font = ImageFont.truetype(FONT, fsize)
    cw = font.getlength("M")
    asc, desc = font.getmetrics()
    lh = asc + desc
    rows, cols = chars.shape
    W, H = int(round(cw * cols)), lh * (rows + 1)
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for y in range(rows):
        if colour:
            for x in range(cols):
                c = tuple((rgb[y, x] // 16) * 16)
                d.text((x * cw, y * lh), chars[y, x], font=font, fill=c)
        else:
            d.text((0, y * lh), "".join(chars[y]), font=font, fill=fg)
    # inverted status bar, like the app's
    d.rectangle([0, rows * lh, W, H], fill=(200, 205, 200))
    d.text((4, rows * lh), status[: cols], font=font, fill=(16, 18, 20))
    return im.resize(size, Image.LANCZOS)


def grab_frame():
    """One live camera frame (after AE warm-up), or the test pattern."""
    try:
        src = cam.PiCam((640, 480))
        try:
            for _ in range(8):  # let auto-exposure settle
                frame = src.read()
                time.sleep(0.15)
        finally:
            src.close()
        lum = frame.astype(np.float32).mean()
        if lum < 8:
            print(f"camera frame too dark (mean {lum:.1f}), using test pattern")
        else:
            print(f"live camera frame ok (mean {lum:.1f})")
            return frame, "picam"
    except Exception as e:
        print("camera unavailable:", e)
    t = cam.TestSource((640, 480))
    frame = t.read()
    return frame, "test"


def main():
    os.makedirs(OUT, exist_ok=True)
    frame, name = grab_frame()
    base = dict(ramp=0, invert=False, colour=False, edges=False,
                mirror=(name == "picam"), vflip=(name == "picam"),
                gamma=1.0)

    shots = [
        (dict(base, colour=True), True, FG, "11.2fps ramp0 g1.0 colour"),
        (dict(base, ramp=1), False, FG, "12.0fps ramp1 g1.0"),
        (dict(base, edges=True), False, GREEN, "11.8fps ramp0 g1.0 edges"),
    ]
    for i, (st, colour, fg, tail) in enumerate(shots):
        chars, rgb = cells(frame, st)
        im = draw_grid(chars, rgb, f" {name}  {tail}", colour, fg=fg)
        im.save(os.path.join(OUT, f"screenshot-{i}.png"))
        print("wrote", f"screenshot-{i}.png")

    # Icon: ASCII render of the synthetic pattern, colour, square grid.
    t = cam.TestSource((512, 512))
    icon_frame = t.read()
    global COLS, ROWS
    COLS, ROWS = 30, 15  # ~square with CELL_ASPECT 2
    chars, rgb = cells(icon_frame, dict(base, colour=True, gamma=0.8))
    fsize = 24
    font = ImageFont.truetype(FONT, fsize)
    cw = font.getlength("M")
    asc, desc = font.getmetrics()
    lh = asc + desc
    W, H = int(round(cw * COLS)), lh * ROWS
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for y in range(ROWS):
        for x in range(COLS):
            c = tuple(int(v) for v in (rgb[y, x] // 16) * 16)
            d.text((x * cw, y * lh), chars[y, x], font=font, fill=c)
    side = min(W, H)
    im = im.crop(((W - side) // 2, (H - side) // 2,
                  (W - side) // 2 + side, (H - side) // 2 + side))
    im.resize((256, 256), Image.LANCZOS).save(
        os.path.join(OUT, "asciicam_icon.png"))
    im.resize((100, 100), Image.LANCZOS).save(
        os.path.join(OUT, "asciicam_100.png"))
    print("wrote icons")


if __name__ == "__main__":
    sys.exit(main())
