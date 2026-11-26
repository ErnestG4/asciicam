#!/usr/bin/env python3
"""asciicam - live camera to ASCII in any terminal.

Runs on the device's own console or over SSH; the grid follows the terminal size.

Sources (tried in order unless --source is given):
  picam  - Raspberry Pi camera via picamera2 (CardputerZero, Pi Zero 2 W)
  cv     - any V4L2/USB webcam via OpenCV (desktop testing)
  test   - synthetic moving pattern (no camera needed)

Keys:
  q quit      r next ramp     i invert      c colour on/off
  e edges     m mirror        f flip Y      [ ] gamma     s save frame
"""
import argparse
import os
import select
import shutil
import sys
import termios
import time
import tty

import numpy as np

VERSION = "0.2.0"

RAMPS = [
    " .:-=+*#%@",
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    " .oO@",
    " ░▒▓█",
]
CELL_ASPECT = 2.0  # a character cell is about twice as tall as it is wide
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
RAMP_CHARS = [np.array(list(r)) for r in RAMPS]


# ---------- sources: each returns an HxWx3 uint8 RGB frame ----------

class PiCam:
    def __init__(self, size):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": size, "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()

    def read(self):
        # picamera2's "RGB888" arrays come out in BGR byte order
        return self.cam.capture_array()[..., 2::-1]

    def close(self):
        self.cam.stop()


class CvCam:
    def __init__(self, size, index=0):
        import cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("no OpenCV camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame[..., ::-1]

    def close(self):
        self.cap.release()


class TestSource:
    def __init__(self, size):
        w, h = size
        self.y, self.x = np.mgrid[0:h, 0:w].astype(np.float32)
        self.w, self.h = w, h
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        cx = self.w * (0.5 + 0.3 * np.cos(t))
        cy = self.h * (0.5 + 0.3 * np.sin(1.3 * t))
        d = np.hypot(self.x - cx, self.y - cy)
        g = 0.5 + 0.5 * np.cos(d / 9.0 - 3 * t)
        g *= np.clip(1.4 - d / (0.6 * self.w), 0, 1)
        rgb = np.stack([g, g * (self.x / self.w), g * (self.y / self.h)], -1)
        return (rgb * 255).astype(np.uint8)

    def close(self):
        pass


def open_source(name, size):
    order = [name] if name != "auto" else ["picam", "cv"]
    errors = []
    for n in order:
        try:
            if n == "picam":
                return PiCam(size), n
            if n == "cv":
                return CvCam(size), n
            if n == "test":
                return TestSource(size), n
        except Exception as e:  # missing module, no camera, busy device
            errors.append(f"{n}: {e}")
    sys.exit("no camera source available\n  " + "\n  ".join(errors)
             + "\n  (try --source test)")


# ---------- image -> grid ----------

def crop_to_aspect(img, aspect):
    """Centre-crop so width/height == aspect."""
    h, w = img.shape[:2]
    if w / h > aspect:
        nw = max(1, int(round(h * aspect)))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = max(1, int(round(w / aspect)))
    y0 = (h - nh) // 2
    return img[y0:y0 + nh]


def area_resize(img, cols, rows):
    """Box-average down to rows x cols (falls back to nearest if upscaling)."""
    h, w = img.shape[:2]
    if h < rows or w < cols:
        yi = np.arange(rows) * h // rows
        xi = np.arange(cols) * w // cols
        return img[yi][:, xi].astype(np.float32)
    ye = np.linspace(0, h, rows + 1).astype(int)
    xe = np.linspace(0, w, cols + 1).astype(int)
    f = img.astype(np.float32)
    f = np.add.reduceat(f, ye[:-1], axis=0)
    f = np.add.reduceat(f, xe[:-1], axis=1)
    counts = np.outer(np.diff(ye), np.diff(xe)).astype(np.float32)
    return f / (counts[..., None] if f.ndim == 3 else counts)


def edges(lum):
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    return np.hypot(gx, gy)


def stretch(lum, st):
    """Contrast-stretch with smoothed bounds so levels don't pulse frame
    to frame (worst in the coarse block ramp)."""
    lo, hi = np.percentile(lum, (2, 98))
    if st.get("lo") is not None:
        lo = 0.8 * st["lo"] + 0.2 * lo
        hi = 0.8 * st["hi"] + 0.2 * hi
    st["lo"], st["hi"] = lo, hi
    if hi - lo < 1e-6:
        return np.zeros_like(lum)
    return np.clip((lum - lo) / (hi - lo), 0, 1)


def render(frame, cols, rows, st):
    img = crop_to_aspect(frame, cols / (rows * CELL_ASPECT))
    small = area_resize(img, cols, rows)
    if st["vflip"]:
        small = small[::-1]
    if st["mirror"]:
        small = small[:, ::-1]
    lum = small @ LUMA
    if st["edges"]:
        lum = edges(lum)
    lum = stretch(lum, st) ** st["gamma"]
    if st["invert"]:
        lum = 1.0 - lum
    ramp = RAMPS[st["ramp"]]
    idx = np.minimum((lum * len(ramp)).astype(int), len(ramp) - 1)
    chars = RAMP_CHARS[st["ramp"]][idx]
    plain = ["".join(r) for r in chars]
    if not st["colour"]:
        return plain, plain
    # Colour is the cell's averaged hue with its brightness replaced by the
    # same stretched luminance that picked the glyph, so colour and glyph
    # density always agree; quantised so flat areas form long runs.
    cell_lum = np.maximum(small @ LUMA, 1.0)[..., None]
    rgb = np.clip(small * (lum[..., None] * 255.0 / cell_lum),
                  0, 255).astype(np.uint8)
    rgb &= 0xF0
    out = []
    for y in range(rows):
        row, cells = rgb[y], plain[y]
        cut = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]).any(1)])
        parts = []
        for i, s in enumerate(cut):
            e = cut[i + 1] if i + 1 < len(cut) else cols
            parts.append("\x1b[38;2;%d;%d;%dm%s"
                         % (row[s, 0], row[s, 1], row[s, 2], cells[s:e]))
        out.append("".join(parts) + "\x1b[0m")
    return out, plain


# ---------- terminal ----------

def key_pressed():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto",
                    choices=["auto", "picam", "cv", "test"])
    ap.add_argument("--size", default="320x240", help="capture size WxH")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=0,
                    help="stop after N frames (0 = run until q)")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))

    src, name = open_source(args.source, size)
    # the Cardputer Zero camera module is mounted rotated 180 degrees
    st = dict(ramp=0, invert=False, colour=False, edges=False,
              mirror=(name == "picam"), vflip=(name == "picam"), gamma=1.0)
    interactive = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if interactive else None
    out = sys.stdout
    msg, n, fps, last = "", 0, 0.0, time.time()
    try:
        if interactive:
            tty.setcbreak(sys.stdin.fileno())
        out.write("\x1b[?25l\x1b[2J")
        while True:
            t0 = time.time()
            cols, lines = shutil.get_terminal_size((53, 17))
            rows = max(1, lines - 1)
            shown, plain = render(src.read(), cols, rows, st)
            status = (" %s %4.1ffps ramp%d g%.1f %s" % (
                name, fps, st["ramp"], st["gamma"], msg))[:cols]
            out.write("\x1b[H" + "\n".join(shown) + "\n\x1b[7m"
                      + status.ljust(cols) + "\x1b[0m")
            out.flush()

            k = key_pressed() if interactive else None
            if k == "q":
                break
            elif k == "r":
                st["ramp"] = (st["ramp"] + 1) % len(RAMPS)
            elif k == "i":
                st["invert"] = not st["invert"]
            elif k == "c":
                st["colour"] = not st["colour"]
            elif k == "e":
                st["edges"] = not st["edges"]
                st["lo"] = st["hi"] = None  # edge stats differ; re-seed
            elif k == "m":
                st["mirror"] = not st["mirror"]
            elif k == "f":
                st["vflip"] = not st["vflip"]
            elif k == "[":
                st["gamma"] = max(0.3, round(st["gamma"] - 0.1, 1))
            elif k == "]":
                st["gamma"] = min(3.0, round(st["gamma"] + 0.1, 1))
            elif k == "s":
                fn = time.strftime("ascii-%Y%m%d-%H%M%S.txt")
                with open(fn, "w", encoding="utf-8") as f:
                    f.write("\n".join(plain) + "\n")
                msg = "saved " + fn

            n += 1
            if args.frames and n >= args.frames:
                break
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        out.write("\x1b[0m\x1b[?25h\n")
        out.flush()


if __name__ == "__main__":
    main()
