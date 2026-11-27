#!/usr/bin/env python3
"""asciicam - live camera to ASCII in any terminal.

Runs on the device's own console or over SSH; the grid follows the terminal size.

Sources (tried in order unless --source is given):
  picam  - Raspberry Pi camera via picamera2 (CardputerZero, Pi Zero 2 W)
  cv     - any V4L2/USB webcam via OpenCV (desktop testing)
  test   - synthetic moving pattern (no camera needed)

Keys:
  q quit      r next ramp     i invert      c colour on/off
  e edges     m mirror        f flip Y      [ ] gamma
  s snapshot (.txt + .ans + rendered .png + raw .jpg)
  v record / stop (asciinema .cast; render to video with --render)

Captures land in ~/Pictures/asciicam/.
"""
import argparse
import json
import os
import re
import select
import shutil
import sys
import termios
import time
import tty

import numpy as np

VERSION = "0.4.0"

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


def cell_grid(frame, cols, rows, st):
    """Front half of the pipeline, shared by every output: cell averages,
    stretched luminance and the glyph index per cell. A temporal EMA on
    the cell grid keeps sensor noise from touching stable cells."""
    img = crop_to_aspect(frame, cols / (rows * CELL_ASPECT))
    small = area_resize(img, cols, rows)
    if st["vflip"]:
        small = small[::-1]
    if st["mirror"]:
        small = small[:, ::-1]
    acc = st.get("acc")
    if acc is None or acc.shape != small.shape:
        st["acc"] = small
    else:
        st["acc"] = 0.65 * acc + 0.35 * small
    small = st["acc"]
    lum = small @ LUMA
    if st["edges"]:
        lum = edges(lum)
    lum = stretch(lum, st) ** st["gamma"]
    if st["invert"]:
        lum = 1.0 - lum
    ramp = RAMPS[st["ramp"]]
    idx = np.minimum((lum * len(ramp)).astype(int), len(ramp) - 1)
    return small, idx


def cell_colour(small, st):
    """Cell colours 0..1, through the same global contrast stretch as the
    glyphs: one linear map, no per-cell gain. None when colour is off."""
    if not st["colour"]:
        return None
    lo, hi = st["lo"], st["hi"]
    return np.clip((small - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def colour_codes(rgb, truecolor):
    """Per-cell colour keys plus the escape builder for them."""
    if truecolor:
        q = (rgb * 255).astype(np.int32) & 0xE0
        col = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]

        def sgr(v):
            return "\x1b[38;2;%d;%d;%dm" % (v >> 16, v >> 8 & 255, v & 255)
    else:
        cube = (rgb * 5.999).astype(np.int32)
        col = 16 + 36 * cube[..., 0] + 6 * cube[..., 1] + cube[..., 2]

        def sgr(v):
            return "\x1b[38;5;%dm" % v
    return col, sgr


def ansi_full(idx, rgb, st):
    """One whole frame as ANSI art: rows joined by CRLF, colour as run-
    length escapes. Used for .ans snapshots and .cast recording frames."""
    chars = RAMP_CHARS[st["ramp"]][idx]
    plain = ["".join(r) for r in chars]
    if rgb is None:
        return "\r\n".join(plain)
    col, sgr = colour_codes(rgb, st["truecolor"])
    rows, cols = idx.shape
    out = []
    for y in range(rows):
        row = col[y]
        cut = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        out.append("".join(
            sgr(row[s]) + plain[y][s:(cut[i + 1] if i + 1 < len(cut)
                                      else cols)]
            for i, s in enumerate(cut)))
    return "\r\n".join(out) + "\x1b[0m"


def render(idx, rgb, cols, rows, st):
    """One terminal update string (a diff against the previous frame) plus
    the plain-text grid. The slow link is the terminal painting cells, not
    this code, so only cells whose glyph or colour actually changed are
    rewritten."""
    chars = RAMP_CHARS[st["ramp"]][idx]
    plain = ["".join(r) for r in chars]
    if rgb is not None:
        col, sgr = colour_codes(rgb, st["truecolor"])
    else:
        col = None

    pch, pcol = st.get("pchars"), st.get("pcol")
    full = (pch is None or pch.shape != chars.shape
            or (col is None) != (pcol is None)
            or st.get("frame_i", 0) % 120 == 0)
    if full:
        changed = np.ones(chars.shape, bool)
    else:
        changed = chars != pch
        if col is not None:
            changed |= col != pcol
    st["pchars"], st["pcol"], st["full"] = chars, col, full

    parts = ["\x1b[2J"] if full else []
    for y in np.flatnonzero(changed.any(1)):
        xs = np.flatnonzero(changed[y])
        # segments of changed cells; gaps of a few unchanged cells are
        # cheaper to rewrite than a fresh cursor move
        brk = np.flatnonzero(np.diff(xs) > 4)
        for g0, g1 in zip(np.r_[0, brk + 1], np.r_[brk, len(xs) - 1]):
            s, e = xs[g0], xs[g1] + 1
            parts.append("\x1b[%d;%dH" % (y + 1, s + 1))
            if col is None:
                parts.append(plain[y][s:e])
            else:
                row = col[y]
                cut = np.flatnonzero(
                    np.r_[True, row[s + 1:e] != row[s:e - 1]]) + s
                for i, cs in enumerate(cut):
                    ce = cut[i + 1] if i + 1 < len(cut) else e
                    parts.append(sgr(row[cs]) + plain[y][cs:ce])
    if col is not None and len(parts) > (1 if full else 0):
        parts.append("\x1b[0m")
    return "".join(parts), plain


# ---------- glyph rasterizing ----------

class GlyphAtlas:
    """Antialiased glyph masks at one cell size, shared by the LCD
    framebuffer output, PNG snapshots and cast-to-video rendering."""
    CHARS = [chr(i) for i in range(32, 127)] + list("░▒▓█")
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

    def __init__(self, cw, ch):
        from PIL import Image, ImageDraw, ImageFont
        self.cw, self.ch = cw, ch
        self.glyph_id = {c: i for i, c in enumerate(self.CHARS)}
        font = ImageFont.truetype(self.FONT, ch)
        a = np.zeros((len(self.CHARS), ch, cw), np.float32)
        for i, c in enumerate(self.CHARS):
            im = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(im).text((0, -1), c, font=font, fill=255)
            a[i] = np.asarray(im, np.float32) / 255.0
        # the shade blocks read better as flat coverage at small sizes
        for c, lvl in zip("░▒▓█", (0.25, 0.5, 0.75, 1.0)):
            a[self.glyph_id[c]] = lvl
        self.a = a
        self.ramp_ids = {}

    def ids(self, ramp):
        ids = self.ramp_ids.get(ramp)
        if ids is None:
            ids = np.array([self.glyph_id.get(c, 0) for c in ramp])
            self.ramp_ids[ramp] = ids
        return ids

    def compose(self, gids, col):
        """RGB float image of a glyph-id grid; col is per-cell RGB 0-255
        or None for plain light-grey."""
        rows, cols = gids.shape
        mask = self.a[gids]                         # rows,cols,ch,cw
        if col is None:
            col = np.full(gids.shape + (3,), 230.0, np.float32)
        px = mask[..., None] * col[:, :, None, None, :]
        return px.transpose(0, 2, 1, 3, 4).reshape(rows * self.ch,
                                                   cols * self.cw, 3)


# ---------- framebuffer output ----------

class FbOut:
    """Draw the grid straight to the LCD framebuffer (RGB565), the same
    path the launcher UI uses - no terminal in the middle, so colour and
    block ramps cost the same as plain ASCII."""
    CW, CH = 6, 10

    def __init__(self, path="/dev/fb_lcd"):
        base = "/sys/class/graphics/" + os.path.basename(
            os.path.realpath(path))
        self.w, self.h = (int(v) for v in
                          open(base + "/virtual_size").read().split(","))
        bpp = int(open(base + "/bits_per_pixel").read())
        if bpp != 16:
            raise RuntimeError("fb is %d bpp, expected 16 (RGB565)" % bpp)
        self.f = open(path, "r+b", buffering=0)
        self.cols = self.w // self.CW
        self.rows = self.h // self.CH - 1  # bottom row is the status bar
        self.atlas = GlyphAtlas(self.CW, self.CH)

    def show(self, idx, col, ramp, status):
        px = self.atlas.compose(self.atlas.ids(ramp)[idx], col)
        frame = np.zeros((self.h, self.w, 3), np.float32)
        frame[:px.shape[0], :px.shape[1]] = px
        self._status(frame, status)
        f16 = frame.astype(np.uint16)
        buf = (((f16[..., 0] >> 3) << 11) | ((f16[..., 1] >> 2) << 5)
               | (f16[..., 2] >> 3))
        self.f.seek(0)
        self.f.write(buf.tobytes())

    def _status(self, frame, text):
        y0 = self.rows * self.CH
        frame[y0:, :] = 205.0
        ids = [self.atlas.glyph_id.get(c, 0) for c in text[: self.cols]]
        if not ids:
            return
        strip = self.atlas.a[np.array(ids)].transpose(1, 0, 2).reshape(
            self.CH, -1)[:, : self.w, None]
        frame[y0:y0 + self.CH, : strip.shape[1]] = (
            205.0 * (1 - strip) + 20.0 * strip)

    def close(self):
        self.f.close()


# ---------- capture ----------

SAVE_DIR = os.path.expanduser("~/Pictures/asciicam")
_snap_atlas = None


def save_path(prefix, ext):
    os.makedirs(SAVE_DIR, exist_ok=True)
    return os.path.join(
        SAVE_DIR, "%s-%s%s" % (prefix, time.strftime("%Y%m%d-%H%M%S"), ext))


def snapshot(idx, rgb, st, frame):
    """Save the moment: plain .txt, coloured .ans, a 2x-rendered .png of
    the ASCII art, and the raw camera frame as .jpg."""
    global _snap_atlas
    base = save_path("ascii", "")
    plain = "\n".join("".join(r) for r in RAMP_CHARS[st["ramp"]][idx])
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(plain + "\n")
    if rgb is not None:
        with open(base + ".ans", "w", encoding="utf-8") as f:
            f.write("\x1b[2J\x1b[H" + ansi_full(idx, rgb, st) + "\n")
    try:
        from PIL import Image
        if _snap_atlas is None:
            _snap_atlas = GlyphAtlas(12, 20)
        px = _snap_atlas.compose(
            _snap_atlas.ids(RAMPS[st["ramp"]])[idx],
            None if rgb is None else rgb * 255.0)
        Image.fromarray(px.astype(np.uint8)).save(base + ".png")
        pic = frame
        if st["vflip"]:
            pic = pic[::-1]
        if st["mirror"]:
            pic = pic[:, ::-1]
        Image.fromarray(np.ascontiguousarray(pic)).save(
            base + "-raw.jpg", quality=90)
    except Exception:
        return "saved %s (.txt only, PIL missing)" % os.path.basename(base)
    return "saved " + os.path.basename(base) + ".*"


class CastRecorder:
    """asciinema v2 recording of full ANSI frames."""

    def __init__(self, path, cols, rows):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.f.write(json.dumps(
            {"version": 2, "width": cols, "height": rows,
             "timestamp": int(self.t0),
             "env": {"TERM": "xterm-256color"}}) + "\n")
        self.write("\x1b[2J\x1b[?25l")

    def write(self, s):
        self.f.write(json.dumps(
            [round(time.time() - self.t0, 4), "o", s]) + "\n")

    def frame(self, idx, rgb, st):
        self.write("\x1b[H" + ansi_full(idx, rgb, st))

    def close(self):
        self.write("\x1b[0m\x1b[?25h")
        self.f.close()


def parse_ansi_frame(s, cols, rows, chars, col):
    """Replay one of OUR recorded frames (cursor home, CRLF, SGR 0/38;5/
    38;2) onto the given char and colour grids."""
    cube = (0, 95, 135, 175, 215, 255)
    x = y = 0
    cur = (230.0, 230.0, 230.0)
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\x1b":
            m = re.match(r"\x1b\[([0-9;]*)([A-Za-z])", s[i:])
            if not m:
                i += 1
                continue
            args, fin = m.group(1), m.group(2)
            if fin == "H":
                p = args.split(";")
                y = int(p[0]) - 1 if p[0] else 0
                x = int(p[1]) - 1 if len(p) > 1 and p[1] else 0
            elif fin == "m":
                p = [int(v) for v in args.split(";") if v] or [0]
                if p[0] == 0:
                    cur = (230.0, 230.0, 230.0)
                elif p[:2] == [38, 5] and len(p) > 2 and p[2] >= 16:
                    v = p[2] - 16
                    cur = (cube[v // 36], cube[v // 6 % 6], cube[v % 6])
                elif p[:2] == [38, 2] and len(p) > 4:
                    cur = tuple(float(v) for v in p[2:5])
            i += m.end()
            continue
        if c == "\r":
            x = 0
        elif c == "\n":
            y += 1
        else:
            if 0 <= y < rows and 0 <= x < cols:
                chars[y][x] = c
                col[y][x] = cur
            x += 1
        i += 1


def render_cast(path, out, fps_out=15.0):
    """Offline: rasterize a recording made by this program into a video
    file via ffmpeg."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found - sudo apt install ffmpeg")
    import subprocess
    lines = open(path, encoding="utf-8").read().splitlines()
    head = json.loads(lines[0])
    cols, rows = head["width"], head["height"]
    events = []
    for ln in lines[1:]:
        t, kind, s = json.loads(ln)
        if kind == "o" and "\x1b[H" in s:
            events.append((t, s))
    if not events:
        sys.exit("no frames in %s" % path)
    atlas = GlyphAtlas(12, 20)
    w, h = cols * atlas.cw, rows * atlas.ch
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
         "-r", "%g" % fps_out, "-i", "-", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE)
    chars = [[" "] * cols for _ in range(rows)]
    col = [[(230.0, 230.0, 230.0)] * cols for _ in range(rows)]
    fi, img = -1, None
    t, end = events[0][0], events[-1][0] + 1.0 / fps_out
    while t <= end:
        nfi = fi
        while nfi + 1 < len(events) and events[nfi + 1][0] <= t:
            nfi += 1
        if nfi != fi:
            fi = max(nfi, 0)
            parse_ansi_frame(events[fi][1], cols, rows, chars, col)
            gids = np.array([[atlas.glyph_id.get(c, 0) for c in row]
                             for row in chars])
            px = atlas.compose(gids, np.array(col, np.float32))
            img = px.astype(np.uint8).tobytes()
        ff.stdin.write(img)
        t += 1.0 / fps_out
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg failed")
    print("wrote %s (%dx%d, %d source frames)" % (out, w, h, len(events)))


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
    ap.add_argument("--truecolor", action="store_true",
                    help="24-bit colour escapes (default: xterm 256-colour)")
    ap.add_argument("--fb", action="store_true",
                    help="draw straight to the LCD framebuffer "
                         "(/dev/fb_lcd) instead of the terminal")
    ap.add_argument("--render", metavar="CAST",
                    help="rasterize a .cast recording to video and exit")
    ap.add_argument("--out", help="output file for --render "
                                  "(default: <cast>.mp4)")
    args = ap.parse_args()
    if args.render:
        render_cast(args.render,
                    args.out or re.sub(r"\.cast$", "", args.render)
                    + ".mp4")
        return
    size = tuple(int(v) for v in args.size.lower().split("x"))

    src, name = open_source(args.source, size)
    # the Cardputer Zero camera module is mounted rotated 180 degrees
    st = dict(ramp=0, invert=False, colour=False, edges=False,
              mirror=(name == "picam"), vflip=(name == "picam"),
              truecolor=args.truecolor, gamma=1.0)
    fb = None
    if args.fb:
        try:
            fb = FbOut()
        except Exception as e:
            src.close()
            sys.exit("--fb unavailable: %s" % e)
    interactive = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if interactive else None
    out = sys.stdout
    msg, n, fps, rec = "", 0, 0.0, None
    win_n, win_t0 = 0, time.time()
    try:
        if interactive:
            tty.setcbreak(sys.stdin.fileno())
        out.write("\x1b[?25l\x1b[2J")
        last_status = None
        while True:
            t0 = time.time()
            if fb is not None:
                cols, rows = fb.cols, fb.rows
            else:
                cols, lines = shutil.get_terminal_size((53, 17))
                rows = max(1, lines - 1)
            st["frame_i"] = n
            status = (" %s %4.1ffps ramp%d g%.1f%s %s" % (
                name, fps, st["ramp"], st["gamma"],
                " REC" if rec else "", msg))[:cols].ljust(cols)
            frame = src.read()
            small, idx = cell_grid(frame, cols, rows, st)
            rgb = cell_colour(small, st)
            if rec is not None:
                rec.frame(idx, rgb, st)
            if fb is not None:
                fb.show(idx, None if rgb is None else rgb * 255.0,
                        RAMPS[st["ramp"]], status)
            else:
                upd, plain = render(idx, rgb, cols, rows, st)
                if status != last_status or st["full"]:
                    upd += "\x1b[%d;1H\x1b[0;7m%s\x1b[0m" % (rows + 1,
                                                            status)
                    last_status = status
                if upd:
                    out.write(upd)
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
                msg = snapshot(idx, rgb, st, frame)
            elif k == "v":
                if rec is None:
                    rec = CastRecorder(save_path("rec", ".cast"),
                                       cols, rows)
                    msg = "recording " + os.path.basename(rec.path)
                else:
                    rec.close()
                    msg = "saved " + os.path.basename(rec.path)
                    rec = None
            if k and k != "s":
                st["pchars"] = None  # mode changed: full repaint
                if k in "fm":
                    st["acc"] = None  # geometry changed: restart denoise

            n += 1
            win_n += 1
            if args.frames and n >= args.frames:
                break
            # frames actually delivered per wall-clock second: a blocked
            # terminal write slows this honestly, where averaging
            # instantaneous rates inflated it
            now = time.time()
            if now - win_t0 >= 1.0:
                fps = win_n / (now - win_t0)
                win_n, win_t0 = 0, now
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if rec is not None:
            rec.close()
        if fb is not None:
            fb.close()
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        out.write("\x1b[0m\x1b[?25h\n")
        out.flush()


if __name__ == "__main__":
    main()
