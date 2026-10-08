"""Branded QR code for https://wildmatch.gmum.net: round dots shading from the logo's blue to its red,
rounded finder eyes, the WildMatch mark in a white badge in the centre (error correction H).

Needs `qrcode[pil]` (8.2), which the package environment does not include; run it in a throwaway env:
    python paper/tools/make_qr.py docs/assets/logo/wildmatch-logo-512.png <output dir>
Writes `wildmatch_qr.png` (2460 px), `wildmatch_qr.svg` and `wildmatch_qr_label.png` (with the address).
The README copy, `docs/assets/qr/wildmatch-qr.png`, is the PNG resized to 600 px and reduced to 64 colours;
check any new version with a phone-style decoder (zxing-cpp) before committing it.
"""

import base64
import io
import math
import sys
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont

URL = "https://wildmatch.gmum.net"
LOGO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

BLUE = (0x2B, 0x63, 0x8C)  # brand blue #3a7eab darkened for scanner contrast
RED = (0xB5, 0x3A, 0x26)  # brand red #cf4832 darkened likewise
GREY = (0x58, 0x59, 0x5B)

qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
qr.add_data(URL)
qr.make(fit=True)
m = qr.get_matrix()
n = len(m)

CELL = 60
QUIET = 4
size = (n + 2 * QUIET) * CELL
SS = 1  # drawn directly at high resolution


def mix(t):
    return tuple(round(a + (b - a) * t) for a, b in zip(BLUE, RED))


def in_finder(r, c):
    return (r < 7 and c < 7) or (r < 7 and c >= n - 7) or (r >= n - 7 and c < 7)


# centre badge: keep about 22 % of the width clear (well within level H)
badge_modules = int(round(n * 0.26)) | 1
b0 = (n - badge_modules) // 2
b1 = b0 + badge_modules


def in_badge(r, c):
    return b0 <= r < b1 and b0 <= c < b1


img = Image.new("RGB", (size, size), "white")
d = ImageDraw.Draw(img)
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n + 2 * QUIET} {n + 2 * QUIET}">',
    f'<rect width="{n + 2 * QUIET}" height="{n + 2 * QUIET}" fill="#fff"/>',
]
hexc = lambda col: "#%02x%02x%02x" % col  # noqa: E731

for r in range(n):
    for c in range(n):
        if not m[r][c] or in_finder(r, c) or in_badge(r, c):
            continue
        t = 1 / (1 + math.exp(-(c - (n - 1) / 2) / 1.6))  # blue half | red half, short blend
        col = mix(t)
        x = (c + QUIET) * CELL
        y = (r + QUIET) * CELL
        pad = CELL * 0.08
        d.ellipse([x + pad, y + pad, x + CELL - pad, y + CELL - pad], fill=col)
        svg.append(f'<circle cx="{c + QUIET + 0.5}" cy="{r + QUIET + 0.5}" r="0.42" fill="{hexc(col)}"/>')

# finder eyes: rounded outer ring + rounded inner square; top-left/bottom-left blue, top-right red
for (r, c), col in (((0, 0), BLUE), ((0, n - 7), RED), ((n - 7, 0), BLUE)):
    x = (c + QUIET) * CELL
    y = (r + QUIET) * CELL
    d.rounded_rectangle([x, y, x + 7 * CELL, y + 7 * CELL], radius=2.2 * CELL, fill=col)
    d.rounded_rectangle([x + CELL, y + CELL, x + 6 * CELL, y + 6 * CELL], radius=1.5 * CELL, fill="white")
    d.rounded_rectangle([x + 2 * CELL, y + 2 * CELL, x + 5 * CELL, y + 5 * CELL], radius=0.9 * CELL, fill=col)
    X, Y = c + QUIET, r + QUIET
    svg.append(f'<rect x="{X}" y="{Y}" width="7" height="7" rx="2.2" fill="{hexc(col)}"/>')
    svg.append(f'<rect x="{X + 1}" y="{Y + 1}" width="5" height="5" rx="1.5" fill="#fff"/>')
    svg.append(f'<rect x="{X + 2}" y="{Y + 2}" width="3" height="3" rx="0.9" fill="{hexc(col)}"/>')

# centre badge with the logo
bx0 = (b0 + QUIET) * CELL
bx1 = (b1 + QUIET) * CELL
inset = CELL * 0.35
d.rounded_rectangle(
    [bx0 + inset, bx0 + inset, bx1 - inset, bx1 - inset],
    radius=CELL * 1.6,
    fill="white",
    outline=(0xD1, 0xD3, 0xD4),
    width=max(2, CELL // 10),
)
logo = Image.open(LOGO).convert("RGBA")
box = int((bx1 - bx0) * 0.86)
logo.thumbnail((box, box), Image.LANCZOS)
lx = (size - logo.width) // 2
ly = (size - logo.height) // 2
img.paste(logo, (lx, ly), logo)

img.save(OUT / "wildmatch_qr.png", dpi=(600, 600))

# SVG: badge + embedded logo (base64 PNG)
buf = io.BytesIO()
logo.save(buf, "PNG")
B0, B1 = b0 + QUIET, b1 + QUIET
svg.append(
    f'<rect x="{B0 + 0.35}" y="{B0 + 0.35}" width="{B1 - B0 - 0.7}" height="{B1 - B0 - 0.7}" rx="1.6" '
    f'fill="#fff" stroke="#d1d3d4" stroke-width="0.1"/>'
)
lw = logo.width / CELL
lh = logo.height / CELL
tot = n + 2 * QUIET
svg.append(
    f'<image x="{(tot - lw) / 2}" y="{(tot - lh) / 2}" width="{lw}" height="{lh}" '
    f'href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'
)
svg.append("</svg>")
(OUT / "wildmatch_qr.svg").write_text("\n".join(svg))

# poster variant: QR with the address underneath
font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
    if Path(p).exists():
        font = ImageFont.truetype(p, int(CELL * 2.4))
        break
font = font or ImageFont.load_default()
label = "wildmatch.gmum.net"
pad_b = int(CELL * 4.5)
card = Image.new("RGB", (size, size + pad_b), "white")
card.paste(img, (0, 0))
cd = ImageDraw.Draw(card)
tw = cd.textlength(label, font=font)
cd.text(((size - tw) / 2, size - CELL * 1.2), label, font=font, fill=GREY)
card.save(OUT / "wildmatch_qr_label.png", dpi=(600, 600))
print("modules", n, "version", qr.version, "badge", badge_modules, "png", img.size)
