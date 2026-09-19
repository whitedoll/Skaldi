"""페이지의 말풍선(bubble_box)마다 원본/결과를 나란히 확대한 점검용 시트.
사용: python scripts/bubble_sheet.py 08 [출력파일]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

name = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"output/zoom/bubbles_{name}.png")
out.parent.mkdir(parents=True, exist_ok=True)
page = json.load(open(f"output/json/{name}.json", encoding="utf-8"))
src = Image.open(page["source"]).convert("RGB")
res = Image.open(f"output/pillow/{name}.webp").convert("RGB")

tiles = []
for r in page["regions"]:
    if r["kind"] != "bubble_text" or not r["bubble_box"]:
        continue
    x1, y1, x2, y2 = r["bubble_box"]
    pad = 12
    box = (max(0, x1 - pad), max(0, y1 - pad), min(src.width, x2 + pad), min(src.height, y2 + pad))
    a, b = src.crop(box), res.crop(box)
    scale = min(3.0, 360 / max(a.height, 1))
    a = a.resize((int(a.width * scale), int(a.height * scale)))
    b = b.resize((int(b.width * scale), int(b.height * scale)))
    t = Image.new("RGB", (a.width + b.width + 8, a.height + 18), (60, 60, 60))
    t.paste(a, (0, 18))
    t.paste(b, (a.width + 8, 18))
    ImageDraw.Draw(t).text((2, 2), f"id={r['id']} widened={r['widened']}", fill=(255, 255, 0))
    tiles.append(t)

cols = 3
rows = (len(tiles) + cols - 1) // cols
cw = max(t.width for t in tiles)
ch = max(t.height for t in tiles)
sheet = Image.new("RGB", (cols * cw + (cols - 1) * 6, rows * ch + (rows - 1) * 6), (30, 30, 30))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * (cw + 6), (i // cols) * (ch + 6)))
sheet.save(out)
print(out, sheet.size, len(tiles))
