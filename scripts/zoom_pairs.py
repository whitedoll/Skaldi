"""원본/결과 비교 확대 이미지 생성.  사용: python scripts/zoom_pairs.py <출력폴더>"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/zoom")
out_dir.mkdir(parents=True, exist_ok=True)

SPOTS = [
    ("08", (900, 1300, 1432, 2023), 0.9),
    ("247", (0, 100, 1120, 1100), 0.75),
    ("250", (1040, 770, 1300, 1050), 1.6),
    ("11", (700, 0, 1432, 560), 0.9),
]

for name, box, scale in SPOTS:
    o = Image.open(f"samples/{name}.webp").convert("RGB").crop(box)
    a = Image.open(f"output/pillow/{name}.webp").convert("RGB").crop(box)
    sh = Image.new("RGB", (o.width * 2 + 10, o.height), (80, 80, 80))
    sh.paste(o, (0, 0))
    sh.paste(a, (o.width + 10, 0))
    if scale != 1.0:
        sh = sh.resize((int(sh.width * scale), int(sh.height * scale)))
    sh.save(out_dir / f"q_{name}.png")
print("ok")
