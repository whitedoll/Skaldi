"""탐지기와 OCR 동작 확인용 스크립트. 샘플 4장에 박스를 그리고, 첫 장의 글자를 OCR 한다."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from skaldi.config import load_config  # noqa: E402
from skaldi.detect import Detector, attach_bubbles  # noqa: E402
from skaldi.ocr import crop_region, make_ocr  # noqa: E402
from skaldi.order import annotated_page  # noqa: E402
from skaldi.page import Region  # noqa: E402

cfg = load_config()
out = Path("output/smoke")
out.mkdir(parents=True, exist_ok=True)

t = time.time()
det = Detector(cfg.detector)
print(f"탐지기 로드 {time.time()-t:.1f}s")

pages = {}
for src in sorted(Path("samples").glob("*.webp")):
    img = Image.open(src).convert("RGB")
    t = time.time()
    dets = det.detect(img)
    regions = []
    for i, (d, bubble) in enumerate(attach_bubbles(dets)):
        regions.append(Region(id=i, kind="bubble_text" if d.label == "text_bubble" else "free_text",
                              box=d.box, bubble_box=bubble, score=d.score))
    n_b = sum(1 for d in dets if d.label == "bubble")
    n_tb = sum(1 for r in regions if r.kind == "bubble_text")
    n_tf = sum(1 for r in regions if r.kind == "free_text")
    print(f"{src.name}: {time.time()-t:.2f}s  bubble={n_b} text_bubble={n_tb} text_free={n_tf}")
    annotated_page(img, regions, 1400, cfg.abs(cfg.paths.font)).save(out / f"{src.stem}_det.png")
    pages[src.name] = (img, regions)

t = time.time()
ocr = make_ocr(cfg)
print(f"OCR 로드 {time.time()-t:.1f}s")
for name in ("08.webp", "247.webp"):
    img, regions = pages[name]
    print(f"--- {name} ---")
    t = time.time()
    for r in regions:
        crop = crop_region(img, r.box, cfg.ocr.crop_padding)
        text = ocr.read(crop)
        print(f"[{r.id:2d}] {r.kind:11s} {r.box}  {text}")
    print(f"OCR {len(regions)}개 {time.time()-t:.1f}s")
