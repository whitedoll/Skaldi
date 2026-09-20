"""여러 렌더 결과를 가로로 붙인 비교 이미지."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def compare_sheet(items: list[tuple[str, Image.Image]], font_path: str, max_h: int = 1600) -> Image.Image:
    if not items:
        raise ValueError("비교할 이미지가 없습니다")
    scale = min(1.0, max_h / max(im.height for _, im in items))
    label_h = 40
    scaled = [(name, im.convert("RGB").resize((int(im.width * scale), int(im.height * scale))))
              for name, im in items]
    total_w = sum(im.width for _, im in scaled) + 10 * (len(scaled) - 1)
    total_h = max(im.height for _, im in scaled) + label_h
    sheet = Image.new("RGB", (total_w, total_h), (60, 60, 60))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(font_path, 24)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for name, im in scaled:
        draw.text((x + 8, 8), name, fill=(255, 255, 255), font=font)
        sheet.paste(im, (x, label_h))
        x += im.width + 10
    return sheet
