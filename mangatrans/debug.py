"""검수용 디버그 그림: 글자 상자·말풍선 상자·배치 상자(또는 다각형)를 결과 위에 겹쳐 그린다.

무엇이 어디에 그려질지 눈으로 확인하려고 만든다. 배치가 이상할 때 탐지 상자가 잘못인지, 말풍선 본체
측정이 잘못인지, 배치 상자를 나눈 방식이 잘못인지 한 장으로 구분할 수 있다.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .page import Page, Region

BOX_COLOR = (230, 30, 30)          # 탐지한 글자 상자
BUBBLE_COLOR = (20, 110, 255)      # 탐지한 말풍선 상자
BODY_COLOR = (0, 170, 60)          # 픽셀로 잰 말풍선 본체
TARGET_COLOR = (255, 140, 0)       # 실제로 글자를 넣은 자리 (네모)
POLY_COLOR = (200, 0, 200)         # 실제로 글자를 넣은 자리 (다각형)
SKIP_COLOR = (130, 130, 130)       # 그리지 않은 영역 (효과음·원본 유지)


def _label(r: Region) -> str:
    """영역 요약: 번호·종류·글꼴·크기·자리 방식."""
    bits = [f"#{r.id}", r.category]
    if not r.render:
        bits.append("원본유지")
    if r.needs_review:
        bits.append("검토")
    bits.append(r.style + ("/굵게" if r.weight == "bold" else ""))
    if r.font_size:
        bits.append(f"{r.font_size}px")
    if r.writing != "auto":
        bits.append(r.writing)
    if r.poly:
        bits.append("다각형")
    if r.widened:
        bits.append("넓힘")
    if r.group is not None:
        bits.append(f"묶음{r.group}")
    return " ".join(bits)


def debug_image(rendered: Image.Image, page: Page, font_path: str | None = None) -> Image.Image:
    """렌더 결과 위에 상자들을 겹쳐 그린 이미지."""
    img = rendered.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(14, int(page.height * 0.011))
    try:
        font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    w = max(2, int(page.height * 0.0015))
    for r in page.ordered():
        if r.bubble_box:
            draw.rectangle(r.bubble_box, outline=BUBBLE_COLOR, width=w)
        if r.body_box:
            draw.rectangle(r.body_box, outline=BODY_COLOR, width=w)
        draw.rectangle(r.box, outline=BOX_COLOR if r.render else SKIP_COLOR, width=w)
        if r.render and r.poly:
            draw.line([tuple(p) for p in r.poly] + [tuple(r.poly[0])], fill=POLY_COLOR, width=w + 1)
        elif r.render and r.target_box:
            draw.rectangle(r.target_box, outline=TARGET_COLOR, width=w + 1)
        text = _label(r)
        tw = draw.textlength(text, font=font)
        x, y = r.box[0], max(0, r.box[1] - size - 4)
        draw.rectangle([x, y, x + tw + 6, y + size + 4], fill=(0, 0, 0, 190))
        draw.text((x + 3, y + 2), text, fill=(255, 255, 255), font=font)
    legend = [("글자 상자", BOX_COLOR), ("말풍선", BUBBLE_COLOR), ("본체", BODY_COLOR),
              ("글자 자리", TARGET_COLOR), ("다각형 자리", POLY_COLOR), ("안 그림", SKIP_COLOR)]
    y = 6
    for name, color in legend:
        draw.rectangle([6, y, 6 + size, y + size], fill=color)
        draw.rectangle([6 + size + 4, y, 6 + size + 10 + draw.textlength(name, font=font), y + size],
                       fill=(0, 0, 0, 190))
        draw.text((6 + size + 7, y), name, fill=(255, 255, 255), font=font)
        y += size + 6
    return img
