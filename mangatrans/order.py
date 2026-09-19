"""읽기 순서 판정과 말풍선 밖 글자 분류 (비전 모델 호출 1회)."""
from __future__ import annotations

import re

from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .llm import OllamaClient
from .page import Page, Region

_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "category": {"type": "string", "enum": ["dialogue", "narration", "label", "sfx"]},
                },
                "required": ["id", "category"],
            },
        },
    },
    "required": ["order", "categories"],
}


def heuristic_order(regions: list[Region], page_w: int) -> list[int]:
    """일본 만화 관례: 위에서 아래로 띠(row)를 나누고, 띠 안에서는 오른쪽에서 왼쪽으로.
    세로로 겹치는 박스끼리 같은 띠로 묶는다."""
    items = sorted(regions, key=lambda r: r.box[1])
    bands: list[dict] = []
    for r in items:
        cy = (r.box[1] + r.box[3]) / 2
        for b in bands:
            if b["y1"] <= cy <= b["y2"]:
                b["regs"].append(r)
                b["y1"], b["y2"] = min(b["y1"], r.box[1]), max(b["y2"], r.box[3])
                break
        else:
            bands.append({"y1": r.box[1], "y2": r.box[3], "regs": [r]})
    bands.sort(key=lambda b: b["y1"])
    out: list[int] = []
    for b in bands:
        regs = sorted(b["regs"], key=lambda r: (-(r.box[0] + r.box[2]) / 2, r.box[1]))
        out.extend(r.id for r in regs)
    return out


def annotated_page(image: Image.Image, regions: list[Region], long_side: int, font_path) -> Image.Image:
    """박스에 번호를 그린 축소 페이지."""
    scale = min(1.0, long_side / max(image.size))
    img = image.convert("RGB").resize((int(image.width * scale), int(image.height * scale)))
    draw = ImageDraw.Draw(img)
    fsize = max(14, int(img.height * 0.016))
    try:
        font = ImageFont.truetype(str(font_path), fsize)
    except OSError:
        font = ImageFont.load_default()
    for r in regions:
        x1, y1, x2, y2 = [int(v * scale) for v in r.box]
        color = (255, 0, 0) if r.kind == "bubble_text" else (0, 90, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(r.id)
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, y1 - fsize - 4, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - fsize - 3), label, fill=(255, 255, 255), font=font)
    return img


def order_and_classify(cfg: Config, client: OllamaClient, image: Image.Image, page: Page) -> None:
    """page.regions 에 order 와 category 를 채운다. 실패하면 휴리스틱으로 대체."""
    regions = page.regions
    if not regions:
        return
    fallback = heuristic_order(regions, page.width)
    prompt = (
        "이 이미지는 일본 만화 한 페이지이고, 글자 영역마다 번호 박스가 그려져 있다. "
        "빨간 박스는 말풍선 안 글자, 파란 박스는 말풍선 밖 글자다.\n"
        "1) order: 일본 만화 읽기 순서(오른쪽 위에서 시작해 왼쪽 아래로, 컷 순서를 따름)대로 "
        f"모든 번호를 나열하라. 번호 목록: {[r.id for r in regions]}\n"
        "2) categories: 모든 번호를 분류하라.\n"
        "  dialogue = 인물이 입으로 하는 말. 말풍선이 없어도 인사·감사·설명·권유하는 말투면 dialogue 다.\n"
        "  narration = 인물의 속마음이나 상황 설명 서술.\n"
        "  label = 누가 말하는 게 아닌 글자만. 인물 이름표, 작품·챕터 제목, 간판·표지판·화면 글자.\n"
        "  sfx = 효과음·의성어·의태어. 말풍선 안이라도 손글씨 소리 표현이면 sfx 다.\n"
        "  주의: 문장이 길거나 말끝이 대화체(です/ます/ね/よ/～/♥)면 label 이 아니다. 애매하면 dialogue.\n"
        "JSON으로만 답하라."
    )
    img = annotated_page(image, regions, cfg.llm.page_long_side, cfg.abs(cfg.paths.font))
    try:
        data = client.chat_json(cfg.llm.vision_model, prompt, images=[img], schema=_SCHEMA)
        order = [int(i) for i in data.get("order", [])]
        cats = {int(d["id"]): d["category"] for d in data.get("categories", [])}
    except Exception as e:  # noqa: BLE001
        page.warnings.append(f"순서/분류 호출 실패, 휴리스틱 사용: {e}")
        order, cats = [], {}

    valid = {r.id for r in regions}
    seq = [i for i in dict.fromkeys(order) if i in valid]
    if len(seq) < len(valid) * 0.8:
        if order:
            page.warnings.append("모델이 준 순서가 불완전해 휴리스틱으로 대체")
        seq = fallback
    else:
        seq += [i for i in fallback if i not in seq]  # 빠진 번호는 휴리스틱 위치로

    pos = {rid: i for i, rid in enumerate(seq)}
    # 분류 결과에 없는 번호를 unknown 으로 두면 말풍선 밖 대사가 통째로 사라진다.
    # (모델이 categories 배열을 끝까지 안 채우는 일이 잦다) 순서 쪽 휴리스틱처럼 안전망을 둔다.
    missing = [r.id for r in regions if r.id not in cats]
    if missing:
        page.warnings.append(f"분류에 없는 번호 {len(missing)}개 {missing[:10]} 는 대사로 둠")
    for r in regions:
        r.order = pos.get(r.id)
        apply_category(r, cats.get(r.id, "dialogue"))
    sibling_sfx(regions)


def sibling_sfx(regions: list[Region]) -> None:
    """같은 말풍선 안에 효과음(규칙으로 확정된 것)이 있으면, 함께 들어 있는 짧은 글자도 효과음으로 본다.
    (손글씨 효과음이 여러 덩어리로 잡혀 한 덩어리만 OCR 오류로 대사처럼 읽히는 경우)"""
    by_bubble: dict[tuple, list[Region]] = {}
    for r in regions:
        if r.kind == "bubble_text" and r.bubble_box:
            by_bubble.setdefault(tuple(r.bubble_box), []).append(r)
    for group in by_bubble.values():
        if len(group) < 2 or not any(g.category == "sfx" for g in group):
            continue
        for g in group:
            core = _SFX_STRIP.sub("", g.text_ja)
            if g.category != "sfx" and len(core) <= 5 and not is_vocal(g.text_ja):
                g.category, g.render, g.erase = "sfx", False, "none"
                g.notes = "같은 말풍선의 효과음과 묶음"


_SFX_STRIP = re.compile(r"[\s。、．・…‥〜～ー！？!?♡♥♪☆★「」『』（）()\-]+")
_KATAKANA_ONLY = re.compile(r"^[ァ-ヴヵヶッ]+$")


def looks_like_sfx(text: str, kind: str) -> bool:
    """규칙 기반 효과음 판정. 같은 글자 반복(おおおお, ボボボボ)은 어디서나,
    짧은 가타카나만 있는 글자(ゴキュッ, ドン)는 말풍선 밖에서만 효과음으로 본다."""
    core = _SFX_STRIP.sub("", text)
    if not core:
        return False
    if len(core) >= 4 and len(set(core)) <= 2:
        return True
    return kind == "free_text" and len(core) <= 4 and bool(_KATAKANA_ONLY.match(core))


_LAUGH = ("クス", "フフ", "ハハ", "ハァ", "ヘヘ", "ヒヒ", "ホホ", "ケケ", "ニヤ", "アハ", "ウフ", "エヘ", "ンフ",
          "くす", "ふふ", "はは", "へへ", "ひひ", "ほほ", "けけ", "あは", "うふ", "えへ", "んふ")


def is_vocal(text: str) -> bool:
    """웃음·콧소리처럼 사람이 내는 소리 표현은 대사로 번역한다 (クスクス → 큭큭)."""
    core = _SFX_STRIP.sub("", text)
    return any(core.startswith(l) for l in _LAUGH)


def too_long_for_sfx(text: str, limit: int = 8) -> bool:
    """효과음이라기엔 너무 긴 글자. 이 길이를 넘으면 문장으로 본다.
    (실측: 두 작품의 진짜 손글씨 효과음은 전부 7자 이하, 버려진 대사는 전부 9자 이상)"""
    return len(_SFX_STRIP.sub("", text)) >= limit


def sfx_allowed_in_bubble(text: str) -> bool:
    """말풍선 안 글자를 효과음으로 인정하는 조건: 가타카나·기호만으로 된 소리 표현이거나 같은 글자 반복.
    そう, へぇ〜, ああっ 같은 히라가나 감탄사·대답과 クスクス 같은 웃음은 대사로 남긴다."""
    core = _SFX_STRIP.sub("", text)
    if is_vocal(text):
        return False
    return bool(core) and (bool(_KATAKANA_ONLY.match(core)) or (len(core) >= 4 and len(set(core)) <= 2))


def apply_category(r: Region, cat: str | None) -> None:
    """분류 결과에 따라 category / render / erase 를 정한다."""
    rule = looks_like_sfx(r.text_ja, r.kind)     # 규칙으로 확정된 효과음인가
    if rule:
        cat = "sfx"
    if r.kind == "bubble_text":
        r.category = "sfx" if (cat == "sfx" and sfx_allowed_in_bubble(r.text_ja)) else "dialogue"
        # 규칙으로 확정된 것만 바로 원본 유지. 비전 모델만 sfx 라고 한 것은 잠정(notes)으로 두고
        # 번역 모델도 sfx 라고 해야 확정한다 (クスクス 같은 웃음은 대사로 번역되게).
        if r.category == "sfx" and not rule:
            r.category, r.notes = "dialogue", "vision:sfx"
        if r.category == "sfx":
            r.render, r.erase = False, "none"      # 말풍선 안 손글씨 효과음은 원본 유지
        else:
            r.render, r.erase = True, "white"
        return
    if cat is None:
        cat = "dialogue"         # 분류가 없으면 대사로 본다. 버리면 원문 일본어가 그대로 남는다.
    # 말풍선 밖 글자도 모델이 단독으로 sfx 라고 한 것은 규칙으로 걸러 낸다.
    # (「」로 묶인 긴 대사를 효과음으로 답해 그대로 버려지는 일이 있었다)
    # 규칙이 확정한 효과음(rule)은 건드리지 않고, 짧은 글자도 건드리지 않는다 —
    # ごそ．．． づポ… 같은 손글씨 효과음까지 지우고 다시 그리면 그림이 상한다.
    if (cat == "sfx" and not rule and not sfx_allowed_in_bubble(r.text_ja)
            and too_long_for_sfx(r.text_ja)):
        cat, r.notes = "dialogue", "vision:sfx"
    r.category = cat if cat in ("dialogue", "narration", "label", "sfx") else "unknown"  # type: ignore[assignment]
    if r.category in ("dialogue", "narration"):
        r.render, r.erase = True, "lama"
    else:
        r.render, r.erase = False, "none"


_STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "styles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "style": {"type": "string", "enum": ["gothic", "mincho", "hand"]},
                },
                "required": ["id", "style"],
            },
        }
    },
    "required": ["styles"],
}


def style_montage(image: Image.Image, regions: list[Region], short_side: int = 130, cols: int = 4,
                  font_path=None, max_long: int = 3) -> Image.Image:
    """글자 크롭들을 번호와 함께 격자로 붙인 이미지 (글꼴 계열 판정용).

    세로 글자는 좁고 길어서 높이 기준으로 줄이면 폭이 수십 픽셀로 뭉개져 글꼴을 알아볼 수 없다.
    그래서 '짧은 변'을 기준으로 확대·축소하고, 긴 변은 앞부분만 잘라 쓴다."""
    tiles = []
    for r in regions:
        x1, y1, x2, y2 = r.box
        crop = image.crop((max(0, x1 - 4), max(0, y1 - 4), min(image.width, x2 + 4), min(image.height, y2 + 4)))
        sc = short_side / max(1, min(crop.width, crop.height))
        crop = crop.resize((max(8, int(crop.width * sc)), max(8, int(crop.height * sc))))
        limit = short_side * max_long
        crop = crop.crop((0, 0, min(crop.width, limit), min(crop.height, limit)))
        tiles.append((r.id, crop))
    try:
        font = ImageFont.truetype(str(font_path), 22) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    tw = max(t.width for _, t in tiles) + 16
    th = max(t.height for _, t in tiles) + 34
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (200, 200, 200))
    d = ImageDraw.Draw(sheet)
    for i, (rid, t) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(t, (x + 8, y + 30))
        d.rectangle([x + 2, y + 4, x + 62, y + 28], fill=(255, 0, 0))
        d.text((x + 6, y + 5), f"#{rid}", fill=(255, 255, 255), font=font)
    if max(sheet.size) > 1800:                     # 너무 크면 모델 입력에 맞게 줄인다
        sc = 1800 / max(sheet.size)
        sheet = sheet.resize((int(sheet.width * sc), int(sheet.height * sc)))
    return sheet


def classify_styles(cfg: Config, client: OllamaClient, image: Image.Image, page: Page) -> None:
    """그릴 영역의 글꼴 계열(gothic/mincho/hand)을 크롭 몽타주 한 장으로 판정한다."""
    regions = [r for r in page.regions if r.render]
    if not regions:
        return
    sheet = style_montage(image, regions, font_path=cfg.abs(cfg.paths.font))
    prompt = (
        "이 이미지는 일본 만화에서 잘라낸 글자 조각들을 번호(#n)와 함께 붙인 것이다. 각 조각의 글꼴을 판정하라.\n"
        "만화 대사는 거의 전부 컴퓨터로 조판한 인쇄체다. 기본값은 gothic 이고, 확실한 근거가 있을 때만 다르게 답하라.\n"
        "  gothic = 인쇄체. 같은 글자가 어디서나 똑같은 모양이고 획 굵기가 고르며 글자들이 줄에 반듯하게 놓여 있다.\n"
        "  mincho = 인쇄된 명조체. 가로획이 눈에 띄게 얇고 세로획이 굵으며 획 끝에 삼각 장식(세리프)이 있다.\n"
        "  hand = 붓이나 펜으로 직접 쓴 글씨. 획이 떨리거나 번지고, 같은 글자도 모양이 제각각이며, "
        "글자 크기와 기울기가 들쭉날쭉하다. 주로 효과음에만 쓰인다.\n"
        "가늘거나 작다는 이유로 hand 라고 하지 말라. 조판된 글자는 전부 gothic 이다.\n"
        f"번호 목록: {[r.id for r in regions]}\nJSON으로만 답하라."
    )
    try:
        data = client.chat_json(cfg.llm.vision_model, prompt, images=[sheet], schema=_STYLE_SCHEMA)
        styles = {int(d["id"]): d["style"] for d in data.get("styles", [])}
    except Exception as e:  # noqa: BLE001
        page.warnings.append(f"글꼴 분류 호출 실패, gothic 사용: {e}")
        styles = {}
    for r in regions:
        st = styles.get(r.id, "gothic")
        r.style = st if st in ("gothic", "mincho", "hand") else "gothic"
    # 만화 대사는 대부분 인쇄체다. 모델이 거의 전부를 손글씨라고 하면 판정을 믿지 않는다.
    # (책에 따라 인쇄체를 전부 hand 로 답하는 실패가 있었고, 그때 가독성이 크게 떨어진다)
    hands = sum(1 for r in regions if r.style == "hand")
    if len(regions) >= 4 and hands / len(regions) > 0.7:
        page.warnings.append(f"글꼴 판정이 {hands}/{len(regions)} 를 손글씨로 봐 신뢰하지 않고 고딕으로 둠")
        for r in regions:
            r.style = "gothic"
