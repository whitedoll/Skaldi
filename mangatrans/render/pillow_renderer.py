"""Pillow 렌더러: 가로쓰기, 자동 줄바꿈, 크기 이진탐색, 그림 위 글자는 흰 외곽선.
기운 원문은 같은 각도로 돌려 그리고, 작은 글자는 크게 그려 줄인다."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ..config import Config
from ..erase import region_target_box
from ..normalize import normalize_ko
from ..page import Page, Region
from ..textfit import fit_text, fit_vertical, font, has_glyph, split_runs, split_variation, text_width

# 세로쓰기에서 가로 모양 그대로 쓰면 어색한 문장부호: 세로 전용 자형(CJK 호환 형태)으로 바꾼다
VERTICAL_FORMS = {
    "…": "︙", "‥": "︰", "、": "︑", "。": "︒", "，": "︐", "：": "︓", "；": "︔",
    "「": "﹁", "」": "﹂", "『": "﹃", "』": "﹄", "（": "︵", "）": "︶", "(": "︵", ")": "︶",
    "【": "︻", "】": "︼", "〔": "︹", "〕": "︺", "〈": "︿", "〉": "﹀", "《": "︽", "》": "︾",
    "｛": "︷", "｝": "︸", "—": "︱", "–": "︲",
}
# 세로 자형이 따로 없는 길게 늘이는 기호: 글자를 90° 돌려 세운다
VERTICAL_ROTATE = set("ー〜～~-－―")


class PillowRenderer:
    name = "pillow"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.font_path = str(cfg.abs(cfg.paths.font))
        # 손글씨 폰트에 없는 기호(♥ 〜 …)를 대신 그릴 폰트
        fb = cfg.render.fonts.get("fallback") or str(cfg.paths.font)
        fb_base, fb_var = split_variation(str(fb))
        self.fallback_path = str(cfg.abs(Path(fb_base))) + (f"#{fb_var}" if fb_var else "")

    def pick_font(self, r: Region) -> tuple[str, float]:
        """영역의 글꼴 계열·굵기에 맞는 폰트 경로와 크기 보정 배율."""
        rc = self.cfg.render
        # 계열 판정(match_style)은 꺼도 굵기는 픽셀에서 잰 값이라 믿을 만하므로 항상 반영한다.
        key = r.style if (rc.match_style and r.style in ("gothic", "mincho", "hand")) else "gothic"
        if r.weight == "bold":
            key_b = key + "_bold"
            path = rc.fonts.get(key_b) or rc.fonts.get(key)
            scale = rc.font_scale.get(key_b, rc.font_scale.get(key, 1.0))
        else:
            path = rc.fonts.get(key)
            scale = rc.font_scale.get(key, 1.0)
        if not path:
            return self.font_path, 1.0
        base, var = split_variation(path)
        ap = self.cfg.abs(Path(base))
        full = str(ap) + (f"#{var}" if var else "")
        return (full if ap.exists() else self.font_path), float(scale)

    def colors(self, r: Region) -> tuple[tuple, tuple]:
        """(글자색, 외곽선색). 외곽선을 실제로 그릴지·얼마나 두껍게 그릴지는 호출부가 정한다.

        측정한 획 색은 '유채색일 때만' 쓴다. 흑백 페이지에서는 안티앨리어싱과 흰 외곽선 때문에
        획 색 측정이 회색으로 흔들려서, 배경 밝기로 정하는 기존 규칙(검정/흰색/검정+흰 외곽선)이
        더 안정적이다."""
        rc = self.cfg.render
        base_fill = (255, 255, 255) if r.text_color == "white" else (0, 0, 0)
        base_stroke = (0, 0, 0) if r.text_color == "white" else (255, 255, 255)
        if rc.match_color and r.text_rgb and max(r.text_rgb) - min(r.text_rgb) > 40:
            fill = tuple(r.text_rgb)
            if r.outline_rgb and max(r.outline_rgb) - min(r.outline_rgb) > 40:
                return fill, tuple(r.outline_rgb)
            lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
            return fill, ((0, 0, 0) if lum > 128 else (255, 255, 255))
        return base_fill, base_stroke

    def render(self, clean: Image.Image, original: Image.Image, page: Page) -> Image.Image:
        img = clean.convert("RGB").copy()
        rc = self.cfg.render
        base = max(8, int(page.height * rc.font_ratio))
        minimum = max(6, int(page.height * rc.min_font_ratio))
        for r in page.regions:
            if not r.render or not r.text_ko.strip():
                continue
            # 정규화 이전에 만든 JSON 으로 다시 그릴 때도 표기를 정리한다 (여러 번 해도 결과가 같다).
            # 폰트가 '…'를 가운데 점(⋯)으로 그려 좁은 말풍선에서 콜론처럼 보이므로 '...'로 통일
            r.text_ko = normalize_ko(r.text_ko).replace("…", "...").replace("‥", "..")
            self._draw_region(img, r, page, base, minimum)
        return img

    def _glyph_masks(self, stroke_m: Image.Image | None, fill_m: Image.Image, x: float, y: float,
                     text: str, font_path: str, size: int, outline_w: int, bold_w: int) -> float:
        """한 줄(또는 한 글자)을 외곽선 마스크와 글자 마스크에 그리고 다음 x 를 돌려준다.

        색을 바로 칠하지 않고 모양만 그려 두면 통째로 돌리거나(기운 글자) 크게 그려 줄일 수(작은 글자)
        있다. 합성은 외곽선 색 → 글자 색 순서라 예전의 draw.text(stroke_width=…) 와 결과가 같다.
        폰트에 없는 글자(♥ 등)는 대체 폰트로 이어 그리고, 진짜 굵은 자형이 없는 폰트는 같은 색 테두리로
        두껍게 만든다."""
        ds = ImageDraw.Draw(stroke_m) if stroke_m is not None else None
        df = ImageDraw.Draw(fill_m)
        for chunk, path in split_runs(text, font_path, self.fallback_path):
            f = font(path, size)
            if ds is not None and outline_w:
                ds.text((x, y), chunk, font=f, fill=255, stroke_width=outline_w + bold_w, stroke_fill=255)
            if bold_w:
                df.text((x, y), chunk, font=f, fill=255, stroke_width=bold_w, stroke_fill=255)
            else:
                df.text((x, y), chunk, font=f, fill=255)
            x += df.textlength(chunk, font=f)
        return x

    def _draw_region(self, img: Image.Image, r: Region, page: Page, base: int, minimum: int) -> None:
        rc = self.cfg.render
        angle = r.angle if (r.angle and r.rot_box) else 0.0
        if angle:                                   # 기운 글자: 세운 좌표의 상자에 맞추고 나중에 돌린다
            cx, cy, bw, bh = r.rot_box              # type: ignore[misc]
            box_w, box_h = max(10, int(bw)), max(10, int(bh))
        else:
            x1, y1, x2, y2 = region_target_box(r, self.cfg, page.width, page.height)
            box_w, box_h = max(10, x2 - x1), max(10, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        fill, stroke_fill = self.colors(r)
        # 그림 위 글자는 두껍게, 말풍선 안 글자는 얇게 (원문도 대개 얇은 흰 테두리를 두른다)
        on_art = r.kind == "free_text" or r.text_color == "outline"
        ratio = rc.stroke_ratio if on_art else rc.bubble_stroke_ratio
        font_path, fscale = self.pick_font(r)
        fake_bold = r.weight == "bold" and r.style in set(rc.fake_bold_styles)
        base, minimum = max(6, int(base * fscale)), max(6, int(minimum * fscale))

        vertical = rc.vertical_for_narrow and box_h / box_w >= rc.narrow_ratio
        if vertical:                                # 좁고 긴 상자는 (옵션) 세로쓰기
            text = r.text_ko.replace("...", "…").replace("..", "‥")
            size, cols, overflow = fit_vertical(text, box_w, box_h, base, minimum, rc.line_spacing)
        else:
            size, lines, overflow = fit_text(r.text_ko, font_path, box_w, box_h, base, minimum,
                                             rc.line_spacing, fallback=self.fallback_path)
        r.font_size, r.overflow, r.vertical = size, overflow, vertical
        outline_w = max(1, round(size * ratio)) if ratio > 0 else 0
        bold_w = max(1, round(size * rc.fake_bold_ratio)) if fake_bold else 0

        # 작은 글자는 크게 그려 줄이고(계단 현상), 돌릴 글자도 크게 그려 돌린 뒤 줄인다(회전 보간 번짐)
        ss = 3 if size < rc.supersample_below else (2 if angle else 1)
        S = size * ss
        ow, bw_ = outline_w * ss, bold_w * ss
        edge = (outline_w + bold_w + 2) * ss        # 외곽선이 상자 밖으로 번지는 여유
        if vertical:
            step = int(size * rc.line_spacing) * ss
            col_w = size * 1.08 * ss
            text_w = col_w * len(cols)
            text_h = step * max(len(c) for c in cols)
        else:
            line_h = int(size * rc.line_spacing) * ss
            widths = [text_width(l, font_path, S, self.fallback_path) for l in lines]
            text_w = max(widths)
            text_h = line_h * len(lines) + S         # 마지막 줄의 아래 획 여유
        W = int(max(box_w * ss, text_w) + 2 * edge)
        H = int(max(box_h * ss, text_h) + 2 * edge)
        stroke_m = Image.new("L", (W, H), 0) if outline_w else None
        fill_m = Image.new("L", (W, H), 0)
        mx, my = W / 2, H / 2                       # 마스크 중심 = 페이지의 (cx, cy)
        if vertical:
            right = mx + text_w / 2
            for ci, col in enumerate(cols):
                ccx = right - col_w * (ci + 0.5)
                top = my - step * len(col) / 2
                for k, ch in enumerate(col):
                    self._vertical_glyph(stroke_m, fill_m, ccx, top + k * step, ch, font_path, S, ow, bw_)
        else:
            top = my - line_h * len(lines) / 2
            for i, (line, lw) in enumerate(zip(lines, widths)):
                self._glyph_masks(stroke_m, fill_m, mx - lw / 2, top + i * line_h, line, font_path, S, ow, bw_)

        masks = [m for m in (stroke_m, fill_m) if m is not None]
        if angle:
            masks = [m.rotate(angle, resample=Image.BICUBIC, expand=True) for m in masks]
        if ss > 1:
            w2, h2 = max(1, round(masks[0].width / ss)), max(1, round(masks[0].height / ss))
            masks = [m.resize((w2, h2), Image.LANCZOS) for m in masks]
        px, py = round(cx - masks[0].width / 2), round(cy - masks[0].height / 2)
        if stroke_m is not None:
            img.paste(stroke_fill, (px, py), masks[0])
        img.paste(fill, (px, py), masks[-1])

    def _vertical_glyph(self, stroke_m: Image.Image | None, fill_m: Image.Image, ccx: float, y: float,
                        ch: str, font_path: str, size: int, ow: int, bw_: int) -> None:
        """세로쓰기 한 글자. 문장부호는 세로 자형으로 바꾸고, 자형이 없으면 가로 글자를 90° 돌린다."""
        form = VERTICAL_FORMS.get(ch)
        if form and (has_glyph(font_path, form) or has_glyph(self.fallback_path, form)):
            ch, turn = form, False
        else:
            turn = ch in VERTICAL_ROTATE or form is not None
        if not turn:
            tw = text_width(ch, font_path, size, self.fallback_path)
            self._glyph_masks(stroke_m, fill_m, ccx - tw / 2, y, ch, font_path, size, ow, bw_)
            return
        # 한 글자짜리 작은 마스크에 그려 시계 방향으로 90° 돌린 뒤 제자리에 붙인다
        pad = ow + bw_ + 2
        side = int(size * 1.4) + 2 * pad
        gs = Image.new("L", (side, side), 0) if stroke_m is not None else None
        gf = Image.new("L", (side, side), 0)
        tw = text_width(ch, font_path, size, self.fallback_path)
        self._glyph_masks(gs, gf, side / 2 - tw / 2, pad, ch, font_path, size, ow, bw_)
        ox, oy = round(ccx - side / 2), round(y - pad)
        for src, dst in ((gs, stroke_m), (gf, fill_m)):
            if src is not None and dst is not None:
                dst.paste(255, (ox, oy), src.transpose(Image.ROTATE_270))
