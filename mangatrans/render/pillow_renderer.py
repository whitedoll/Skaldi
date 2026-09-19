"""Pillow 렌더러: 가로쓰기, 자동 줄바꿈, 단계 축소, 그림 위 글자는 흰 외곽선."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ..config import Config
from ..erase import region_target_box
from ..page import Page, Region
from ..textfit import fit_text, fit_vertical, font, split_runs, split_variation, text_width


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
        draw = ImageDraw.Draw(img)
        rc = self.cfg.render
        base = max(8, int(page.height * rc.font_ratio))
        minimum = max(6, int(page.height * rc.min_font_ratio))
        for r in page.regions:
            if not r.render or not r.text_ko.strip():
                continue
            # 폰트가 '…'를 가운데 점(⋯)으로 그려 좁은 말풍선에서 콜론처럼 보이므로 '...'로 통일
            r.text_ko = r.text_ko.replace("…", "...").replace("‥", "..")
            self._draw_region(draw, r, page, base, minimum)
        return img

    def _draw_line(self, draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font_path: str,
                   size: int, fill, stroke_fill, outline_w: int, bold_w: int) -> None:
        """한 줄을 그린다. 폰트에 없는 글자(♥ 등)는 대체 폰트로 이어 그리고,
        진짜 굵은 자형이 없는 폰트는 같은 색 테두리를 덧대 두껍게 만든다."""
        for chunk, path in split_runs(text, font_path, self.fallback_path):
            f = font(path, size)
            if outline_w:                       # 바깥 외곽선 → 그다음 획 두껍게
                draw.text((x, y), chunk, font=f, fill=fill,
                          stroke_width=outline_w + bold_w, stroke_fill=stroke_fill)
            if bold_w:
                draw.text((x, y), chunk, font=f, fill=fill, stroke_width=bold_w, stroke_fill=fill)
            if not outline_w and not bold_w:
                draw.text((x, y), chunk, font=f, fill=fill)
            x += draw.textlength(chunk, font=f)

    def _draw_region(self, draw: ImageDraw.ImageDraw, r: Region, page: Page, base: int, minimum: int) -> None:
        rc = self.cfg.render
        x1, y1, x2, y2 = region_target_box(r, self.cfg, page.width, page.height)
        box_w, box_h = max(10, x2 - x1), max(10, y2 - y1)
        fill, stroke_fill = self.colors(r)
        # 그림 위 글자는 두껍게, 말풍선 안 글자는 얇게 (원문도 대개 얇은 흰 테두리를 두른다)
        on_art = r.kind == "free_text" or r.text_color == "outline"
        ratio = rc.stroke_ratio if on_art else rc.bubble_stroke_ratio
        font_path, fscale = self.pick_font(r)
        fake_bold = r.weight == "bold" and r.style in set(rc.fake_bold_styles)
        outline_w = lambda size: (max(1, round(size * ratio)) if ratio > 0 else 0)   # noqa: E731
        bold_w = lambda size: max(1, round(size * rc.fake_bold_ratio)) if fake_bold else 0  # noqa: E731
        base, minimum = max(6, int(base * fscale)), max(6, int(minimum * fscale))

        # 좁고 긴 상자는 (옵션) 세로쓰기
        if rc.vertical_for_narrow and box_h / box_w >= rc.narrow_ratio:
            size, cols, overflow = fit_vertical(r.text_ko, box_w, box_h, base, minimum, rc.line_spacing)
            r.font_size, r.overflow, r.vertical = size, overflow, True
            col_w = size * 1.08
            step = int(size * rc.line_spacing)
            right = (x1 + x2) / 2 + col_w * len(cols) / 2
            cy = (y1 + y2) / 2
            for ci, col in enumerate(cols):
                cx = right - col_w * (ci + 0.5)
                top = cy - step * len(col) / 2
                for k, ch in enumerate(col):
                    tw = text_width(ch, font_path, size, self.fallback_path)
                    self._draw_line(draw, cx - tw / 2, top + k * step, ch, font_path, size,
                                    fill, stroke_fill, outline_w(size), bold_w(size))
            return

        size, lines, overflow = fit_text(r.text_ko, font_path, box_w, box_h, base, minimum,
                                         rc.line_spacing, fallback=self.fallback_path)
        r.font_size, r.overflow, r.vertical = size, overflow, False
        line_h = int(size * rc.line_spacing)
        top = (y1 + y2) / 2 - line_h * len(lines) / 2
        cx = (x1 + x2) / 2
        for i, line in enumerate(lines):
            tw = text_width(line, font_path, size, self.fallback_path)
            self._draw_line(draw, cx - tw / 2, top + i * line_h, line, font_path, size,
                            fill, stroke_fill, outline_w(size), bold_w(size))
