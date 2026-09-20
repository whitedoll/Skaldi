"""AnyText v1.1 (diffusers 이식판) 렌더러.

말풍선 크롭(512×512)에 한 줄당 위치 마스크 하나를 주고 편집 모드로 한국어를 써넣은 뒤,
글자 박스 영역만 원본 위치에 다시 붙인다. 코드는 models/anytext_code (diffusers research project) 를 쓴다.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..config import Config
from ..erase import region_target_box
from ..page import Page, Region
from ..textfit import fit_text


class AnyTextRenderer:
    name = "anytext"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.font_path = str(cfg.abs(cfg.paths.font))
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        code_dir = self.cfg.abs(self.cfg.paths.models_dir) / "anytext_code"
        if not (code_dir / "anytext.py").exists():
            raise RuntimeError(f"AnyText 코드가 없습니다: {code_dir} (scripts/download_models.py 실행)")
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from anytext_controlnet import AnyTextControlNetModel  # type: ignore
        from diffusers import DiffusionPipeline

        a = self.cfg.anytext
        cn = AnyTextControlNetModel.from_pretrained(a.controlnet_repo, dtype=torch.float16, variant="fp16")
        # 파이프라인 코드는 HF 저장소 것이 아니라 로컬에 받아 둔 사본(models/anytext_code)을 쓴다.
        pipe = DiffusionPipeline.from_pretrained(
            a.repo, custom_pipeline=str(code_dir / "anytext.py"), trust_remote_code=True,
            font_path=self.font_path, controlnet=cn, dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def render(self, clean: Image.Image, original: Image.Image, page: Page) -> Image.Image:
        pipe = self._load()
        if torch.cuda.is_available():
            pipe.to("cuda")
        try:
            return self._render_all(pipe, clean, page)
        finally:
            # 다른 렌더러(Qwen/ComfyUI)가 VRAM을 쓸 수 있게 내려 둔다
            if torch.cuda.is_available():
                pipe.to("cpu")
                torch.cuda.empty_cache()

    def _render_all(self, pipe, clean: Image.Image, page: Page) -> Image.Image:
        img = clean.convert("RGB").copy()
        rc = self.cfg.render
        base = max(8, int(page.height * rc.font_ratio))
        minimum = max(6, int(page.height * rc.min_font_ratio))
        draw = ImageDraw.Draw(img)
        for r in page.regions:
            if not r.render or not r.text_ko.strip():
                continue
            try:
                self._render_region(pipe, img, draw, r, page, base, minimum)
            except Exception as e:  # noqa: BLE001
                page.warnings.append(f"anytext 실패 (id={r.id}): {e}")
        return img

    def _render_region(self, pipe, img: Image.Image, draw, r: Region, page: Page, base: int, minimum: int) -> None:
        S = self.cfg.anytext.crop_size
        tx1, ty1, tx2, ty2 = region_target_box(r, self.cfg, page.width, page.height)
        tw, th = max(8, tx2 - tx1), max(8, ty2 - ty1)
        size, lines, overflow = fit_text(r.text_ko, self.font_path, tw, th, base, minimum,
                                         self.cfg.render.line_spacing)
        r.font_size, r.overflow = size, overflow
        lines = [l for l in lines if l.strip()][:8]
        if not lines:
            return

        # 목표 박스가 크롭의 약 70% 안에 들어오도록 정사각 크롭 영역을 정한다.
        span = max(tw, th) / 0.7
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        half = span / 2
        cx1, cy1 = int(cx - half), int(cy - half)
        cx2, cy2 = int(cx + half), int(cy + half)
        # 페이지 밖으로 나가면 흰색으로 채운 캔버스에 붙인다.
        canvas = Image.new("RGB", (cx2 - cx1, cy2 - cy1), (255, 255, 255))
        sx1, sy1 = max(0, cx1), max(0, cy1)
        sx2, sy2 = min(page.width, cx2), min(page.height, cy2)
        canvas.paste(img.crop((sx1, sy1, sx2, sy2)), (sx1 - cx1, sy1 - cy1))
        scale = S / canvas.width
        crop = canvas.resize((S, S), Image.LANCZOS)

        # 줄마다 위치 마스크(검은 사각형, 흰 배경). AnyText는 위→아래 순으로 정렬해서 텍스트와 짝짓는다.
        mask = Image.new("RGB", (S, S), (255, 255, 255))
        md = ImageDraw.Draw(mask)
        line_h = size * self.cfg.render.line_spacing
        total_h = line_h * len(lines)
        top = cy - total_h / 2
        gap = line_h * 0.12  # 줄 사각형이 서로 붙으면 AnyText가 한 덩어리로 인식하므로 간격을 둔다
        for i, line in enumerate(lines):
            lw = draw.textlength(line, font=_font(self.font_path, size))
            lx1 = (cx - lw / 2 - size * 0.15 - cx1) * scale
            lx2 = (cx + lw / 2 + size * 0.15 - cx1) * scale
            ly1 = (top + i * line_h + gap / 2 - cy1) * scale
            ly2 = (top + (i + 1) * line_h - gap / 2 - cy1) * scale
            md.rectangle([lx1, ly1, lx2, ly2], fill=(0, 0, 0))

        quoted = " ".join(f'"{l}"' for l in lines)
        prompt = f"black and white manga page, speech bubble with clean printed Korean text {quoted}, high contrast"
        out = pipe(
            prompt,
            negative_prompt="blurry, low quality, watermark, color, distorted glyphs",
            num_inference_steps=self.cfg.anytext.steps,
            mode="edit",
            draw_pos=mask,
            ori_image=np.array(crop),
            guidance_scale=9.0,
            generator=torch.Generator(device="cpu").manual_seed(0),
        ).images[0]
        out = out.resize(canvas.size, Image.LANCZOS)

        # 글자 박스 영역만 원본에 붙인다 (마스크 바깥은 원본 유지).
        m = mask.resize(canvas.size, Image.NEAREST).convert("L").point(lambda v: 255 if v < 128 else 0)
        region = Image.composite(out, canvas, m)
        img.paste(region.crop((sx1 - cx1, sy1 - cy1, sx2 - cx1, sy2 - cy1)), (sx1, sy1))


def _font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size)
