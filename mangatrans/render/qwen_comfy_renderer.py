"""Qwen-Image-Edit-2511 렌더러 (ComfyUI HTTP API, GGUF + Lightning LoRA).

말풍선 크롭을 ComfyUI에 올리고 "빈 말풍선 안에 이 한국어 문장을 써라"는 편집을 요청한 뒤,
글자 박스 영역만 원본 위치에 다시 붙인다.
"""
from __future__ import annotations

import random

from PIL import Image, ImageDraw

from ..comfy import ComfyClient, ComfyServer
from ..config import Config
from ..erase import region_target_box
from ..page import Page, Region


def build_graph(cfg: Config, image_name: str, prompt: str, seed: int) -> dict:
    q = cfg.qwen
    lightning = q.lightning_lora and bool(q.lora_file)
    g: dict[str, dict] = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q.unet_file}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": q.clip_file, "type": "qwen_image", "device": q.clip_device or "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": q.vae_file}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
        "7": {"class_type": "CFGNorm", "inputs": {"model": ["6", 0], "strength": 1.0, "pre_cfg": False}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["5", 0]}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["5", 0]}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "mangatrans_qwen"}},
    }
    model_ref = ["7", 0]
    if lightning:
        g["8"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["7", 0], "lora_name": q.lora_file, "strength_model": 1.0}}
        model_ref = ["8", 0]
        steps, cfg_scale = 4, 1.0
    else:
        steps, cfg_scale = max(1, q.steps), 4.0
    g["14"] = {"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["13", 0],
        "seed": seed, "steps": steps, "cfg": cfg_scale, "sampler_name": "euler", "scheduler": "simple",
        "denoise": 1.0}}
    return g


class QwenComfyRenderer:
    name = "qwen"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.server = ComfyServer(cfg.qwen)
        self.client: ComfyClient | None = None

    def close(self) -> None:
        """우리가 띄운 ComfyUI만 종료한다."""
        self.server.stop()

    def _ensure(self) -> ComfyClient:
        if self.client is None:
            q = self.cfg.qwen
            if not (q.unet_file and q.clip_file and q.vae_file):
                raise RuntimeError("config.yaml qwen.unet_file/clip_file/vae_file 이 비어 있습니다 (install_comfyui.py 참고)")
            self.server.ensure()
            self.client = ComfyClient(q.url, q.timeout_sec)
        return self.client

    def render(self, clean: Image.Image, original: Image.Image, page: Page) -> Image.Image:
        client = self._ensure()
        img = clean.convert("RGB").copy()
        for r in page.regions:
            if not r.render or not r.text_ko.strip():
                continue
            try:
                self._render_region(client, img, r, page)
            except Exception as e:  # noqa: BLE001
                page.warnings.append(f"qwen 실패 (id={r.id}): {e}")
        return img

    def _render_region(self, client: ComfyClient, img: Image.Image, r: Region, page: Page) -> None:
        S = self.cfg.qwen.crop_size
        tx1, ty1, tx2, ty2 = region_target_box(r, self.cfg, page.width, page.height)
        tw, th = max(8, tx2 - tx1), max(8, ty2 - ty1)
        span = max(tw, th) / 0.6
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        half = span / 2
        cx1, cy1, cx2, cy2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
        canvas = Image.new("RGB", (cx2 - cx1, cy2 - cy1), (255, 255, 255))
        sx1, sy1 = max(0, cx1), max(0, cy1)
        sx2, sy2 = min(page.width, cx2), min(page.height, cy2)
        canvas.paste(img.crop((sx1, sy1, sx2, sy2)), (sx1 - cx1, sy1 - cy1))
        crop = canvas.resize((S, S), Image.LANCZOS)

        where = "empty speech bubble" if r.kind == "bubble_text" else "blank area"
        text = r.text_ko.strip().replace("\n", " ")
        prompt = (
            f'Write the Korean text "{text}" inside the {where} at the center of this black-and-white manga panel. '
            "Use a clean black printed manga font, horizontal lines, centered, sized to fit the bubble. "
            "Keep every other part of the image exactly the same."
        )
        name = client.upload_image(crop, f"mangatrans_{page.source.split('/')[-1].split(chr(92))[-1]}_{r.id}.png")
        seed = random.randint(0, 2**31 - 1)
        out = client.run(build_graph(self.cfg, name, prompt, seed))[0]
        out = out.resize(canvas.size, Image.LANCZOS)

        # 글자 박스(+여백)만 붙인다.
        m = Image.new("L", canvas.size, 0)
        pad = int(min(tw, th) * 0.15)
        ImageDraw.Draw(m).rectangle([tx1 - pad - cx1, ty1 - pad - cy1, tx2 + pad - cx1, ty2 + pad - cy1], fill=255)
        region = Image.composite(out, canvas, m)
        img.paste(region.crop((sx1 - cx1, sy1 - cy1, sx2 - cx1, sy2 - cy1)), (sx1, sy1))
