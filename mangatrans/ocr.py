"""OCR 백엔드: Baberu OCR(기본) 또는 LLM 비전."""
from __future__ import annotations

import sys

from PIL import Image

from .config import Config


class BaberuBackend:
    name = "baberu"

    def __init__(self, cfg: Config):
        model_dir = cfg.abs(cfg.paths.models_dir) / "baberu-ocr"
        required = ["model.safetensors", "inference.py", "modeling_baberu.py",
                    "configuration_baberu.py", "tokenization_baberu.py", "config.json"]
        if not all((model_dir / f).exists() for f in required):
            from huggingface_hub import snapshot_download

            snapshot_download(
                cfg.ocr.baberu_repo,
                local_dir=str(model_dir),
                allow_patterns=["*.py", "*.json", "*.txt", "model.safetensors", "tokenizer/*"],
            )
        # inference.py는 같은 폴더의 모듈을 상대 import 하므로 경로를 추가한다.
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from inference import BaberuOCR  # type: ignore

        self.ocr = BaberuOCR(model_dir)

    def read(self, crop: Image.Image) -> str:
        return self.ocr(crop).strip()


class LlmBackend:
    name = "llm"

    def __init__(self, cfg: Config):
        from .llm import OllamaClient

        self.client = OllamaClient(cfg.llm)
        self.model = cfg.llm.vision_model

    def read(self, crop: Image.Image) -> str:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        prompt = (
            "이 이미지는 만화 말풍선 하나를 잘라낸 것이다. 안에 있는 일본어 글자를 있는 그대로 읽어라. "
            "번역하지 말고, 후리가나(작은 글자)는 빼고, 줄바꿈 없이 한 줄로 이어서 text 필드에 넣어라. "
            "글자가 없으면 빈 문자열."
        )
        data = self.client.chat_json(self.model, prompt, images=[crop], schema=schema)
        return str(data.get("text", "")).strip()


def make_ocr(cfg: Config):
    if cfg.ocr.backend == "llm":
        return LlmBackend(cfg)
    return BaberuBackend(cfg)


def crop_region(image: Image.Image, box: list[int], pad: int) -> Image.Image:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))
