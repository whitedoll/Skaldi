"""렌더러 레지스트리. 이름으로 렌더러 인스턴스를 만든다."""
from __future__ import annotations

from ..config import Config

RENDERERS = ("pillow", "anytext", "qwen")


def make_renderer(name: str, cfg: Config):
    if name == "pillow":
        from .pillow_renderer import PillowRenderer

        return PillowRenderer(cfg)
    if name == "anytext":
        from .anytext_renderer import AnyTextRenderer

        return AnyTextRenderer(cfg)
    if name == "qwen":
        from .qwen_comfy_renderer import QwenComfyRenderer

        return QwenComfyRenderer(cfg)
    raise ValueError(f"알 수 없는 렌더러: {name}")


def resolve_names(arg: str | None, cfg: Config) -> list[str]:
    name = arg or cfg.render.default
    if name == "both":
        return list(RENDERERS)
    return [n.strip() for n in name.split(",") if n.strip()]
