"""config.yaml 로딩. 프로젝트 루트 기준 경로를 절대 경로로 바꿔 준다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class PathsCfg(BaseModel):
    models_dir: Path = Path("models")
    font: Path = Path("fonts/NanumSquareRoundB.ttf")
    glossary: Path = Path("glossary.yaml")
    output: Path = Path("output")
    output_suffix: str = "_ko"      # 입력 이름 뒤에 붙여 출력 폴더·zip 이름을 만든다


class DetectorCfg(BaseModel):
    repo: str = "ogkalu/comic-text-and-bubble-detector"
    threshold: float = 0.3
    input_size: int = 640
    device: str = "cuda"


class OcrCfg(BaseModel):
    backend: str = "baberu"
    baberu_repo: str = "genshiai-daichi/baberu-ocr"
    crop_padding: int = 4


class LlmCfg(BaseModel):
    backend: str = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen3.5:9b"
    translate_model: str = "qwen3.5:9b"
    translate_fallbacks: list[str] = Field(default_factory=list)
    temperature: float = 0.1
    page_long_side: int = 1536
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key_env: str = "GEMINI_API_KEY"


class TranslateCfg(BaseModel):
    source_lang: str = "일본어"
    target_lang: str = "한국어"


class EraseCfg(BaseModel):
    bubble_padding: int = 6
    flood_fill: bool = True
    flood_tolerance: int = 40
    lama: bool = True
    lama_dilate: int = 8
    widen_bubbles: bool = True      # 번역문이 자연스럽게 안 들어가는 좁은 말풍선을 가로로 넓힌다
    widen_max_lines: int = 3        # 이 줄 수 이내로 들어가도록 넓힌다
    widen_max_ratio: float = 0.6    # 원래 폭 대비 최대 확장 비율
    widen_min_font_ratio: float = 0.8  # 이 비율 이상 크기로 들어가면 넓히지 않음


class RenderCfg(BaseModel):
    default: str = "pillow"
    font_ratio: float = 0.018
    min_font_ratio: float = 0.011
    line_spacing: float = 1.15
    bubble_inner_margin: float = 0.08
    bubble_fit: float = 0.7             # 말풍선 곡선에 얼마나 맞출지. 1=글자가 절대 안 나감,
                                        # 0=본체 사각형 그대로(모서리가 크게 삐져나옴)
    stroke_ratio: float = 0.12          # 그림 위 글자(나레이션)의 외곽선 두께 비율
    bubble_stroke_ratio: float = 0.05   # 말풍선 안 글자의 외곽선 두께 비율 (0이면 없음)
    jpeg_quality: int = 95
    compare: bool = True
    vertical_for_narrow: bool = False   # 좁고 긴 상자는 세로쓰기
    narrow_ratio: float = 2.5           # 높이/폭 이 이 값 이상이면 '좁고 긴' 상자
    match_style: bool = True            # 글꼴 계열 판정 (굵기는 이 값과 무관하게 항상 적용)
    match_color: bool = True            # 원문 획·외곽선 색을 따라감
    fonts: dict[str, str] = Field(default_factory=dict)        # {gothic, gothic_bold, mincho, mincho_bold, hand, hand_bold}
    font_scale: dict[str, float] = Field(default_factory=dict) # 폰트별 크기 보정 (손글씨체는 작게 보여서 키움)
    bold_stroke_ratio: float = 0.16     # 획 두께/글자 크기 가 이 값 이상이면 bold
    fake_bold_ratio: float = 0.025      # 진짜 굵은 자형이 없는 폰트에서 획을 덧대는 두께 비율
    fake_bold_styles: list[str] = Field(default_factory=lambda: ["hand"])
    follow_angle: bool = True           # 원문이 기울어 있으면 번역문도 같은 각도로 돌려 그림
    supersample_below: int = 35         # 이 크기(px) 미만 글자는 크게 그려 줄여서 계단 현상을 줄임


class AnyTextCfg(BaseModel):
    repo: str = "tolgacangoz/anytext"
    controlnet_repo: str = "tolgacangoz/anytext-controlnet"
    steps: int = 20
    crop_size: int = 512


class QwenCfg(BaseModel):
    comfyui_dir: Path = Path("C:/ComfyUI")
    url: str = "http://127.0.0.1:8188"
    autostart: bool = True
    lightning_lora: bool = True
    steps: int = 4
    crop_size: int = 1024
    timeout_sec: int = 900
    extra_args: list[str] = Field(default_factory=list)   # ComfyUI 자동 실행 시 추가 인자
    unet_file: str = ""            # models/unet 또는 diffusion_models 안의 파일명 (설치 스크립트가 채움)
    clip_file: str = ""
    vae_file: str = ""
    lora_file: str = ""
    clip_device: str = "cpu"       # 텍스트 인코더 위치: cpu 면 VRAM을 확산 모델에 몰아준다


class Config(BaseModel):
    paths: PathsCfg = PathsCfg()
    detector: DetectorCfg = DetectorCfg()
    ocr: OcrCfg = OcrCfg()
    llm: LlmCfg = LlmCfg()
    translate: TranslateCfg = TranslateCfg()
    erase: EraseCfg = EraseCfg()
    render: RenderCfg = RenderCfg()
    anytext: AnyTextCfg = AnyTextCfg()
    qwen: QwenCfg = QwenCfg()

    def abs(self, p: Path) -> Path:
        """상대 경로를 프로젝트 루트 기준 절대 경로로."""
        p = Path(p)
        return p if p.is_absolute() else ROOT / p


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
