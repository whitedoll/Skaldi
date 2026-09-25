"""로컬 모델·코드·폰트를 미리 내려받는다 (파이프라인은 없으면 자동으로 받지만, 미리 받아 두면 빠르다).

  python scripts/download_models.py            # 탐지기, Baberu OCR, 분할 모델, LaMa, 폰트, AnyText 코드
  python scripts/download_models.py --anytext  # AnyText 가중치까지 (약 2GB)
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skaldi.config import load_config  # noqa: E402

ANYTEXT_CODE_BASE = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/research_projects/anytext"
ANYTEXT_CODE_FILES = ["README.md", "anytext.py", "anytext_controlnet.py"]
ANYTEXT_OCR_FILES = ["RNN.py", "RecCTCHead.py", "RecModel.py", "RecMv1_enhance.py", "RecSVTR.py",
                     "common.py", "en_dict.txt"]
FONT_URL = "https://raw.githubusercontent.com/innks/NanumSquareRound/master/NanumSquareRoundB.ttf"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anytext", action="store_true", help="AnyText 가중치까지 내려받기")
    args = ap.parse_args()
    cfg = load_config()
    models = cfg.abs(cfg.paths.models_dir)

    print("[폰트]")
    fetch(FONT_URL, cfg.abs(cfg.paths.font))

    print("[탐지기]")
    from huggingface_hub import snapshot_download

    snapshot_download(cfg.detector.repo, allow_patterns=["*.json", "*.safetensors"])

    print("[Baberu OCR]")
    snapshot_download(cfg.ocr.baberu_repo, local_dir=str(models / "baberu-ocr"),
                      allow_patterns=["*.py", "*.json", "*.txt", "model.safetensors", "tokenizer/*"])

    print("[말풍선·글자 분할(koharu)]")
    from huggingface_hub import hf_hub_download

    hf_hub_download(cfg.layout.repo, "model.safetensors",
                    local_dir=str(models / "koharu-layout"))

    print("[LaMa]")
    from simple_lama_inpainting import SimpleLama

    SimpleLama()

    print("[AnyText 코드]")
    code = models / "anytext_code"
    for f in ANYTEXT_CODE_FILES:
        fetch(f"{ANYTEXT_CODE_BASE}/{f}", code / f)
    for f in ANYTEXT_OCR_FILES:
        fetch(f"{ANYTEXT_CODE_BASE}/ocr_recog/{f}", code / "ocr_recog" / f)

    if args.anytext:
        print("[AnyText 가중치]")
        snapshot_download(cfg.anytext.controlnet_repo, allow_patterns=["*.json", "*fp16*"])
        snapshot_download(cfg.anytext.repo, allow_patterns=["*.json", "*.txt", "*.py", "*fp16*",
                                                            "text_embedding_module/*"])
    print("완료")


if __name__ == "__main__":
    main()
