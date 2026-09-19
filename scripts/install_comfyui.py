"""ComfyUI + Qwen-Image-Edit-2511 (GGUF) 자동 설치.

  uv run python scripts/install_comfyui.py                 # config.yaml 의 qwen.comfyui_dir 에 설치
  uv run python scripts/install_comfyui.py --dir D:/ComfyUI
  uv run python scripts/install_comfyui.py --skip-models   # 코드·환경만

받는 것: ComfyUI(git) + 전용 venv(torch cu130) + ComfyUI-GGUF 노드 + 모델 약 22GB
  models/unet/qwen-image-edit-2511-Q4_0.gguf                      11.9GB
  models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors       8.7GB
  models/vae/qwen_image_vae.safetensors                            0.25GB
  models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors  0.85GB
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mangatrans.config import load_config  # noqa: E402

COMFY_GIT = "https://github.com/comfyanonymous/ComfyUI"
GGUF_NODE_GIT = "https://github.com/city96/ComfyUI-GGUF"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# (repo, 저장소 안 경로, 대상 하위 폴더, 저장 파일명)
MODELS = [
    ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_0.gguf", "unet", "qwen-image-edit-2511-Q4_0.gguf"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae", "qwen_image_vae.safetensors"),
    ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
     "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    comfy = Path(args.dir or cfg.qwen.comfyui_dir)
    print(f"설치 위치: {comfy}")

    if not (comfy / "main.py").exists():
        run(["git", "clone", "--depth", "1", COMFY_GIT, comfy])
    else:
        print("ComfyUI 이미 있음, git pull")
        run(["git", "pull", "--ff-only"], cwd=comfy)

    venv = comfy / ".venv"
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        run(["uv", "venv", venv, "--python", "3.12"])
    run(["uv", "pip", "install", "--python", py, "torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX])
    run(["uv", "pip", "install", "--python", py, "-r", comfy / "requirements.txt"])

    node = comfy / "custom_nodes" / "ComfyUI-GGUF"
    if not node.exists():
        run(["git", "clone", "--depth", "1", GGUF_NODE_GIT, node])
    run(["uv", "pip", "install", "--python", py, "-r", node / "requirements.txt"])

    if args.skip_models:
        print("모델 다운로드 생략")
        return
    from huggingface_hub import hf_hub_download

    for repo, fname, sub, dest_name in MODELS:
        dest = comfy / "models" / sub / dest_name
        if dest.exists():
            print(f"있음: {dest}")
            continue
        print(f"다운로드: {repo}/{fname}")
        got = Path(hf_hub_download(repo, fname))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, dest)  # HF 캐시에 원본이 남는다. 공간이 아까우면 캐시를 지운다.
        print(f"  → {dest}")
    print("완료. config.yaml 의 qwen 항목 파일명이 위와 같은지 확인하세요.")


if __name__ == "__main__":
    main()
