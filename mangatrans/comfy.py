"""ComfyUI 서버 실행 관리와 HTTP API 클라이언트."""
from __future__ import annotations

import io
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image

from .config import QwenCfg


class ComfyServer:
    """설정된 URL에 ComfyUI가 없으면 자식 프로세스로 띄우고, 우리가 띄운 것만 종료한다."""

    def __init__(self, cfg: QwenCfg):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None

    def is_up(self, timeout: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{self.cfg.url}/system_stats", timeout=timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def ensure(self) -> None:
        if self.is_up():
            return
        if not self.cfg.autostart:
            raise RuntimeError(f"ComfyUI가 {self.cfg.url} 에 없습니다. 먼저 실행하거나 qwen.autostart 를 켜세요.")
        comfy = Path(self.cfg.comfyui_dir)
        main_py = comfy / "main.py"
        py = comfy / ".venv" / "Scripts" / "python.exe"
        if not main_py.exists() or not py.exists():
            raise RuntimeError(f"ComfyUI 설치가 없습니다: {comfy} (scripts/install_comfyui.py 실행)")
        port = self.cfg.url.rsplit(":", 1)[-1].strip("/")
        cmd = [str(py), str(main_py), "--listen", "127.0.0.1", "--port", port, "--disable-auto-launch"]
        cmd += self.cfg.extra_args
        log = (comfy / "comfyui_autostart.log").open("ab")
        self.proc = subprocess.Popen(cmd, cwd=str(comfy), stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.is_up():
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI 실행 실패 (로그: {comfy / 'comfyui_autostart.log'})")
            time.sleep(2)
        raise RuntimeError("ComfyUI 가 180초 안에 뜨지 않았습니다")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class ComfyClient:
    def __init__(self, url: str, timeout_sec: int = 900):
        self.url = url.rstrip("/")
        self.timeout = timeout_sec
        self.client_id = uuid.uuid4().hex
        self.http = httpx.Client(timeout=60)

    def upload_image(self, image: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        r = self.http.post(f"{self.url}/upload/image",
                           files={"image": (name, buf.getvalue(), "image/png")},
                           data={"overwrite": "true"})
        r.raise_for_status()
        return r.json()["name"]

    def queue(self, graph: dict) -> str:
        r = self.http.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI /prompt 오류 {r.status_code}: {r.text[:500]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = self.http.get(f"{self.url}/history/{prompt_id}")
            r.raise_for_status()
            hist = r.json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI 실행 오류: {msgs[-1] if msgs else '?'}")
                if entry.get("outputs"):
                    return entry["outputs"]
            time.sleep(1.0)
        raise TimeoutError(f"ComfyUI 작업이 {self.timeout}초 안에 끝나지 않았습니다")

    def fetch_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        r = self.http.get(f"{self.url}/view",
                          params={"filename": filename, "subfolder": subfolder, "type": folder_type})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def run(self, graph: dict) -> list[Image.Image]:
        outputs = self.wait(self.queue(graph))
        images: list[Image.Image] = []
        for node_out in outputs.values():
            for im in node_out.get("images", []):
                images.append(self.fetch_image(im["filename"], im.get("subfolder", ""), im.get("type", "output")))
        if not images:
            raise RuntimeError("ComfyUI 출력 이미지가 없습니다")
        return images
