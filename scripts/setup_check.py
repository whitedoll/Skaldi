"""설치 뒤 동작 점검. setup.ps1 / setup.sh 가 마지막에 부른다.

  uv run python scripts/setup_check.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skaldi.config import load_config  # noqa: E402


def main() -> int:
    ok = True
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: 못 찾음 — CPU 로 돌면 한 장에 몇 분씩 걸립니다")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"  GPU: 확인 실패 ({e})")
        ok = False

    cfg = load_config()
    try:
        with urllib.request.urlopen(cfg.llm.ollama_host.rstrip("/") + "/api/tags", timeout=3) as r:
            names = {m.get("name", "") for m in __import__("json").load(r).get("models", [])}
        print(f"  Ollama: 실행 중 (모델 {len(names)}개)")
        for m in dict.fromkeys([cfg.llm.vision_model, cfg.llm.translate_model]):
            mark = "있음" if any(n == m or n.startswith(m) for n in names) else "없음 → ollama pull 필요"
            print(f"    - {m}: {mark}")
            ok = ok and "없음" not in mark
    except Exception:
        print(f"  Ollama: 응답 없음 ({cfg.llm.ollama_host}) — 번역하려면 Ollama 를 실행하세요")
        ok = False

    font = cfg.abs(cfg.paths.font)
    print(f"  기본 글꼴: {'있음' if font.exists() else '없음 → scripts/download_models.py'} ({font.name})")
    print("  점검 결과:", "모두 준비됨" if ok else "위의 항목을 채우면 됩니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
