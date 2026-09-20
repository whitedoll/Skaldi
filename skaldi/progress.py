"""자식 프로세스(skaldi CLI)의 진행 상황을 부모(GUI)에게 알리는 한 줄 프로토콜.

GUI 는 파이프라인을 자식 프로세스로 돌리고 stdout 만 읽는다. 사람이 읽는 로그만으로는
"몇 장 중 몇 장째"를 알 수 없으므로, 환경변수 ``SKALDI_PROGRESS=1`` 이 켜져 있을 때만
``@@PROG {json}`` 형태의 줄을 함께 찍는다. GUI 는 이 줄을 로그에서 걸러 내고 진행 막대에 쓴다.
CLI 로 직접 쓸 때는 환경변수가 없으므로 아무것도 찍히지 않는다.

이벤트 종류(``ev``)
  plan   total=전체 작업 수                     — 실행 시작
  job    index, total, name, kind               — 입력(폴더·zip) 하나 시작
  pages  total                                  — 현재 입력의 페이지 수
  page   index, total, name                     — 페이지 하나 시작
  stage  name                                   — 페이지 안의 단계(탐지·OCR·번역·렌더링…)
  done   ok                                     — 전체 종료
"""
from __future__ import annotations

import json
import os
import sys

PREFIX = "@@PROG "
STAGES = ("탐지", "OCR", "순서·분류", "번역", "지우기", "렌더링")


def enabled() -> bool:
    # 부모가 환경변수를 넘겼는지 매번 확인한다(모듈 import 시점에 고정하지 않는다).
    return os.environ.get("SKALDI_PROGRESS") == "1"


def emit(ev: str, **fields) -> None:
    if not enabled():
        return
    try:
        sys.stdout.write(PREFIX + json.dumps({"ev": ev, **fields}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001  진행 표시가 본 작업을 망치면 안 된다
        pass


def parse(line: str) -> dict | None:
    """로그 한 줄이 진행 이벤트면 dict 로, 아니면 None."""
    if not line.startswith(PREFIX):
        return None
    try:
        return json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return None
