"""페이지 단위 결과 데이터 모델과 JSON 입출력."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["bubble_text", "free_text"]
Category = Literal["dialogue", "narration", "label", "sfx", "unknown"]
EraseMode = Literal["white", "lama", "none"]


class Region(BaseModel):
    id: int
    kind: Kind                                # 탐지기 클래스 (말풍선 안 / 밖)
    box: list[int]                            # 글자 박스 [x1, y1, x2, y2]
    bubble_box: list[int] | None = None       # 감싸는 말풍선 박스
    score: float = 0.0
    text_ja: str = ""
    text_ko: str = ""
    category: Category = "unknown"            # dialogue/narration은 그림, label은 JSON만, sfx는 무시
    order: int | None = None                  # 읽기 순서 (0부터)
    render: bool = True                       # 이미지에 그릴지
    erase: EraseMode = "white"
    font_size: int | None = None              # 렌더러가 실제 사용한 크기
    text_color: str = "black"                 # 말풍선 배경이 어두우면 white (지우기 단계에서 결정)
    vertical: bool = False                    # 세로쓰기로 그렸는지
    style: str = "gothic"                     # 글꼴 계열: gothic(고딕/인쇄체) | mincho(명조) | hand(손글씨)
    weight: str = "regular"                   # regular | bold (획 두께로 측정)
    text_rgb: list[int] | None = None         # 원문 획 색 (측정값). None 이면 text_color 규칙
    outline_rgb: list[int] | None = None      # 원문 외곽선 색 (있을 때)
    needs_review: bool = False                # 원문이 불명확하거나 번역 모델이 뜻을 못 잡음 → 원본 유지
    body_box: list[int] | None = None         # 말풍선 안쪽에 실제로 글자가 들어가는 사각형 (픽셀 측정값)
    target_box: list[int] | None = None       # 글자를 넣을 영역 (말풍선 넓히기 등으로 조정된 값)
    angle: float = 0.0                        # 원문 글자 기울기(도, 반시계 +). 0 이면 똑바로 그린다
    rot_box: list[float] | None = None        # 기운 글자의 배치 상자: [중심x, 중심y, 폭, 높이] (세운 좌표의 크기)
    writing: str = "auto"                     # auto(가로쓰기) | vertical(세로 한 열 라벨) | vcols(여러 열 세로쓰기) | sideways(가로 한 줄을 90° 돌림)
    poly: list[list[int]] | None = None       # 겹친 말풍선에서 폴리곤 배치를 고른 경우의 글자 자리 다각형 [[x, y], ...]
    group: int | None = None                  # 겹친 말풍선을 나눠 쓴 묶음 번호. 같은 묶음은 글자 크기를 맞춘다
    widened: bool = False                     # 말풍선을 넓혔는지
    overflow: bool = False                    # 최소 크기에서도 넘침
    refused: bool = False                     # 번역 거부·순화 감지
    ocr_backend: str = ""
    notes: str = ""

    def w(self) -> int:
        return self.box[2] - self.box[0]

    def h(self) -> int:
        return self.box[3] - self.box[1]


class Page(BaseModel):
    version: int = 1
    source: str                               # 원본 이미지 경로
    width: int
    height: int
    regions: list[Region] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def ordered(self) -> list[Region]:
        """읽기 순서대로 정렬. 순서가 없으면 id 순."""
        return sorted(self.regions, key=lambda r: (r.order if r.order is not None else 10_000, r.id))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Page":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
