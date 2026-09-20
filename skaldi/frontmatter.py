"""표지·속표지 같은 앞장을 가려내 번역에서 빼는 판정.

앞에서부터 차례로 보다가 본문 페이지를 처음 만나면 멈춘다. "앞 N장 건너뛰기" 로 고정하지
않는 이유는 앞장 수가 책마다 1~3장으로 제각각이고, 중간의 무대사 페이지까지 빼먹으면
안 되기 때문이다.

판정에 쓰는 신호 (샘플 7종 실측값):

  말풍선 안 글자 수   표지·속표지 0~1개 / 본문 3~12개   ← 가장 확실
  제일 큰 '말풍선 밖 글자' 의 면적 비율
                      표지 0.10~0.20 / 본문 0.00~0.01   ← 제목 로고
  채도                흑백 본문 위의 컬러 표지. 절대값이 아니라 그 책 본문의 중앙값과
                      견준다. 풀컬러 작품은 본문 채도도 높아 이 신호가 저절로 무효가 된다.
  크기·종횡비         본문과 판형이 다른 장 (홍보 배너, 따로 받은 표지)

압축 안 첫 이미지는 사실상 언제나 표지라 신호를 보지 않고 앞장으로 친다.
"""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .config import FrontMatterCfg

# 본문 기준값을 잴 위치 (전체 장수 대비). 앞뒤 표지를 피해 가운데에서 고른다.
BASE_POINTS = (0.35, 0.45, 0.55, 0.65, 0.75)


def _saturation(image: Image.Image) -> float:
    """HSV 의 S 평균. 흑백 페이지는 0 에 가깝다."""
    w, h = image.size
    small = image.convert("RGB").resize((128, max(1, int(128 * h / w))))
    a = np.asarray(small).astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0).mean())


class Baseline:
    """그 책 '본문' 의 기준값. 앞장인지는 이 값과의 차이로 본다."""

    def __init__(self, sat: float, size: tuple[int, int]):
        self.sat = sat
        self.size = size
        self.ratio = size[1] / max(1, size[0])

    @classmethod
    def measure(cls, images: list[Path]) -> "Baseline":
        picks, seen = [], set()
        for p in BASE_POINTS:
            i = int(len(images) * p)
            if i < len(images) and i not in seen:
                seen.add(i)
                picks.append(images[i])
        sats, sizes = [], []
        for path in picks:
            try:
                with Image.open(path) as im:
                    sizes.append(im.size)
                    sats.append(_saturation(im))
            except Exception:  # noqa: BLE001 - 못 읽는 장은 기준에서 뺀다
                continue
        if not sats:
            return cls(0.0, (1, 1))
        return cls(statistics.median(sats), Counter(sizes).most_common(1)[0][0])


def judge(image: Image.Image, labels: list[str], boxes: list[list[int]],
          base: Baseline, cfg: FrontMatterCfg) -> str | None:
    """앞장이면 근거 문자열을, 본문이면 None 을 돌려준다.

    labels/boxes 는 탐지기 결과 (label 은 bubble/text_bubble/text_free)."""
    bubble_text = sum(1 for lb in labels if lb == "text_bubble")
    if bubble_text > cfg.max_bubble_text:
        return None                       # 대사가 여럿이면 본문이다

    w, h = image.size
    area = w * h
    free = [b for lb, b in zip(labels, boxes) if lb == "text_free"]
    logo = max((((b[2] - b[0]) * (b[3] - b[1])) / area for b in free), default=0.0)
    sat = _saturation(image)

    why = []
    if logo >= cfg.logo_area:
        why.append(f"제목 로고({logo:.2f})")
    if sat > base.sat + cfg.sat_margin:
        why.append(f"컬러({sat:.2f}>{base.sat:.2f})")
    if (w, h) != base.size and abs(h / max(1, w) - base.ratio) > cfg.ratio_margin:
        why.append(f"판형 다름({w}x{h})")
    if bubble_text == 0 and not free:
        why.append("글자 없음")
    return ", ".join(why) if why else None


def find(images: list[Path], detect, cfg: FrontMatterCfg,
         first_is_cover: bool = False) -> dict[Path, str]:
    """건너뛸 앞장 → 근거. detect(image) 는 (labels, boxes) 를 돌려주는 함수.

    first_is_cover (압축 입력) 면 첫 장은 신호가 약해도 표지로 친다. 흑백에 제목이 작은
    표지는 어느 신호에도 안 걸리기 때문이다. 다만 대사가 여럿인 장은 그래도 본문으로 둔다
    — 표지 없이 만화부터 시작하는 책이 있다."""
    if not cfg.skip or not images:
        return {}
    base = Baseline.measure(images)
    # 짧은 책은 본문 기준값을 믿을 수 없어(앞장을 본문으로 잘못 고르면 기준 채도가 올라가
    # 나머지 판정이 전부 무너진다) 첫 장만 본다
    limit = cfg.max_pages if len(images) >= cfg.max_pages * 2 else (1 if first_is_cover else 0)

    front: dict[Path, str] = {}
    for i, path in enumerate(images[:limit]):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                labels, boxes = detect(im)
                why = judge(im, labels, boxes, base, cfg)
                if (why is None and i == 0 and first_is_cover
                        and sum(1 for lb in labels if lb == "text_bubble") <= cfg.max_bubble_text):
                    why = "압축 첫 장"
        except Exception:  # noqa: BLE001 - 못 읽으면 본문으로 두고 평소대로 처리
            break
        if why is None:
            break                          # 본문을 만났다. 뒤는 더 보지 않는다
        front[path] = why
    return front
