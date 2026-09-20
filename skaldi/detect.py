"""RT-DETR-v2 기반 말풍선/글자 탐지기.

클래스: 0 = bubble(말풍선), 1 = text_bubble(말풍선 안 글자), 2 = text_free(말풍선 밖 글자)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from .config import DetectorCfg

LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}


@dataclass
class Det:
    label: str
    box: list[int]
    score: float


def iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def contain_ratio(inner: list[int], outer: list[int]) -> float:
    """inner 박스 면적 중 outer 안에 들어 있는 비율."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


class Detector:
    def __init__(self, cfg: DetectorCfg):
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.proc = RTDetrImageProcessor.from_pretrained(
            cfg.repo, size={"width": cfg.input_size, "height": cfg.input_size}
        )
        self.model = RTDetrV2ForObjectDetection.from_pretrained(cfg.repo).to(self.device).eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[Det]:
        img = image.convert("RGB")
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.proc.post_process_object_detection(
            out, threshold=self.cfg.threshold, target_sizes=[img.size[::-1]]
        )[0]
        dets: list[Det] = []
        w, h = img.size
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            dets.append(Det(LABELS.get(int(label), "unknown"), [x1, y1, x2, y2], float(score)))
        return dedupe(dets)


def _area(b: list[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def dedupe(dets: list[Det], iou_thr: float = 0.7, cross_thr: float = 0.5) -> list[Det]:
    """중복·묶음 박스 정리.
    1) 같은 클래스끼리 IoU가 크면 점수 높은 것만 남긴다.
    2) 같은 클래스의 다른 박스를 두 개 이상 품는 '묶음 박스'는 버린다 (열 전체를 하나로 잡은 경우).
    3) 하나가 다른 하나를 거의 품으면 점수 높은 쪽만 남긴다.
    4) text_free 가 text_bubble 과 겹치면 text_bubble 을 우선한다."""
    keep: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if any(k.label == d.label and iou(k.box, d.box) > iou_thr for k in keep):
            continue
        keep.append(d)

    def children(k: Det) -> list[Det]:
        return [o for o in keep if o is not k and o.label == k.label and contain_ratio(o.box, k.box) > 0.8]

    keep = [k for k in keep if len(children(k)) < 2]

    out: list[Det] = []
    for d in sorted(keep, key=lambda d: -d.score):
        if any(k.label == d.label and (contain_ratio(d.box, k.box) > 0.8 or contain_ratio(k.box, d.box) > 0.8)
               for k in out):
            continue
        out.append(d)

    final: list[Det] = []
    for d in sorted(out, key=lambda d: ({"text_bubble": 0, "text_free": 1, "bubble": 2}.get(d.label, 3), -d.score)):
        if d.label == "text_free" and any(
            k.label == "text_bubble" and (iou(k.box, d.box) > cross_thr or contain_ratio(k.box, d.box) > 0.8)
            for k in final
        ):
            continue
        final.append(d)
    return final


def attach_bubbles(dets: list[Det]) -> list[tuple[Det, list[int] | None]]:
    """글자 박스마다 감싸는 말풍선 박스를 찾아 붙인다. 말풍선 자체는 결과에서 제외."""
    bubbles = [d for d in dets if d.label == "bubble"]
    out: list[tuple[Det, list[int] | None]] = []
    for d in dets:
        if d.label == "bubble":
            continue
        best, best_r = None, 0.0
        for b in bubbles:
            r = contain_ratio(d.box, b.box)
            if r > best_r:
                best, best_r = b, r
        out.append((d, best.box if best is not None and best_r >= 0.6 else None))
    return out
