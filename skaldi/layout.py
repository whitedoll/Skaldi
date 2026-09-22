"""페이지 배치 분할: 말풍선·글자·효과음을 모양(마스크)으로 찾는다 (koharu-layout-rfdetr-seg).

지금까지는 말풍선 안쪽을 색으로 채워 나가며(flood fill) 추정했다. 빗금·가시 테두리·옅어지는 바탕·반투명·
망점에서 채우기가 막히거나 새어, 글자가 작아지거나(Kamaboko 異世界 03·05쪽) 한쪽으로 밀렸다(test05).
분할 모델은 모양을 학습해 이런 말풍선에서도 윤곽을 잡는다. 네 모델을 견줘 이 모델을 골랐다:
kitsumed yolov8m-seg 는 반투명 말풍선 절반을 놓쳤고, ShadowB YOLO26s-seg 는 손글씨 효과음 말풍선을
못 잡고 효과음 구분이 없다.

모델은 Manga109(학술·비상업 조건)로 학습돼 저장소에 넣지 않고 처음 쓸 때 Hugging Face 에서 받는다.
결과는 페이지마다 <작업폴더>/layout/<이름>.json 에 다각형으로 남겨, 다시 그릴 때 모델을 돌리지 않는다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import Config

CLASSES = ["text", "onomatopoeia", "bubble", "panel"]


class KoharuLayout:
    def __init__(self, cfg: Config):
        from huggingface_hub import hf_hub_download
        from rfdetr import RFDETRSeg2XLarge
        from rfdetr.config import PretrainWeightsCompatibilityWarning
        from safetensors.torch import load_file

        lc = cfg.layout
        self.cfg = lc
        path = hf_hub_download(lc.repo, "model.safetensors",
                               local_dir=str(cfg.abs(cfg.paths.models_dir) / "koharu-layout"))
        # 모델 저장소의 load_model.py 와 같은 방식으로 만든다
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
            model = RFDETRSeg2XLarge(pretrain_weights=None, resolution=1152, num_select=160,
                                     num_classes=len(CLASSES))
        bad = model.model.model.load_state_dict(load_file(path, device="cpu"), strict=True)
        if bad.missing_keys or bad.unexpected_keys:
            raise RuntimeError(f"koharu 레이아웃 가중치가 맞지 않습니다: {bad}")
        model.model.class_names = CLASSES.copy()
        self.model = model

    def predict(self, image: Image.Image) -> list[dict]:
        """[{cls, conf, box, polys}] — polys 는 마스크 바깥 윤곽 다각형들(글자 마스크는 글자마다 조각이 여럿이다)."""
        import torch

        th = {"text": self.cfg.text_threshold, "onomatopoeia": self.cfg.sfx_threshold,
              "bubble": self.cfg.bubble_threshold, "panel": 1.1}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                det = self.model.predict(image.convert("RGB"), threshold=min(th.values()))
        finally:
            # 계산 중 임시 메모리가 2.7GB 라, 번역 모델(11GB)과 같은 카드에서 쓰려면 바로 비운다
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out = []
        for i in range(len(det)):
            cls = CLASSES[int(det.class_id[i])]
            conf = float(det.confidence[i])
            if conf < th[cls]:
                continue
            m = det.mask[i].astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = [cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).tolist() for c in cnts if cv2.contourArea(c) >= 4]
            if polys:
                out.append({"cls": cls, "conf": round(conf, 3),
                            "box": [int(v) for v in det.xyxy[i]], "polys": polys})
        return out


def save(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "items": items}, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("items")
    except Exception:  # noqa: BLE001
        return None


def rasterize(item: dict, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(p, np.int32) for p in item["polys"]], 255)
    return m


def bubble_masks(items: list[dict] | None, shape: tuple[int, int]) -> list[tuple[list[int], np.ndarray]]:
    """말풍선 마스크 목록 [(상자, 페이지 크기 마스크 0/255)]."""
    if not items:
        return []
    return [(it["box"], rasterize(it, shape)) for it in items if it["cls"] == "bubble"]


def bubble_for(box: list[int], bubbles: list[tuple[list[int], np.ndarray]], min_cover: float = 0.5
               ) -> np.ndarray | None:
    """글자 상자를 가장 많이 덮는 말풍선 마스크. 글자 상자의 min_cover 이상을 덮어야 그 말풍선의 글로 본다."""
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    best, best_cov = None, min_cover
    for bb, m in bubbles:
        if bb[2] <= x1 or bb[0] >= x2 or bb[3] <= y1 or bb[1] >= y2:
            continue
        cov = float((m[y1:y2, x1:x2] > 0).sum()) / area
        if cov >= best_cov:
            best, best_cov = m, cov
    return best


SFX_NOTE = "분할:효과음"


def mark_sfx(page, items: list[dict] | None, shape: tuple[int, int], min_cover: float = 0.15) -> int:
    """분할 모델이 효과음으로 본 글자는 번역해 그리지 않고 원본을 둔다. 바꾼 영역 수를 돌려준다.

    손글씨 효과음은 번역하지 않기로 했다. 그동안은 비전·번역 모델의 판단과 글꼴 판정(손글씨인가)을
    엮어 가렸는데, 손글씨 효과음을 OCR 이 헛읽으면('射的精米' 'おはようございます' '．．．') 대사로
    번역돼 그려졌다. 분할 모델은 모양으로 효과음을 따로 구분한다.

    영역 상자 안에서 효과음 마스크가 min_cover 이상이고 글자 마스크보다 넓을 때만 효과음으로 본다.
    실측(작품 6종·테스트 페이지 721 영역): 지금 번역해 그리는데 이 기준에 걸린 것 55개 중 52개가
    손글씨 효과음·신음이었다. 예외는 원 안의 로고 글자('催眠'), 손글씨 단어('また'), 손글씨 나레이션
    ('朝の空', 0.14 라 기준에 안 걸린다). test02·04·05 의 인쇄체 나레이션·대사는 하나도 걸리지 않았다."""
    if not items:
        return 0
    # 같은 글자를 효과음과 글자로 겹쳐 잡는 경우가 흔하다(Kamaboko 08쪽 'オオォォォ': 효과음 0.50, 글자
    # 0.44 로 같은 픽셀). 마스크를 그냥 합치면 두 비율이 같아져 가를 수 없으므로, 픽셀마다 확신도가 더
    # 높은 쪽 분류로 칠한다
    sfx_conf = np.zeros(shape, np.float32)
    txt_conf = np.zeros(shape, np.float32)
    for it in items:
        if it["cls"] in ("onomatopoeia", "text"):
            m = rasterize(it, shape) > 0
            tgt = sfx_conf if it["cls"] == "onomatopoeia" else txt_conf
            np.maximum(tgt, np.where(m, it["conf"], 0).astype(np.float32), out=tgt)
    sfx = sfx_conf > txt_conf
    txt = txt_conf > sfx_conf
    if not sfx.any():
        return 0
    n = 0
    for r in page.regions:
        if r.category == "sfx" or not r.text_ja.strip():
            continue
        x1, y1, x2, y2 = r.box
        area = max(1, (x2 - x1) * (y2 - y1))
        sc = float(sfx[y1:y2, x1:x2].sum()) / area
        tc = float(txt[y1:y2, x1:x2].sum()) / area
        if sc >= min_cover and sc > tc:
            r.category, r.render, r.erase = "sfx", False, "none"
            r.notes = (r.notes + f" {SFX_NOTE}({sc:.2f}/{tc:.2f})").strip()
            n += 1
    return n



def text_mask(items: list[dict] | None, shape: tuple[int, int]) -> np.ndarray | None:
    """글자·효과음 마스크를 합친 페이지 크기 마스크(0/255). 없으면 None."""
    if not items:
        return None
    m = np.zeros(shape, np.uint8)
    for it in items:
        if it["cls"] in ("text", "onomatopoeia"):
            m |= rasterize(it, shape)
    return m if m.any() else None


def limit_to_text(mask: np.ndarray, tmask: np.ndarray, tbox: tuple[int, int, int, int], halo: int,
                  lum: np.ndarray | None = None, bg_lum: float | None = None, restrict: bool = True,
                  region: np.ndarray | None = None) -> np.ndarray | None:
    """지우기 마스크를 분할 모델 글자 근처로 줄이고, 모델 글자 자체는 꼭 넣는다. 모델이 이 글을 못 잡았으면 None.

    색으로 만든 지우기 마스크는 흰 외곽선까지 잡으려고 바탕과 조금만 달라도 넣어, 말풍선 안에 비치는 그림 선·
    집중선까지 지웠다(test04 23쪽·Kamaboko 03쪽 반투명 말풍선의 흰 조각). 모델 글자 마스크는 글자 채움에
    딱 맞고 외곽선은 빠지므로 셋을 남긴다.
    - 모델 글자에서 halo 안: 마스크 그대로
    - 3×halo 안: 바탕보다 밝은 것(흰 외곽선)만. 거리로만 자르면 두꺼운 흰 외곽선이 글자 모양 흰 윤곽으로
      남고, LaMa 는 그 흰 테두리를 보고 글자 자리를 흰색으로 메웠다(test04 '舌が勝手に')
    - 모델 글자 halo 안의 짙은 픽셀(색 마스크가 놓친 획)
    - 모델이 못 잡은 작은 글자·기호(후리가나 'わらわ', ♥): 글자 상자 안의 짙은 조각 중 글자 한 칸 남짓 이하
    restrict=False 면 줄이지 않고 보충만 한다(마스크 ∪ 위의 둘째·셋째 것). region 을 주면 보충은 그 안에서만
    한다(말풍선 안쪽 — 테두리 가까운 글자의 halo 가 말풍선 밖 그림까지 넘지 않게).
    mask·tmask·lum 은 같은 크롭 크기, tbox 는 크롭 좌표의 글자 상자."""
    x1, y1, x2, y2 = (max(0, v) for v in tbox)
    got = int((tmask[y1:y2, x1:x2] > 0).sum())
    had = int((mask[y1:y2, x1:x2] > 0).sum())
    if got < 30 or got < 0.1 * had:
        return None
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))  # noqa: E731
    near = cv2.dilate(tmask, el(halo))
    keep = cv2.bitwise_and(mask, near)
    extra = np.zeros_like(mask)
    if lum is not None and bg_lum is not None:
        if bg_lum >= 128:
            # 흰 외곽선은 밝은 바탕에서만 따진다. 검은 말풍선에서는 모든 픽셀이 '바탕보다 밝아' 말풍선 밖
            # 머리카락까지 지워 검게 칠했다(Tsusauto 106쪽)
            far = cv2.dilate(tmask, el(3 * halo))
            light = ((lum >= bg_lum + 8) | (lum >= 250)).astype(np.uint8) * 255
            keep = cv2.bitwise_or(keep, cv2.bitwise_and(cv2.bitwise_and(mask, far), light))
        dark = (np.abs(lum - bg_lum) > 40).astype(np.uint8)
        # 모델 글자 근처의 짙은 픽셀은 색 마스크에 없어도 지운다. 색 마스크가 글자를 통째로 놓친 곳이 있고
        # (Tsusauto 106쪽 'あっ'), 모델 마스크는 획 끝을 조금 놓친다
        extra = cv2.bitwise_and(dark * 255, near)
        char = halo / 0.2
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        small = np.zeros_like(mask)
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in st[i])
            # 후리가나·하트처럼 모델이 글자로 안 본 기호: 글자 상자 안, 글자 한 칸 남짓 이하
            if (area >= 6 and bx >= x1 and by >= y1 and bx + bw <= x2 and by + bh <= y2
                    and max(bw, bh) <= 1.2 * char and area >= 0.1 * bw * bh
                    and not (tmask[lab == i] > 0).any()):
                small[lab == i] = 255
        if small.any():
            keep = cv2.bitwise_or(keep, cv2.bitwise_and(mask, cv2.dilate(small, el(max(2, halo // 2)))))
    core = cv2.dilate(tmask, el(2))
    lim = np.zeros_like(mask)
    lim[max(0, y1 - halo):y2 + halo, max(0, x1 - halo):x2 + halo] = 255
    extra = cv2.bitwise_and(cv2.bitwise_or(extra, core), lim)
    if region is not None:
        extra = cv2.bitwise_and(extra, region)
    return cv2.bitwise_or(keep if restrict else mask, extra)
