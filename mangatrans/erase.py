"""원문 글자 지우기.

말풍선 안 글자: 사각형을 통째로 칠하지 않고 글자 획 픽셀만 찾아 주변 배경색으로 메운다.
  → 말풍선 테두리가 보존되고, 회색·톤 배경 말풍선도 배경이 그대로 남는다.
그림 위 글자: LaMa 인페인팅.
말풍선 넓히기(옵션): 번역문이 자연스럽게 들어가지 않는 좁은 말풍선은 내부를 가로로 늘리고 테두리를 다시 그린다.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np
from PIL import Image

from .config import Config
from .order import pixel_style_check
from .page import Page, Region
from .skew import line_count, rotated_body, rotated_extent, text_angle
from .textfit import fit_text, natural_width


def _clip(box: list[int], w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def glyph_mask(crop: np.ndarray, tbox: tuple[int, int, int, int], diff: int = 40
               ) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """크롭(말풍선 전체) 안에서 글자 상자 tbox(크롭 좌표) 의 글자 획 마스크를 구한다.
    1) 배경 밝기 = 글자 상자 바로 바깥 띠(말풍선 안쪽)의 중앙값.
    2) 배경과 diff 이상 다른 픽셀(검은 획, 흰 외곽선)이 글자 후보. 주변의 옅은 잔상도 포함.
    3) '글자에 인접한 배경색 연결 성분'을 말풍선 내부로 보고 구멍(글자)을 메운 뒤, 그 안의 글자만 남긴다.
       크롭이 말풍선 전체라 글자가 크롭 가장자리에 닿지 않고, 테두리 선(어두움)과 그 바깥은
       배경 성분에 끼지 못하므로 지워지지 않는다.
    돌려주는 값: (글자 마스크 0/255, 배경 마스크 0/255, 배경 밝기)"""
    lum = _lum(crop)
    h, w = lum.shape
    tx1, ty1, tx2, ty2 = tbox
    # 배경 밝기: 글자 상자의 안쪽 가장자리 띠(2~10px 안쪽)에서 잰다. 제목 상자처럼 글자 상자와
    # 말풍선 테두리가 붙어 있어도 바깥 여백을 재지 않는다. 글자가 띠에 조금 걸쳐도 중앙값이라 견딘다.
    ring = np.zeros((h, w), bool)
    ix1, iy1, ix2, iy2 = max(0, tx1 + 2), max(0, ty1 + 2), min(w, tx2 - 2), min(h, ty2 - 2)
    ring[iy1:iy2, ix1:ix2] = True
    ring[max(0, ty1 + 10):max(0, ty2 - 10), max(0, tx1 + 10):max(0, tx2 - 10)] = False
    if ring.sum() < 50:
        ring[max(0, ty1):min(h, ty2), max(0, tx1):min(w, tx2)] = True
    ring_vals = lum[ring]
    bg_lum = float(np.median(ring_vals))
    q1, q3 = np.percentile(ring_vals, [25, 75])
    # 사선·들쭉날쭉한 말풍선은 축 정렬 글자 상자가 말풍선 밖까지 걸쳐서, 배경을 한 값으로 잡을 수 없다.
    # (띠의 사분위 범위가 넓다 = 밝기가 두 가지 이상 섞여 있다) 그때는 국소 배경(중앙값 흐리기)과 비교한다.
    mixed = (q3 - q1) > 35
    if mixed:
        k = int(np.clip(min(tx2 - tx1, ty2 - ty1) * 0.5, 21, 81)) | 1
        local = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), k).astype(np.float32)
        strict = (np.abs(lum - local) > diff).astype(np.uint8) * 255
        strict = _drop_line_parts(strict)   # 말풍선 테두리처럼 가늘고 긴 선은 글자가 아니다
    else:
        strict = (np.abs(lum - bg_lum) > diff).astype(np.uint8) * 255
    glyph = cv2.dilate(strict, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    near = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    loose = ((np.abs(lum - bg_lum) > 12).astype(np.uint8) * 255) & near
    glyph = cv2.bitwise_or(glyph, loose)

    # 글자 상자(+여백) 안만 지운다. 같은 말풍선의 다른 글자(효과음 등)는 건드리지 않는다.
    limit = np.zeros((h, w), np.uint8)
    limit[max(0, ty1 - 6):min(h, ty2 + 6), max(0, tx1 - 6):min(w, tx2 + 6)] = 255
    glyph = cv2.bitwise_and(glyph, limit)

    if mixed:
        # 배경 연결 영역을 못 믿으므로 글자 상자로만 제한하고,
        # 채울 색은 글자 바로 둘레에서만 뽑는다 (말풍선 밖 색이 섞여 얼룩지지 않게).
        inside = limit
        halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        glyph = cv2.bitwise_and(glyph, inside)
        bg = cv2.bitwise_and(halo, cv2.bitwise_not(glyph))
    else:
        inside = _bubble_region(lum, bg_lum, glyph, min(diff, 30), tbox)
        glyph = cv2.bitwise_and(glyph, inside)
        bg = cv2.bitwise_and(cv2.bitwise_not(glyph), inside)
    if not bg.any():
        bg = cv2.bitwise_not(glyph)
    return glyph, bg, bg_lum, mixed


def _label_writing(r: Region, cfg: Config) -> str:
    """원문이 세로 한 열인 라벨(세로 제목, 세로 소개 문구)은 번역문도 세로로 그린다.
    가로로 바꾸면 긴 세로 자리에 짧은 가로줄이 떠 원본 배치와 달라진다(196쪽 제목)."""
    mode = cfg.render.label_vertical
    if mode == "off" or r.category != "label" or r.kind != "free_text" or not r.render:
        return "auto"
    vertical, lines = line_count(r.box, r.text_ja)
    if not vertical or lines >= 1.6:
        return "auto"
    return "sideways" if mode == "sideways" else "vertical"


def _art_text_mask(gray: np.ndarray, box: list[int], dilate: int,
                   clip: tuple[int, int, int, int]) -> np.ndarray:
    """그림 위 글자를 LaMa 로 지울 마스크(clip 범위 크기). 상자 전체가 아니라 글자 획 둘레만.

    상자를 통째로 지우면 큰 글자(세로 제목 238×988)에서 메울 면적이 너무 커져 LaMa 가 회색으로
    뭉갠다(실측: 196쪽 제목 자리에 코트·몸을 덮는 얼룩). 획(검은 글씨와 흰 외곽선 모두)을 찾아
    글자 크기의 1/4 정도로 닫아 획 속과 글자 안 틈을 메우고, 조금 넓힌다. 획을 거의 못 찾거나
    마스크가 상자 대부분을 덮으면 예전처럼 상자 전체를 쓴다."""
    from .skew import stroke_mask
    x1, y1, x2, y2 = clip
    full = np.full((y2 - y1, x2 - x1), 255, np.uint8)
    bx1, by1, bx2, by2 = box
    pad = bx1 - x1 if bx1 - x1 == by1 - y1 else dilate
    m = stroke_mask(gray, box, pad=pad)
    if m is None or m.shape != full.shape:
        return full
    k = max(9, int(min(bx2 - bx1, by2 - by1) * 0.25)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    cover = float((m > 0).mean())
    if cover < 0.05 or cover > 0.85:
        return full
    return m


def _drop_line_parts(mask: np.ndarray, long_ratio: float = 3.0, fill_max: float = 0.22) -> np.ndarray:
    """글자 덩어리 크기의 중앙값보다 훨씬 길면서 속이 빈 성분(테두리 선, 그림 윤곽)을 마스크에서 뺀다."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    dims = [max(stats[i, 2], stats[i, 3]) for i in range(1, n) if stats[i, 4] >= 20]
    if len(dims) < 2:
        return mask
    char = float(np.median(dims))
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 12:
            continue
        long_side = max(bw, bh)
        fill = area / max(1.0, float(bw * bh))
        if long_side > long_ratio * char and fill < fill_max:
            out[labels == i] = 0
    return out


def _bubble_region(lum: np.ndarray, bg_lum: float, glyph: np.ndarray, diff: int,
                   tbox: tuple[int, int, int, int]) -> np.ndarray:
    """글자에 인접한 배경색 픽셀에서 이어지는 연결 성분들(= 말풍선 내부)을 잡고 구멍을 메운 마스크.
    닫힘 연산을 쓰지 않으므로 얇은 테두리 선을 넘어 바깥으로 번지지 않는다."""
    h, w = lum.shape
    like = (np.abs(lum - bg_lum) <= diff).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(like, connectivity=4)
    if n <= 1:
        return np.full(lum.shape, 255, np.uint8)
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    seed = (halo > 0) & (glyph == 0) & (like > 0)
    ids = np.unique(labels[seed])
    ids = ids[ids != 0]
    if len(ids) == 0:
        ids = np.array([1 + int(np.argmax(np.bincount(labels.ravel())[1:]))])
    region = np.isin(labels, ids).astype(np.uint8) * 255
    inv = cv2.bitwise_not(region)
    ff = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # '바깥' 시작점은 글자 상자 범위 밖의 가장자리 픽셀만. 글자 상자 범위 안의 가장자리에 닿은 것은
    # 글자(외곽선)일 수 있으므로 바깥으로 보지 않는다.
    tx1, ty1, tx2, ty2 = tbox
    border = [(x, 0) for x in range(w) if not (tx1 - 2 <= x <= tx2 + 2)] +              [(x, h - 1) for x in range(w) if not (tx1 - 2 <= x <= tx2 + 2)] +              [(0, y) for y in range(h) if not (ty1 - 2 <= y <= ty2 + 2)] +              [(w - 1, y) for y in range(h) if not (ty1 - 2 <= y <= ty2 + 2)]
    if not border:
        border = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in border:
        if ff[sy, sx] == 255:
            cv2.floodFill(ff, mask, (sx, sy), 0)
    region = cv2.bitwise_or(region, ff)          # 바깥과 이어지지 않은 구멍(글자)을 채운다
    return cv2.erode(region, np.ones((3, 3), np.uint8))   # 테두리 선 자체는 건드리지 않게 1px 안쪽


def erase_bubble_text(img: np.ndarray, r: Region, pad: int, mask_out: np.ndarray | None = None,
                      lama_mask: np.ndarray | None = None) -> None:
    """img(RGB, 제자리 수정)에서 r.box 안의 글자만 지운다. r.text_color 를 정한다.
    크롭은 말풍선 박스(bubble_box)와 글자 상자를 합친 영역이다."""
    h, w = img.shape[:2]
    if r.bubble_box:
        bb = r.bubble_box
        ux = [min(bb[0], r.box[0]), min(bb[1], r.box[1]), max(bb[2], r.box[2]), max(bb[3], r.box[3])]
        x1, y1, x2, y2 = _clip(ux, w, h, 4)
    else:
        x1, y1, x2, y2 = _clip(r.box, w, h, max(pad, 12))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return
    tbox = (r.box[0] - x1, r.box[1] - y1, r.box[2] - x1, r.box[3] - y1)
    glyph, bg, _, mixed = glyph_mask(crop, tbox)
    bg_px = crop[bg > 0]
    if len(bg_px) == 0:
        return
    med = np.median(bg_px, axis=0)
    lum = float(0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2])
    # 밝은 배경: 검은 글자 / 어두운 배경: 흰 글자 / 중간 회색: 원본 관례대로 검은 글자 + 흰 외곽선
    r.text_color = "black" if lum >= 170 else ("white" if lum < 60 else "outline")
    if mask_out is not None:
        mask_out[y1:y2, x1:x2] = np.maximum(mask_out[y1:y2, x1:x2], glyph)
        return
    if mixed and lama_mask is not None:
        # 사선 말풍선처럼 배경이 섞인 경우: 주변 색으로 메우면 얼룩지므로 LaMa 로 넘긴다
        lama_mask[y1:y2, x1:x2] = np.maximum(lama_mask[y1:y2, x1:x2], glyph)
        return
    # 배경 픽셀만으로 정규화 블러를 해서 글자 자리의 배경색을 추정한다.
    bgm = (bg > 0).astype(np.float32)
    sigma = max(6.0, min(r.box[2] - r.box[0], r.box[3] - r.box[1]) / 10)
    num = cv2.GaussianBlur(crop.astype(np.float32) * bgm[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(bgm, (0, 0), sigma)
    est = num / np.maximum(den, 1e-3)[..., None]
    est[den < 0.05] = med
    alpha = cv2.GaussianBlur((glyph > 0).astype(np.float32), (0, 0), 1.5)[..., None]
    alpha = np.maximum(alpha, (glyph > 0)[..., None].astype(np.float32))
    blended = crop.astype(np.float32) * (1 - alpha) + est * alpha
    crop[:] = np.clip(blended, 0, 255).astype(np.uint8)


def measure_text_style(img: np.ndarray, r: Region, bold_ratio: float) -> None:
    """원문 글자의 획 색·외곽선 색·굵기를 측정해 r.text_rgb / r.outline_rgb / r.weight 에 넣는다.
    글자 상자(+6px) 크롭에서 배경(안쪽 가장자리 띠 중앙값)과 다른 픽셀을 글자 후보로 잡고,
    3x3 침식으로 얇은 외곽선을 걷어낸 '획' 픽셀의 중앙값 색을 획 색으로, 나머지를 외곽선 색으로 본다."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip(r.box, w, h, 6)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return
    lum = _lum(crop)
    ch, cw = lum.shape
    ring = np.zeros((ch, cw), bool)
    ring[6:ch - 6, 6:cw - 6] = True
    ring[14:max(14, ch - 14), 14:max(14, cw - 14)] = False
    if ring.sum() < 50:
        ring[:] = True
    bg_lum = float(np.median(lum[ring]))
    cand = (np.abs(lum - bg_lum) > 40).astype(np.uint8)
    # 가장자리에 닿은 길고 성긴 성분(테두리 선)은 제외
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    sizes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= cw or y + bh >= ch
        if touches and (bw >= 0.85 * cw or bh >= 0.85 * ch) and area < 0.3 * bw * bh:
            cand[labels == i] = 0
        elif area >= 12:
            sizes.append(max(bw, bh))
    if cand.sum() < 20:
        return
    stroke = cv2.erode(cand, np.ones((3, 3), np.uint8))
    if stroke.sum() < 10:
        stroke = cand
    px = crop[stroke > 0].astype(np.float32)
    col = np.median(px, axis=0)
    r.text_rgb = _snap_color(col)
    # 외곽선: 획 주변에 있으면서 획과 밝기가 크게 다른 후보 픽셀
    around = cv2.bitwise_and(cand, cv2.bitwise_not(cv2.dilate(stroke, np.ones((3, 3), np.uint8))))
    if around.sum() >= 0.15 * cand.sum():
        opx = crop[around > 0].astype(np.float32)
        ocol = np.median(opx, axis=0)
        snapped = _snap_color(ocol)
        # 획 둘레 한 겹은 안티앨리어싱이라 늘 회색으로 나온다. 순백·순흑이나 유채색일 때만
        # 진짜 외곽선으로 인정한다.
        deliberate = snapped in ([255, 255, 255], [0, 0, 0]) or max(snapped) - min(snapped) > 40
        if deliberate and abs(_lum(ocol[None, None, :])[0, 0] - _lum(col[None, None, :])[0, 0]) > 80:
            r.outline_rgb = snapped
    # 굵기: 획 폭(거리변환 상위값의 2배) / 글자 크기(성분 크기 중앙값)
    if sizes:
        dist = cv2.distanceTransform(stroke, cv2.DIST_L2, 3)
        vals = dist[dist > 0]
        if len(vals):
            stroke_w = 2.0 * float(np.percentile(vals, 90))
            char = float(np.median(sizes))
            if char > 0:
                r.weight = "bold" if stroke_w / char >= bold_ratio else "regular"


def _snap_color(col: np.ndarray) -> list[int]:
    """거의 검정/흰색은 순수 검정/흰색으로 스냅하고, 나머지는 측정값 그대로."""
    c = [int(round(float(v))) for v in col]
    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    sat = max(c) - min(c)
    if sat < 40 and lum < 70:
        return [0, 0, 0]
    if sat < 40 and lum > 200:
        return [255, 255, 255]
    return [max(0, min(255, v)) for v in c]


def bubble_interior(gray: np.ndarray, r: Region, tol: int, margin: float = 0.06) -> np.ndarray | None:
    """글자 박스 중심에서 flood-fill 해 말풍선 내부(밝은 영역) 마스크를 구한다.

    fill 범위는 말풍선 박스를 margin 만큼만 넘도록 잘라 둔다. 페이지 전체를 대상으로 하면
    말풍선 둘이 맞붙어 있거나 테두리가 열린 곳이 있을 때 옆 말풍선과 컷 배경까지 한 덩어리로
    번져서(실측: 257×368 말풍선이 447×761 로 번짐) 내부 판정이 통째로 실패한다.
    그러면 body_box 를 못 재고 탐지 박스로 물러나므로 글자가 말풍선 중앙에서 벗어난다.
    말풍선 박스의 1.3배를 넘으면 실패로 본다."""
    H, W = gray.shape
    bb = r.bubble_box or r.box
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    mx, my = int(bw * margin) + 4, int(bh * margin) + 4
    x0, y0 = max(0, bb[0] - mx), max(0, bb[1] - my)
    x1, y1 = min(W, bb[2] + mx), min(H, bb[3] + my)
    sub = gray[y0:y1, x0:x1]
    h, w = sub.shape
    if h < 8 or w < 8:
        return None
    # 검은 바탕 말풍선(흰 글씨)은 밝기를 뒤집어 '밝은 내부'로 만든다. 그대로 두면 글자 상자 모서리에
    # 걸친 말풍선 바깥의 흰 종이에서 채우기를 시작해 바깥 여백을 내부로 잡는다(실측: 196쪽 검은
    # 말풍선이 오른쪽 아래 모서리 93×104 조각으로 잡혀 글자가 밖에서 15px 로 그려짐).
    # 지운 뒤라 글자 상자 안은 거의 말풍선 바탕색이다.
    tb = gray[max(0, r.box[1]):r.box[3], max(0, r.box[0]):r.box[2]]
    if tb.size and float(np.median(tb)) < 100:
        sub = 255 - sub
    cx, cy = (r.box[0] + r.box[2]) // 2 - x0, (r.box[1] + r.box[3]) // 2 - y0
    cx, cy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    if sub[cy, cx] < 160:  # 글자 위일 수 있으니 박스 안에서 밝은 점을 찾는다
        gx1, gy1 = max(0, r.box[0] - x0), max(0, r.box[1] - y0)
        gx2, gy2 = min(w, r.box[2] - x0), min(h, r.box[3] - y0)
        if gx2 <= gx1 or gy2 <= gy1:
            return None
        ys, xs = np.where(sub[gy1:gy2, gx1:gx2] >= 200)
        if len(xs) == 0:
            return None
        k = len(xs) // 2
        cx, cy = gx1 + int(xs[k]), gy1 + int(ys[k])
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(sub.copy(), mask, (cx, cy), 255, tol, tol,
                  cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
    m = mask[1:-1, 1:-1]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    if (xs.max() - xs.min()) > 1.3 * bw or (ys.max() - ys.min()) > 1.3 * bh:
        return None
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = m
    return out


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """True 가 가장 길게 이어지는 구간 [시작, 끝](양끝 포함). 하나도 없으면 None."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return None
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [len(idx) - 1]))
    k = int(np.argmax(ends - starts))
    return int(idx[starts[k]]), int(idx[ends[k]])


def bubble_body(mask: np.ndarray, ratio: float = 0.4) -> tuple[int, int, int, int] | None:
    """말풍선 내부 마스크에서 뿔·꼬리를 뺀 '본체' 사각형 (x1, y1, x2, y2).

    flood fill 은 폭발형 말풍선의 뾰족한 뿔이나 아래로 뻗은 꼬리를 타고 새어 나간다. 그 마스크의
    bounding box 를 그대로 글자 상자로 쓰면 상자 중심이 본체 중심에서 벗어나(뿔이 위로 뻗으면 위로)
    글자가 한쪽으로 치우쳐 말풍선 밖으로 삐져나온다. 폭이 최대의 ratio 에 못 미치는 줄을 가장자리에서
    잘라 본체만 남긴다. 가운데가 끊긴 마스크를 대비해 '가장 긴 연속 구간'을 쓴다."""
    rows = (mask > 0).sum(axis=1)
    if not rows.any():
        return None
    span = _longest_run(rows >= rows.max() * ratio)
    if span is None:
        return None
    y1, y2 = span
    cols = (mask[y1:y2 + 1] > 0).sum(axis=0)
    if not cols.any():
        return None
    span = _longest_run(cols >= cols.max() * ratio)
    if span is None:
        return None
    x1, x2 = span
    return x1, y1, x2, y2


def inner_rect(mask: np.ndarray, body: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """본체 안에서 글자를 놓아도 말풍선 밖으로 나가지 않는 사각형.

    말풍선은 둥근데 글자 상자는 사각형이라, 본체 사각형에서 여백 비율만 빼면 위·아래 줄의 양 끝이
    말풍선을 뚫고 나간다(폭 84%·높이 84% 는 타원 조건 (w/W)²+(h/H)² ≤ 1 을 크게 어긴다).
    세로 중앙에서 위아래로 똑같이 넓혀 가며 그 띠에 공통으로 들어가는 폭을 잰다. 세로 중앙을
    유지하므로 글자가 말풍선 한쪽으로 쏠리지 않는다."""
    x1, y1, x2, y2 = body
    sub = mask[y1:y2 + 1, x1:x2 + 1] > 0
    h, w = sub.shape
    lefts = np.full(h, w, np.int32)
    rights = np.full(h, -1, np.int32)
    any_row = sub.any(axis=1)
    lefts[any_row] = sub.argmax(axis=1)[any_row]
    rights[any_row] = w - 1 - sub[:, ::-1].argmax(axis=1)[any_row]
    cy = h // 2
    # 고를 기준은 넓이가 아니라 '두 변 중 더 많이 깎인 쪽'이다. 넓이로 고르면 납작한 말풍선
    # (채팅 버블 등)에서 폭을 다 쓰는 대신 높이를 절반까지 깎아 버려 글자가 위아래로 넘친다.
    best_score, best = 0.0, None
    for half in range(1, h // 2 + 1):
        a, b = cy - half, cy + half
        if a < 0 or b >= h:
            break
        # 최댓값/최솟값 대신 퍼센타일을 쓴다. 말풍선에 인물이나 집중선이 걸쳐 한 줄만 안쪽으로
        # 파여도 max/min 을 쓰면 상자 전체가 그만큼 좁아져(실측: 584px 말풍선의 왼쪽 153px 손실)
        # 글자가 반대쪽으로 쏠린다. 그런 줄에서 글자가 살짝 걸치는 것은 bubble_fit 이 감당한다.
        left = int(np.percentile(lefts[a:b + 1], 85))
        right = int(np.percentile(rights[a:b + 1], 15))
        if right - left < 8:
            break                                       # 더 넓히면 폭이 무너진다
        score = min((right - left + 1) / w, (b - a + 1) / h)
        if score > best_score:
            best_score, best = score, (x1 + left, y1 + a, x1 + right, y1 + b)
    return best or body


def _blend(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int],
           fit: float) -> list[int]:
    """내접 사각형과 본체 사각형 사이를 fit 비율로 섞는다.

    완전 내접(fit=1)은 글자가 말풍선을 절대 안 벗어나지만 상자가 좁아 글자도 작아진다.
    말풍선을 늘려 고치기보다 글자가 곡선에 조금 걸치는 편이 원본을 덜 건드린다."""
    f = max(0.0, min(1.0, fit))
    return [int(round(i * f + o * (1 - f))) for i, o in zip(inner, outer)]


def _rect_like(mask: np.ndarray, fill: float = 0.9) -> bool:
    """말풍선이 직사각형에 가까운가 (타원은 0.785 근처, 직사각형은 1에 가깝다).

    반드시 마스크 '전체 범위'를 기준으로 재야 한다. 본체 사각형(bubble_body) 안에서 재면 이미
    좁은 가장자리를 잘라낸 뒤라 둥근 말풍선도 0.93 쯤으로 꽉 차 보여 사각형으로 잘못 판정된다."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return len(xs) >= fill * max(1, area)


def _spiky(mask: np.ndarray, body: tuple[int, int, int, int], ratio: float = 0.22) -> bool:
    """뾰족한 뿔이 뻗은 폭발형 말풍선인가. 마스크 전체 범위와 본체 사각형이 어느 한 변에서
    그 축 길이의 ratio 넘게 차이 나면 뿔로 본다 (평범한 말풍선 꼬리는 이보다 짧다)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    w, h = max(1, bx2 - bx1), max(1, by2 - by1)
    return (max(body[0] - bx1, bx2 - body[2]) > w * ratio
            or max(body[1] - by1, by2 - body[3]) > h * ratio)


def _straight_sides(interior: np.ndarray, ix1: int, ix2: int, iy1: int, iy2: int) -> tuple[bool, bool]:
    """내부 마스크의 왼쪽/오른쪽 경계가 세로 직선(컷 테두리에 잘림)인지. 경계 x가 최솟값/최댓값 2px 안에
    머무는 행이 높이의 45% 이상이면 직선으로 본다."""
    rows = interior[iy1:iy2 + 1]
    if rows.shape[0] < 10:
        return False, False
    left, right = [], []
    for row in rows:
        xs = np.where(row > 0)[0]
        if len(xs):
            left.append(xs[0]); right.append(xs[-1])
    if not left:
        return False, False
    left, right = np.array(left), np.array(right)
    l_straight = (np.abs(left - left.min()) <= 2).mean() >= 0.45
    r_straight = (np.abs(right - right.max()) <= 2).mean() >= 0.45
    return bool(l_straight), bool(r_straight)


def outline_thickness(gray: np.ndarray, interior: np.ndarray, max_t: int = 10) -> int:
    """말풍선 내부 마스크 바깥으로 1px씩 띠를 넓히며 어두운 픽셀 비율이 절반 이상인 동안을
    테두리 두께로 본다."""
    prev = interior
    t = 0
    for k in range(1, max_t + 1):
        cur = cv2.dilate(interior, np.ones((3, 3), np.uint8), iterations=k)
        band = cv2.bitwise_and(cur, cv2.bitwise_not(prev))
        px = gray[band > 0]
        if len(px) == 0 or (px < 128).mean() < 0.5:
            break
        t = k
        prev = cur
    return max(2, t)


class Eraser:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lama = None

    def _lama_model(self):
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama

            self._lama = SimpleLama()
        return self._lama

    def clean(self, image: Image.Image, page: Page) -> Image.Image:
        """모든 render 대상 영역의 원문을 지운 이미지를 돌려준다."""
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]
        lama_mask = np.zeros((h, w), np.uint8)
        e = self.cfg.erase

        # 기울기는 지우기 전 원본에서 잰다 (지우는 도중에는 이웃 글자가 이미 사라져 있을 수 있다)
        orig_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pixel_style_check(orig_gray, page)        # 획이 고른 글자는 모델 판정과 달라도 인쇄체로
        for r in page.regions:
            r.angle, r.rot_box = 0.0, None
            r.writing = _label_writing(r, self.cfg)
            # 기울기는 글줄에서 나온다. 기호를 뺀 글자가 3자 미만이면('ん♡') 글줄이라 할 수 없다
            if (r.render and r.erase != "none" and self.cfg.render.follow_angle
                    and len(re.findall(r"\w", r.text_ja)) >= 3):
                try:
                    r.angle = text_angle(orig_gray, r.box)
                except Exception as ex:  # noqa: BLE001
                    page.warnings.append(f"기울기 측정 실패 (id={r.id}): {ex}")

        for r in page.regions:
            r.body_box, r.target_box, r.widened = None, None, False
            r.text_rgb, r.outline_rgb, r.weight = None, None, "regular"
            if not r.render or r.erase == "none":
                continue
            if self.cfg.render.match_color or self.cfg.render.match_style:
                try:
                    measure_text_style(img, r, self.cfg.render.bold_stroke_ratio)
                except Exception as ex:  # noqa: BLE001
                    page.warnings.append(f"스타일 측정 실패 (id={r.id}): {ex}")
            if r.erase == "white":
                erase_bubble_text(img, r, e.bubble_padding,
                                  lama_mask=lama_mask if e.lama else None)
            elif r.erase == "lama":
                x1, y1, x2, y2 = _clip(r.box, w, h, e.lama_dilate)
                lama_mask[y1:y2, x1:x2] = np.maximum(lama_mask[y1:y2, x1:x2],
                                                     _art_text_mask(orig_gray, r.box, e.lama_dilate, (x1, y1, x2, y2)))

        if lama_mask.any() and e.lama:
            try:
                out = self._lama_model()(Image.fromarray(img), Image.fromarray(lama_mask))
                out = np.array(out.convert("RGB"))
                if out.shape != img.shape:
                    out = cv2.resize(out, (w, h))
                img = out
            except Exception as ex:  # noqa: BLE001
                page.warnings.append(f"LaMa 인페인팅 실패, 흰색으로 대체: {ex}")
                img[lama_mask > 0] = 255
        elif lama_mask.any():
            img[lama_mask > 0] = 255

        # 말풍선 본체를 재서 글자 상자의 기준으로 삼는다. 넓히기를 끄더라도 이건 필요하다
        # (탐지기가 준 bubble_box 는 꼬리·뿔까지 감싸서 중심이 본체와 어긋난다).
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tilted: dict[int, tuple[float, float, float, float]] = {}     # id(r) → 세운 말풍선 상자
        for r in page.regions:
            if not (r.render and r.erase == "white" and r.kind == "bubble_text"):
                continue
            try:
                interior = bubble_interior(gray, r, e.flood_tolerance)
                body = self._measure_body(r, interior)
                if body and e.widen_bubbles and r.text_ko.strip():
                    self._widen(img, gray, r, page, interior, body)
                if r.angle and interior is not None and r.body_box and not r.widened:
                    center = ((r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2)
                    rb = rotated_body(interior, center, r.angle, self.cfg.render.bubble_fit,
                                      self.cfg.render.bubble_inner_margin / 2)
                    if rb:
                        tilted[id(r)] = rb
            except Exception as ex:  # noqa: BLE001
                page.warnings.append(f"말풍선 넓히기 실패 (id={r.id}): {ex}")
        assign_target_boxes(page, self.cfg)
        self._assign_rot_boxes(page, orig_gray, tilted)
        return Image.fromarray(img)

    def _assign_rot_boxes(self, page: Page, gray: np.ndarray,
                          tilted: dict[int, tuple[float, float, float, float]]) -> None:
        """기운 글자의 배치 상자(세운 좌표)를 정한다. 못 정하면 각도를 0 으로 돌려 똑바로 그린다.

        말풍선 안: 말풍선을 글자 각도만큼 세워서 잰 본체. 단, 겹침 처리(_stack_overlapping)가
        배치 상자를 잘랐다면 한 말풍선을 여러 영역이 나눠 쓰는 경우라 세운 본체를 통째로 쓸 수 없다.
        말풍선 밖: 기운 글자 덩어리를 세운 크기에, 이웃을 보고 넓혀 준 폭(target_box - box)을 더한다."""
        rc = self.cfg.render
        for r in page.regions:
            if not r.angle:
                continue
            if not r.render or not r.target_box or r.widened:
                r.angle = 0.0
                continue
            if r.kind == "bubble_text" and (r.body_box or r.bubble_box):
                rb = tilted.get(id(r))
                untouched = r.target_box == _bubble_inner_box(r, rc.bubble_inner_margin, page.width, page.height)
                if rb and untouched:
                    r.rot_box = [round(v, 1) for v in rb]
                else:
                    r.angle = 0.0
                continue
            ext = rotated_extent(gray, r.box, r.angle)
            if not ext:
                r.angle = 0.0
                continue
            if r.writing != "auto":      # 세로 라벨: 세운 글자 덩어리 크기 그대로 (넓히지 않으니 이웃과 안 겹친다)
                r.rot_box = [round(v, 1) for v in ext]
                continue
            fit = _rotated_fit(r.target_box, ext[2] + (r.target_box[2] - r.target_box[0]) - (r.box[2] - r.box[0]),
                               ext[3], r.angle)
            if fit is None:
                r.angle = 0.0
                continue
            r.rot_box = [round(v, 1) for v in fit]

    def _measure_body(self, r: Region, interior: np.ndarray | None) -> tuple[int, int, int, int] | None:
        """말풍선 내부 마스크에서 본체 사각형을 재어 r.body_box 에 넣는다.

        flood fill 이 글자 획 안에 갇히거나(아주 작은 마스크) 말풍선 밖으로 크게 새면 믿을 수 없으므로
        글자 박스 넓이의 절반은 넘어야 인정한다."""
        r.body_box = None
        if interior is None:
            return None
        area = int((interior > 0).sum())
        box_area = max(1, (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
        if area < 0.5 * box_area:
            return None                                # 글자 박스보다 작은 영역이면 말풍선 내부가 아니다
        body = bubble_body(interior)
        if body is None or body[2] - body[0] < 10 or body[3] - body[1] < 10:
            return None
        # 본체는 원문 글자를 품고 있어야 한다. 글자 상자와 거의 안 겹치면 엉뚱한 곳(말풍선 바깥 여백,
        # 옆 칸)을 잰 것이므로 버리고 탐지기의 말풍선 상자로 물러난다.
        bx1, by1, bx2, by2 = r.box
        ov = max(0, min(bx2, body[2]) - max(bx1, body[0])) * max(0, min(by2, body[3]) - max(by1, body[1]))
        if ov < 0.5 * box_area:
            return None
        r.body_box = _blend(inner_rect(interior, body), body, self.cfg.render.bubble_fit)
        return body

    def _widen(self, img: np.ndarray, gray: np.ndarray, r: Region, page: Page,
               interior: np.ndarray, body: tuple[int, int, int, int]) -> None:
        e, rc = self.cfg.erase, self.cfg.render
        font_path = str(self.cfg.abs(self.cfg.paths.font))
        base = max(8, int(page.height * rc.font_ratio))
        minimum = max(6, int(page.height * rc.min_font_ratio))
        # 지금 상자로도 무리 없이 들어가면(크게 줄이지 않고, 넘치지 않고) 넓히지 않는다.
        # 기준 상자는 방금 잰 body_box 를 반영한 값이라 실제 말풍선 안쪽과 맞다.
        tb = region_target_box(r, self.cfg, page.width, page.height)
        size, lines, overflow = fit_text(r.text_ko, font_path, max(10, tb[2] - tb[0]),
                                         max(10, tb[3] - tb[1]), base, minimum, rc.line_spacing)
        if not overflow and size >= base * e.widen_min_font_ratio and len(lines) <= e.widen_max_lines:
            return
        if r.text_color != "black":
            return                                     # 회색·검은 상자는 넓히지 않는다 (내부 검출이 불안정)
        # 뿔이 뻗은 폭발형 말풍선은 넓히지 않는다. 가로로 늘이면 뿔이 뭉개져 원래 모양을 잃고,
        # 흐림·이진화에서 얇은 뿔이 사라지면 옛 윤곽선만 남아 테두리가 두 겹으로 보인다.
        if _spiky(interior, body):
            return
        # 확대·윤곽선 그리기는 마스크 '전체'를 기준으로 한다. 본체만 늘이면 꼬리가 따로 놀아
        # 모양이 깨진다. 본체는 글자 상자를 정할 때만 쓴다.
        ys, xs = np.where(interior > 0)
        ix1, ix2, iy1, iy2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cur_w = ix2 - ix1
        margin = int(cur_w * rc.bubble_inner_margin)
        usable = cur_w - 2 * margin
        need = natural_width(r.text_ko, font_path, base, e.widen_max_lines) + 2 * margin + base // 2
        if need <= usable:
            return
        extra = min(need - usable, int(cur_w * e.widen_max_ratio))
        if extra < 4:
            return
        # 내부 마스크를 가로만 확대(affine)해서 둥근 모양을 유지한다.
        # 컷 테두리에 잘려 한쪽이 직선인 말풍선은 그 쪽을 고정하고 반대쪽으로만 넓힌다.
        clip_l, clip_r = _straight_sides(interior, ix1, ix2, iy1, iy2)
        if clip_l and clip_r:
            return
        sx = (cur_w + extra) / max(1, cur_w)
        anchor = float(ix1) if clip_l else (float(ix2) if clip_r else (ix1 + ix2) / 2)
        # 마스크를 픽셀째로 확대하면 계단이 남아 말풍선이 다각형처럼 각져 보인다(블러로도 덜 지워진다).
        # 대신 윤곽선(점 목록) 자체를 가로로 늘인다. 원래 곡선과 꼬리 모양이 그대로 유지된다.
        contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return
        pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        pts[:, 0] = np.clip(anchor + (pts[:, 0] - anchor) * sx, 0, img.shape[1] - 1)
        poly = np.round(pts).astype(np.int32)
        new_int = np.zeros_like(interior)
        cv2.fillPoly(new_int, [poly], 255)
        # 측정값은 안티앨리어싱 띠까지 포함해 실제보다 두꺼우므로 조금 줄인다
        t = max(2, min(int(round(outline_thickness(gray, interior) * 0.75)), int(page.height * 0.003)))
        bg = np.median(img[interior > 0], axis=0).astype(np.uint8)
        line = (255, 255, 255) if r.text_color == "white" else (0, 0, 0)
        # 옛 테두리까지 덮도록 '새 내부 ∪ 옛 내부'를 두께만큼 넓힌 영역을 배경색으로 칠한 뒤 다시 그린다.
        # 옛 내부를 빼면 넓히지 않은 쪽(뿔·꼬리)의 옛 윤곽선이 남아 테두리가 두 겹으로 보인다.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 3, 2 * t + 3))
        cover = cv2.dilate(np.maximum(new_int, interior), kernel)
        img[cover > 0] = bg
        cv2.polylines(img, [poly], True, line, thickness=t,
                      lineType=cv2.LINE_8 if _rect_like(interior) else cv2.LINE_AA)
        nbody = bubble_body(new_int)
        if nbody is None:
            return
        r.body_box = _blend(inner_rect(new_int, nbody), nbody, rc.bubble_fit)
        r.target_box = _text_box(r, rc.bubble_inner_margin)
        r.widened = True


def assign_target_boxes(page: Page, cfg: Config, margin: int = 10) -> None:
    """그릴 영역의 글자 배치 상자를 서로 겹치지 않게 정한다.

    말풍선 밖 세로 글자는 가로쓰기로 바꾸려면 상자를 옆으로 넓혀야 하는데, 이웃 글자를 보지 않고
    넓히면 서로 침범해 글자가 겹쳐 보인다. 이웃(모든 영역의 원본 상자)과 페이지 경계까지 남은
    공간만큼만 넓히고, 이웃도 넓어질 영역이면 그 사이 공간을 절반씩 나눠 갖는다."""
    rc = cfg.render
    fixed: list[list[int]] = []          # 이미 확정된 배치 상자 (넓힌 말풍선 등)
    expanding: list[Region] = []
    for r in page.regions:
        if not r.render:
            continue
        if r.target_box:                 # 말풍선 넓히기가 이미 정한 것은 그대로 둔다
            fixed.append(list(r.target_box))
        elif r.writing != "auto":        # 세로로 그리는 라벨은 원문 자리 그대로 (옆으로 넓히지 않는다)
            r.target_box = [max(0, r.box[0]), max(0, r.box[1]), min(page.width, r.box[2]), min(page.height, r.box[3])]
            fixed.append(list(r.target_box))
        elif (r.body_box or r.bubble_box) and r.kind == "bubble_text":
            r.target_box = _bubble_inner_box(r, rc.bubble_inner_margin, page.width, page.height)
            fixed.append(list(r.target_box))
        else:
            expanding.append(r)

    grows = {id(r) for r in expanding}
    for r in expanding:
        x1, y1, x2, y2 = r.box
        want = max(0, int((y2 - y1) * 0.6) - (x2 - x1)) // 2
        left = right = want
        for o in page.regions:
            if o is r:
                continue
            ox1, oy1, ox2, oy2 = o.box
            if oy2 <= y1 or oy1 >= y2:          # 세로로 안 겹치면 신경 쓸 필요 없다
                continue
            share = 2 if id(o) in grows else 1  # 이웃도 넓어지면 사이 공간을 반씩
            if ox2 <= x1:
                left = min(left, max(0, (x1 - ox2 - margin) // share))
            elif ox1 >= x2:
                right = min(right, max(0, (ox1 - x2 - margin) // share))
            else:
                left = right = 0                # 이미 가로로 겹쳐 있으면 넓히지 않는다
        for fx1, fy1, fx2, fy2 in fixed:
            if fy2 <= y1 or fy1 >= y2:
                continue
            if fx2 <= x1:
                left = min(left, max(0, x1 - fx2 - margin))
            elif fx1 >= x2:
                right = min(right, max(0, fx1 - x2 - margin))
        left = min(left, max(0, x1 - margin))
        right = min(right, max(0, page.width - x2 - margin))
        r.target_box = [x1 - left, y1, x2 + right, y2]

    drawn = [r for r in page.regions if r.render and r.target_box]
    _stack_overlapping([r for r in drawn if _in_bubble(r)], sizer=_make_sizer(page, cfg))
    _separate_leftovers(drawn)


def _rotated_fit(target: list[int], want_w: float, want_h: float, angle: float,
                 keep: float = 0.6) -> tuple[float, float, float, float] | None:
    """angle 만큼 돌린 사각형이 target(이웃을 보고 정한 축 정렬 상자) 안에 들어가는 가장 넓은 크기.

    돌리면 외접 사각형이 커져서, 원하는 크기(세운 글자 덩어리 + 옆으로 넓힌 폭) 그대로 돌리면 이웃 글자와
    붙는다(실측: 095쪽 서류 글씨가 왼쪽 대사에 닿음). 높이를 조금씩 줄여 가며 폭이 가장 넓게 남는 조합을
    고른다. 똑바로 그릴 때 쓸 수 있는 넓이(target 전체)의 keep 배도 못 건지면 None — 좁은 틀 안에서
    돌리면 폭이 크게 줄어 글자가 작아지고 잘게 쪼개지므로(실측: 38px 두 줄 → 31px 네 줄), 원본 각도를
    따르는 것보다 똑바로 크게 그리는 편이 낫다. 돌려주는 값: (중심 x, y, 폭, 높이)."""
    tx1, ty1, tx2, ty2 = target
    W, H = tx2 - tx1, ty2 - ty1
    c, s = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    best: tuple[float, float] | None = None
    for i in range(20, 4, -1):
        h = want_h * i / 20
        w = min(want_w, (W - h * s) / c, (H - h * c) / s if s > 1e-6 else want_w)
        if w > 0 and (best is None or w * h > best[0] * best[1]):
            best = (w, h)
    if best is None or best[0] * best[1] < keep * W * H:
        return None
    return (tx1 + tx2) / 2, (ty1 + ty2) / 2, best[0], best[1]


def _make_sizer(page: Page, cfg: Config):
    """(영역, 폭, 높이) → 그 상자에 번역문을 넣었을 때의 글자 크기. 렌더러와 같은 기준 크기·최소 크기를
    쓰되 글꼴은 기본 글꼴로 어림한다(겹친 말풍선 둘을 비교하는 용도라 상대값이면 충분하다)."""
    rc = cfg.render
    base = max(8, int(page.height * rc.font_ratio))
    minimum = max(6, int(page.height * rc.min_font_ratio))
    font_path = str(cfg.abs(cfg.paths.font))

    def size(r: Region, w: int, h: int) -> int:
        text = (r.text_ko or "").replace("…", "...").strip()
        if not text:
            return base
        s, _, overflow = fit_text(text, font_path, max(10, w), max(10, h), base, minimum,
                                  rc.line_spacing, fallback=font_path)
        return s // 2 if overflow else s
    return size


def _best_cut(up: Region, dn: Region, lo: int, hi: int, gap: int, min_h: int, sizer) -> int | None:
    """겹친 구간 [lo, hi] 안에서 위(up)·아래(dn) 상자를 가를 y. 두 번역문의 글자 크기 중 작은 쪽이
    가장 커지는 곳, 같으면 두 크기 차가 작은 곳, 그래도 같으면 가운데에 가까운 곳."""
    ux1, uy1, ux2, _ = up.target_box        # type: ignore[misc]
    dx1, _, dx2, dy2 = dn.target_box        # type: ignore[misc]
    mid = (lo + hi) / 2
    best, best_key = None, None
    step = max(2, (hi - lo) // 20)
    for c in range(lo, hi + 1, step):
        a_h, b_h = c - gap - uy1, dy2 - (c + gap)
        if a_h < min_h or b_h < min_h:
            continue
        sa, sb = sizer(up, ux2 - ux1, a_h), sizer(dn, dx2 - dx1, b_h)
        key = (min(sa, sb), -abs(sa - sb), -abs(c - mid))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def _stack_overlapping(regions: list[Region], gap: int = 6, min_h: int = 28,
                       min_ratio: float = 0.12, min_v: float = 0.5, rounds: int = 3,
                       sizer=None) -> None:
    """상자가 겹치는 말풍선 안 글자들을 세로로 나눠 갖게 한다.

    원문이 세로쓰기면 한 말풍선 안의 줄마다 다른 영역으로 탐지되곤 한다(구름 말풍선 하나에
    '준짱' / '오랜만이야' / '다시 만나서 기뻐'). 그 영역들은 같은 말풍선을 가리키므로 똑같은
    상자를 받아 글자가 같은 자리에 겹쳐 그려진다. 가로쓰기로 옮기면 위→아래가 자연스러우니
    세로로 나눈다.

    나누는 위치는 겹친 구간 안에서 두 번역문의 글자 크기가 최대한 같아지는 곳이다(sizer).
    나누는 것은 '겹치는 구간'뿐이다. 그룹 전체 높이를 균등 분할하면 원래 겹치지도 않던 부분까지
    잘려 글자가 말풍선 아래쪽으로 몰린다. 위아래 순서는 상자의 세로 중심을 따르되, 중심이 거의
    같으면(한 말풍선에 나란히 쓰인 경우) 읽기 순서를 따른다."""
    def key(r: Region) -> tuple:
        return (r.order if r.order is not None else 10_000, r.id)

    for _ in range(rounds):
        moved = False
        for i, a in enumerate(regions):
            for b in regions[i + 1:]:
                ax1, ay1, ax2, ay2 = a.target_box   # type: ignore[misc]
                bx1, by1, bx2, by2 = b.target_box   # type: ignore[misc]
                ow = min(ax2, bx2) - max(ax1, bx1)
                oh = min(ay2, by2) - max(ay1, by1)
                if ow <= 0 or oh <= 0:
                    continue
                # 세로로 나란히 겹칠 때만 나눈다. 대각선으로 인접한 두 말풍선은 사각형 상자의
                # 모서리만 스치는데(세로 겹침 40% 수준), 그걸 나누면 아래쪽 글자의 위가 잘려
                # 말풍선 중앙이 아니라 바닥으로 몰린다.
                if oh < min_v * max(1, min(ay2 - ay1, by2 - by1)):
                    continue
                small = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                if ow * oh < min_ratio * max(1, small):
                    continue                        # 살짝 스치는 정도는 글자까지 겹치지 않는다
                acy, bcy = (ay1 + ay2) / 2, (by1 + by2) / 2
                if abs(acy - bcy) >= 20:
                    up, dn = (a, b) if acy < bcy else (b, a)
                else:
                    up, dn = (a, b) if key(a) <= key(b) else (b, a)
                ux1, uy1, ux2, uy2 = up.target_box  # type: ignore[misc]
                dx1, dy1, dx2, dy2 = dn.target_box  # type: ignore[misc]
                # 겹친 구간의 가운데에서 자르면 긴 문장이 든 쪽이 높이를 잃어 혼자 작아진다
                # (실측: 11쪽 '다 큰 어른이 한심하네~♥' 22px, 옆의 '큭큭♥' 36px). 글자 양을 보고 자른다.
                mid = _best_cut(up, dn, min(dy1, uy2), max(dy1, uy2), gap, min_h, sizer) if sizer else None
                if mid is None:
                    mid = (uy2 + dy1) // 2
                nuy2, ndy1 = mid - gap, mid + gap
                if nuy2 - uy1 < min_h or dy2 - ndy1 < min_h:
                    continue                        # 나누면 글자가 못 들어갈 만큼 좁다
                up.target_box = [ux1, uy1, ux2, nuy2]
                dn.target_box = [dx1, ndy1, dx2, dy2]
                moved = True
        if not moved:
            break


def _in_bubble(r: Region) -> bool:
    """자기 말풍선 안에 놓이는 글자인가. 그렇다면 상자가 이웃과 겹쳐도 글자는 겹치지 않는다."""
    return r.kind == "bubble_text" and bool(r.body_box or r.bubble_box)


def _separate_leftovers(regions: list[Region], min_w: int = 24) -> None:
    """탐지 상자끼리 이미 조금 겹쳐 있던 경우처럼 남은 겹침은 서로 물러나게 한다.

    말풍선 안 글자는 움직이지 않는다. 말풍선 둘이 맞붙어 있으면 상자는 겹쳐도 글자는 각자의
    말풍선 안에 들어가므로 밀어낼 이유가 없고, 밀면 오히려 상자 중심이 말풍선 중심에서 벗어나
    글자가 한쪽으로 쏠린다. 그림 위 글자끼리, 또는 그림 위 글자와 말풍선 사이의 겹침만 푼다."""
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            a_fixed, b_fixed = _in_bubble(a), _in_bubble(b)
            if a_fixed and b_fixed:
                continue
            ax1, ay1, ax2, ay2 = a.target_box  # type: ignore[misc]
            bx1, by1, bx2, by2 = b.target_box  # type: ignore[misc]
            if ay2 <= by1 or by2 <= ay1:
                continue
            ov = min(ax2, bx2) - max(ax1, bx1)
            if ov <= 0:
                continue
            # 한쪽이 고정이면 다른 쪽이 겹침을 다 물러난다
            cut = (ov + 1) if (a_fixed or b_fixed) else (ov // 2 + 1)
            left, right = (a, b) if (ax1 + ax2) <= (bx1 + bx2) else (b, a)
            lx1, ly1, lx2, ly2 = left.target_box  # type: ignore[misc]
            rx1, ry1, rx2, ry2 = right.target_box  # type: ignore[misc]
            if not _in_bubble(left) and lx2 - lx1 - cut >= min_w:
                left.target_box = [lx1, ly1, lx2 - cut, ly2]
            if not _in_bubble(right) and rx2 - rx1 - cut >= min_w:
                right.target_box = [rx1 + cut, ry1, rx2, ry2]


def _inset(box: list[int] | tuple[int, int, int, int], ratio: float) -> list[int]:
    x1, y1, x2, y2 = box
    mw, mh = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
    return [x1 + mw, y1 + mh, x2 - mw, y2 - mh]


def _text_box(r: Region, ratio: float) -> list[int]:
    """말풍선 안 글자 상자. 잰 body_box 가 있으면 그것을 쓴다.

    body_box 는 이미 말풍선 안쪽에 들어가는 사각형이라 여백을 절반만 뺀다. 탐지기가 준 bubble_box
    는 꼬리·뿔까지 감싼 바깥 상자여서 여백을 다 빼고도 글자가 말풍선을 뚫고 나가곤 한다."""
    if r.body_box:
        return _inset(r.body_box, ratio / 2)
    return _inset(r.bubble_box, ratio)  # type: ignore[arg-type]


def _bubble_inner_box(r: Region, ratio: float, page_w: int, page_h: int) -> list[int]:
    box = _text_box(r, ratio)
    return [max(0, box[0]), max(0, box[1]), min(page_w, box[2]), min(page_h, box[3])]


def region_target_box(r: Region, cfg: Config, page_w: int, page_h: int) -> list[int]:
    """글자를 배치할 영역. 넓히기로 정해진 값이 있으면 그것, 말풍선이면 본체(없으면 탐지 박스)에서
    안쪽 여백을 뺀 박스, 그 밖이면 글자 박스를 가로로 넓힌 것."""
    if r.target_box:
        b = r.target_box
        return [max(0, b[0]), max(0, b[1]), min(page_w, b[2]), min(page_h, b[3])]
    if (r.body_box or r.bubble_box) and r.kind == "bubble_text":
        box = _text_box(r, cfg.render.bubble_inner_margin)
    else:
        # 세로쓰기 글자 박스는 좁고 길다. 가로쓰기용으로 가로를 넓히고 세로를 유지한다.
        x1, y1, x2, y2 = r.box
        bw, bh = x2 - x1, y2 - y1
        extra = max(0, int(bh * 0.6) - bw) // 2
        box = [x1 - extra, y1, x2 + extra, y2]
    return [max(0, box[0]), max(0, box[1]), min(page_w, box[2]), min(page_h, box[3])]
