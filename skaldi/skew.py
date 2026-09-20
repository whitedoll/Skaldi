"""원문 글자 덩어리의 기울기 측정.

비스듬히 놓인 글자(기운 서류, 대각선으로 배치된 손글씨)는 번역문도 같은 각도로 돌려 그려야
원본 배치를 따른다. 문제는 오탐이다 — 똑바른 글자를 기울었다고 판정하면 멀쩡한 대사를 돌려 그린다.

방식 두 가지를 따로 재서 둘이 같은 각도를 가리킬 때만 믿는다. 두 방식은 실패하는 경우가 서로 다르다.
- 투영(projection profile): 획 픽셀을 돌려 가며 열/행 투영이 가장 날카로운 각도. 글자가 여러 개면
  정확하지만, 글자가 1~2개면 한 글자 안의 대각선 획을 읽어 ±30~40° 로 오판한다
  (실측: 1789개 영역 중 약 100개 오탐, 대부분 'ん♡' 'あっ' 같은 짧은 말풍선).
- 외곽(최소 회전 사각형): 획을 이어 붙인 덩어리의 외곽. 글자 수와 무관하지만, 세로쓰기 열마다 길이가
  다르면 외곽이 계단 모양이라 기울었다고 오판한다(실측: 32개 판정 중 대부분이 똑바른 세로글).
두 방식이 3° 안에서 일치하고, 덩어리가 길쭉하며(글줄 방향이 뚜렷), 투영이 0° 보다 확실히 날카로울 때만
기울기로 인정한다. 이 조건으로 4개 작품 1856개 영역 중 3개만 판정됐고 모두 실제로 기운 글자였다.

각도 규약: PIL Image.rotate 와 같다(화면에서 반시계 방향이 +).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_ANGLE = 4.0          # 이보다 작은 기울기는 무시 (돌려 그려도 차이가 안 보이고 흐려지기만 한다)
MAX_ANGLE = 35.0         # 이보다 크면 측정을 믿지 않는다 (±40° 근처는 거의 오탐이었다)
AGREE = 3.0              # 두 방식의 허용 차이
MIN_ASPECT = 1.8         # 덩어리 긴 변 / 짧은 변
MIN_GAIN = 1.2           # 투영 날카로움: 최적 각도 / 0°


def stroke_mask(gray: np.ndarray, box: list[int], pad: int = 4) -> np.ndarray | None:
    """글자 상자 안의 획 마스크(0/255). 주변 국소 밝기와 40 이상 다른 픽셀을 획으로 본다.
    검은 글씨·흰 외곽선 모두 잡히고, 말풍선 테두리처럼 가늘고 긴 선은 뺀다."""
    from .erase import _drop_line_parts                          # erase 가 이 모듈을 쓰므로 지연 import
    H, W = gray.shape
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape) < 8:
        return None
    k = int(np.clip(min(crop.shape) * 0.5, 15, 81)) | 1
    local = cv2.medianBlur(crop, k).astype(np.int16)
    m = (np.abs(crop.astype(np.int16) - local) > 40).astype(np.uint8) * 255
    return _drop_line_parts(m)


def _hull(mask: np.ndarray) -> tuple[float, float]:
    """외곽 방식: (기울기, 길쭉함). 획을 글줄 덩어리로 이은 뒤 최소 회전 사각형의 긴 변 방향."""
    h, w = mask.shape
    k = max(3, int(min(h, w) * 0.12)) | 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return 0.0, 0.0
    total = st[1:, 4].sum()
    keep = [i for i in range(1, n) if st[i, 4] >= 0.05 * total]      # 잡티 제외
    pts = np.column_stack(np.nonzero(np.isin(lab, keep)))[:, ::-1].astype(np.float32)
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    if min(rw, rh) < 1:
        return 0.0, 0.0
    long_ang = ang if rw >= rh else ang + 90.0                        # 긴 변의 방향 (이미지 좌표)
    dev = ((long_ang + 45.0) % 90.0) - 45.0                           # 가까운 축에서 벗어난 정도
    return -dev, max(rw, rh) / min(rw, rh)                            # y 가 아래로 → 반시계 + 로 뒤집음


def _profile(mask: np.ndarray, lim: float = 40.0) -> tuple[float, float]:
    """투영 방식: (기울기, 날카로움 비). a° 기운 것을 되돌렸을 때 열·행 투영의 제곱합이 가장 크다."""
    h, w = mask.shape
    d = int(np.hypot(h, w)) + 4
    canvas = np.zeros((d, d), np.uint8)
    oy, ox = (d - h) // 2, (d - w) // 2
    canvas[oy:oy + h, ox:ox + w] = (mask > 0).astype(np.uint8)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D((d / 2, d / 2), -a, 1.0)
        r = cv2.warpAffine(canvas, M, (d, d), flags=cv2.INTER_NEAREST)
        c = r.sum(0).astype(np.float64)
        rr = r.sum(1).astype(np.float64)
        return max(float((c * c).sum()), float((rr * rr).sum()))

    angs = np.arange(-lim, lim + 1e-6, 1.0)
    sc = np.array([score(a) for a in angs])
    return float(angs[int(sc.argmax())]), float(sc.max() / max(1.0, score(0.0)))


def text_angle(gray: np.ndarray, box: list[int]) -> float:
    """글자 덩어리의 기울기(도). 확신할 수 없으면 0."""
    mask = stroke_mask(gray, box)
    if mask is None or int((mask > 0).sum()) < 150:
        return 0.0
    hull_a, aspect = _hull(mask)
    if aspect < MIN_ASPECT:
        return 0.0                                                    # 글줄 방향이 없는 덩어리 (글자 1~2개)
    prof_a, gain = _profile(mask)
    if not (MIN_ANGLE <= abs(prof_a) <= MAX_ANGLE) or gain < MIN_GAIN or abs(prof_a - hull_a) > AGREE:
        return 0.0
    return prof_a


def _rotate_about(mask: np.ndarray, center: tuple[float, float], angle: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """mask 를 center 기준으로 angle 만큼 '되돌린'(-angle) 큰 캔버스와, 되돌린 좌표 → 원래 좌표 행렬."""
    h, w = mask.shape
    cx, cy = float(center[0]), float(center[1])
    # 중심이 가운데가 아니어도 잘리지 않게, 중심에서 가장 먼 모서리까지를 반지름으로 잡는다
    d = int(2 * max(np.hypot(cx - x, cy - y) for x in (0, w) for y in (0, h))) + 4
    # 원래 좌표 → 캔버스 좌표: center 를 캔버스 중심으로 옮긴 뒤 -angle 회전
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    M[0, 2] += d / 2 - cx
    M[1, 2] += d / 2 - cy
    out = cv2.warpAffine(mask, M, (d, d), flags=cv2.INTER_NEAREST)
    inv = cv2.invertAffineTransform(M)
    return out, inv


def rotated_extent(gray: np.ndarray, box: list[int], angle: float) -> tuple[float, float, float, float] | None:
    """기운 글자 덩어리를 똑바로 세웠을 때의 크기와 원래 좌표의 중심: (cx, cy, 폭, 높이)."""
    mask = stroke_mask(gray, box)
    if mask is None or not mask.any():
        return None
    pad = 4
    ox, oy = max(0, box[0] - pad), max(0, box[1] - pad)
    h, w = mask.shape
    rot, inv = _rotate_about(mask, (w / 2, h / 2), angle)
    ys, xs = np.nonzero(rot)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    c = inv @ np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.0])
    return float(c[0] + ox), float(c[1] + oy), float(x2 - x1), float(y2 - y1)


def rotated_body(interior: np.ndarray, center: tuple[float, float], angle: float, fit: float,
                 inset: float) -> tuple[float, float, float, float] | None:
    """말풍선 내부 마스크를 글자 각도만큼 되돌려 세운 뒤 본체·내접 사각형을 잰다.
    기운 타원 말풍선에 축 정렬 사각형을 넣으면 폭이 크게 줄지만, 세워서 재면 제 크기가 나온다.
    돌려주는 값: (원래 좌표 중심 x, y, 세운 좌표의 폭, 높이)."""
    from .erase import _blend, _inset, bubble_body, inner_rect   # 순환 import 방지

    ys, xs = np.nonzero(interior)
    if len(xs) == 0:
        return None
    pad = 4
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = xs.max() + pad + 1, ys.max() + pad + 1
    sub = interior[y0:y1, x0:x1]
    rot, inv = _rotate_about(sub, (center[0] - x0, center[1] - y0), angle)
    body = bubble_body(rot)
    if body is None or body[2] - body[0] < 10 or body[3] - body[1] < 10:
        return None
    rect = _inset(_blend(inner_rect(rot, body), body, fit), inset)
    rx1, ry1, rx2, ry2 = rect
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return None
    c = inv @ np.array([(rx1 + rx2) / 2, (ry1 + ry2) / 2, 1.0])
    return float(c[0] + x0), float(c[1] + y0), float(rx2 - rx1), float(ry2 - ry1)


def line_count(box: list[int], text: str) -> tuple[bool, float]:
    """원문 글자 덩어리가 (세로쓰기인가, 추정 글줄 수).

    픽셀 투영으로 열을 세면 그림 위 글자의 흰 외곽선이 옆 열과 맞붙어 여러 열도 한 덩어리로 보인다
    (실측: 3열짜리 대사까지 전부 1줄). 대신 글자 수와 상자 비율로 어림한다. 세로 한 열이면
    높이/폭 ≈ 글자 수이고, 열이 k개면 높이/폭 ≈ 글자 수/k² 이므로 k ≈ √(글자 수 × 폭/높이).
    말줄임 점(．・…)은 칸을 덜 차지하므로 1/3 글자로 센다.
    한 줄짜리 라벨(세로 제목, 이름표)을 세로쓰기로 그릴지 정하는 데 쓴다."""
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    vertical = h >= w
    n = 0.0
    for ch in text:
        if ch.isspace():
            continue
        n += 1 / 3 if ch in "．・…‥.。、，," else 1.0
    if n <= 0:
        return vertical, 0.0
    ratio = (w / h) if vertical else (h / w)
    return vertical, float(np.sqrt(n * ratio))
