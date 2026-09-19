"""말풍선 안 글자의 줄 높이·줄 간격·획 두께를 픽셀에서 잰다.

말풍선(큰 흰 덩어리)을 찾고, 그 안의 검은 글자를 수평 투영해 줄로 나눈 뒤
줄 높이(=글자 크기에 가깝다), 줄 간격, 획 두께(distance transform)를 모은다.
언어와 무관하므로 원문·번역본·우리 결과를 같은 잣대로 비교할 수 있다."""
import sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image


def bubbles(gray: np.ndarray, page_h: int) -> list[tuple[int, int, int, int]]:
    """말풍선 후보: 충분히 크고 꽉 찬 흰 영역."""
    white = (gray >= 205).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < page_h * 0.03 or w < page_h * 0.03:
            continue
        if area < 0.45 * w * h:                 # 컷 배경처럼 성긴 영역 제외
            continue
        if w > page_h * 0.9 or h > page_h * 0.9:
            continue
        out.append((x, y, w, h))
    return out


def lines_in(gray: np.ndarray, box) -> list[tuple[int, int]]:
    """말풍선 안에서 글자 줄의 (시작y, 끝y). 가장자리 테두리는 안쪽으로 잘라 피한다."""
    x, y, w, h = box
    pad = max(2, int(min(w, h) * 0.10))
    sub = gray[y + pad:y + h - pad, x + pad:x + w - pad]
    if sub.size == 0:
        return []
    ink = (sub < 128)
    rows = ink.sum(axis=1)
    thr = max(1, int(sub.shape[1] * 0.02))
    on = rows >= thr
    runs, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(on) - 1))
    return [(a, b) for a, b in runs if b - a >= 6]


def stroke_width(gray: np.ndarray, box, runs=None) -> float:
    """획 두께. 글자 줄이 있는 띠에서만 잰다 — 말풍선 테두리가 섞이면 두껍게 나온다."""
    x, y, w, h = box
    pad = max(2, int(min(w, h) * 0.10))
    sub = gray[y + pad:y + h - pad, x + pad:x + w - pad]
    if sub.size == 0:
        return 0.0
    if runs:
        rows = np.zeros(sub.shape[0], bool)
        for a, b in runs:
            rows[a:b + 1] = True
        sub = sub[rows]
        if sub.size == 0:
            return 0.0
    ink = (sub < 128).astype(np.uint8)
    if ink.sum() < 50:
        return 0.0
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    return float(np.percentile(vals, 80) * 2) if len(vals) else 0.0


def measure(path: Path):
    """대사 말풍선만 본다. 2줄 이상이어야 줄 간격을 잴 수 있고, 효과음은 대개 한 줄에
    아주 큰 글자라 이 조건에서 자연히 빠진다. 그래도 큰 것은 높이로 한 번 더 거른다."""
    img = np.array(Image.open(path).convert("L"))
    H = img.shape[0]
    heights, gaps, strokes = [], [], []
    for b in bubbles(img, H):
        ls = lines_in(img, b)
        if len(ls) < 2:
            continue
        hs = [c - a + 1 for a, c in ls]
        gs = [ls[i][0] - ls[i - 1][0] for i in range(1, len(ls))]
        if np.median(hs) > H * 0.035 or np.median(gs) > H * 0.045:
            continue                                 # 효과음·제목처럼 큰 글자
        heights += hs
        gaps += gs
        sw = stroke_width(img, b, ls)
        if sw:
            strokes.append(sw)
    return heights, gaps, strokes, H


def run(label: str, files: list[Path]):
    H_all, G_all, S_all, page_h = [], [], [], 0
    for f in files:
        h, g, s, ph = measure(f)
        H_all += h; G_all += g; S_all += s; page_h = ph
    def q(v, p):
        return float(np.percentile(v, p)) if v else 0.0
    print(f"[{label}] 페이지높이 {page_h} · 줄 {len(H_all)}개 · 줄간격 {len(G_all)}개")
    print(f"   글자 줄높이  중앙값 {q(H_all,50):.1f}px  (25%~75%: {q(H_all,25):.0f}~{q(H_all,75):.0f})"
          f"  → 페이지높이 대비 {q(H_all,50)/page_h:.4f}")
    print(f"   줄간격(line_h) 중앙값 {q(G_all,50):.1f}px  → 줄높이 대비 {q(G_all,50)/max(1,q(H_all,50)):.2f}")
    print(f"   획 두께      중앙값 {q(S_all,50):.2f}px  → 줄높이 대비 {q(S_all,50)/max(1,q(H_all,50)):.3f}")
    return q(H_all,50), q(G_all,50), q(S_all,50), page_h


if __name__ == "__main__":
    label, d = sys.argv[1], Path(sys.argv[2])
    names = sys.argv[3:]
    files = [d / n for n in names] if names else sorted(d.glob("*.webp"))
    run(label, [f for f in files if f.exists()])
