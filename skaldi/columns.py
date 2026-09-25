"""세로쓰기 글자 덩어리를 열 단위로 나눈다.

탐지기는 나란한 세로 열 여러 개(나레이션 3열 + 옆의 대사 1열)를 한 상자로 묶어 준다. 그대로 두면
- Baberu 는 크롭을 224x224 로 줄여 읽는데, 열이 많고 긴 상자는 글자가 뭉개져 가짜 문장을 지어낸다
- 비전 모델도 긴 상자에서는 열 몇 개를 빠뜨리고 읽는다(test02 316쪽: 분홍 대사 한 열이 통째로 빠졌다)
- 색이 다른 대사와 나레이션이 한 덩어리로 번역·지우기·렌더링된다

그래서 열을 찾아 열마다 읽고, 색이 다른 열 묶음은 별도 영역으로 떼어 낸다.

열 찾기: 픽셀 투영(열마다 잉크가 있는지)으로는 못 한다. 그림 위 글자의 흰 외곽선이 옆 열과 맞붙어
여러 열이 한 덩어리로 보인다(skew.line_count 주석). 대신 '흰색이 아닌 픽셀' 덩어리를 글자 조각으로 보고,
조각의 가로 중심을 면적 가중치로 쌓아 봉우리를 찾는다. 같은 열의 글자는 가로 중심이 한 줄로 늘어서고,
열 사이 흰 테두리 틈으로 보이는 배경 조각은 작아서 봉우리를 만들지 못한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

INK_LUM = 190           # 이보다 어두우면 잉크(검정·분홍·파랑·초록 글자 모두 걸린다. 흰 외곽선은 빠진다)
COLOR_SPLIT = 90        # 이웃 열의 글자 색 차이(RGB 거리)가 이보다 크면 다른 글로 본다


@dataclass
class Column:
    x1: int             # 크롭 안 좌표
    x2: int
    y1: int
    y2: int
    peak: int           # 글자 가로 중심
    rgb: tuple[int, int, int]
    n: int              # 글자 크기만 한 조각 수
    halo: tuple[int, int, int] | None = None   # 글자 바로 둘레의 색(흰 외곽선을 두른 글씨면 흰색)


def find_columns(rgb: np.ndarray) -> tuple[list[Column], float]:
    """세로쓰기 글자 크롭(RGB 배열)의 열 목록(왼쪽→오른쪽)과 글자 크기(px). 못 찾으면 ([], 0)."""
    h, w = rgb.shape[:2]
    if h < 16 or w < 8:
        return [], 0.0
    a = rgb.astype(np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    n, lab, st, _ = cv2.connectedComponentsWithStats((lum < INK_LUM).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in st[i])
        if area < 8 or max(cw, ch) > 0.3 * max(h, w):
            continue                       # 잡티, 또는 테두리·그림 같은 큰 덩어리
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue                       # 크롭 가장자리에 걸친 것은 그림 조각이다
        comps.append((x, y, cw, ch, area, i))
    if len(comps) < 4:
        return [], 0.0
    g = float(np.percentile([max(c[2], c[3]) for c in comps], 80))
    comps = [c for c in comps if 0.2 * g <= max(c[2], c[3]) <= 1.6 * g]
    if not comps:
        return [], 0.0

    hist = np.zeros(w + 1)
    for c in comps:
        hist[int(c[0] + c[2] / 2)] += c[4]
    sig = max(1.0, 0.15 * g)
    kern = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    sm = np.convolve(hist, kern, mode="same")
    peaks: list[int] = []
    for x in np.argsort(-sm):
        if sm[x] < 0.12 * sm.max():
            break
        if all(abs(int(x) - q) >= 0.75 * g for q in peaks):
            peaks.append(int(x))
    peaks.sort()

    members: dict[int, list] = {q: [] for q in peaks}
    for c in comps:
        cx = c[0] + c[2] / 2
        q = min(peaks, key=lambda q: abs(q - cx))
        if abs(q - cx) <= 0.6 * g:
            members[q].append(c)
    cols: list[Column] = []
    for q in peaks:
        big = [c for c in members[q] if max(c[2], c[3]) >= 0.5 * g]
        if len(big) < 2:
            continue                       # 루비(작은 글자)나 떨어진 기호
        glyph = np.isin(lab, [c[5] for c in big])
        px = rgb[glyph]
        ring = cv2.dilate(glyph.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=max(2, int(0.08 * g))).astype(bool) & ~glyph & (lum >= INK_LUM)
        halo = tuple(int(v) for v in np.median(rgb[ring], axis=0)) if ring.sum() >= 20 else None
        cols.append(Column(
            x1=min(c[0] for c in big), x2=max(c[0] + c[2] for c in big),
            y1=min(c[1] for c in big), y2=max(c[1] + c[3] for c in big),
            peak=q, rgb=tuple(int(v) for v in np.median(px, axis=0)), n=len(big), halo=halo))
    return cols, g


def dominant_colors(cols: list[Column]) -> tuple[list[int], list[int] | None]:
    """(글자 채움 색, 둘레 색). 조각이 많은 열의 값을 따른다."""
    main = max(cols, key=lambda c: c.n)
    return list(main.rgb), (list(main.halo) if main.halo else None)


def pitch(cols: list[Column], g: float) -> float:
    """열 간격. 열이 하나면 글자 크기의 1.25 배로 어림한다."""
    if len(cols) < 2:
        return 1.25 * g
    return float(np.median(np.diff([c.peak for c in cols])))


def glyph_size(cols: list[Column], g: float) -> int:
    """원문 글자 크기(px). 열 폭이 곧 전각 글자 폭이다. 쉼표·괄호가 옆으로 삐져나와 폭이 조금 커지므로
    0.9 를 곱하고, 조각 크기로 잰 값(g)과 크게 어긋나지 않게 묶는다."""
    widths = [c.x2 - c.x1 for c in cols if c.n >= 3] or [c.x2 - c.x1 for c in cols]
    return int(np.clip(0.9 * float(np.median(widths)), 0.8 * g, 1.3 * g))


def column_boxes(box: list[int], cols: list[Column], g: float, page_w: int, page_h: int) -> list[list[int]]:
    """열마다 읽을 페이지 좌표 상자. 오른쪽 열부터(세로쓰기 읽는 순서)."""
    x0, y0 = box[0], box[1]
    half = pitch(cols, g) / 2
    pad = int(0.3 * g)
    out = []
    for c in sorted(cols, key=lambda c: -c.peak):
        out.append([max(0, int(x0 + c.peak - half)), max(0, y0 + c.y1 - pad),
                    min(page_w, int(x0 + c.peak + half)), min(page_h, y0 + c.y2 + pad)])
    return out


def color_groups(cols: list[Column]) -> list[list[Column]]:
    """이웃 열끼리 글자 색이 비슷하면 한 묶음(왼쪽→오른쪽). 분홍 대사 열과 검정 나레이션 열을 가른다."""
    groups: list[list[Column]] = []
    for c in cols:
        if groups and np.linalg.norm(np.subtract(c.rgb, groups[-1][-1].rgb)) <= COLOR_SPLIT:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def is_vertical_block(box: list[int]) -> bool:
    """세로쓰기 글자 덩어리로 볼 만큼 세로로 긴가."""
    return (box[3] - box[1]) >= 1.3 * (box[2] - box[0])


def measure(image: Image.Image, box: list[int]) -> tuple[list[Column], float]:
    x1, y1, x2, y2 = box
    return find_columns(np.asarray(image.convert("RGB").crop((x1, y1, x2, y2))))


def split_by_color(image: Image.Image, regions: list) -> list:
    """말풍선 밖 세로 글자 덩어리 중 글자 색이 다른 열 묶음이 섞인 것을 묶음마다 별도 영역으로 나눈다.

    test02 316쪽: 나레이션 3열(검정)과 '美咲「あっ…♥タケルっ…♥来てたんだっ…♥」'(분홍) 한 열이 한 상자로
    잡혀, 대사가 나레이션 번역에 섞이거나 통째로 빠졌다. 색이 같은 열끼리는 한 문단으로 둔다
    (세로쓰기 문장은 열을 넘어 이어지므로 열마다 쪼개면 문장이 끊긴다)."""
    from .page import Region

    out = []
    for r in regions:
        if r.kind != "free_text" or not is_vertical_block(r.box):
            out.append(r)
            continue
        cols, g = measure(image, r.box)
        groups = [grp for grp in color_groups(cols) if sum(c.n for c in grp) >= 3]
        if len(groups) < 2:
            out.append(r)
            continue
        x0, y0 = r.box[0], r.box[1]
        pad = int(0.3 * g)
        for grp in groups:
            box = [max(r.box[0], x0 + min(c.x1 for c in grp) - pad), max(r.box[1], y0 + min(c.y1 for c in grp) - pad),
                   min(r.box[2], x0 + max(c.x2 for c in grp) + pad), min(r.box[3], y0 + max(c.y2 for c in grp) + pad)]
            out.append(Region(id=r.id, kind=r.kind, box=box, bubble_box=r.bubble_box, score=r.score,
                              split_from=r.id))
    return out


def ink_glyph_mask(rgb: np.ndarray, inner: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """글자 채움 픽셀 마스크(크롭 크기, bool). find_columns 와 같은 기준으로 고른 글자 조각이다.
    흰 외곽선·바깥 테두리·그림은 들어가지 않는다. inner(크롭 좌표)를 주면 그 상자에 걸친 조각만 고른다
    — 탐지 상자가 글자를 가로지르면 글자가 크롭 가장자리에 걸려 빠지므로, 넓게 잘라 보고 이렇게 거른다."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), bool)
    a = rgb.astype(np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    n, lab, st, _ = cv2.connectedComponentsWithStats((lum < INK_LUM).astype(np.uint8), connectivity=8)
    comps = [i for i in range(1, n) if st[i][4] >= 8 and max(st[i][2], st[i][3]) <= 0.3 * max(h, w)
             and not (st[i][0] == 0 or st[i][1] == 0 or st[i][0] + st[i][2] >= w or st[i][1] + st[i][3] >= h)]
    if not comps:
        return out
    g = float(np.percentile([max(st[i][2], st[i][3]) for i in comps], 80))
    keep = [i for i in comps if max(st[i][2], st[i][3]) <= 1.6 * g]
    if inner:
        ix1, iy1, ix2, iy2 = inner
        keep = [i for i in keep if st[i][0] < ix2 and st[i][0] + st[i][2] > ix1
                and st[i][1] < iy2 and st[i][1] + st[i][3] > iy1]
    return np.isin(lab, keep)


_QUOTE = re.compile(r"^.{0,8}?[「『“\"]")     # '美咲「…' 처럼 앞 몇 글자 안에 여는 따옴표


def merge_unquoted(page) -> None:
    """색으로 나눈 조각 중 따옴표 대사가 하나도 없으면 한 영역으로 다시 합친다.

    색이 다른 열이 늘 다른 사람의 말은 아니다. test02 313쪽은 한 문장 안에서 이름·구절만 분홍으로
    강조했는데('美咲が一番嫌悪するはずの男…' 분홍 / 'なのに美咲は…' 검정), 색으로 나누니 한 문장이 세
    조각으로 따로 번역·배치됐다. 따로 두어야 하는 건 316쪽처럼 '美咲「あっ…♥」' 식의 대사 줄이다.
    판독이 확정된 뒤(비전 채택 후)에 본다 — Baberu 는 이 대사를 'あはははっ…' 로 헛읽어 따옴표가 없었다.
    조각 하나라도 원본 유지(needs_review)면 합치지 않는다. 헛읽은 문장이 멀쩡한 번역에 섞이면 안 된다."""
    groups: dict[int, list] = {}
    for r in page.regions:
        if r.split_from is not None:
            groups.setdefault(r.split_from, []).append(r)
    for parts in groups.values():
        # 멀쩡한 조각들 사이에 끼인 원본 유지 조각은 버린다. 흰 외곽선 글씨에서 열 사이 틈이 딴 색 열로
        # 잡혀 생긴 가짜 조각이다(Kamaboko 異世界 18쪽 '心までワシの / (가짜) / 言いなり♥' 가 합쳐지지 않아
        # '마음까지 내' / '말대로♥' 로 따로 번역됨). 합친 상자가 그 자리를 덮으므로 지우기에서도 빠지지 않는다
        good = [r for r in parts if not r.needs_review]
        if len(good) >= 2:
            ux1, ux2 = min(r.box[0] for r in good), max(r.box[2] for r in good)
            bad = [r for r in parts if r.needs_review]
            if bad and all(r.box[0] >= ux1 and r.box[2] <= ux2 for r in bad):
                drop_ids = {id(r) for r in bad}
                page.regions = [r for r in page.regions if id(r) not in drop_ids]
                parts = good
        if len(parts) < 2:
            continue
        if any(r.needs_review for r in parts) or any(_QUOTE.match(r.text_ja.strip()) for r in parts):
            continue
        parts.sort(key=lambda r: -r.box[0])            # 세로쓰기: 오른쪽 조각부터 읽는다
        head = max(parts, key=lambda r: (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
        keep = parts[0]
        keep.text_ja = "".join(r.text_ja for r in parts)
        keep.box = [min(r.box[0] for r in parts), min(r.box[1] for r in parts),
                    max(r.box[2] for r in parts), max(r.box[3] for r in parts)]
        keep.cols = [c for r in parts for c in (r.cols or [])] or None
        confs = [r.ocr_conf for r in parts if r.ocr_conf is not None]
        keep.ocr_conf = min(confs) if confs else None
        keep.category, keep.render, keep.erase = head.category, head.render, head.erase
        orders = [r.order for r in parts if r.order is not None]
        keep.order = min(orders) if orders else keep.order
        keep.split_from = None
        drop = {id(r) for r in parts[1:]}
        page.regions = [r for r in page.regions if id(r) not in drop]


def merge_free_columns(regions: list) -> list:
    """말풍선 밖 세로 글자 중 탐지기가 열마다 따로 잡은 것을 한 영역으로 합친다.

    Kamaboko 異世界 18쪽 '心までワシの / 言いなり♥' 는 두 열이 따로 잡혀(가운데에 가짜 상자도 하나) 열마다
    따로 번역됐고('마음까지 내' / '말대로♥'), 열 폭이 달라 글자 크기(34·24px)와 글꼴 판정도 갈렸다.
    합치는 조건: 둘 다 한 열 폭의 세로로 긴 상자, 폭이 비슷하고(0.6배 이상), 윗줄이 글자 하나 차이 안에서
    맞고, 세로로 절반 넘게 겹치며, 좌우 틈이 글자 폭의 0.6배 이하(겹쳐도 됨). 색이 다른 열은 뒤이어
    split_by_color 가 다시 나눈다."""
    def column_like(r) -> bool:
        return r.kind == "free_text" and r.h() >= 2 * r.w()

    def joinable(a, b) -> bool:
        wa, wb = a.w(), b.w()
        if min(wa, wb) < 0.6 * max(wa, wb):
            return False
        if abs(a.box[1] - b.box[1]) > max(wa, wb):
            return False
        if min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]) < 0.5 * min(a.h(), b.h()):
            return False
        gap = max(a.box[0], b.box[0]) - min(a.box[2], b.box[2])
        return gap <= 0.6 * min(wa, wb)

    out = list(regions)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(out):
            if not column_like(a):
                continue
            for b in out[i + 1:]:
                if column_like(b) and joinable(a, b):
                    keep = a if a.h() >= b.h() else b
                    keep.box = [min(a.box[0], b.box[0]), min(a.box[1], b.box[1]),
                                max(a.box[2], b.box[2]), max(a.box[3], b.box[3])]
                    out.remove(b if keep is a else a)
                    merged = True
                    break
            if merged:
                break
    return out


_DEDUP_CORE = re.compile(r"[぀-ヿ一-鿿A-Za-z0-9]")


def dedupe_overlap(page, min_chars: int = 2) -> None:
    """상자가 겹친 두 영역이 같은 글자를 나눠 읽었으면 앞 영역에서 뺀다.

    갈래가 둘인 말풍선에서 탐지기가 한 갈래 상자에 옆 갈래의 첫 열까지 담으면, 그 열이 양쪽에 읽힌다
    (Kamaboko 異世界 28쪽 '催眠で感度上げてるからなド淫乱' + 'ド淫乱♀エルフ♥' → '이 초음란녀' / '초음란 엘프♥').
    앞 영역 끝과 뒤 영역 첫머리가 같은 글자(min_chars 글자 이상)면 앞 영역에서 지운다. 상자는 줄이지 않는다
    — 줄이면 그 자리의 다른 열('からな')까지 지우기에서 빠진다."""
    rs = [r for r in page.regions if r.render and r.text_ja.strip()]
    for a in rs:
        for b in rs:
            if a is b or not (min(a.box[2], b.box[2]) > max(a.box[0], b.box[0])
                              and min(a.box[3], b.box[3]) > max(a.box[1], b.box[1])):
                continue
            ta, tb = a.text_ja, b.text_ja
            n = next((k for k in range(min(len(ta), len(tb)) - 1, 0, -1) if ta.endswith(tb[:k])), 0)
            if len(_DEDUP_CORE.findall(tb[:n])) < min_chars or n >= len(ta):
                continue
            a.text_ja = ta[:-n].rstrip()
            a.notes = (a.notes + f" 겹친 글자 중복 제거('{tb[:n]}', id={b.id})").strip()
