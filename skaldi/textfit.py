"""글자 크기·줄바꿈 맞춤. 렌더러와 지우기(말풍선 넓히기) 양쪽에서 쓴다."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

_measure_draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def split_variation(path: str) -> tuple[str, str | None]:
    """"경로#굵기" 를 (경로, 굵기이름)으로 나눈다. 가변 폰트(Noto Sans KR VF 처럼 한 파일에
    Thin~Black 이 들어 있는 것)에서 어느 굵기를 쓸지 설정 한 줄로 지정하기 위한 표기."""
    if "#" in path:
        base, _, var = path.rpartition("#")
        return base, (var or None)
    return path, None


@lru_cache(maxsize=64)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    base, var = split_variation(path)
    f = ImageFont.truetype(base, size)
    if var:
        try:
            f.set_variation_by_name(var)
        except Exception:  # noqa: BLE001  고정 폰트이거나 그런 이름이 없으면 기본 굵기로 둔다
            pass
    return f


@lru_cache(maxsize=8192)
def has_glyph(path: str, ch: str) -> bool:
    """폰트에 그 글자의 자형이 있는가. 손글씨 폰트에는 ♥ ♡ 〜 … 같은 기호가 없는 경우가 많은데,
    없으면 빈칸으로 그려지면서 자리만 차지해 글줄이 한쪽으로 밀린다."""
    if not ch.strip():
        return True
    try:
        f = font(path, 40)
        m = f.getmask(ch)
        if m.getbbox() is None:
            return False
        # 없는 글자를 빈 네모(.notdef)로 그리는 폰트가 있다. 그 네모도 bbox 가 있어 '있음'으로 잘못 판정되면
        # 세로 전용 자형(︙ ﹁)이 네모로 찍힌다(나눔스퀘어라운드). 확실히 없는 글자의 모양과 같으면 없는 것.
        nd = f.getmask("\U0010FFFD")
        return not (m.size == nd.size and bytes(m) == bytes(nd))
    except Exception:  # noqa: BLE001
        return False


def split_runs(text: str, font_path: str, fallback: str | None) -> list[tuple[str, str]]:
    """폰트에 없는 글자는 대체 폰트로 그리도록 (조각, 폰트경로) 목록으로 나눈다."""
    if not fallback or fallback == font_path:
        return [(text, font_path)]
    runs: list[tuple[str, str]] = []
    for ch in text:
        p = font_path if has_glyph(font_path, ch) else fallback
        if runs and runs[-1][1] == p:
            runs[-1] = (runs[-1][0] + ch, p)
        else:
            runs.append((ch, p))
    return runs


def unsupported(text: str, font_path: str, fallback: str | None) -> str:
    """글자 폰트에도 대체 폰트에도 자형이 없는 글자들(있으면 그 글자). 그대로 그리면 빈 네모가 찍힌다."""
    bad = [ch for ch in dict.fromkeys(text or "")
           if ch.strip() and not has_glyph(font_path, ch) and not (fallback and has_glyph(fallback, ch))]
    return "".join(bad)


def text_width(text: str, font_path: str, size: int, fallback: str | None = None) -> float:
    return sum(_measure_draw.textlength(t, font=font(p, size))
               for t, p in split_runs(text, font_path, fallback))


# 어절 안에서 끊어도 어색하지 않은 지점: 조사(한테/에게/에서/으로/부터/까지/이/가/은/는/을/를/의/도/만/와/과/로/에)
# 와 '하-' 계열 용언(배설|하는, 준비|해줘) 앞·뒤. 완벽한 형태소 분석은 아니지만 좁은 말풍선에서
# "마마한/테" 대신 "마마한테 / 배설 / 하는 거"처럼 끊는다.
_PARTICLES = ("한테서", "에게서", "으로써", "으로서", "한테", "에게", "에서", "으로", "부터", "까지", "처럼", "보다",
              "마다", "조차", "이나", "이든", "이라", "이", "가", "은", "는", "을", "를", "의", "도", "만", "와", "과",
              "로", "에", "께", "요")
_VERB_HEADS = ("하", "되", "해", "했", "합", "할", "함", "시키", "당하", "받")


def soft_break_points(word: str) -> list[int]:
    """word 안에서 끊을 수 있는 위치(뒤쪽 조각의 시작 인덱스) 목록. 앞뒤 조각이 최소 2글자."""
    pts: set[int] = set()
    n = len(word)
    for i in range(2, n - 1):
        rest = word[i:]
        for pt in _PARTICLES:
            if rest.startswith(pt):
                pts.add(i)                              # 조사 앞
                if n - (i + len(pt)) >= 2:
                    pts.add(i + len(pt))                # 조사 뒤 (마마한테|배설)
                break
        for vh in _VERB_HEADS:
            if rest.startswith(vh):
                pts.add(i)                              # 용언 앞 (배설|하는)
                break
    return sorted(p for p in pts if 2 <= p <= n - 2)


def split_word_soft(word: str, tl, max_w: int) -> list[str] | None:
    """소프트 브레이크 지점만 써서 word 를 max_w 이하 조각들로 나눈다. 불가능하면 None."""
    pieces: list[str] = []
    cur = word
    while tl(cur) > max_w:
        pts = [p for p in soft_break_points(cur) if tl(cur[:p]) <= max_w]
        if not pts:
            return None
        p = max(pts)
        pieces.append(cur[:p])
        cur = cur[p:]
    pieces.append(cur)
    return pieces


def wrap(text: str, font_path: str, size: int, max_w: int, allow_char_break: bool,
         fallback: str | None = None) -> list[str] | None:
    """띄어쓰기 단위로 줄을 나눈다. 한 단어가 폭을 넘으면 조사·어미 단위 소프트 브레이크를 먼저 시도하고,
    그래도 안 되면 allow_char_break 일 때만 글자 단위로 자르고, 아니면 None(이 크기로는 못 맞춤)."""
    tl = lambda t: text_width(t, font_path, size, fallback)  # noqa: E731
    lines: list[str] = []
    for para in text.replace("\r", "").split("\n"):
        words = [w for w in para.split(" ") if w]
        cur = ""
        for w in words:
            cand = (cur + " " + w) if cur else w
            if tl(cand) <= max_w:
                cur = cand
                continue
            if cur:
                lines.append(cur)
            cur = w
            if tl(cur) > max_w:
                soft = split_word_soft(cur, tl, max_w)
                if soft is not None:
                    lines.extend(soft[:-1])
                    cur = soft[-1]
                    continue
                if not allow_char_break:
                    return None
                while tl(cur) > max_w and len(cur) > 1:
                    cut = len(cur)
                    while cut > 1 and tl(cur[:cut]) > max_w:
                        cut -= 1
                    lines.append(cur[:cut])
                    cur = cur[cut:]
        if cur:
            lines.append(cur)
    return lines or [""]


def _fits(lines: list[str], font_path: str, size: int, box_w: int, box_h: int,
          spacing: float, fallback: str | None) -> bool:
    return int(size * spacing) * len(lines) <= box_h and all(
        text_width(l, font_path, size, fallback) <= box_w for l in lines)


# 원문 글자 수와 상자 넓이에서 글자 크기를 되짚을 때 쓰는 계수. CJK 글자는 정사각형에 가까워
# 한 글자가 대략 (글자크기)^2 만큼을 차지하므로 sqrt(상자넓이/글자수) 가 글자 크기에 비례한다.
# 자간·행간과 상자 여백 때문에 그대로는 크게 나와 0.79 를 곱한다. 기존 작품 8종의 JSON 으로 맞춘
# 값이다 — 글자 크기가 천장에 걸리지 않았던 작품(Sevengar·brake2·samples)에서 실제로 쓰인
# 크기와 3% 안쪽으로 일치한다.
SRC_FONT_K = 0.79
SRC_FONT_MIN_CHARS = 4      # 이보다 짧은 글('응!')은 상자를 채우지 않아 추정이 튄다


def source_font_size(text_ja: str, box_w: int, box_h: int, k: float = SRC_FONT_K) -> int | None:
    """원문 글자 상자와 원문 글자 수로 원문 글자 크기를 추정한다. 못 재면 None.

    페이지 높이에 비례한 기준 크기는 판형이 바뀌면 무너진다(1920x1080 가로형에서 19px 이 되어
    원문의 절반도 안 된다). 원문 글자 크기를 직접 되짚으면 판형과 무관하게 맞는다."""
    n = len(re.sub(r"\s", "", text_ja))
    if n < SRC_FONT_MIN_CHARS or box_w < 8 or box_h < 8:
        return None
    size = k * math.sqrt(box_w * box_h / n)
    return max(6, min(int(size), min(box_w, box_h)))    # 한 글자가 상자 짧은 변보다 클 수는 없다


def fit_text(text: str, font_path: str, box_w: int, box_h: int, base: int, minimum: int,
             spacing: float, fallback: str | None = None) -> tuple[int, list[str], bool]:
    """base 이하에서 박스에 들어가는 가장 큰 크기를 이진탐색으로 찾고, 줄 길이를 고르게 나눈다.

    예전에는 base 에서 약 8%씩(size // 12) 계단식으로 줄여 최대 8% 가까이 작게 그릴 수 있었다.
    단어 중간 끊기는 최소 크기에서도 안 들어갈 때만 허용한다. 돌려주는 값: (크기, 줄 목록, overflow)"""
    minimum = min(minimum, base)

    def attempt(size: int) -> list[str] | None:
        lines = wrap(text, font_path, size, box_w, allow_char_break=False, fallback=fallback)
        if lines is not None and int(size * spacing) * len(lines) <= box_h:
            return lines
        return None

    if attempt(minimum) is not None:
        if attempt(base) is not None:
            best = base
        else:
            lo, hi = minimum, base                      # lo 는 들어가고 hi 는 안 들어간다
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if attempt(mid) is not None:
                    lo = mid
                else:
                    hi = mid
            best = lo
        lines = attempt(best) or [text]
        return best, balance_lines(text, lines, font_path, best, box_w, fallback), False
    # 최소 크기에서도 단어를 안 끊고는 안 들어간다. 글자 단위 줄바꿈을 허용하고, 그래도 넘치면
    # 최소 크기 아래로 더 줄인다. 글자가 조금 작아지는 편이 말풍선 밖으로 크게 삐져나오는 것보다 낫다.
    floor = max(8, minimum // 2)
    size = minimum
    while True:
        lines = wrap(text, font_path, size, box_w, allow_char_break=True, fallback=fallback) or [text]
        if _fits(lines, font_path, size, box_w, box_h, spacing, fallback):
            return size, lines, False
        if size <= floor:
            return size, lines, True
        size -= 1


def balance_lines(text: str, lines: list[str], font_path: str, size: int, box_w: int,
                  fallback: str | None = None) -> list[str]:
    """줄 수는 그대로 두고 줄 길이를 고르게 다시 나눈다.

    탐욕 줄바꿈은 앞줄을 폭까지 꽉 채워 마지막 줄에 한두 글자만 남기곤 한다('봉사해 / 봐♡').
    같은 줄 수가 나오는 가장 좁은 폭으로 다시 나누면 가장 긴 줄이 최소가 되어 줄 길이가 고르다.
    말풍선은 가운데가 넓어서, 좁고 고른 글자 덩어리가 곡선에도 덜 걸친다.
    단어 하나가 폭을 넘어 조사 단위로 쪼갠 경우와 줄바꿈 문자가 든 경우는 건드리지 않는다."""
    if len(lines) < 2 or "\n" in text:
        return lines
    words = [w for w in text.split(" ") if w]
    if not words or max(text_width(w, font_path, size, fallback) for w in words) > box_w:
        return lines
    width = natural_width(text, font_path, size, len(lines), fallback)
    if width >= box_w:
        return lines
    out = wrap(text, font_path, size, width, allow_char_break=False, fallback=fallback)
    return out if out is not None and len(out) == len(lines) else lines


def natural_width(text: str, font_path: str, size: int, max_lines: int,
                  fallback: str | None = None) -> int:
    """base 크기에서 단어를 끊지 않고 max_lines 줄 이내로 넣을 수 있는 최소 폭."""
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    if not words:
        return 0
    lo = int(max(text_width(w, font_path, size, fallback) for w in words))
    hi = int(text_width(" ".join(words), font_path, size, fallback)) + 1
    while lo < hi:
        mid = (lo + hi) // 2
        lines = wrap(text, font_path, size, mid, allow_char_break=False, fallback=fallback)
        if lines is not None and len(lines) <= max_lines:
            hi = mid
        else:
            lo = mid + 1
    return lo


HALF_CELL_PUNCT = ".,"              # 세로쓰기에서 반 칸만 차지하고 앞 글자 옆에 붙는 문장부호


def column_steps(col: str) -> float:
    """세로 열 하나의 길이(글자 칸 단위). 띄어쓰기와 마침표·쉼표는 반 칸."""
    return sum(0.5 if (c == " " or c in HALF_CELL_PUNCT) else 1.0 for c in col)


def fit_vertical(text: str, box_w: int, box_h: int, base: int, minimum: int,
                 spacing: float, max_cols: int | None = None) -> tuple[int, list[str], bool]:
    """세로쓰기: 글자를 위→아래로 쌓고 열은 오른쪽→왼쪽. 돌려주는 값: (크기, 열 목록, overflow)

    띄어쓰기는 지우지 않고 반 칸 틈으로 남긴다(한국어는 띄어쓰기가 없으면 '최면당한남편과'처럼 붙어 읽기
    어렵다). 열을 나눌 때는 단어 경계를 우선하고, 한 단어가 열보다 길 때만 글자 단위로 자른다.
    max_cols 를 주면 그 열 수 안에 들 때까지 줄인다(원문이 한 열인 세로 제목은 한 열로)."""
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    size = base
    while True:
        step = max(1, int(size * spacing))
        cap = max(1.0, float(box_h // step))              # 한 열에 들어가는 칸 수
        cols: list[str] = []
        cur = ""
        for w in words:
            cand = (cur + " " + w) if cur else w
            if column_steps(cand) <= cap:
                cur = cand
                continue
            if cur:
                cols.append(cur)
            while len(w) > cap:                            # 열보다 긴 단어는 글자 단위로 자른다
                cols.append(w[:int(cap)])
                w = w[int(cap):]
            cur = w
        if cur:
            cols.append(cur)
        cols = cols or [""]
        if size * 1.08 * len(cols) <= box_w and (max_cols is None or len(cols) <= max_cols):
            return size, cols, False
        if size <= minimum:
            return size, cols, True
        size = max(minimum, size - max(1, size // 12))


# ---- 다각형 자리 (겹친 말풍선의 폴리곤 배치) -------------------------------------

def _band_run(mask, y0: int, y1: int) -> tuple[int, int] | None:
    """띠 [y0, y1) 의 모든 행에서 마스크 안인 열 중 가장 긴 연속 구간 (x1, x2)."""
    import numpy as np

    band = mask[max(0, y0):y1]
    if band.shape[0] == 0:
        return None
    ok = np.concatenate([[0], band.all(axis=0).astype(np.int8), [0]])
    d = np.diff(ok)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(starts) == 0:
        return None
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _poly_attempt(text: str, font_path: str, mask, size: int, spacing: float, fallback: str | None,
                  step: int = 3) -> list[tuple[str, float, int]] | None:
    """size 로 마스크 안에 줄마다 그 높이의 실제 폭만큼 채운다. 블록 중심이 마스크 무게중심에 가장 가까운
    시작 높이를 고른다. 돌려주는 값: [(줄, 중심 x, 위쪽 y)] (마스크 좌표) 또는 None."""
    import numpy as np

    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return None
    cy = float(np.where(mask)[0].mean())
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    lh = int(size * spacing)
    tl = lambda t: text_width(t, font_path, size, fallback)  # noqa: E731
    runs: dict[int, tuple[int, int] | None] = {}
    best = None
    for top in range(int(ys[0]), int(ys[-1]) - size + 1, step):
        lines, y, queue = [], top, list(words)
        while queue:
            if y not in runs:
                runs[y] = _band_run(mask, y, y + size)
            run = runs[y]
            if run is None or run[1] - run[0] < size:
                break
            wmax = run[1] - run[0]
            cur = ""
            while queue:
                cand = (cur + " " + queue[0]) if cur else queue[0]
                if tl(cand) <= wmax:
                    cur = cand
                    queue.pop(0)
                    continue
                if not cur:                         # 한 단어가 폭을 넘으면 조사·어미 단위로만 끊는다
                    soft = split_word_soft(queue[0], tl, wmax)
                    if soft and len(soft) >= 2:
                        cur, queue[0] = soft[0], "".join(soft[1:])
                break
            if not cur:
                break
            lines.append((cur, (run[0] + run[1]) / 2, y))
            y += lh
        if queue:
            continue
        d = abs(top + len(lines) * lh / 2 - cy)
        if best is None or d < best[0]:
            best = (d, lines)
    return best[1] if best else None


def fit_polygon(text: str, font_path: str, mask, base: int, minimum: int, spacing: float,
                fallback: str | None = None) -> tuple[int, list[tuple[str, float, int]]] | None:
    """다각형 마스크(bool 2차원 배열) 안에 들어가는 가장 큰 크기와 줄 배치. 최소 크기에서도 안 되면 None.

    네모 상자는 말풍선 안에 '완전히 들어가는 사각형'이라 둥근 모서리·불룩한 부분을 버린다. 다각형은 줄마다
    그 높이에서 실제로 쓸 수 있는 폭을 써서, 가시형·한쪽이 불룩한 말풍선에서 더 크게 들어간다
    (02_00_1 가시 말풍선: 네모 31px, 다각형 40px)."""
    minimum = min(minimum, base)
    if _poly_attempt(text, font_path, mask, minimum, spacing, fallback) is None:
        return None
    lo, hi = minimum, base
    if _poly_attempt(text, font_path, mask, base, spacing, fallback) is not None:
        lo = base
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _poly_attempt(text, font_path, mask, mid, spacing, fallback) is not None:
            lo = mid
        else:
            hi = mid
    lines = _poly_attempt(text, font_path, mask, lo, spacing, fallback, step=1)
    return (lo, lines) if lines else None
