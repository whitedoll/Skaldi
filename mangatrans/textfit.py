"""글자 크기·줄바꿈 맞춤. 렌더러와 지우기(말풍선 넓히기) 양쪽에서 쓴다."""
from __future__ import annotations

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
        return font(path, 40).getmask(ch).getbbox() is not None
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


def fit_text(text: str, font_path: str, box_w: int, box_h: int, base: int, minimum: int,
             spacing: float, fallback: str | None = None) -> tuple[int, list[str], bool]:
    """base 크기부터 minimum 까지 줄여 가며 박스에 들어가는 크기를 찾는다.
    단어 중간 끊기는 최소 크기에서도 안 들어갈 때만 허용한다. 돌려주는 값: (크기, 줄 목록, overflow)"""
    size = base
    while True:
        lines = wrap(text, font_path, size, box_w, allow_char_break=False, fallback=fallback)
        if lines is not None and int(size * spacing) * len(lines) <= box_h:
            return size, lines, False
        if size <= minimum:
            break
        size = max(minimum, size - max(1, size // 12))
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


def fit_vertical(text: str, box_w: int, box_h: int, base: int, minimum: int,
                 spacing: float) -> tuple[int, list[str], bool]:
    """세로쓰기: 글자를 위→아래로 쌓고 열은 오른쪽→왼쪽. 돌려주는 값: (크기, 열 목록, overflow)"""
    chars = [c for c in text.replace("\n", " ") if not c.isspace()]
    size = base
    while True:
        step = max(1, int(size * spacing))
        per_col = max(1, box_h // step)
        cols = ["".join(chars[i:i + per_col]) for i in range(0, len(chars), per_col)] or [""]
        if size * 1.08 * len(cols) <= box_w:
            return size, cols, False
        if size <= minimum:
            return size, cols, True
        size = max(minimum, size - max(1, size // 12))
