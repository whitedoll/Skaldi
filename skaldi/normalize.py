"""번역문 정규화: 모델이 남기는 일본어 표기·기호를 한국어 식자에 맞게 고친다.

모델마다 남기는 찌꺼기가 다르다(실측, Sevengar 8쪽 96영역):
- gemma4 31B/26B 2비트: 「」 를 16~17영역에 그대로 두고, 따옴표 종류가 섞이고 짝이 안 맞는다(「…ㅋㅋ").
  26B 는 장음 'ー' 를 4곳에 남긴다("에ーーー!").
- gemma4 31B Q4: 토큰이 바이트로 새어 나온다(ンほッ → '응<0xED><0x9D><0xA3>').
- 여러 모델: 웃음 표시 ｗ 를 그대로 두거나, 호칭을 가나로 남긴다(세이나ちゃん).
뜻을 바꾸는 규칙은 넣지 않는다. 표기만 고치고, 고칠 수 없는 가나는 호출부가 경고로 남긴다.
"""
from __future__ import annotations

import re

_HANGUL = "가-힣ㄱ-ㅎㅏ-ㅣ"
_BYTES = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")
_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
# 한글 뒤에 붙은 호칭만 바꾼다
_HONORIFIC = [("ちゃん", "쨩"), ("チャン", "쨩"), ("さん", "상"), ("くん", "군"), ("さま", "님"), ("様", "님")]
_CHARMAP = str.maketrans({
    "「": '"', "」": '"', "“": '"', "”": '"', "„": '"', "″": '"', "＂": '"',
    "『": "'", "』": "'", "‘": "'", "’": "'",
    "（": "(", "）": ")", "！": "!", "？": "?", "〜": "~", "～": "~",
    "ｗ": "ㅋ", "Ｗ": "ㅋ",
    "ー": "ㅡ",                                     # 장음: 한국어 식자에서는 'ㅡ' 로 늘인다
})


def _decode_bytes(m: re.Match) -> str:
    """'<0xED><0x9D><0xA3>' 처럼 새어 나온 UTF-8 바이트를 글자로 되돌린다. 안 되면 지운다."""
    raw = bytes(int(h, 16) for h in re.findall(r"<0x([0-9A-Fa-f]{2})>", m.group(0)))
    return raw.decode("utf-8", errors="ignore")


def _balance_quotes(text: str) -> str:
    """따옴표 수가 홀수면 짝을 맞춘다. 「…ㅋㅋ" 처럼 종류만 달랐던 것은 통일하면서 이미 맞는다.

    남은 경우는 대부분 여는 따옴표만 있고 닫는 것이 없는 경우다 — 모델이 닫기를 빠뜨렸거나, 원문 인용이
    다음 영역으로 이어져 이 영역에서 끊긴 경우(047쪽 '이 녀석의 「좋아♡ … 줄게♡'). 끝에 닫아 준다.
    반대로 여는 것 없이 닫는 따옴표가 맨 앞에 있으면 뺀다."""
    for q in ('"', "'"):
        if text.count(q) % 2 == 0:
            continue
        last = text.rfind(q)
        if last == 0 or text[last - 1].isspace():            # 마지막 따옴표가 '여는' 쪽
            text = text + q
        elif text.startswith(q) is False and text.find(q) == len(text) - 1:
            text = text[:-1].rstrip()                         # 짝 없는 닫는 따옴표 하나뿐
    return text


def normalize_ko(text: str) -> str:
    if not text:
        return text
    t = _BYTES.sub(_decode_bytes, text)
    for ja, ko in _HONORIFIC:
        t = re.sub(rf"(?<=[{_HANGUL}])\s?{ja}", ko, t)
    t = t.translate(_CHARMAP)
    # 반각 w 웃음: 한글·문장부호 뒤에서 단어 끝으로 끝나는 w 묶음만 (영어 단어는 앞이 로마자라 건드리지 않음)
    t = re.sub(rf"(?<=[{_HANGUL}?!.~♡♥)\"' ])[wW]+(?=$|[\s\"')!?.,~♡♥])", lambda m: "ㅋ" * len(m.group(0)), t)
    # 문장부호 사이에 홀로 남은 촉음(ッ/っ): 소리가 없는 표기라 지운다 ("♥♥ッ♥♥")
    t = re.sub(rf"(?<![぀-ゟ゠-ヿ])[ッっ](?![぀-ゟ゠-ヿ])", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return _balance_quotes(t)


def leftover_kana(text: str) -> str:
    """정규화 뒤에도 남은 가나(있으면 그 글자들)."""
    return "".join(dict.fromkeys(_KANA.findall(text or "")))
