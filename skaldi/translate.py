"""페이지 단위 번역. 거부·순화 감지 시 재시도 모델로 자동 전환."""
from __future__ import annotations

from typing import Any

from .config import Config
from .glossary import glossary_prompt
from .llm import looks_refused
from .normalize import leftover_cjk, normalize_ko
from .ocr import OCR_MISMATCH
from .order import filler_allowed, sfx_allowed_in_bubble, sibling_sfx
from .page import Page, Region

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "ko": {"type": "string"},
                    "note": {"type": "string", "enum": ["ok", "sfx", "filler", "unclear"]},
                },
                "required": ["id", "ko", "note"],
            },
        }
    },
    "required": ["translations"],
}

_CAT_KO = {"dialogue": "대사", "narration": "나레이션", "label": "라벨/제목", "sfx": "효과음", "unknown": "미분류"}


def _targets(page: Page) -> list[Region]:
    # OCR 교차검증에서 걸린 영역은 원문이 가짜일 가능성이 커서 번역하지 않는다. 문맥으로 넣으면
    # 지어낸 문장이 같은 페이지의 다른 번역까지 흐린다.
    return [r for r in page.ordered()
            if r.text_ja.strip() and r.category in ("dialogue", "narration", "label", "unknown", "sfx")
            and not r.notes.startswith(OCR_MISMATCH)]


def _system(cfg: Config, glossary: dict) -> str:
    base = (
        f"당신은 {cfg.translate.source_lang} 만화를 {cfg.translate.target_lang}로 옮기는 전문 번역가다. "
        "주어진 원문은 한 페이지의 글자를 읽는 순서대로 나열한 것이다. 문맥을 살려 자연스러운 만화 대사체로 번역하라. "
        "원문의 어조·감정·수위를 바꾸지 말고, 요약하거나 검열하거나 설명을 덧붙이지 말라. "
        "말풍선에 들어가야 하므로 간결하게. 일본어 글자를 남기지 말고 전부 한국어로 옮겨라. "
        "한자는 쓰지 말고 병기도 하지 말라 (무모(無毛) ✕ / 무모 ○). 식자 글꼴에 한자가 없다. "
        "ko 필드에는 번역문만 넣고 번호·괄호·종류 표시 같은 것을 붙이지 말라. "
        "note 필드: 보통 대사면 ok. 사물·동작 소리를 흉내낸 효과음(ドン, ゴキュッ, ボボボ)만 sfx(ko에는 음역). "
        "대답·질문·감탄·웃음처럼 뜻을 전하는 말(はい, そう, へぇ, え？, クスクス, ふふ)은 ok. "
        "filler: 뜻 있는 낱말 없이 모음·ん·は행 소리만 늘인 발성으로, 번역해도 음역밖에 안 되고 "
        "빼도 장면 이해에 지장이 없는 것. 신음·교성·헐떡임·감탄 비명(おおお, ほおお, おおっ, んほっ♥, あっ♡, "
        "あっあっ, はぁはぁ, ふぅ…, はひ…, うおお…)은 filler 다. 이런 발성을 ok 로 두지 말라. "
        "다만 같은 あっ 이라도 무언가를 알아차리거나 부르거나 놀라 반응하는 것이 문맥상 분명하면 ok. "
        "원문이 OCR 오류나 잘림으로 뜻을 알 수 없으면 unclear(ko에는 최선의 추정). "
        "각 항목을 같은 id로 돌려주고 JSON으로만 답하라."
    )
    g = glossary_prompt(glossary)
    return base + ("\n\n" + g if g else "")


def _prompt(regions: list[Region]) -> str:
    lines = [f"[{r.id}] {r.text_ja}" for r in regions]
    hints = []
    for cat in ("narration", "label", "sfx"):
        ids = [r.id for r in regions if r.category == cat]
        if ids:
            hints.append(f"{_CAT_KO[cat]} 항목: {ids}")
    hint = ("\n(" + " / ".join(hints) + ". 나머지는 대사)") if hints else ""
    return "다음 원문을 번역하라." + hint + "\n\n" + "\n".join(lines)


def _call(client, model: str, system: str, prompt: str) -> dict[int, tuple[str, str]]:
    data: dict[str, Any] = client.chat_json(model, prompt, schema=_SCHEMA, system=system, num_ctx=8192)
    out: dict[int, tuple[str, str]] = {}
    for item in data.get("translations", []):
        try:
            out[int(item["id"])] = (str(item.get("ko", "")).strip(), str(item.get("note", "ok")))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def translate_page(cfg: Config, client, page: Page, glossary: dict) -> None:
    regions = _targets(page)
    if not regions:
        return
    system = _system(cfg, glossary)
    models = [cfg.llm.translate_model] + [m for m in cfg.llm.translate_fallbacks if m != cfg.llm.translate_model]
    if cfg.llm.backend == "gemini":
        models = [cfg.llm.gemini_model]

    pending = list(regions)
    used: list[str] = []
    for model in models:
        if not pending:
            break
        try:
            result = _call(client, model, system, _prompt(pending))
        except Exception as e:  # noqa: BLE001
            page.warnings.append(f"번역 호출 실패 ({model}): {e}")
            continue
        used.append(model)
        still: list[Region] = []
        for r in pending:
            ko, note = result.get(r.id, ("", "ok"))
            if looks_refused(r.text_ja, ko):
                still.append(r)
                continue
            r.text_ko = normalize_ko(ko)            # 「」·ｗ·ー·새어 나온 바이트 등 표기 정리
            r.refused = False
            vision_sfx = r.notes == "vision:sfx"
            # 말풍선 안 글자는 번역 모델 혼자 sfx 라고 하면 믿지 않는다(웃음·대답을 효과음으로 볼 수 있어서).
            # 다만 원문이 손으로 그린 글씨면 두 번째 근거가 된다: 250쪽 말풍선 속 'ゴキュッ' 은 비전 모델이
            # 번호를 빠뜨려 대사로 번역됐다. 실측으로 번역 모델은 손글씨 영역 중 진짜 효과음 5개(ゴキュッ ピーッ
            # ジッ！！ …)에만 sfx 라 답했고, 가타카나 규칙이 잘못 걸던 대사(フンッ イベント ヤバ)는 모두 ok 였다.
            hand_drawn = r.kind == "bubble_text" and r.style == "hand"
            if note == "sfx" and r.category != "sfx" and sfx_allowed_in_bubble(r.text_ja) and (
                vision_sfx or r.kind == "free_text" or hand_drawn
            ):
                r.category = "sfx"                    # 비전·번역 모델이 모두 소리 표현이라고 본 경우만 원본 유지
                r.render, r.erase = False, "none"
            elif note in ("filler", "sfx") and r.category in ("dialogue", "narration", "unknown") and filler_allowed(r.text_ja):
                # 번역 모델이 문맥상 뜻 없는 발성이라 보고, 원문도 발성 글자로만 된 경우만 원본 유지.
                # 모델은 신음(んほっ ほおおおおッ)을 filler 대신 sfx 라고 답하는 일이 많아 둘 다 받는다.
                r.category = "filler"
                r.render, r.erase = False, "none"
            elif note == "unclear":
                r.needs_review = True                 # 원문 불명확: 원본 유지, JSON에서 검토
                r.render, r.erase = False, "none"
            else:
                r.needs_review = False
        if still and len(models) > 1:
            page.warnings.append(f"{model}: {len(still)}개 항목 거부/순화 감지, 다음 모델로 재시도")
        pending = still

    for r in pending:
        r.refused = True
        if not r.text_ko:
            r.text_ko = ""
    for r in regions:
        left = leftover_cjk(r.text_ko)
        if left and r.render:
            page.warnings.append(f"번역문에 일본어·한자가 남음 (id={r.id}): {left}")
    sibling_sfx(page.regions)   # 번역 단계에서 효과음으로 확정된 것과 같은 말풍선의 짧은 조각도 묶는다
    page.models["translate"] = ",".join(used) if used else "-"
