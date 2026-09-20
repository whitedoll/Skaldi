"""OCR 백엔드: Baberu OCR(기본) 또는 LLM 비전."""
from __future__ import annotations

import re
import sys

from PIL import Image

from .config import Config


class BaberuBackend:
    name = "baberu"

    def __init__(self, cfg: Config):
        model_dir = cfg.abs(cfg.paths.models_dir) / "baberu-ocr"
        required = ["model.safetensors", "inference.py", "modeling_baberu.py",
                    "configuration_baberu.py", "tokenization_baberu.py", "config.json"]
        if not all((model_dir / f).exists() for f in required):
            from huggingface_hub import snapshot_download

            snapshot_download(
                cfg.ocr.baberu_repo,
                local_dir=str(model_dir),
                allow_patterns=["*.py", "*.json", "*.txt", "model.safetensors", "tokenizer/*"],
            )
        # inference.py는 같은 폴더의 모듈을 상대 import 하므로 경로를 추가한다.
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from inference import BaberuOCR  # type: ignore

        self.ocr = BaberuOCR(model_dir)

    def read(self, crop: Image.Image) -> str:
        return self.ocr(crop).strip()


class LlmBackend:
    name = "llm"

    def __init__(self, cfg: Config):
        from .llm import OllamaClient

        self.client = OllamaClient(cfg.llm)
        self.model = cfg.llm.vision_model

    def read(self, crop: Image.Image) -> str:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        prompt = (
            "이 이미지는 만화 말풍선 하나를 잘라낸 것이다. 안에 있는 일본어 글자를 있는 그대로 읽어라. "
            "번역하지 말고, 후리가나(작은 글자)는 빼고, 줄바꿈 없이 한 줄로 이어서 text 필드에 넣어라. "
            "글자가 없으면 빈 문자열."
        )
        data = self.client.chat_json(self.model, prompt, images=[crop], schema=schema)
        return str(data.get("text", "")).strip()


def make_ocr(cfg: Config):
    if cfg.ocr.backend == "llm":
        return LlmBackend(cfg)
    return BaberuBackend(cfg)


def crop_region(image: Image.Image, box: list[int], pad: int) -> Image.Image:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))


# ---- 비전 모델 교차검증 -------------------------------------------------

OCR_MISMATCH = "OCR 불일치"          # notes 접두사. 이 영역은 번역하지 않고 원본을 둔다

_KEEP = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟA-Za-z0-9]")
_VISION_PROMPT = (
    "이 이미지는 일본 만화에서 글자 하나를 잘라낸 것이다. 안에 있는 일본어 글자를 있는 그대로 읽어라. "
    "번역하지 말고, 후리가나는 빼고, 한 줄로 text 필드에 넣어라. 읽을 수 있는 글자가 없으면 빈 문자열."
)
_TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _norm(text: str) -> str:
    """가나·한자·영숫자만 남기고 세 번 이상 이어진 같은 글자는 둘로 줄인다(ぉぉぉぉ, っっっっ)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(_KEEP.findall(text)))


def ocr_agreement(a: str, b: str, short: int = 5) -> float:
    """두 OCR 결과의 일치도(0~1). 짧은 쪽 기준 겹침 비율(overlap coefficient).

    보통은 두 글자 조합(bigram)으로 잰다. 흔한 히라가나 한 글자(の, は)는 가짜 문장과도 우연히 겹치기
    때문이다. 한쪽이 short 글자 이하로 짧으면 조합이 몇 개 안 나와 흔들리므로 글자 단위로 잰다."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if min(len(na), len(nb)) <= short and max(len(na), len(nb)) <= 2 * short:
        A, B = set(na), set(nb)
    else:
        A = {na[i:i + 2] for i in range(len(na) - 1)} or {na}
        B = {nb[i:i + 2] for i in range(len(nb) - 1)} or {nb}
    return len(A & B) / min(len(A), len(B))


def vision_read(client, model: str, crop: Image.Image, max_tokens: int = 160) -> str:
    """비전 모델로 글자를 읽는다. 같은 글자를 반복하다 잘려 JSON 이 깨지면 남은 앞부분을 쓴다."""
    try:
        data = client.chat_json(model, _VISION_PROMPT, images=[crop], schema=_TEXT_SCHEMA, num_predict=max_tokens)
        return str(data.get("text", "")).strip()
    except ValueError as e:
        m = re.search(r'"text"\s*:\s*"([^"]*)', str(e))
        if m:
            return m.group(1).strip()
        raise


def cross_check(cfg: Config, client, image: Image.Image, page) -> None:
    """말풍선 밖 글자(대사·나레이션·라벨)를 비전 모델로 다시 읽어, Baberu 결과와 거의 안 겹치면
    번역하지 않고 원본을 남긴다(needs_review).

    Baberu 는 흰 바탕 검은 인쇄체용이라, 검은 톤 위에 흰 외곽선을 두른 큰 손글씨는 못 읽고 대신
    'それは．．．このままではないですから' 같은 그럴듯한 문장을 지어낸다(09_07·08_06_1 쪽). 가짜 문장은
    문법이 멀쩡해 번역 모델도 unclear 라고 하지 않으므로 번역 전에 걸러야 한다. 비전 모델도 이런 글씨는
    반쯤만 읽지만(このキッパおがれになりゃ…) 지어낸 문장과는 글자 조합이 거의 겹치지 않는다.
    실측(44영역): 정상 인쇄체 0.25~1.0, 환각 0~0.06. 페이지 가장자리 오탐(글자 없음)은 비전 모델이
    빈 문자열이나 엉뚱한 한 글자를 돌려준다."""
    for r in page.regions:
        if r.kind != "free_text" or r.category not in ("dialogue", "narration", "label") or not r.text_ja.strip():
            continue
        crop = crop_region(image, r.box, cfg.ocr.crop_padding)
        try:
            seen = vision_read(client, cfg.llm.vision_model, crop)
        except Exception as e:  # noqa: BLE001
            page.warnings.append(f"OCR 교차검증 실패 (id={r.id}): {e}")
            continue
        score = ocr_agreement(r.text_ja, seen)
        if score >= cfg.ocr.cross_check_min:
            continue
        r.needs_review, r.render, r.erase = True, False, "none"
        r.notes = f"{OCR_MISMATCH}({score:.2f}) 비전:{seen or '(글자 없음)'}"
        page.warnings.append(f"OCR 불일치로 원본 유지 (id={r.id}): {r.text_ja[:20]} / 비전 {seen[:20]}")
