"""OCR 백엔드: Baberu OCR(기본) 또는 LLM 비전."""
from __future__ import annotations

import re
import sys

from PIL import Image

from .config import Config
from .order import filler_allowed


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
        return self.read_conf(crop)[0]

    def read_conf(self, crop: Image.Image) -> tuple[str, float]:
        """(판독, 확신도). 확신도는 생성한 토큰마다 가장 높은 확률의 평균이다.

        Baberu 는 못 읽는 글자에서도 문법이 멀쩡한 가짜 문장을 지어내는데('それでも、これからは、'),
        그때는 토큰마다 확률이 낮다. 실측: 정상 판독 0.78~1.00, 헛읽기 0.45~0.80.
        디코딩 설정은 모델 카드의 BaberuOCR.__call__ 과 같다(반복 억제 두 가지)."""
        import torch
        from transformers import LogitsProcessorList

        from inference import CapContentRun  # type: ignore

        o = self.ocr
        dtype = next(o.model.parameters()).dtype
        pv = o.image_processor(crop.convert("RGB"), return_tensors="pt")["pixel_values"].to(o.device, dtype=dtype)
        ids = torch.tensor([[o.tok.bos_token_id]], device=o.device)
        with torch.inference_mode():
            out = o.model.generate(
                input_ids=ids, pixel_values=pv, max_new_tokens=128, do_sample=False,
                repetition_penalty=1.2,
                logits_processor=LogitsProcessorList([CapContentRun(o._content_ids, 12)]),
                eos_token_id=o.tok.eos_token_id, pad_token_id=o.tok.pad_token_id,
                bos_token_id=o.tok.bos_token_id, use_cache=True,
                output_scores=True, return_dict_in_generate=True,
            )
        text = o.tok.decode(out.sequences[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
        probs = [float(torch.softmax(sc[0].float(), -1).max()) for sc in out.scores]
        return text, (sum(probs) / len(probs) if probs else 0.0)


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
VISION_ADOPTED = "비전 판독"          # notes 접두사. Baberu 대신 비전 모델 판독으로 번역했다

_KEEP = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟA-Za-z0-9]")
_VISION_PROMPT = (
    "이 이미지는 일본 만화에서 글자 하나를 잘라낸 것이다. 안에 있는 일본어 글자를 있는 그대로 읽어라. "
    "번역하지 말고, 후리가나는 빼고, 한 줄로 text 필드에 넣어라. 읽을 수 있는 글자가 없으면 빈 문자열."
)
_TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


# 일치도를 잴 때 빼는 글자. っ·ー·작은 글자는 어느 판독에나 흔해 가짜 판독끼리도 겹친다
# (52쪽 'はははっっ' ↔ 'ぼっぼっ' 이 っ 하나로 일치 0.5 가 나와 Baberu 헛읽기를 믿었다)
_WEAK = re.compile(r"[ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮーｰ]")


def _norm(text: str) -> str:
    """가나·한자·영숫자만 남기고(っ·ー·작은 글자 제외) 세 번 이상 이어진 같은 글자는 둘로 줄인다(ぉぉぉぉ)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", _WEAK.sub("", "".join(_KEEP.findall(text))))


def fix_ja(text: str) -> str:
    """인식 결과의 버릇을 고친다. Baberu 는 세로쓰기 말줄임표(‥)를 쌍점(：)으로 읽는다.
    만화 대사에 쌍점은 쓰이지 않으므로 말줄임표로 되돌린다. 그대로 두면 번역문에 ':' 가 남는다
    (Kamaboko 異世界 '豚のザーメン：' → '돼지 정액:', 'いい桃尻：っ' → '복숭아 엉덩이:').
    ♀ 기호는 전각 숫자 ９ 로 읽는다('ド淫乱９エルフ' → '음란 9엘프'). 숫자 없이 가나·한자 사이에 홀로 있는
    ９ 만 ♀ 로 되돌린다."""
    text = re.sub(r"[：:]+", "…", text)
    return re.sub(r"(?<=[぀-ヿ一-鿿])９(?=[぀-ヿ一-鿿♥♡])", "♀", text)


def squeeze_runs(text: str, keep: int = 4) -> str:
    """같은 글자가 keep 번 넘게 이어지면 keep 번으로 줄인다. 비전 모델은 신음을 'ほぉぉぉ…'(수십 자)로
    읽기도 하는데, 그대로 번역 모델에 넣으면 번역 모델도 반복에 빠져 호출이 통째로 실패한다
    (test02 320쪽: 세 모델이 연달아 'token repeat limit' 으로 실패하며 181초를 썼다)."""
    return re.sub(r"(.)\1{%d,}" % keep, lambda m: m.group(1) * keep, text)


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


def _vision_stable(client, cfg: Config, image: Image.Image, r, seen: str, page) -> bool:
    """비전 판독을 여백을 달리해 한 번 더 읽어, 두 판독이 맞을 때만 믿는다.

    비전 모델은 인쇄체는 정확히 읽지만 손글씨 효과음에서는 문장을 지어낸다(Kamaboko 異世界 08쪽:
    'んほぉぉ' 를 'おはようございます' 로). 지어낸 판독은 매번 달라진다. 실측: 인쇄체 0.91~1.00,
    지어낸 판독 0.00~0.33 ('おはようございます' ↔ 'んぼまぼま', 'ほんぼぼ' ↔ 'こんにちは')."""
    try:
        again = vision_read(client, cfg.llm.vision_model, crop_region(image, r.box, 16), max_tokens=400)
    except Exception:  # noqa: BLE001
        return False
    agree = ocr_agreement(seen, again)
    if agree < cfg.ocr.vision_consistency:
        page.warnings.append(f"비전 판독이 흔들려 쓰지 않음 (id={r.id}, 일치 {agree:.2f}): {seen[:16]} / {again[:16]}")
        return False
    return True


_ONLY_DOTS = re.compile(r"^[\s．。…・‥\.､、,，]+$")


def dots_over_strokes(image: Image.Image, page, limit: float = 0.22) -> None:
    """점(．．．)으로만 읽혔는데 원문에 큰 획이 있으면 손글씨 효과음을 잘못 읽은 것이다. 원본을 둔다.

    Baberu 는 말풍선을 가로지르는 큰 손글씨 효과음('オオッ' 'イグッ')을 '．．．' 로 읽고, 확신도도 0.8 넘게
    준다(교차검증을 피한다). 그러면 원문을 지우고 '...' 를 그렸다(Kamaboko 異世界 23곳). 진짜 말없음표
    말풍선은 점이 작다. 글자 상자 짧은 변 대비 가장 큰 획 덩어리: 말없음표 0.19 이하, 효과음 0.26 이상
    (다른 작품에서 점으로 읽힌 0.25 이상 8곳도 모두 효과음이었다)."""
    import cv2
    import numpy as np

    arr = None
    for r in page.regions:
        if not r.text_ja.strip() or not _ONLY_DOTS.match(r.text_ja) or r.category == "sfx":
            continue
        if arr is None:
            arr = np.asarray(image.convert("L"))
        x1, y1, x2, y2 = r.box
        g = arr[max(0, y1):y2, max(0, x1):x2]
        if g.size == 0:
            continue
        h, w = g.shape
        n, _, st, _ = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), connectivity=8)
        sizes = [max(st[i][2], st[i][3]) for i in range(1, n) if st[i][4] >= 6
                 and not (st[i][0] == 0 or st[i][1] == 0 or st[i][0] + st[i][2] >= w or st[i][1] + st[i][3] >= h)]
        ratio = max(sizes) / min(h, w) if sizes else 0.0
        if ratio >= limit:
            r.category, r.render, r.erase = "sfx", False, "none"
            r.notes = (r.notes + f" 점 판독+큰 획({ratio:.2f})→효과음").strip()


def oversized_bubbles(image: Image.Image, page, ratio: float, min_count: int = 3) -> set[int]:
    """원문 글자가 같은 페이지 말풍선 대사의 중앙값보다 ratio 배 이상 큰 말풍선 글자 id.

    크기는 인식한 글자 수가 아니라 픽셀(세로 열 폭)로 잰다. 헛읽기는 글자 수가 틀려 글자 수로 잰
    크기도 틀린다. 실측(Kamaboko 異世界): 보통 대사 0.7~1.2, 손글씨 신음 1.5~3.0, 크게 쓴 인쇄체 외침 1.6."""
    from .columns import glyph_size, measure

    sizes: dict[int, int] = {}
    for r in page.regions:
        if r.kind != "bubble_text" or not r.text_ja.strip() or r.category not in ("dialogue", "narration"):
            continue
        cols, g = measure(image, r.box)
        if cols:
            sizes[r.id] = glyph_size(cols, g)
    if len(sizes) < min_count:
        return set()
    vals = sorted(sizes.values())
    mid = vals[len(vals) // 2]
    return {rid for rid, s in sizes.items() if s >= ratio * mid}


def cross_check(cfg: Config, client, image: Image.Image, page) -> None:
    """Baberu 판독을 비전 모델로 다시 읽어 헛읽기를 거른다. 대상은 두 가지다.

    - 말풍선 밖 글자(대사·나레이션·라벨)는 전부. Baberu 는 흰 바탕 검은 인쇄체용이라 검은 톤 위에
      흰 외곽선을 두른 글씨는 못 읽고 그럴듯한 문장을 지어낸다(09_07·08_06_1 쪽). 가짜 문장은 문법이
      멀쩡해 번역 모델도 unclear 라고 하지 않으므로 번역 전에 걸러야 한다.
      일치도 실측(44영역): 정상 인쇄체 0.25~1.0, 환각 0~0.06.
    - 말풍선 안 글자는 Baberu 확신도(r.ocr_conf)가 낮을 때만. 전부 다시 읽으면 장당 수십 초가 붙는다.

    헛읽기로 보이면 비전 판독이 쓸 만할 때 그것으로 바꿔 번역한다(vision_adopt). test02 에서 걸린
    긴 나레이션 9개 중 6개는 비전 모델이 끝까지 정확히 읽었는데, 예전에는 그것을 버리고 원본을 뒀다.
    비전 판독도 너무 짧으면('嫌' 'ッ' 빈칸) 원본을 둔다(needs_review)."""
    oc = cfg.ocr
    big = oversized_bubbles(image, page, oc.oversized_ratio)
    for r in page.regions:
        if r.category not in ("dialogue", "narration", "label") or not r.text_ja.strip():
            continue
        low = r.ocr_conf is not None and r.ocr_conf < oc.min_confidence
        # 글자가 유난히 큰 말풍선은 확신도가 높아도 다시 읽는다. 탁점 붙은 손글씨 신음('お゛ご゛お゛っ')에서
        # Baberu 는 'そういうことで' 같은 문장을 확신도 0.81 로 지어낸다(Kamaboko 異世界 36쪽)
        if r.kind != "free_text" and not low and r.id not in big:
            continue
        crop = crop_region(image, r.box, oc.crop_padding)
        try:
            if low and r.cols and len(r.cols) >= 2:
                # 긴 여러 열 상자는 비전 모델도 열을 빠뜨리고 읽는다(316쪽 분홍 대사). 열마다 읽는다
                seen = "".join(vision_read(client, cfg.llm.vision_model, image.crop(tuple(b)))
                               for b in r.cols)
            else:
                seen = vision_read(client, cfg.llm.vision_model, crop, max_tokens=400)
        except Exception as e:  # noqa: BLE001
            page.warnings.append(f"OCR 교차검증 실패 (id={r.id}): {e}")
            continue
        score = ocr_agreement(r.text_ja, seen)
        if score >= oc.cross_check_min and not low:
            continue                                 # 확신도도 괜찮고 비전과도 맞는다
        conf = f"{r.ocr_conf:.2f}" if r.ocr_conf is not None else "-"
        if score < oc.cross_check_min and (filler_allowed(seen) or filler_allowed(r.text_ja)):
            # 두 판독이 어긋나고 한쪽이 신음 글자로만 되어 있으면 손글씨 신음이다. 비전 모델은 탁점 모음
            # (お゛)을 'ポボボボ' 로 읽어 그대로 번역됐다(12쪽 '포보보보'). 읽은 글자를 믿지 말고 원본을 둔다
            r.category, r.render, r.erase = "filler", False, "none"
            r.notes = f"신음 판독(확신도 {conf}, 일치 {score:.2f}) 비전:{seen[:16]}"
            continue
        if (oc.vision_adopt and len(_norm(seen)) >= oc.vision_adopt_min_chars
                and _vision_stable(client, cfg, image, r, seen, page)):
            # 'vision:sfx' 는 번역 단계가 효과음 확정 근거로 보는 표시라 덮어쓰지 않는다
            # (비전 분류가 효과음이라 했지만 믿지 않고 대사로 둔 영역. 긴 나레이션이 흔히 여기 든다)
            if r.notes != "vision:sfx":
                r.notes = f"{VISION_ADOPTED}(확신도 {conf}, 일치 {score:.2f}) Baberu:{r.text_ja}"
            page.warnings.append(f"비전 판독으로 바꿈 (id={r.id}, 확신도 {conf}): "
                                 f"{r.text_ja[:16]} → {seen[:24]}")
            r.text_ja, r.ocr_backend = fix_ja(squeeze_runs(seen)), "vision"
            continue
        if score >= oc.cross_check_min and r.id not in big:
            continue                                 # 확신도는 낮지만 비전도 못 읽었고 둘이 대체로 맞는다
        r.needs_review, r.render, r.erase = True, False, "none"
        r.notes = f"{OCR_MISMATCH}({score:.2f}) 비전:{seen or '(글자 없음)'}"
        page.warnings.append(f"OCR 불일치로 원본 유지 (id={r.id}): {r.text_ja[:20]} / 비전 {seen[:20]}")
