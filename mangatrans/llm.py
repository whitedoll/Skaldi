"""LLM 클라이언트: Ollama(비전·텍스트), Gemini(텍스트 번역 전용)."""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any

from PIL import Image

from .config import LlmCfg

# 거부 문구. "죄송"만으로는 판정하지 않는다 — すみません 의 정상 번역(죄송합니다)과 구분이 안 된다.
REFUSAL_PATTERNS = [
    r"도와드릴 수 없", r"도움을 드릴 수 없", r"번역할 수 없", r"번역해 드릴 수 없", r"제공할 수 없",
    r"응답할 수 없", r"답변할 수 없", r"부적절한 (?:내용|요청|콘텐츠)", r"정책에 (?:위배|어긋)",
    r"I can(?:'|no)t (?:help|assist|translate|provide)", r"I cannot (?:help|assist|translate|provide)",
    r"I'm sorry,? but", r"I am sorry,? but", r"not able to (?:help|assist|translate)", r"unable to (?:help|assist|translate)",
    r"inappropriate content", r"explicit content", r"against .*polic",
    r"申し訳ありませんが", r"翻訳できません", r"お手伝いできません",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_HANGUL_RE = re.compile(r"[가-힣]")
_JA_RE = re.compile(r"[぀-ヿ一-鿿]")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def extract_json(text: str) -> dict[str, Any]:
    """모델이 JSON 앞뒤에 잡담을 붙였을 때를 대비해 첫 { ... } 블록을 파싱."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"JSON 파싱 실패: {text[:200]!r}")


class OllamaClient:
    def __init__(self, cfg: LlmCfg):
        import ollama

        self.cfg = cfg
        self.client = ollama.Client(host=cfg.ollama_host)

    def chat_json(
        self,
        model: str,
        prompt: str,
        images: list[Image.Image] | None = None,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        num_ctx: int = 8192,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [png_bytes(im) for im in images]
        messages.append(msg)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            format=schema or "json",
            options={"temperature": self.cfg.temperature, "num_ctx": num_ctx},
        )
        try:
            resp = self.client.chat(think=False, **kwargs)
        except Exception as e:  # think 파라미터를 모르는 모델/버전
            if "think" in str(e).lower():
                resp = self.client.chat(**kwargs)
            else:
                raise
        return extract_json(resp.message.content or "")

    def chat_text(self, model: str, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat(model=model, messages=messages,
                                options={"temperature": self.cfg.temperature})
        return resp.message.content or ""


class GeminiClient:
    """무료 등급 주의: 입력이 구글 제품 개선에 사용되며 성인물 입력은 약관 위반이다."""

    def __init__(self, cfg: LlmCfg):
        from google import genai

        key = os.environ.get(cfg.gemini_api_key_env)
        if not key:
            raise RuntimeError(f"환경변수 {cfg.gemini_api_key_env} 가 없습니다.")
        self.cfg = cfg
        self.client = genai.Client(api_key=key)

    def chat_json(self, model, prompt, images=None, schema=None, system=None, num_ctx=None):
        from google.genai import types

        contents: list[Any] = [prompt]
        for im in images or []:
            contents.append(types.Part.from_bytes(data=png_bytes(im), mime_type="image/png"))
        r = self.client.models.generate_content(
            model=model or self.cfg.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=self.cfg.temperature,
            ),
        )
        return extract_json(r.text or "")


def make_client(cfg: LlmCfg):
    if cfg.backend == "gemini":
        return GeminiClient(cfg)
    return OllamaClient(cfg)


def looks_refused(text_ja: str, text_ko: str) -> bool:
    """거부·순화 판정. 빈 출력, 거부 문구, 원문이 있는데 한글이 없는 경우."""
    if not text_ja.strip():
        return False
    if not text_ko.strip():
        return True
    if _REFUSAL_RE.search(text_ko):
        return True
    if not _HANGUL_RE.search(text_ko) and _JA_RE.search(text_ja):
        return True
    return False
