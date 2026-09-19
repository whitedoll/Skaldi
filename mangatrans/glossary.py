"""용어집 로딩과 병합 (루트 + 입력 폴더, 폴더 우선)."""
from __future__ import annotations

from pathlib import Path

import yaml


def _load(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_glossary(root_path: Path, folder: Path | None) -> dict:
    base = _load(root_path)
    local = _load(folder / "glossary.yaml") if folder else {}
    names = {**(base.get("names") or {}), **(local.get("names") or {})}
    style = "\n".join(s for s in [base.get("style", ""), local.get("style", "")] if s and s.strip())
    notes = list(base.get("notes") or []) + list(local.get("notes") or [])
    return {"names": names, "style": style.strip(), "notes": notes}


def glossary_prompt(g: dict) -> str:
    parts: list[str] = []
    if g.get("names"):
        lines = "\n".join(f"- {ja} → {ko}" for ja, ko in g["names"].items())
        parts.append("고유명사 표기 (반드시 따를 것):\n" + lines)
    if g.get("style"):
        parts.append("번역 방침:\n" + g["style"])
    if g.get("notes"):
        parts.append("참고 사항:\n" + "\n".join(f"- {n}" for n in g["notes"]))
    return "\n\n".join(parts)
