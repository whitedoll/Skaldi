"""용어집 로딩과 병합.

전역(프로젝트 루트 glossary.yaml) → 작품별 → 작업 폴더 순으로 합치고, 뒤에 오는 것이 이긴다.
작품별 파일은 입력 옆에 둔다:
  - zip/cbz  `book.zip`  → 같은 폴더의 `book.glossary.yaml`
  - 폴더     `book/`     → `book/glossary.yaml`
  - 어느 쪽이든 결과 작업 폴더 `<출력>/<이름>_ko/glossary.yaml` 도 읽는다 (한 번 돌린 뒤 고치기 좋다)
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _load(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def glossary_paths(root_path: Path, source: Path | None, work: Path | None) -> list[Path]:
    """합칠 용어집 파일 경로를 우선순위가 낮은 것부터 나열한다 (있는 것만)."""
    cands: list[Path] = [root_path]
    if source is not None:
        cands.append(source / "glossary.yaml" if source.is_dir()
                     else source.with_suffix("").with_suffix(".glossary.yaml"))
    if work is not None:
        cands.append(work / "glossary.yaml")
    seen: list[Path] = []
    for p in cands:
        if p and p.exists() and p not in seen:
            seen.append(p)
    return seen


def load_glossary(*paths: Path | None) -> dict:
    """여러 용어집 파일을 순서대로 합친다. names 는 뒤에 오는 것이 덮어쓰고, style·notes 는 이어 붙인다."""
    names: dict = {}
    styles: list[str] = []
    notes: list = []
    for p in paths:
        g = _load(p)
        names.update(g.get("names") or {})
        if (g.get("style") or "").strip():
            styles.append(g["style"].strip())
        notes += list(g.get("notes") or [])
    return {"names": names, "style": "\n".join(styles), "notes": notes}


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
