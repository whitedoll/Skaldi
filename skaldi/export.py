"""처리된 작업 폴더에서 결과 이미지만 뽑아 zip 또는 폴더로 내보낸다.

작업 폴더에는 중간 산출물이 함께 들어 있다(json/ 원문·번역, clean/ 글자 지운 판, src/ 압축을 푼
원본). 남에게 주거나 뷰어로 볼 때 필요한 것은 렌더 결과뿐이므로 그것만 원래 이름으로 추출한다.

zip 입력으로 만든 작업 폴더에는 src/ 가 있다. 그 경우 원본 압축의 항목 이름과 순서를 그대로
쓰고, 렌더 결과가 없는 항목(표지 정보 같은 이미지 아닌 파일, 건너뛴 페이지)은 원본을 넣는다.
그래야 결과 zip 이 원본과 같은 구조가 된다.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_DIRS = {"json", "clean", "src", "compare"}


def list_works(root: Path) -> list[str]:
    """저장 폴더 안에서 내보낼 수 있는 작업 폴더 이름 목록 (렌더 결과가 있는 것만)."""
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and renderers_in(d):
            out.append(d.name)
    return out


def renderers_in(work: Path) -> list[str]:
    """작업 폴더에 결과가 들어 있는 렌더러 이름들."""
    if not work.is_dir():
        return []
    return [d.name for d in sorted(work.iterdir())
            if d.is_dir() and d.name not in SKIP_DIRS
            and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file())]


def _pairs(work: Path, renderer: str, order: list[str] | None) -> list[tuple[str, Path]]:
    """(내보낼 이름, 실제 파일) 목록."""
    rdir = work / renderer
    src_dir = work / "src"
    if order is None and src_dir.is_dir():
        order = [p.relative_to(src_dir).as_posix() for p in sorted(src_dir.rglob("*")) if p.is_file()]
    if order is not None:
        pairs = []
        for name in order:
            if name.endswith("/"):
                continue
            orig = src_dir / name
            if orig.is_dir():
                continue
            rendered = rdir / Path(name).name
            if rendered.exists():
                pairs.append((name, rendered))
            elif orig.exists():
                pairs.append((name, orig))          # 렌더 결과가 없으면 원본 그대로
        return pairs
    # 폴더 입력이라 src/ 가 없다 — 렌더 결과 자체가 전부다
    return [(p.name, p) for p in sorted(rdir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def export_result(work: Path, renderer: str, dest: Path, as_zip: bool = True,
                  order: list[str] | None = None) -> Path:
    """work(작업 폴더)의 renderer 결과를 dest 로 내보낸다. 만들어진 경로를 돌려준다.

    as_zip 이면 dest 는 zip 파일 경로, 아니면 폴더 경로. order 를 주면 그 순서·이름을 쓴다
    (원본 압축의 항목 목록)."""
    work = Path(work)
    if not (work / renderer).is_dir():
        raise FileNotFoundError(f"렌더 결과가 없습니다: {work / renderer}")
    pairs = _pairs(work, renderer, order)
    if not pairs:
        raise FileNotFoundError(f"내보낼 파일이 없습니다: {work / renderer}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if as_zip:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, path in pairs:
                zf.write(path, name)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for name, path in pairs:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return dest
