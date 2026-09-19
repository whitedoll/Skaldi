"""페이지 처리 파이프라인: 탐지 → OCR → 순서·분류 → 번역 → JSON → 지우기 → 렌더링."""
from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.markup import escape

from .config import Config
from .detect import Detector, attach_bubbles
from .erase import Eraser
from .export import export_result
from .glossary import load_glossary
from .llm import make_client
from .ocr import crop_region, make_ocr
from .order import classify_styles, order_and_classify, pixel_style_check
from .page import Page, Region
from .progress import emit
from .render import make_renderer
from .render.compare import compare_sheet
from .translate import translate_page

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


MANUAL_NOTE = "수동 지정"           # 검수 탭에서 사람이 그리기 여부를 바꾼 영역


def apply_label_policy(page: Page, cfg: Config, is_cover: bool) -> None:
    """라벨(제목·이름표·간판·화면 글자)을 그릴지 정한다. 번역문은 분석 단계에서 이미 JSON 에 있다.

    분석 때가 아니라 그리기 직전에 정하므로, 예전 JSON 을 --rerender 해도 이 규칙이 적용된다.
    zip 의 첫 이미지(표지)는 제목 로고가 그림의 일부라 원본을 둔다(cover_labels 로 바꿀 수 있다).
    검수 탭에서 사람이 직접 정한 라벨(notes == MANUAL_NOTE)은 건드리지 않는다."""
    draw = cfg.render.draw_labels and (cfg.render.cover_labels or not is_cover)
    for r in page.regions:
        if r.category != "label" or r.notes == MANUAL_NOTE:
            continue
        r.render = draw and bool(r.text_ko.strip()) and not r.needs_review
        r.erase = ("white" if r.kind == "bubble_text" else "lama") if r.render else "none"
console = Console()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def save_image(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(path, quality=quality, subsampling=0)
    elif ext == ".webp":
        img.convert("RGB").save(path, quality=quality)
    else:
        img.save(path)


class Pipeline:
    """무거운 모델은 처음 필요할 때 한 번만 로드한다."""

    def __init__(self, cfg: Config, out_dir: Path | None = None):
        self.cfg = cfg
        self.out = out_dir or cfg.abs(cfg.paths.output)
        self.base_out = self.out          # named_dir 의 기준. self.out 은 입력마다 바뀐다.
        self._detector = None
        self._ocr = None
        self._client = None
        self._eraser = None
        self._renderers: dict[str, object] = {}
        self._cover: Path | None = None       # zip 의 첫 이미지(표지). 라벨을 그리지 않는다

    # ---- lazy loaders -------------------------------------------------
    @property
    def detector(self) -> Detector:
        if self._detector is None:
            console.print("[dim]탐지기 로드 중...[/dim]")
            self._detector = Detector(self.cfg.detector)
        return self._detector

    @property
    def ocr(self):
        if self._ocr is None:
            console.print(f"[dim]OCR({self.cfg.ocr.backend}) 로드 중...[/dim]")
            self._ocr = make_ocr(self.cfg)
        return self._ocr

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.cfg.llm)
        return self._client

    @property
    def eraser(self) -> Eraser:
        if self._eraser is None:
            self._eraser = Eraser(self.cfg)
        return self._eraser

    def renderer(self, name: str):
        if name not in self._renderers:
            self._renderers[name] = make_renderer(name, self.cfg)
        return self._renderers[name]

    # ---- paths --------------------------------------------------------
    def json_path(self, src: Path) -> Path:
        return self.out / "json" / f"{src.stem}.json"

    def clean_path(self, src: Path) -> Path:
        return self.out / "clean" / f"{src.stem}.png"

    def render_path(self, src: Path, renderer: str) -> Path:
        return self.out / renderer / src.name

    def close(self) -> None:
        for r in self._renderers.values():
            if hasattr(r, "close"):
                r.close()
        self._renderers.clear()           # 닫은 렌더러를 다음 입력에서 재사용하지 않도록

    # ---- stages -------------------------------------------------------
    def analyze(self, src: Path, glossary: dict) -> Page:
        """탐지·OCR·순서·번역까지 수행해 Page 를 만든다 (이미지는 만들지 않음)."""
        image = Image.open(src).convert("RGB")
        page = Page(source=str(src), width=image.width, height=image.height)
        t0 = time.time()

        emit("stage", name="탐지")
        dets = self.detector.detect(image)
        regions = [
            Region(id=i, kind="bubble_text" if d.label == "text_bubble" else "free_text",
                   box=d.box, bubble_box=bubble, score=round(d.score, 3))
            for i, (d, bubble) in enumerate(attach_bubbles(dets))
        ]
        # 번호를 휴리스틱 읽기 순서로 다시 매긴다. 비전 모델이 번호를 그대로 돌려줘도
        # 대체로 맞는 순서가 되고, 모델은 틀린 부분만 고치면 된다.
        from .order import heuristic_order
        by_id = {r.id: r for r in regions}
        for new_id, old_id in enumerate(heuristic_order(regions, page.width)):
            r = by_id[old_id]
            r.id = new_id
            page.regions.append(r)
        page.regions.sort(key=lambda r: r.id)
        page.models["detector"] = self.cfg.detector.repo
        t1 = time.time()

        emit("stage", name="OCR")
        ocr = self.ocr
        for r in page.regions:
            crop = crop_region(image, r.box, self.cfg.ocr.crop_padding)
            try:
                r.text_ja = ocr.read(crop)
            except Exception as e:  # noqa: BLE001
                page.warnings.append(f"OCR 실패 (id={r.id}): {e}")
            r.ocr_backend = ocr.name
        page.models["ocr"] = ocr.name
        t2 = time.time()

        emit("stage", name="순서·분류")
        if self.cfg.llm.backend == "ollama":
            order_and_classify(self.cfg, self.client, image, page)
            if self.cfg.render.match_style:
                classify_styles(self.cfg, self.client, image, page)
            # 번역 전에 글꼴 판정을 원문 획으로 보정한다 (번역 단계의 손글씨 효과음 판정이 이 값을 쓴다)
            pixel_style_check(np.array(image.convert("L")), page)
            page.models["vision"] = self.cfg.llm.vision_model
        else:
            from .order import heuristic_order
            pos = {rid: i for i, rid in enumerate(heuristic_order(page.regions, page.width))}
            for r in page.regions:
                r.order = pos[r.id]
                r.category = "dialogue" if r.kind == "bubble_text" else "unknown"
                r.render = r.kind == "bubble_text"
                r.erase = "white" if r.render else "none"
        t3 = time.time()

        emit("stage", name="번역")
        translate_page(self.cfg, self.client, page, glossary)
        t4 = time.time()
        console.print(
            f"  탐지 {len(page.regions)}개 {t1-t0:.1f}s · OCR {t2-t1:.1f}s · 순서 {t3-t2:.1f}s · 번역 {t4-t3:.1f}s"
        )
        return page

    def render(self, src: Path, page: Page, renderers: list[str], force: bool = False) -> dict[str, Path]:
        image = Image.open(src).convert("RGB")
        cpath = self.clean_path(src)
        if cpath.exists() and not force:
            clean = Image.open(cpath).convert("RGB")
        else:
            emit("stage", name="지우기")
            clean = self.eraser.clean(image, page)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            clean.save(cpath)

        results: dict[str, Path] = {}
        outputs: list[tuple[str, Image.Image]] = [("original", image)]
        for name in renderers:
            emit("stage", name=f"렌더링({name})")
            try:
                out = self.renderer(name).render(clean, image, page)
            except Exception as e:  # noqa: BLE001
                page.warnings.append(f"렌더러 {name} 실패: {e}")
                console.print(f"  [red]렌더러 {name} 실패: {e}[/red]")
                continue
            path = self.render_path(src, name)
            save_image(out, path, self.cfg.render.jpeg_quality)
            results[name] = path
            outputs.append((name, out))
        if len(outputs) > 2 and self.cfg.render.compare:
            sheet = compare_sheet(outputs, str(self.cfg.abs(self.cfg.paths.font)))
            spath = self.out / "compare" / f"{src.stem}.jpg"
            save_image(sheet, spath, 90)
            results["compare"] = spath
        return results

    # ---- zip / cbz ----------------------------------------------------
    def run_archive(self, archive: Path, renderers: list[str], rerender: bool = False, force: bool = False,
                    files: list[Path] | None = None, export_renderer: str | None = None) -> Path:
        """zip/cbz 안의 이미지를 처리하고, 같은 항목 이름으로 새 zip 을 만든다.
        작업 폴더: <out>/<zip이름><접미사>/ (src/ 에 풀고 json/ pillow/ 등 생성).
        결과: <out>/<zip이름><접미사>.zip"""
        t0 = time.time()
        emit("stage", name="압축 풀기")
        work = self.named_dir(archive)
        src_dir = work / "src"
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if not src_dir.exists() or force:
                shutil.rmtree(src_dir, ignore_errors=True)
                src_dir.mkdir(parents=True)
                zf.extractall(src_dir)
        self.out = work
        images = [src_dir / n for n in names if Path(n).suffix.lower() in IMAGE_EXTS]
        # 표지는 압축 안 순서의 첫 이미지 (결과 zip 도 같은 순서로 만든다). 일부 파일만 골라도 표지는 그대로
        self._cover = images[0] if images else None
        if files:
            wanted = {Path(f).name for f in files}
            images = [p for p in images if p.name in wanted]
        try:
            self.run(src_dir, renderers, rerender=rerender, force=force, files=images, nest=False)
        finally:
            self._cover = None

        emit("stage", name="zip 만들기")
        export = export_renderer or renderers[0]
        out_zip = work.parent / (work.name + archive.suffix)
        export_result(work, export, out_zip, as_zip=True, order=names)
        console.print(f"→ {out_zip}  (총 {time.time() - t0:.1f}s, {len(images)}장)", markup=False, style="bold")
        return out_zip

    def named_dir(self, source: Path) -> Path:
        """입력 이름 + 접미사로 된 전용 출력 폴더. 입력마다 결과가 섞이지 않게 한다."""
        name = source.stem if source.suffix else source.name
        return self.base_out / f"{name}{self.cfg.paths.output_suffix}"

    # ---- driver -------------------------------------------------------
    def run(self, input_dir: Path, renderers: list[str], rerender: bool = False, force: bool = False,
            files: list[Path] | None = None, nest: bool = True) -> None:
        if nest:
            self.out = self.named_dir(input_dir)
        images = files or list_images(input_dir)
        glossary = load_glossary(self.cfg.abs(self.cfg.paths.glossary), input_dir)
        emit("pages", total=len(images), out=str(self.out))
        console.print(f"[bold]{len(images)}장 처리, 렌더러: {', '.join(renderers)}[/bold]")
        t0 = time.time()
        try:
            self._run_pages(images, glossary, renderers, rerender, force)
        finally:
            self.close()
        console.print(f"[dim]처리 시간 {time.time() - t0:.1f}s ({len(images)}장, 장당 {(time.time() - t0) / max(1, len(images)):.1f}s)[/dim]")

    def _run_pages(self, images, glossary, renderers, rerender, force) -> None:
        total = len(images)
        for i, src in enumerate(images):
            emit("page", index=i, total=total, name=src.name)
            jpath = self.json_path(src)
            console.print(f"[cyan]{escape(src.name)}[/cyan]")
            if rerender:
                if not jpath.exists():
                    console.print("  [yellow]JSON이 없어 건너뜀[/yellow]")
                    continue
                page = Page.load(jpath)
            elif jpath.exists() and not force:
                console.print("  JSON 있음, 분석 건너뜀 (--force 로 다시)")
                page = Page.load(jpath)
            else:
                page = self.analyze(src, glossary)
                page.save(jpath)
            apply_label_policy(page, self.cfg, is_cover=self._cover is not None and src == self._cover)
            # 지우기·렌더링 단계 경고는 이번 실행에서 다시 만들어지므로 옛것을 지운다
            # (안 지우면 --rerender 할 때마다 이미 고친 문제의 경고가 계속 쌓인다)
            page.warnings = [w for w in page.warnings if not w.startswith(
                ("렌더러", "anytext", "qwen", "LaMa", "말풍선 넓히기", "스타일 측정"))]
            outs = self.render(src, page, renderers, force=force or rerender)
            page.save(jpath)  # font_size, overflow 등 렌더 결과 반영
            for w in page.warnings:
                console.print(f"  [yellow]! {w}[/yellow]")
            for name, p in outs.items():
                console.print(f"  → {name}: {p}", markup=False)
