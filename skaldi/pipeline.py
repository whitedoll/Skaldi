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
from .columns import column_boxes, is_vertical_block, merge_unquoted, split_by_color
from .columns import measure as measure_columns
from .debug import debug_image
from .detect import Detector, attach_bubbles
from .erase import Eraser
from .export import export_result
from .frontmatter import find as find_frontmatter
from .glossary import glossary_paths, load_glossary
from .llm import make_client
from .ocr import cross_check, crop_region, dots_over_strokes, make_ocr
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

    def __init__(self, cfg: Config, out_dir: Path | None = None, debug: bool = False):
        self.cfg = cfg
        self.debug = debug          # True 면 <작업폴더>/debug/ 에 상자를 겹쳐 그린 그림을 남긴다
        self.out = out_dir or cfg.abs(cfg.paths.output)
        self.base_out = self.out          # named_dir 의 기준. self.out 은 입력마다 바뀐다.
        self._detector = None
        self._ocr = None
        self._client = None
        self._eraser = None
        self._layout = None
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

    @property
    def layout_model(self):
        if self._layout is None:
            from .layout import KoharuLayout
            console.print("[dim]말풍선 분할 모델 로드 중...[/dim]")
            self._layout = KoharuLayout(self.cfg)
        return self._layout

    def layout_items(self, src: Path, image: Image.Image, force: bool = False) -> list | None:
        """분할 결과. layout/<이름>.json 에 있으면 그것을, 없으면 모델을 돌려 저장한다. 꺼져 있거나 실패하면 None."""
        if not self.cfg.layout.enabled:
            return None
        from . import layout as L
        path = self.out / "layout" / f"{src.stem}.json"
        if not force:
            items = L.load(path)
            if items is not None:
                return items
        try:
            emit("stage", name="말풍선 분할")
            items = self.layout_model.predict(image)
        except Exception as e:  # noqa: BLE001 - 모델이 없거나 실패하면 예전 방식으로 그린다
            console.print(f"  [yellow]말풍선 분할 실패, 색으로 추정합니다: {e}[/yellow]")
            return None
        L.save(path, items)
        return items

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

    # ---- 앞장(표지) ----------------------------------------------------
    def find_front(self, images: list[Path]) -> dict[Path, str]:
        """번역하지 않고 원본을 둘 앞장 → 그렇게 본 근거."""
        if not self.cfg.frontmatter.skip or not images:
            return {}

        def detect(im):
            dets = self.detector.detect(im)
            return [d.label for d in dets], [d.box for d in dets]

        first_is_cover = self._cover is not None and images[0] == self._cover
        return find_frontmatter(images, detect, self.cfg.frontmatter, first_is_cover)

    def _read_columns(self, image: Image.Image, r: Region, ocr) -> tuple[str, float] | None:
        """여러 열 세로 글자를 열마다 읽어 (이어 붙인 글, 글자 수 가중 확신도). 열이 하나뿐이면 None."""
        cols, g = measure_columns(image, r.box)
        if len(cols) < 2:
            return None
        r.cols = column_boxes(r.box, cols, g, image.width, image.height)
        texts, weighted, total = [], 0.0, 0
        for b in r.cols:
            t, c = ocr.read_conf(image.crop(tuple(b)))
            texts.append(t)
            weighted += c * max(1, len(t))
            total += max(1, len(t))
        return "".join(texts), weighted / total

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
        # 색이 다른 세로 열 묶음(나레이션 + 분홍 대사)이 한 상자로 잡혔으면 나눈다
        regions = split_by_color(image, regions)
        for i, r in enumerate(regions):
            r.id = i
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
                if hasattr(ocr, "read_conf"):
                    r.text_ja, conf = ocr.read_conf(crop)
                    # 확신이 없고 세로로 긴 덩어리면 열마다 따로 읽어 본다. 224x224 로 줄이면 여러 열이
                    # 뭉개지지만 한 열씩이면 Baberu 가 학습한 크기다
                    if conf < self.cfg.ocr.min_confidence and is_vertical_block(r.box):
                        by_col = self._read_columns(image, r, ocr)
                        if by_col and by_col[1] > conf:
                            r.text_ja, conf = by_col
                    r.ocr_conf = round(conf, 3)
                else:
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
            # Baberu 가 못 읽는 손글씨에서 지어낸 문장은 번역 전에 거른다 (LLM OCR 이면 같은 모델이라 의미 없음)
            if self.cfg.ocr.cross_check and self.ocr.name == "baberu":
                emit("stage", name="OCR 교차검증")
                cross_check(self.cfg, self.client, image, page)
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

        dots_over_strokes(image, page)       # 점으로 잘못 읽은 손글씨 효과음은 원본을 둔다
        merge_unquoted(page)                 # 색으로 나눴지만 한 문장이었던 조각은 다시 합친다
        emit("stage", name="번역")
        translate_page(self.cfg, self.client, page, glossary)
        t4 = time.time()
        console.print(
            f"  탐지 {len(page.regions)}개 {t1-t0:.1f}s · OCR {t2-t1:.1f}s · 순서 {t3-t2:.1f}s · 번역 {t4-t3:.1f}s"
        )
        return page

    def render(self, src: Path, page: Page, renderers: list[str], force: bool = False) -> dict[str, Path]:
        image = Image.open(src).convert("RGB")
        page.layout = self.layout_items(src, image)
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
            if self.debug:
                dpath = self.out / "debug" / f"{src.stem}{'' if name == self.cfg.render.default else '_' + name}.jpg"
                save_image(debug_image(out, page, self.cfg.abs(self.cfg.paths.font)), dpath, 88)
                results[f"debug({name})"] = dpath
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
            self.run(src_dir, renderers, rerender=rerender, force=force, files=images, nest=False,
                     source=archive)
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
            files: list[Path] | None = None, nest: bool = True, source: Path | None = None) -> None:
        """source 는 사용자가 준 원래 입력(zip 이면 그 zip). 작품별 용어집을 그 옆에서 찾는다."""
        if nest:
            self.out = self.named_dir(input_dir)
        images = files or list_images(input_dir)
        paths = glossary_paths(self.cfg.abs(self.cfg.paths.glossary), source or input_dir, self.out)
        glossary = load_glossary(*paths)
        if len(paths) > 1:
            console.print("  용어집: " + ", ".join(str(p) for p in paths), markup=False, style="dim")
        emit("pages", total=len(images), out=str(self.out))
        front = self.find_front(images)
        if front:
            console.print(f"  앞장 {len(front)}장은 번역하지 않고 원본을 둡니다", style="dim")
        console.print(f"[bold]{len(images)}장 처리, 렌더러: {', '.join(renderers)}[/bold]")
        t0 = time.time()
        try:
            self._run_pages(images, glossary, renderers, rerender, force, front)
        finally:
            self.close()
        console.print(f"[dim]처리 시간 {time.time() - t0:.1f}s ({len(images)}장, 장당 {(time.time() - t0) / max(1, len(images)):.1f}s)[/dim]")

    def _run_pages(self, images, glossary, renderers, rerender, force, front=None) -> None:
        total = len(images)
        front = front or {}
        for i, src in enumerate(images):
            emit("page", index=i, total=total, name=src.name)
            jpath = self.json_path(src)
            console.print(f"[cyan]{escape(src.name)}[/cyan]")
            if src in front:
                console.print(f"  앞장 — 그리지 않고 원본을 둠 ({front[src]})", style="dim")
                # 그리지는 않아도 제목 원문·번역문은 JSON 에 남긴다. 표지 식자는 사람이 직접
                # 하게 되는데(글자를 지울 수 없다) 그때 번역문이 있어야 쓸모가 있다.
                if self.cfg.frontmatter.analyze and not rerender:
                    if jpath.exists() and not force:
                        console.print("  JSON 있음, 분석 건너뜀", style="dim")
                    else:
                        page = self.analyze(src, glossary)
                        page.save(jpath)
                        console.print(f"  → json: {jpath}", markup=False)
                        for w in page.warnings:
                            console.print(f"  [yellow]! {w}[/yellow]")
                else:
                    emit("stage", name="앞장")
                # 폴더 입력은 내보내기가 렌더 폴더만 훑으므로(export._pairs) 원본을 거기
                # 복사해 두어야 결과에서 빠지지 않는다.
                for name in renderers:
                    dest = self.render_path(src, name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                continue
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
