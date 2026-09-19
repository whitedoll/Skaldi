"""로컬 모델 벤치마크: 번역 후보 모델과 비전(순서·분류) 후보 모델을 샘플로 비교한다.

사용: python bench_models.py samples [--translate m1,m2,...] [--vision m1,m2] [--out output/bench]
결과: output/bench/report.md (원문·모델별 번역 나란히, 소요 시간, 거부 건수)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from PIL import Image

from mangatrans.config import load_config
from mangatrans.glossary import load_glossary
from mangatrans.llm import OllamaClient, looks_refused
from mangatrans.order import heuristic_order, order_and_classify
from mangatrans.page import Page
from mangatrans.pipeline import Pipeline, list_images
from mangatrans.translate import translate_page

DEFAULT_TRANSLATE = ["qwen3.5:9b", "gemma4:12b", "exaone3.5:7.8b", "qwen2.5:14b", "aya-expanse:8b"]
DEFAULT_VISION = ["qwen3.5:9b", "gemma4:12b"]


def prepare_pages(pipe: Pipeline, images: list[Path], cache: Path) -> dict[str, Page]:
    """탐지 + OCR 결과를 캐시. 순서는 휴리스틱, 말풍선 밖 글자는 미분류로 둔다."""
    pages: dict[str, Page] = {}
    for src in images:
        cpath = cache / f"{src.stem}.base.json"
        if cpath.exists():
            pages[src.name] = Page.load(cpath)
            continue
        image = Image.open(src).convert("RGB")
        page = Page(source=str(src), width=image.width, height=image.height)
        from mangatrans.detect import attach_bubbles
        from mangatrans.ocr import crop_region
        from mangatrans.page import Region

        regs = [Region(id=i, kind="bubble_text" if d.label == "text_bubble" else "free_text",
                       box=d.box, bubble_box=b, score=round(d.score, 3))
                for i, (d, b) in enumerate(attach_bubbles(pipe.detector.detect(image)))]
        by_id = {r.id: r for r in regs}
        for new_id, old_id in enumerate(heuristic_order(regs, page.width)):
            r = by_id[old_id]
            r.id = new_id
            page.regions.append(r)
        page.regions.sort(key=lambda r: r.id)
        for r in page.regions:
            r.text_ja = pipe.ocr.read(crop_region(image, r.box, pipe.cfg.ocr.crop_padding))
            r.order = r.id
            r.category = "dialogue" if r.kind == "bubble_text" else "unknown"
        page.save(cpath)
        pages[src.name] = page
    return pages


def bench_vision(cfg, client, images: list[Path], pages: dict[str, Page], models: list[str]) -> dict:
    out: dict[str, dict] = {}
    for m in models:
        c = copy.deepcopy(cfg)
        c.llm.vision_model = m
        res: dict[str, dict] = {}
        for src in images:
            page = copy.deepcopy(pages[src.name])
            image = Image.open(src).convert("RGB")
            t = time.time()
            order_and_classify(c, client, image, page)
            dt = time.time() - t
            seq = [r.id for r in page.ordered()]
            cats = {r.id: r.category for r in page.regions if r.kind == "free_text"}
            res[src.name] = {"time": dt, "order": seq, "cats": cats, "warnings": list(page.warnings),
                             "changed": sum(1 for i, rid in enumerate(seq) if rid != i)}
            print(f"[vision {m}] {src.name}: {dt:.1f}s, 순서 변경 {res[src.name]['changed']}개, "
                  f"분류 {cats}")
        out[m] = res
    return out


def bench_translate(cfg, client, images: list[Path], pages: dict[str, Page], models: list[str],
                    glossary: dict) -> dict:
    out: dict[str, dict] = {}
    for m in models:
        c = copy.deepcopy(cfg)
        c.llm.translate_model = m
        c.llm.translate_fallbacks = []
        res: dict[str, dict] = {}
        for src in images:
            page = copy.deepcopy(pages[src.name])
            # 벤치마크에서는 말풍선 밖 글자도 나레이션으로 간주해 번역 대상에 넣는다(효과음 제외 못함)
            for r in page.regions:
                if r.kind == "free_text":
                    r.category = "narration"
            t = time.time()
            translate_page(c, client, page, glossary)
            dt = time.time() - t
            items = {r.id: r.text_ko for r in page.regions}
            refused = sum(1 for r in page.regions if r.text_ja.strip() and looks_refused(r.text_ja, r.text_ko))
            res[src.name] = {"time": dt, "items": items, "refused": refused,
                             "warnings": list(page.warnings)}
            print(f"[translate {m}] {src.name}: {dt:.1f}s, 거부/누락 {refused}개")
        out[m] = res
    return out


def write_report(path: Path, images: list[Path], pages: dict[str, Page], vis: dict, tr: dict) -> None:
    L: list[str] = ["# 모델 벤치마크 보고", ""]
    if tr:
        L += ["## 번역 모델 요약", "", "| 모델 | 총 시간(s) | 거부/누락 | 경고 |", "|---|---:|---:|---|"]
        for m, res in tr.items():
            tt = sum(v["time"] for v in res.values())
            rf = sum(v["refused"] for v in res.values())
            warns = sum(len(v["warnings"]) for v in res.values())
            L.append(f"| {m} | {tt:.1f} | {rf} | {warns} |")
        L.append("")
        for src in images:
            page = pages[src.name]
            L += [f"### {src.name} 번역 비교", ""]
            heads = ["#", "원문"] + list(tr.keys())
            L.append("| " + " | ".join(heads) + " |")
            L.append("|" + "---|" * len(heads))
            for r in page.ordered():
                if not r.text_ja.strip():
                    continue
                row = [str(r.id), r.text_ja.replace("|", "\\|")]
                for m in tr:
                    ko = tr[m][src.name]["items"].get(r.id, "")
                    mark = " ⚠" if looks_refused(r.text_ja, ko) else ""
                    row.append((ko or "(없음)").replace("|", "\\|").replace("\n", " ") + mark)
                L.append("| " + " | ".join(row) + " |")
            times = ", ".join(f"{m}: {tr[m][src.name]['time']:.1f}s" for m in tr)
            L += ["", f"소요 시간: {times}", ""]
    if vis:
        L += ["## 비전 모델 (읽기 순서·분류)", "", "| 모델 | 페이지 | 시간(s) | 휴리스틱 대비 변경 | 분류 | 경고 |",
              "|---|---|---:|---:|---|---|"]
        for m, res in vis.items():
            for name, v in res.items():
                cats = ", ".join(f"{k}:{c}" for k, c in v["cats"].items())
                L.append(f"| {m} | {name} | {v['time']:.1f} | {v['changed']} | {cats} | {len(v['warnings'])} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--translate", default=",".join(DEFAULT_TRANSLATE))
    ap.add_argument("--vision", default=",".join(DEFAULT_VISION))
    ap.add_argument("--out", type=Path, default=Path("output/bench"))
    args = ap.parse_args()

    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input_dir)
    pipe = Pipeline(cfg)
    pages = prepare_pages(pipe, images, args.out)
    client = OllamaClient(cfg.llm)
    glossary = load_glossary(cfg.abs(cfg.paths.glossary), args.input_dir)

    tmodels = [m for m in args.translate.split(",") if m]
    vmodels = [m for m in args.vision.split(",") if m]
    tr = bench_translate(cfg, client, images, pages, tmodels, glossary) if tmodels else {}
    vis = bench_vision(cfg, client, images, pages, vmodels) if vmodels else {}
    (args.out / "raw.json").write_text(json.dumps({"translate": tr, "vision": vis}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    write_report(args.out / "report.md", images, pages, vis, tr)
    print(f"보고서: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
