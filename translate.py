"""만화 번역 CLI.

사용 예)
  python translate.py samples                       # 기본 렌더러(config: render.default)
  python translate.py samples --render both         # pillow + anytext + qwen + 비교 이미지
  python translate.py samples --rerender            # JSON만 읽어 다시 그림
  python translate.py samples --force               # JSON이 있어도 다시 분석
  python translate.py samples --files 08.webp       # 특정 파일만
  python translate.py "samples/book.zip"            # zip/cbz → output/book.zip (같은 항목 이름)
  python translate.py a.zip b.zip samples           # 여러 개를 한 번에 (모델은 한 번만 로드)
  python translate.py samples --out-root D:/결과    # 저장 폴더 지정 → D:/결과/samples_ko/
  python translate.py samples --out D:/결과/여기    # 그 폴더에 바로 (이름 접미사 없음)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mangatrans.config import load_config
from mangatrans.export import export_result
from mangatrans.pipeline import Pipeline
from mangatrans.progress import emit
from mangatrans.render import resolve_names

ARCHIVE_EXTS = (".zip", ".cbz")


def _export_only(args, cfg, renderers) -> int:
    """이미 처리된 작업 폴더에서 결과만 내보낸다. 입력은 작업 폴더(<이름>_ko) 또는 원래 입력 이름."""
    root = cfg.abs(cfg.paths.output)
    rc = 0
    for target in args.inputs:
        work = target if (target / (args.export or renderers[0])).is_dir() else None
        if work is None:                       # 원래 입력 이름을 줬으면 작업 폴더를 찾아 준다
            name = target.stem if target.suffix else target.name
            cand = (args.out or root / f"{name}{cfg.paths.output_suffix}")
            work = cand if cand.is_dir() else target
        renderer = args.export or renderers[0]
        try:
            if args.zip:
                dest = work.parent / (work.name + ".zip")
            else:
                dest = work.parent / (work.name + "_결과")
            made = export_result(work, renderer, dest, as_zip=args.zip)
            print(f"→ {made}", flush=True)
        except FileNotFoundError as e:
            print(f"내보내기 실패: {e}", file=sys.stderr)
            rc = 2
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="일본어 만화 페이지를 한국어로 식자한다.")
    ap.add_argument("inputs", type=Path, nargs="+", help="페이지 이미지 폴더 또는 zip/cbz 파일 (여러 개 가능)")
    ap.add_argument("--out", type=Path, default=None,
                    help="결과를 이 폴더에 바로 넣는다 (이름 접미사를 붙이지 않음). 입력이 하나일 때만")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="결과를 담을 상위 폴더 (기본: config 의 paths.output). "
                         "입력마다 그 안에 <입력이름><접미사> 폴더를 만든다")
    ap.add_argument("--render", default=None, help="pillow | anytext | qwen | both | 쉼표 구분 목록")
    ap.add_argument("--rerender", action="store_true", help="모델을 돌리지 않고 JSON으로 다시 그림")
    ap.add_argument("--force", action="store_true", help="JSON이 있어도 다시 분석")
    ap.add_argument("--backend", choices=["ollama", "gemini"], default=None, help="번역 백엔드 덮어쓰기")
    ap.add_argument("--ocr", choices=["baberu", "llm"], default=None, help="OCR 백엔드 덮어쓰기")
    ap.add_argument("--files", nargs="*", default=None, help="입력 폴더 안에서 처리할 파일명")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml 경로")
    ap.add_argument("--export", default=None, help="내보낼 렌더러 (기본: 첫 번째 렌더러)")
    ap.add_argument("--zip", action="store_true",
                    help="폴더 입력도 결과를 zip 으로 내보낸다 (zip 입력은 원래부터 zip 으로 나온다)")
    ap.add_argument("--debug", action="store_true",
                    help="<작업폴더>/debug/ 에 글자 상자·말풍선 상자·글자 자리를 겹쳐 그린 그림을 남긴다")
    ap.add_argument("--export-only", action="store_true",
                    help="모델을 돌리지 않고, 이미 처리된 작업 폴더에서 결과만 내보낸다")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.llm.backend = args.backend
    if args.ocr:
        cfg.ocr.backend = args.ocr
    if args.out and len(args.inputs) > 1:
        print("--out 은 입력이 하나일 때만 쓸 수 있습니다. 여러 개면 --out-root 를 쓰세요.", file=sys.stderr)
        return 2
    if args.out and args.out_root:
        print("--out 과 --out-root 는 함께 쓸 수 없습니다.", file=sys.stderr)
        return 2
    if args.out_root:
        cfg.paths.output = args.out_root        # 입력마다 이 폴더 안에 <이름><접미사> 로 들어간다

    renderers = resolve_names(args.render, cfg)
    if args.export_only:
        return _export_only(args, cfg, renderers)
    # 입력이 여러 개여도 파이프라인(=모델)은 하나만 만들어 재사용한다.
    pipe = Pipeline(cfg, out_dir=args.out, debug=args.debug)
    total = len(args.inputs)
    emit("plan", total=total, renderers=renderers)
    failed = 0
    try:
        for i, target in enumerate(args.inputs):
            is_archive = target.is_file() and target.suffix.lower() in ARCHIVE_EXTS
            emit("job", index=i, total=total, name=target.name, kind="zip" if is_archive else "dir")
            if total > 1:
                print(f"\n=== [{i + 1}/{total}] {target} ===", flush=True)
            if is_archive:
                pipe.run_archive(target, renderers, rerender=args.rerender, force=args.force,
                                 files=[Path(f) for f in args.files] if args.files else None,
                                 export_renderer=args.export)
                continue
            if not target.is_dir():
                print(f"입력 폴더가 없습니다: {target}", file=sys.stderr)
                failed += 1
                continue
            files = [target / f for f in args.files] if args.files else None
            # --out 을 직접 주면 그 경로를 그대로 쓴다 (이름을 덧붙이지 않음)
            pipe.run(target, renderers, rerender=args.rerender, force=args.force, files=files,
                     nest=args.out is None)
            if args.zip:
                out_zip = pipe.out.parent / (pipe.out.name + ".zip")
                export_result(pipe.out, args.export or renderers[0], out_zip, as_zip=True)
                print(f"→ {out_zip}", flush=True)
    finally:
        pipe.close()
    emit("done", ok=failed == 0)
    return 2 if failed and failed == total else 0


if __name__ == "__main__":
    raise SystemExit(main())
