"""검수·실행 GUI (gradio).

  uv run python gui.py            # http://127.0.0.1:7860
  uv run python gui.py --port 7870

탭 1 실행: 폴더·zip 을 여러 개 골라 파이프라인으로 처리
           (translate.py 를 자식 프로세스로 한 번만 띄우고, 로그와 진행 막대를 실시간 표시)
탭 2 검수: 페이지별 원본/결과 보기, 번역문·분류·그리기 여부를 표에서 고친 뒤 저장하고 다시 그리기
"""
from __future__ import annotations

import argparse
import atexit
import html
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from mangatrans.config import ROOT, load_config
from mangatrans.export import export_result, list_works, renderers_in
from mangatrans.page import Page
from mangatrans.pipeline import MANUAL_NOTE, Pipeline
from mangatrans.progress import parse as parse_progress

CFG = load_config()
DEFAULT_OUT = CFG.abs(CFG.paths.output)
OUT = DEFAULT_OUT          # 현재 저장 폴더. 실행 탭에서 바꾸면 검수 탭도 이 폴더를 본다


def set_out(path: str) -> Path:
    """저장 폴더를 바꾼다. 비우면 config 의 paths.output 으로 되돌린다."""
    global OUT
    OUT = Path(path.strip().strip(chr(34))).expanduser() if path.strip() else DEFAULT_OUT
    return OUT
COLS = ["id", "order", "category", "render", "text_ja", "text_ko", "needs_review", "style", "weight", "notes"]
CATEGORIES = ["dialogue", "narration", "label", "sfx", "unknown"]
UPLOADS = CFG.abs(Path("input"))
IMAGE_SUFFIXES = {".webp", ".jpg", ".jpeg", ".png", ".bmp"}


def list_configs() -> list[str]:
    """프로젝트 루트의 config*.yaml 목록. 기본 config.yaml 을 맨 앞에 둔다."""
    names = sorted(q.name for q in ROOT.glob("config*.yaml"))
    if "config.yaml" in names:
        names.remove("config.yaml")
        names.insert(0, "config.yaml")
    return names or ["config.yaml"]
ARCHIVE_SUFFIXES = {".zip", ".cbz"}


# ---------------------------------------------------------------- 경로 선택
# tkinter 대화상자는 메인 스레드에서만 열 수 있는데 gradio 콜백은 작업 스레드에서 돈다.
# 그래서 대화상자 전용 프로세스를 하나 띄워 두고(상주), 요청을 한 줄씩 주고받는다.
# 클릭마다 새로 띄우면 매번 0.4초가 기동에 쓰이고, 그 사이 눌린 클릭이 밀려 안 열린 것처럼 보인다.
# 응답은 탭으로 이어 붙인 여러 경로다 (윈도 경로에는 탭이 들어갈 수 없다).
_PICKER = r'''
import sys, tkinter as tk
from tkinter import filedialog

# 경로에 일본어가 섞이면 기본 출력 인코딩(cp949)으로는 쓸 수 없어 프로세스가 죽는다.
# 그러면 부모는 빈 줄을 받고 "선택이 안 된" 것처럼 보이므로, UTF-8 바이트로 직접 쓴다.
def emit(text):
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()

FILETYPES = [("만화 압축·이미지", "*.zip *.cbz *.webp *.jpg *.jpeg *.png"),
             ("압축 파일", "*.zip *.cbz"), ("이미지", "*.webp *.jpg *.jpeg *.png"), ("모든 파일", "*.*")]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

def refocus():
    # 앞선 대화상자의 잔여 이벤트를 처리하고 포커스를 되찾는다.
    # (특히 폴더 대화상자 다음에 이걸 안 하면 다음 창이 뒤에 숨거나 안 열린다)
    root.update()
    root.deiconify(); root.lift(); root.focus_force(); root.withdraw()

emit("ready")

for raw in sys.stdin.buffer:
    kind = raw.decode("utf-8", "replace").strip()
    if not kind:
        continue
    if kind == "quit":
        break
    refocus()
    picked = []
    try:
        if kind == "dir":
            # 폴더 대화상자는 여러 개를 한 번에 고를 수 없으므로, 취소를 누를 때까지 반복해서 연다.
            while True:
                title = "폴더 추가 선택 (취소하면 마침)" if picked else "페이지 이미지 폴더 선택"
                p = filedialog.askdirectory(parent=root, title=title)
                if not p:
                    break
                picked.append(p)
                refocus()
        else:
            picked = list(filedialog.askopenfilenames(
                parent=root, title="zip/cbz 또는 이미지 선택 (여러 개 가능)", filetypes=FILETYPES))
    except Exception:
        picked = []
    root.update()
    emit("\t".join(picked))
'''

_picker_proc: subprocess.Popen | None = None
_picker_lock = threading.Lock()


def _picker() -> subprocess.Popen:
    """상주 대화상자 프로세스. 없거나 죽었으면 새로 띄운다."""
    global _picker_proc
    if _picker_proc is None or _picker_proc.poll() is not None:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        _picker_proc = subprocess.Popen(
            [sys.executable, "-c", _PICKER], cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _picker_proc.stdout.readline()          # "ready"
    return _picker_proc


def warm_picker() -> None:
    """앱 시작 때 미리 띄워 첫 클릭도 즉시 열리게 한다."""
    try:
        _picker()
    except Exception:  # noqa: BLE001
        pass


@atexit.register
def _close_picker() -> None:
    proc = _picker_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=3)
    except Exception:  # noqa: BLE001
        proc.kill()


def _ask(kind: str) -> list[str]:
    """상주 프로세스에 요청 한 번. 프로세스가 죽어 있었으면 한 번 다시 띄워 재시도한다."""
    for attempt in (1, 2):
        proc = _picker()
        try:
            proc.stdin.write(kind + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError):
            line = ""
        if line:
            return [p for p in line.rstrip("\r\n").split("\t") if p]
        if attempt == 1:                        # 죽은 프로세스였다면 새로 띄워 한 번 더
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            globals()["_picker_proc"] = None
    return []


def split_paths(text: str) -> list[str]:
    """입력창(한 줄에 하나)을 경로 목록으로. 붙여 넣을 때 따라오는 따옴표는 떼어 낸다."""
    return [ln.strip().strip('"') for ln in (text or "").splitlines() if ln.strip()]


def add_paths(current: str, new: list[str]) -> str:
    """이미 목록에 있는 경로는 건너뛰고 뒤에 덧붙인다."""
    paths = split_paths(current)
    have = {p.lower() for p in paths}
    for p in new:
        if p.lower() not in have:
            paths.append(p)
            have.add(p.lower())
    return "\n".join(paths)


def pick_paths(kind: str, current: str) -> str:
    """네이티브 대화상자로 파일(kind='file') 또는 폴더(kind='dir')를 고른다. 여러 개 선택 가능.
    고른 결과는 현재 목록 뒤에 덧붙는다. 취소하면 목록은 그대로."""
    if not _picker_lock.acquire(blocking=False):
        gr.Info("대화상자가 이미 열려 있습니다. 그 창에서 고르거나 닫아 주세요.")
        return current
    try:
        picked = _ask(kind)
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"파일 대화상자를 열지 못했습니다: {e}")
        return current
    finally:
        _picker_lock.release()
    if not picked:
        return current
    # 이미지 낱장을 골랐으면 그것이 든 폴더를 대상으로 삼는다 (파이프라인 입력 단위가 폴더·압축이므로)
    resolved: list[str] = []
    folded = False
    for raw in picked:
        p = Path(raw)
        if p.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(str(p.parent))
            folded = True
        else:
            resolved.append(str(p))
    if folded:
        gr.Info("이미지는 그것이 든 폴더를 입력으로 잡았습니다.")
    updated = add_paths(current, resolved)
    gr.Info(f"{len(split_paths(updated))}개 입력이 목록에 있습니다.")
    return updated


def pick_out(current: str) -> str:
    """저장 폴더 하나를 고른다. 취소하면 지금 값 유지. (pick_paths 는 목록에 덧붙이므로 따로 둔다)"""
    if not _picker_lock.acquire(blocking=False):
        gr.Info("대화상자가 이미 열려 있습니다.")
        return current
    try:
        picked = _ask("dir")
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"파일 대화상자를 열지 못했습니다: {e}")
        return current
    finally:
        _picker_lock.release()
    return picked[0] if picked else current


def accept_upload(files, current: str) -> str:
    """브라우저 업로드(원격 접속용). zip/cbz 는 각각 input/ 으로 복사하고, 이미지 여러 장은 input/<시각>/ 에 모은다."""
    if not files:
        return current
    paths = [Path(f if isinstance(f, str) else f.name) for f in files]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    archives = [p for p in paths if p.suffix.lower() in ARCHIVE_SUFFIXES]
    for a in archives:
        dest = UPLOADS / a.name
        shutil.copy2(a, dest)
        added.append(str(dest))
    images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    if images:
        folder = UPLOADS / datetime.now().strftime("upload_%Y%m%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        for p in images:
            shutil.copy2(p, folder / p.name)
        added.append(str(folder))
    if added:
        gr.Info(f"압축 {len(archives)}개 · 이미지 {len(images)}장을 올렸습니다.")
    return add_paths(current, added)


# ---------------------------------------------------------------- 진행 막대
def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}초"
    if sec < 3600:
        return f"{sec // 60}분 {sec % 60}초"
    return f"{sec // 3600}시간 {sec % 3600 // 60}분"


# gradio 의 HTML 컴포넌트는 .prose(max-width:65ch) 안에 들어가서 자식 div 폭이 65ch 로 잘린다.
# 막대가 화면 폭을 다 쓰도록 이 요소들에는 max-width 를 풀어 준다.
FULL = "max-width:none;width:100%"


def _bar(label: str, frac: float, right: str, color: str) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return (
        f'<div style="{FULL};margin:6px 0 10px">'
        f'<div style="{FULL};display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;opacity:.9">'
        f'<span>{label}</span><span style="opacity:.75">{right}</span></div>'
        f'<div style="{FULL};height:14px;border-radius:7px;background:rgba(128,128,128,.22);overflow:hidden">'
        # CSS transition 은 쓰지 않는다. 브라우저가 탭을 백그라운드로 돌리면 전환이 멈춘 채로 남아
        # 막대 길이가 실제 값과 어긋난다. 갱신이 잦으므로 애니메이션 없이도 충분히 부드럽다.
        f'<div style="max-width:none;height:100%;width:{pct:.1f}%;background:{color};'
        'border-radius:7px"></div></div></div>'
    )


def progress_html(st: dict) -> str:
    """실행 상태 dict 로 전체·현재 진행 막대 두 개를 그린다."""
    jobs = max(1, st.get("jobs_total", 1))
    pages = st.get("pages_total", 0)
    page_frac = (st.get("page_index", 0) / pages) if pages else 0.0
    overall = min(1.0, (st.get("job_index", 0) + page_frac) / jobs)
    if st.get("finished"):
        overall = 1.0

    elapsed = time.time() - st["t0"]
    right = f"경과 {fmt_time(elapsed)}"
    if 0.02 < overall < 1.0:
        right += f" · 남은 시간 약 {fmt_time(elapsed / overall - elapsed)}"

    done_jobs = jobs if st.get("finished") else min(jobs, st.get("job_index", 0))
    head = f"<b>전체 진행</b> &nbsp;{done_jobs}/{jobs} 작업 · {overall * 100:.0f}%"
    out = [_bar(head, overall, right, "linear-gradient(90deg,#f97316,#fbbf24)")]

    if st.get("finished"):
        note = st.get("note", "완료")
        out.append(f'<div style="max-width:none;font-size:13px;opacity:.85">{html.escape(note)}</div>')
        return "".join(out)

    job = st.get("job_name") or "준비 중"
    if pages:
        idx = st.get("page_index", 0)
        sub = f'<b>{html.escape(job)}</b> &nbsp;{idx + 1}/{pages} 장 · {(idx + 1) / pages * 100:.0f}%'
        right2 = html.escape(st.get("stage", ""))
        per = st.get("per_page")               # 장당 평균(초). 한 장이라도 끝나야 생긴다
        if per:
            right2 += f" · 장당 {fmt_time(per)}"
            left_pages = pages - idx - 1
            if left_pages > 0:
                right2 += f" · 이 작업 {fmt_time(per * left_pages)} 남음"
        out.append(_bar(sub, (idx + 1) / pages, right2, "linear-gradient(90deg,#3b82f6,#22d3ee)"))
        # 지금 처리 중인 파일 이름을 한 줄로 크게 (여러 장을 돌릴 때 어디까지 갔는지 바로 보이게)
        name = st.get("page_name", "")
        if name:
            out.append('<div style="max-width:none;font-size:13px;margin-top:-4px;opacity:.9">'
                       f'처리 중: <b>{html.escape(name)}</b></div>')
    else:
        out.append(f'<div style="max-width:none;font-size:13px;opacity:.85"><b>{html.escape(job)}</b> · '
                   f'{html.escape(st.get("stage", "모델 로드 중..."))}</div>')
    return "".join(out)


def idle_html(msg: str = "실행을 누르면 여기에 진행 상황이 표시됩니다.") -> str:
    return f'<div style="max-width:none;font-size:13px;opacity:.7;padding:6px 0">{html.escape(msg)}</div>'


# ---------------------------------------------------------------- 실행 탭
def run_pipeline(paths_text: str, out_root: str, config_name: str, renderer: str, force: bool, rerender: bool,
                 debug: bool = False, as_zip: bool = False):
    """translate.py 를 자식 프로세스로 실행하고 로그·진행도를 실시간으로 흘려보낸다.

    입력이 여러 개여도 프로세스는 하나만 띄운다 (모델을 한 번만 로드하기 위해)."""
    targets = split_paths(paths_text)
    missing = [p for p in targets if not Path(p).exists()]
    if not targets:
        yield idle_html("입력 경로가 없습니다. zip/폴더를 골라 주세요."), ""
        return
    if missing:
        yield idle_html("경로가 없습니다: " + ", ".join(missing)), "경로가 없습니다:\n" + "\n".join(missing)
        return

    cmd = [sys.executable, str(ROOT / "translate.py"), *targets, "--render", renderer]
    if config_name and config_name != "config.yaml":
        cmd += ["--config", str(ROOT / config_name)]
    out_root = out_root.strip().strip(chr(34))
    if out_root:
        cmd += ["--out-root", out_root]
    set_out(out_root)                       # 검수 탭이 같은 폴더를 보도록
    if force:
        cmd.append("--force")
    if rerender:
        cmd.append("--rerender")
    if debug:
        cmd.append("--debug")
    if as_zip:
        cmd.append("--zip")             # 폴더 입력도 zip 으로 (zip/cbz 입력은 원래부터 zip 으로 나온다)
    log = ("$ translate.py " + " ".join(f'"{t}"' for t in targets)
           + f" --render {renderer}"
           + (f" --config {config_name}" if config_name != "config.yaml" else "")
           + (f' --out-root "{out_root}"' if out_root else "")
           + (" --debug" if debug else "") + (" --zip" if as_zip else "") + "\n")
    st: dict = {"t0": time.time(), "jobs_total": len(targets), "job_index": 0,
                "job_name": Path(targets[0]).name, "stage": "모델 로드 중..."}
    yield progress_html(st), log

    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", "MANGATRANS_PROGRESS": "1",
                                 "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    assert proc.stdout is not None
    for line in proc.stdout:
        ev = parse_progress(line.rstrip("\r\n"))
        if ev is not None:                      # 진행 이벤트는 로그에 남기지 않는다
            kind = ev.get("ev")
            if kind == "plan":
                st["jobs_total"] = ev.get("total", st["jobs_total"])
            elif kind == "job":
                st.update(job_index=ev.get("index", 0), job_name=ev.get("name", ""),
                          pages_total=0, page_index=0, page_name="", stage="준비 중...")
                st.pop("pages_t0", None)
                st.pop("per_page", None)
            elif kind == "pages":
                st.update(pages_total=ev.get("total", 0), page_index=0)
            elif kind == "page":
                now = time.time()
                first = st.setdefault("pages_t0", now)      # 이 작업의 첫 장 시작 시각
                idx = ev.get("index", 0)
                if idx > 0:
                    st["per_page"] = (now - first) / idx     # 끝난 장 수로 나눈 평균
                st.update(page_index=idx, pages_total=ev.get("total", st.get("pages_total", 0)),
                          page_name=ev.get("name", ""))
            elif kind == "stage":
                st["stage"] = ev.get("name", "")
            elif kind == "done":
                st.update(page_index=st.get("pages_total", 0), stage="마무리")
            yield progress_html(st), log
            continue
        if "warn" in line.lower() or "symlink" in line.lower():
            continue
        log += line
        yield progress_html(st), log
    proc.wait()
    log += f"\n[종료 코드 {proc.returncode}]"
    st["finished"] = True
    st["note"] = (f"{st['jobs_total']}개 작업 완료 · 총 {fmt_time(time.time() - st['t0'])}"
                  if proc.returncode == 0 else f"실패 (종료 코드 {proc.returncode})")
    yield progress_html(st), log


# ---------------------------------------------------------------- 내보내기
def refresh_works():
    """저장 폴더 안의 작업 목록을 다시 읽는다."""
    works = list_works(OUT)
    return gr.update(choices=works, value=works[0] if works else None)


def work_renderers(work: str):
    rs = renderers_in(OUT / work) if work else []
    return gr.update(choices=rs, value=rs[0] if rs else None)


def do_export(work: str, renderer: str, as_zip: str, dest_dir: str) -> str:
    """고른 작업의 렌더 결과만 zip 또는 폴더로 내보낸다."""
    if not work or not renderer:
        return "작업과 렌더러를 고르세요. (목록이 비어 있으면 '작업 목록 새로고침')"
    src = OUT / work
    root = Path(dest_dir.strip().strip(chr(34))) if dest_dir.strip() else OUT
    zipped = as_zip.startswith("zip")
    dest = root / (work + (".zip" if zipped else "_결과"))
    try:
        made = export_result(src, renderer, dest, as_zip=zipped)
    except Exception as e:  # noqa: BLE001
        return f"내보내기 실패: {e}"
    if zipped:
        return f"저장했습니다: {made}  ({made.stat().st_size / 1e6:.1f}MB)"
    return f"저장했습니다: {made}  ({len(list(made.rglob('*')))}개 파일)"


# ---------------------------------------------------------------- 검수 탭
def list_pages() -> list[str]:
    """output/json/*.json 과 output/<zip>/json/*.json 을 모두 나열 (output 기준 상대 경로)."""
    items: list[str] = []
    if not OUT.exists():
        return items
    for jp in sorted(OUT.glob("json/*.json")) + sorted(OUT.glob("*/json/*.json")):
        items.append(str(jp.relative_to(OUT)).replace("\\", "/"))
    return items


def _paths(rel: str) -> tuple[Path, Path, Path]:
    """(json 경로, 출력 폴더, 원본 이미지 경로)"""
    jp = OUT / rel
    out_dir = jp.parent.parent
    page = Page.load(jp)
    src = Path(page.source)
    if not src.is_absolute():
        src = ROOT / src
    return jp, out_dir, src


def load_page(rel: str, renderer: str):
    if not rel:
        return None, None, [], ""
    jp, out_dir, src = _paths(rel)
    page = Page.load(jp)
    rendered = out_dir / renderer / src.name
    rows = [[r.id, r.order, r.category, r.render, r.text_ja, r.text_ko, r.needs_review, r.style, r.weight, r.notes]
            for r in page.ordered()]
    warn = "\n".join(page.warnings)
    orig = Image.open(src).convert("RGB") if src.exists() else None
    out = Image.open(rendered).convert("RGB") if rendered.exists() else None
    return orig, out, rows, warn


def save_and_render(rel: str, renderer: str, rows):
    if not rel:
        return None, "페이지를 선택하세요"
    jp, out_dir, src = _paths(rel)
    page = Page.load(jp)
    by_id = {r.id: r for r in page.regions}
    # gradio Dataframe 은 list[list] 또는 DataFrame 으로 온다
    try:
        rows = rows.values.tolist()  # pandas
    except AttributeError:
        pass
    for row in rows:
        try:
            rid = int(row[0])
        except (TypeError, ValueError):
            continue
        r = by_id.get(rid)
        if r is None:
            continue
        cat = str(row[2]).strip()
        if cat in CATEGORIES:
            r.category = cat  # type: ignore[assignment]
        before = r.render
        r.render = str(row[3]).strip().lower() in ("true", "1", "yes", "y", "✓")
        if r.category == "label" and r.render != before:
            r.notes = MANUAL_NOTE           # 라벨 규칙(표지 제외 등)이 다음 실행에서 되돌리지 않게
        r.text_ko = str(row[5]) if row[5] is not None else ""
        r.needs_review = str(row[6]).strip().lower() in ("true", "1", "yes", "y")
        st = str(row[7]).strip()
        if st in ("gothic", "mincho", "hand"):
            r.style = st
        wt = str(row[8]).strip()
        if wt in ("regular", "bold"):
            r.weight = wt
        # 그리기 여부에 맞춰 지우기 방식도 맞춘다
        if not r.render:
            r.erase = "none"
        elif r.erase == "none":
            r.erase = "white" if r.kind == "bubble_text" else "lama"
    page.save(jp)
    pipe = Pipeline(CFG, out_dir=out_dir)
    try:
        outs = pipe.render(src, page, [renderer], force=True)
    finally:
        pipe.close()
    page.save(jp)
    rendered = outs.get(renderer)
    img = Image.open(rendered).convert("RGB") if rendered and Path(rendered).exists() else None
    return img, "저장하고 다시 그렸습니다: " + (str(rendered) if rendered else "(실패)") + (
        ("\n" + "\n".join(page.warnings)) if page.warnings else "")


def build() -> gr.Blocks:
    with gr.Blocks(title="manga-transimage") as demo:
        gr.Markdown("## manga-transimage — 일본어 만화 → 한국어 식자")
        with gr.Tab("실행"):
            with gr.Row():
                path = gr.Textbox(label="입력 목록 (한 줄에 하나: 폴더 또는 zip/cbz)", scale=6, lines=4, max_lines=12,
                                  placeholder="G:\\manga_transimage\\samples\nG:\\manga\\book1.zip\nG:\\manga\\book2.zip")
                with gr.Column(scale=1, min_width=160):
                    file_btn = gr.Button("📦 zip/파일 선택")
                    dir_btn = gr.Button("📁 폴더 선택")
                    clear_btn = gr.Button("🗑 목록 비우기")
            gr.Markdown("<sub>파일 대화상자에서는 Ctrl·Shift 로 여러 개를 한 번에 고를 수 있고, "
                        "폴더는 취소를 누를 때까지 반복해서 물어봅니다. 목록 순서대로 처리합니다.</sub>")
            with gr.Row():
                out_box = gr.Textbox(label="저장 폴더", scale=6, value=str(DEFAULT_OUT), lines=1, max_lines=1,
                                     info="입력마다 이 폴더 안에 <이름>_ko 로 저장된다. 비우면 config 의 paths.output")
                out_btn = gr.Button("📂 저장 폴더 선택", scale=1)
            with gr.Accordion("브라우저에서 올리기 (다른 기기에서 접속할 때)", open=False):
                up = gr.File(label="zip/cbz 여러 개 또는 이미지 여러 장", file_count="multiple",
                             file_types=[".zip", ".cbz", ".webp", ".jpg", ".jpeg", ".png"])
            with gr.Row():
                cfg_dd = gr.Dropdown(list_configs(), value="config.yaml", label="설정 파일",
                                     info="작품별 식자 설정(글꼴·크기·줄간격)")
                renderer = gr.Dropdown(["pillow", "both", "pillow,anytext", "pillow,qwen"], value="pillow", label="렌더러")
                force = gr.Checkbox(label="JSON이 있어도 다시 분석 (--force)")
                rerender = gr.Checkbox(label="JSON만으로 다시 그리기 (--rerender)")
            with gr.Row():
                debug_cb = gr.Checkbox(label="디버그 그림 남기기",
                                       info="<작업폴더>/debug/ 에 글자 상자·말풍선 상자·글자 자리를 겹쳐 그린다")
                zip_cb = gr.Checkbox(label="결과를 zip 으로", value=True,
                                     info="zip/cbz 입력은 원래부터 zip 으로 나온다. 폴더 입력도 zip 으로 만든다")
            run_btn = gr.Button("실행", variant="primary")
            prog = gr.HTML(idle_html(), label="진행도")
            log = gr.Textbox(label="로그", lines=20, max_lines=40)

            # queue=False: 파이프라인이 도는 중에도 대화상자가 큐에 밀리지 않고 바로 열린다
            file_btn.click(lambda cur: pick_paths("file", cur), path, path, queue=False)
            dir_btn.click(lambda cur: pick_paths("dir", cur), path, path, queue=False)
            clear_btn.click(lambda: "", None, path, queue=False)
            up.upload(accept_upload, [up, path], path)
            out_btn.click(lambda cur: pick_out(cur), out_box, out_box, queue=False)
            run_btn.click(run_pipeline, [path, out_box, cfg_dd, renderer, force, rerender, debug_cb, zip_cb],
                          [prog, log])

        with gr.Tab("내보내기"):
            gr.Markdown("작업 폴더에는 중간 산출물(json·clean·src)이 함께 있다. "
                        "여기서는 **렌더 결과만** 원래 이름으로 뽑아낸다. "
                        "zip 입력으로 만든 작업은 원본 압축의 항목 이름·순서를 그대로 쓴다.")
            with gr.Row():
                work_dd = gr.Dropdown(list_works(DEFAULT_OUT), label="작업", scale=4)
                exp_rend = gr.Dropdown([], label="렌더러", scale=1)
                works_btn = gr.Button("작업 목록 새로고침", scale=1)
            with gr.Row():
                as_zip = gr.Radio(["zip 하나로", "폴더로"], value="zip 하나로", label="형식", scale=2)
                exp_dir = gr.Textbox(label="내보낼 위치", scale=4, lines=1, max_lines=1,
                                     placeholder="비우면 저장 폴더 안에 만든다")
                exp_btn = gr.Button("내보내기", variant="primary", scale=1)
            exp_msg = gr.Textbox(label="결과", lines=2)

            works_btn.click(refresh_works, None, work_dd)
            work_dd.change(work_renderers, work_dd, exp_rend)
            exp_btn.click(do_export, [work_dd, exp_rend, as_zip, exp_dir], exp_msg)

        with gr.Tab("검수"):
            with gr.Row():
                page_dd = gr.Dropdown(choices=list_pages(), label="페이지 (저장 폴더 기준)", scale=4)
                refresh = gr.Button("목록 새로고침", scale=1)
                rend_dd = gr.Dropdown(["pillow", "anytext", "qwen"], value="pillow", label="렌더러", scale=1)
            out_note = gr.Markdown(f"<sub>보는 중: {DEFAULT_OUT}</sub>")
            with gr.Row():
                orig_img = gr.Image(label="원본", type="pil", height=700)
                out_img = gr.Image(label="결과", type="pil", height=700)
            table = gr.Dataframe(headers=COLS, datatype=["number", "number", "str", "bool", "str", "str", "bool", "str", "str", "str"],
                                 interactive=True, wrap=True, label="영역 (text_ko·category·render·style·weight 수정 가능)")
            warn = gr.Textbox(label="경고", lines=3)
            save_btn = gr.Button("저장 후 다시 그리기", variant="primary")
            status = gr.Textbox(label="상태", lines=3)

            refresh.click(lambda: (gr.update(choices=list_pages()), f"보는 중: {OUT}"), None, [page_dd, out_note])
            page_dd.change(load_page, [page_dd, rend_dd], [orig_img, out_img, table, warn])
            rend_dd.change(load_page, [page_dd, rend_dd], [orig_img, out_img, table, warn])
            save_btn.click(save_and_render, [page_dd, rend_dd, table], [out_img, status])
    return demo


def free_port(start: int, tries: int = 20) -> int:
    """start 부터 실제로 바인딩되는 포트를 찾는다 (7860 은 다른 앱이 쓰는 경우가 많다).
    연결 시도만으로 판정하면 방금 닫힌 포트(TIME_WAIT)를 비어 있다고 잘못 보고 launch 가 실패한다."""
    import socket

    for port in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    port = free_port(args.port)
    if port != args.port:
        print(f"포트 {args.port} 이(가) 사용 중이라 {port} 로 엽니다.")
    warm_picker()          # 첫 클릭도 바로 열리도록 대화상자 프로세스를 미리 띄운다
    build().launch(server_name="127.0.0.1", server_port=port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
