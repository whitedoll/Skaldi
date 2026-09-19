"""페이지 JSON을 읽기 순서대로 한 줄씩 출력한다.  사용: python scripts/show_json.py output/json/08.json"""
from __future__ import annotations

import json
import sys

for path in sys.argv[1:]:
    p = json.load(open(path, encoding="utf-8"))
    print(f"=== {path}  models={p['models']}")
    for w in p.get("warnings", []):
        print(f"  ! {w}")
    regs = sorted(p["regions"], key=lambda r: (r["order"] if r["order"] is not None else 999, r["id"]))
    for r in regs:
        o = r["order"] if r["order"] is not None else "-"
        flags = ("R" if r["render"] else "-") + ("O" if r["overflow"] else "-") + ("X" if r["refused"] else "-")
        print(f"{o:>3} id={r['id']:>2} {r['kind'][:6]} {r['category']:9s} {r['erase']:5s} {flags} "
              f"fs={r['font_size']} | {r['text_ja']}  =>  {r['text_ko']}")
