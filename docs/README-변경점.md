# 옛 README 대비 달라진 점

2026-09-25에 README를 사용자 중심으로 다시 쓰면서, 옛 README에 적혀 있던 내용을 코드·`config.yaml`과 하나씩 대조한 결과다.
**틀린 것**은 고쳐서 새 README에 반영했고, **빠진 것**은 새로 넣었다. 기술적인 배경 설명은 [내부 동작](내부동작.md)으로 옮겼다.

## 1. 틀렸던 것 (그대로 따라 하면 실패)

| 항목 | 옛 README | 실제 |
|---|---|---|
| 실행 명령 | `uv run skaldi samples` | **동작하지 않는다.** 이 프로젝트는 패키지로 설치되지 않아(`tool.uv.package` 미설정) `skaldi` 실행 파일이 만들어지지 않는다. `uv sync` 도 "Skipping installation of entry points" 경고를 낸다. 올바른 명령은 `uv run python -m skaldi samples` |
| `--help` | (언급 없음) | 한국어 윈도우 콘솔(cp949)에서 설명문의 `—` 때문에 `UnicodeEncodeError` 로 죽었다. 이번에 `skaldi/cli.py` 에서 출력 인코딩을 UTF-8로 맞춰 고쳤다 |
| 글자 외곽선 두께 | `stroke_ratio`(0.12), `bubble_stroke_ratio`(0.05) | 둘 다 **0.15** (`config.yaml`) |
| 용어집 | 문서 끝에서 "루트 + 입력 폴더 두 곳 병합"이라고 다시 설명 | 실제는 세 곳(루트 / 작품 옆 / 결과 폴더)이며 문서 앞쪽 표가 맞다. 뒤쪽 중복 설명을 지웠다 |
| 모델 미리 받기 | `download_models.py` 가 전부 받는 것처럼 설명 | 말풍선·글자 분할 모델(koharu)이 빠져 있었다. 이번에 스크립트에 추가했다 |
| Qwen 속도 | "768 기준 말풍선당 약 1분, 1024는 약 8분" | `config.yaml` 주석에는 "768 기준 5~8분"으로 적혀 있어 서로 어긋난다. 새 README에는 실측한 쪽(1분)만 남기고 설정 주석은 그대로 두었다. 다시 재 보고 한쪽으로 맞추는 게 좋다 |

## 2. 파이프라인 설명이 실제와 달라진 부분

옛 README는 6단계(탐지 → OCR → 순서·분류 → 번역 → 지우기 → 렌더링)로 적혀 있었다. 지금은 단계가 늘었다.

1. **말풍선·글자·효과음 분할 모델**(koharu, `layout.*`) — 탐지 직후에 돈다. 장당 약 2.4초. 효과음 판정(`layout.sfx`)과 지우기 범위 제한(`layout.erase`)에 쓴다. 옛 README에는 이 단계가 아예 없다.
2. **OCR 교차검증** — Baberu 판독을 비전 모델로 다시 읽어 헛읽기를 거른다. 관련 설정이 옛 README에 하나도 없었다: `cross_check`, `cross_check_min`(0.2), `min_confidence`(0.8), `vision_adopt`, `vision_adopt_min_chars`(3), `vision_consistency`(0.8), `oversized_ratio`(1.4).
3. **앞장 건너뛰기**(`frontmatter.*`) — 표지·속표지는 번역하지 않고 원본을 둔다. CLI `--no-skip-front` 로 끈다. 옛 README에 없다.
4. **글꼴 계열·굵기·획 색 측정** — 번역 전에 원문 획을 픽셀로 재서 글꼴 판정을 보정한다.

## 3. 새로 생겨서 옛 README에 없던 기능

- **뜻 없는 신음 원본 유지**(`filler`): `おおお`, `んほっ♥` 처럼 번역해도 음역밖에 안 되는 발성은 번역하지 않고 원본을 남긴다. 번역 모델의 문맥 판단과 글자 규칙이 모두 맞을 때만 적용한다.
- **큰 손글씨 신음 헛읽기 차단**: 글자가 페이지 중앙값의 1.4배 이상인 말풍선은 확신도와 상관없이 다시 읽고, 두 판독이 어긋나면 원본을 둔다.
- **인식 결과 교정**: 세로 말줄임표를 쌍점으로 읽는 버릇(`：` → `…`), `♀` 를 `９` 로 읽는 버릇을 되돌린다.
- **글자 크기 맞추기**: `render.bubble_group_floor`(말풍선 대사), `render.free_group_floor`(말풍선 밖 세로 글), `render.group_floor`(겹친 말풍선 묶음).
- **원문 크기 따라가기**: `render.match_size`, `render.src_font_scale`(0.79).
- **세로쓰기 선택**: `render.free_vertical`, `free_vertical_min_len`, `overlap_vertical`, `vertical_short_len`, `overlap_layout`.
- **글꼴 매핑·보정**: `render.fonts`(계열_굵기별 파일), `render.font_scale`(손글씨 1.42배), `fake_bold_styles`.
- **GUI 내보내기 탭**과 진행 막대(옛 README에도 일부 있으나 흩어져 있었다).

## 4. CLI 옵션 중 설명이 없던 것

`--backend {ollama,gemini}`, `--ocr {baberu,llm}`, `--export`, `--no-skip-front`.
(`--out`, `--out-root`, `--render`, `--rerender`, `--force`, `--files`, `--config`, `--zip`, `--debug`, `--export-only` 는 옛 README에도 있었다.)

## 5. JSON 필드 설명이 옛날 것

옛 README는 `overflow`, `refused`, `render`, `category`, `text_color`, `vertical` 만 설명한다. 지금 JSON에는 이런 것들이 더 있다.

`needs_review`(원본 유지 후 검토), `notes`(왜 그렇게 판정했는지), `style`·`weight`·`weight_ratio`(글꼴 계열·굵기), `body_box`(말풍선 안쪽 사각형), `target_box`·`rot_box`·`poly`(글자 자리), `group`(겹친 말풍선 묶음), `writing`(가로/세로), `cols`(세로 열 상자), `glyph_px`·`src_font_size`(원문 글자 크기), `split_from`(색으로 나눈 조각), `ocr_conf`·`ocr_backend`.

## 6. 문서 구조 변경

- 옛 README의 긴 기술 설명(말풍선 안쪽 상자 계산, flood fill 범위, 줄바꿈 규칙, 겹친 말풍선 자리 나누기 등)은 [docs/내부동작.md](내부동작.md)로 옮겼다. 왜 그렇게 했는지가 적혀 있어 지우지 않고 보존했다.
- 용어집 설명이 두 군데 있던 것을 한 군데로 합쳤다.
- 설치는 `scripts/setup.ps1` 한 줄로 끝나게 하고, 수동 설치는 접어 두었다.
