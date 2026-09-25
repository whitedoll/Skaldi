# Skaldi

일본어 만화를 **한국어로 식자까지 끝낸 이미지**로 바꿔 주는 개인용 도구. 폴더나 zip을 넣으면 번역된 폴더·zip이 나온다.

- 말풍선을 찾아 원문을 지우고, 번역문을 원문과 비슷한 글꼴·크기로 그려 넣는다.
- 번역은 내 컴퓨터의 Ollama 모델로 한다(인터넷에 원고를 보내지 않는다).
- 효과음과 뜻 없는 신음은 건드리지 않고 원본 그림을 남긴다.
- 결과가 마음에 안 들면 **검수 화면**에서 번역문만 고치고 그 페이지만 다시 그릴 수 있다.

이름은 고대 노르드어 *skáld*(이야기를 자기 언어로 읊는 시인)에서 왔다.

## 필요한 것

| | 내용 |
|---|---|
| 그래픽카드 | NVIDIA, VRAM 12GB 이상 권장 (RTX 3080 Ti 12GB에서 개발·실측) |
| 저장 공간 | 약 15GB (번역 모델 11GB + 나머지 모델 1.5GB + 파이썬 패키지). Qwen 렌더러까지 쓰면 +25GB |
| [Ollama](https://ollama.com/download) | 번역·분류용. 설치 후 실행 중이어야 한다 |
| 윈도우 | 아래 설치 스크립트는 Windows PowerShell 기준. macOS·Linux는 [수동 설치](#수동-설치) 참고 |

## 설치

PowerShell을 열고 프로젝트 폴더에서 한 줄이면 된다.

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

이 스크립트가 순서대로 해 준다.

1. `uv`(파이썬 환경 관리자)가 없으면 설치
2. `uv sync` 로 파이썬 패키지 설치 (GUI 포함)
3. `config.yaml` 에 적힌 Ollama 모델 내려받기 (약 12GB, 이미 있으면 건너뜀)
4. 탐지·OCR·분할·LaMa 모델과 글꼴 내려받기 (약 1.5GB)
5. GPU와 Ollama가 제대로 보이는지 점검

선택 옵션:

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipOllama   # 번역 모델은 나중에
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gemini       # Gemini 번역 백엔드도 설치
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Anytext      # AnyText 렌더러(비교용)까지
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -ComfyUI      # Qwen 렌더러용 ComfyUI(+25GB)
```

설치가 끝난 뒤 환경만 다시 점검하려면:

```bash
uv run python scripts/setup_check.py
```

<details>
<summary><b>수동 설치</b> (macOS·Linux이거나 스크립트를 쓰고 싶지 않을 때)</summary>

```bash
uv sync                                        # 파이썬 패키지 (GUI 포함)
uv sync --extra gemini --extra anytext         # 선택 기능까지
ollama pull hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q2_K_XL   # 번역·분류 모델
uv run python scripts/download_models.py       # 탐지·OCR·분할·LaMa·글꼴
uv run python scripts/download_models.py --anytext   # AnyText 가중치까지 (약 2GB)
uv run python scripts/install_comfyui.py       # Qwen 렌더러용 ComfyUI (약 25GB)
uv run python scripts/setup_check.py           # 점검
```

`config.yaml` 의 `llm.translate_fallbacks`(gemma4:12b, qwen3.5:9b, exaone3.5:7.8b)는 번역이 거부될 때 쓰는 예비 모델이다. 없어도 돌아간다.
</details>

## 써 보기

가장 쉬운 방법은 GUI다.

```bash
uv run python gui.py
```

브라우저에서 http://127.0.0.1:7860 이 열린다(포트가 쓰이고 있으면 빈 포트를 골라 알려 준다). zip이나 폴더를 고르고 실행을 누르면 된다.

명령줄로도 같다.

```bash
uv run python -m skaldi samples          # samples/ 폴더의 이미지를 번역 → output/samples_ko/
uv run python -m skaldi "book.zip"       # zip/cbz → output/book_ko.zip
uv run python -m skaldi a.zip b.zip      # 여러 개를 한 번에 (모델은 한 번만 올린다)
```

> `uv run skaldi` 는 동작하지 않는다. 이 프로젝트는 패키지로 설치하지 않아서 `uv run python -m skaldi` 를 써야 한다.

한 장에 약 40~60초 걸린다(탐지·OCR·순서 판정·번역·지우기·그리기 전부 포함, RTX 3080 Ti 기준).

### 자주 쓰는 옵션

| 옵션 | 쓰임 |
|---|---|
| `--files 08.webp 09.webp` | 그 파일만 처리 (표지 건너뛰기도 자동으로 꺼진다) |
| `--rerender` | 모델을 돌리지 않고 저장된 JSON으로 다시 그리기만 |
| `--force` | JSON이 있어도 처음부터 다시 분석 |
| `--out-root D:/결과` | 저장 폴더 지정 → `D:/결과/<이름>_ko/` |
| `--out D:/결과/여기` | 그 폴더에 바로 (이름 접미사 없음, 입력 하나일 때만) |
| `--zip` | 폴더 입력도 결과를 zip으로 (zip 입력은 원래부터 zip) |
| `--export-only --zip` | 이미 처리된 작업에서 결과 이미지만 다시 뽑기 |
| `--config config_book.yaml` | 작품별 식자 설정 사용 |
| `--no-skip-front` | 표지·속표지도 번역 |
| `--render both` | pillow·anytext·qwen 전부 그리고 비교 이미지까지 |
| `--debug` | 상자·글자 자리를 겹쳐 그린 그림을 `debug/` 에 |

전체 목록은 `uv run python -m skaldi --help`.

## 결과물

입력마다 전용 폴더가 생긴다. 이름은 `<입력 이름>_ko` (접미사는 `config.yaml` 의 `paths.output_suffix`).

```
output/
  book_ko/
    src/      압축을 푼 원본
    json/     좌표·원문·번역·판정 결과 (여기를 고치고 --rerender)
    clean/    원문을 지운 이미지
    pillow/   ← 완성된 번역 이미지
  book_ko.zip  ← 결과 zip (뷰어로 바로 볼 수 있다)
```

남에게 주거나 뷰어로 볼 때는 결과 이미지만 필요하므로, GUI의 **내보내기 탭**이나 `--export-only` 로 렌더 결과만 뽑아낼 수 있다. zip 입력으로 만든 작업은 원본 압축의 항목 이름과 순서를 그대로 쓴다.

## GUI

```bash
uv run python gui.py          # --port 7870 으로 포트 지정 가능
```

- **실행 탭** — 입력을 한 줄에 하나씩 여러 개 넣고 순서대로 처리한다. `📦 zip/파일 선택`(Ctrl·Shift로 여러 개), `📁 폴더 선택`(취소를 누를 때까지 반복해서 물어본다), `🗑 목록 비우기`. 다른 기기에서 접속했다면 "브라우저에서 올리기"로 업로드한다. 진행 막대에 전체·현재 작업 진행률과 남은 시간, 지금 도는 단계가 실시간으로 뜬다.
- **검수 탭** — 페이지를 고르면 원본과 결과를 나란히 보여 준다. 표에서 `text_ko`(번역문), `category`(대사/나레이션/라벨/효과음/신음), `render`(그릴지), `style`·`weight`(글꼴 계열·굵기)를 고치고 **저장 후 다시 그리기**를 누르면 그 페이지만 다시 그린다.
- **내보내기 탭** — 이미 처리한 작업에서 결과 이미지만 zip이나 폴더로 뽑는다.
- **설정 파일** — 루트의 `config*.yaml` 중에서 고른다. 작품마다 식자 설정을 달리할 때 쓴다(파일을 새로 만들었다면 GUI를 다시 띄워야 목록에 나온다).

## 작품별 설정

### 용어집 — 이름 표기와 말투

인물 이름, 말투 방침, 참고 사항을 적어 두면 번역에 반영된다. 세 곳을 읽어 합치고 뒤에 오는 것이 이긴다.

| 순서 | 위치 | 쓰임 |
|---|---|---|
| 1 | `glossary.yaml` (프로젝트 루트) | 모든 작품 공통 |
| 2 | `<zip이름>.glossary.yaml` (zip 옆) 또는 `<폴더>/glossary.yaml` | 그 작품 전용 |
| 3 | `<결과폴더>/<이름>_ko/glossary.yaml` | 한 번 돌린 뒤 고칠 때 (`--force` 해도 안 지워진다) |

```yaml
# G:/manga/book.glossary.yaml
names:
  高崎梨杏: 타카사키 리안
  藤ねぇ: 후지 누나
style: |
  주인공은 반말, 선배에게는 존댓말.
notes:
  - 3권부터 두 사람은 사귀는 사이
```

`names` 는 같은 키가 있으면 뒤엣것이 덮어쓰고, `style` 과 `notes` 는 이어 붙는다. 어떤 파일을 읽었는지는 실행 로그에 `용어집: ...` 으로 찍힌다.

### 식자 설정 — 글꼴·크기·지우기

그림에 관한 것은 `config.yaml` 이다. 작품별로 다르게 하려면 복사해서 `config_<이름>.yaml` 로 두고 고른다.

```bash
uv run python -m skaldi "book.zip" --config config_book.yaml
```

자주 만지는 값만 추리면 이렇다. 나머지는 파일 안 주석과 [내부 동작](docs/내부동작.md)에 있다.

| 값 | 기본 | 뜻 |
|---|---|---|
| `render.src_font_scale` | 0.79 | 글자 크기 전체 배율. 글자가 작으면 올린다 |
| `render.bubble_group_floor` | 0.8 | 한 페이지 대사 글자 크기를 서로 맞추는 정도. 0이면 끔 |
| `render.bubble_fit` | 0.7 | 1에 가까울수록 글자가 말풍선 곡선을 안 벗어나지만 작아진다 |
| `render.match_style` | true | 원문 글꼴 계열(고딕/명조/손글씨)을 따라간다 |
| `render.bubble_stroke_ratio` | 0.15 | 말풍선 안 글자의 흰 외곽선 두께. 0이면 없음 |
| `erase.widen_bubbles` | false | 좁은 말풍선을 넓혀 글자를 키운다(원본 말풍선 모양이 바뀐다) |
| `layout.enabled` | true | 말풍선·글자·효과음 분할 모델 사용(장당 +2.4초) |
| `frontmatter.skip` | true | 표지·속표지는 번역하지 않고 원본을 둔다 |

## 잘 안 될 때

| 증상 | 확인할 것 |
|---|---|
| 번역이 시작되지 않는다 | Ollama가 실행 중인지, 모델을 받았는지 → `uv run python scripts/setup_check.py` |
| 너무 느리다 | GPU를 못 잡은 경우가 많다. 위 점검에서 GPU 이름이 보이는지 확인 |
| 글자가 일본어로 남아 있다 | 효과음·신음·읽기 실패로 **일부러 원본을 남긴 것**이다. JSON의 `notes` 에 이유가 적힌다. 번역시키려면 검수 탭에서 `category` 를 `dialogue` 로 바꾸고 `render` 를 켠다 |
| 번역문이 말풍선을 넘친다 | JSON에 `overflow: true` 로 표시된다. 검수 탭에서 문장을 줄이는 게 가장 깔끔하다 |
| 글자가 너무 작다 | `render.src_font_scale` 을 올리거나, `erase.widen_bubbles: true` 로 말풍선을 넓힌다 |
| 이름이 매번 다르게 번역된다 | 용어집 `names` 에 적는다 |
| 어떤 모델도 번역을 거부한다 | JSON에 `refused: true` 로 남는다. 검수 탭에서 직접 입력 |

## 더 읽을거리

- [내부 동작](docs/내부동작.md) — 왜 이렇게 만들었는지(헛읽기 거르기, 말풍선 안쪽 계산, 글자 크기 맞추기, 모델 선택 근거)
- [옛 README 대비 변경점](docs/README-변경점.md) — 이 문서를 다시 쓰면서 고친 것과 새로 넣은 것

## 주의

- 개인용 도구다. 번역 결과를 배포할 생각이라면 원저작물의 권리를 먼저 확인할 것.
- Gemini 백엔드(`--backend gemini`)는 번역 전용이다. 무료 등급은 입력이 구글 제품 개선에 쓰이고, 성인물 입력은 약관 위반이다.
- 생성 렌더러(AnyText, Qwen)는 글자 정확도가 Pillow보다 낮다. 비교용으로 둔 것이다. Qwen은 말풍선 하나에 약 1분(`qwen.crop_size` 768 기준, 1024면 약 8분) 걸리고 첫 실행에 ComfyUI 기동·모델 적재로 2~3분이 더 붙는다.
- 분할 모델(koharu)은 Manga109로 학습된 학술·비상업 용도 모델이다.
