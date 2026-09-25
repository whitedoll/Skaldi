# Skaldi 한 번에 설치 (Windows PowerShell)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Anytext -Gemini
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipOllama   # 번역 모델은 나중에
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -ComfyUI      # Qwen 렌더러까지(약 25GB)
#
# 하는 일: uv 확인 → 파이썬 환경 설치 → Ollama 모델 받기 → 로컬 모델·폰트 받기 → 동작 점검

[CmdletBinding()]
param(
    [switch]$Anytext,       # AnyText 렌더러 (비교용 생성 렌더러, 가중치 약 2GB)
    [switch]$Gemini,        # Gemini 번역 백엔드 패키지
    [switch]$ComfyUI,       # Qwen 렌더러용 ComfyUI + 모델 (약 25GB)
    [switch]$SkipOllama,    # Ollama 모델 내려받기 건너뛰기
    [switch]$SkipModels     # 탐지·OCR·분할·LaMa 모델 미리 받기 건너뛰기
)

$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 은 콘솔 기본 인코딩이 ANSI 라 한글 출력이 깨진다
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }
function Warn($text) { Write-Host "  ! $text" -ForegroundColor Yellow }

# ---- 1. uv (파이썬 환경 관리자) -----------------------------------------
Step "uv 확인"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Warn "uv 가 없습니다. 설치합니다 (https://docs.astral.sh/uv/)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv 설치 후에도 찾을 수 없습니다. 새 터미널을 열고 다시 실행하세요."
    }
}
uv --version

# ---- 2. 파이썬 패키지 ----------------------------------------------------
Step "파이썬 환경 설치 (uv sync)"
$syncArgs = @("sync")
if ($Gemini)  { $syncArgs += @("--extra", "gemini") }
if ($Anytext) { $syncArgs += @("--extra", "anytext") }
& uv @syncArgs
if ($LASTEXITCODE -ne 0) { throw "uv sync 실패" }

# ---- 3. Ollama 모델 ------------------------------------------------------
if (-not $SkipOllama) {
    Step "Ollama 모델"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Warn "Ollama 가 없습니다. https://ollama.com/download 에서 설치한 뒤 이 스크립트를 -SkipModels 로 다시 실행하세요."
    } else {
        # config.yaml 에 적힌 모델 이름을 그대로 쓴다 (설정을 바꾸면 여기도 따라간다)
        $models = & uv run python -c "from skaldi.config import load_config; c=load_config().llm; print('\n'.join(dict.fromkeys([c.vision_model, c.translate_model] + list(c.translate_fallbacks))))"
        $have = (& ollama list) -join "`n"
        foreach ($m in $models) {
            $m = $m.Trim()
            if (-not $m) { continue }
            if ($have -match [regex]::Escape($m)) {
                Write-Host "  이미 있음: $m"
            } else {
                Write-Host "  받는 중: $m" -ForegroundColor Green
                & ollama pull $m
                if ($LASTEXITCODE -ne 0) { Warn "내려받기 실패: $m (건너뜁니다)" }
            }
        }
    }
}

# ---- 4. 로컬 모델·폰트 ---------------------------------------------------
if (-not $SkipModels) {
    Step "탐지기·OCR·분할·LaMa 모델과 폰트"
    $dlArgs = @("run", "python", "scripts/download_models.py")
    if ($Anytext) { $dlArgs += "--anytext" }
    & uv @dlArgs
    if ($LASTEXITCODE -ne 0) { Warn "모델 내려받기 실패 — 처음 실행할 때 자동으로 다시 받습니다" }
}

# ---- 5. ComfyUI (선택) ---------------------------------------------------
if ($ComfyUI) {
    Step "ComfyUI + Qwen 모델 (약 25GB)"
    & uv run python scripts/install_comfyui.py
    if ($LASTEXITCODE -ne 0) { Warn "ComfyUI 설치 실패" }
}

# ---- 6. 점검 -------------------------------------------------------------
Step "점검"
& uv run python scripts/setup_check.py

Write-Host "`n설치 완료. 다음으로:" -ForegroundColor Green
Write-Host "  uv run skaldi samples          # 예제 이미지 번역"
Write-Host "  uv run python gui.py           # 브라우저 화면으로 쓰기"
