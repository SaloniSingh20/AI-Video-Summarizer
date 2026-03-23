$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendPath = Join-Path $projectRoot "backend"
$frontendPath = Join-Path $projectRoot "frontend"
if (-not (Test-Path $frontendPath)) {
    $frontendPath = Join-Path (Split-Path -Parent $projectRoot) "frontend"
}
$envPath = Join-Path $projectRoot ".env"
$npmPath = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $npmPath) { $npmPath = "npm.cmd" }

if (-not (Test-Path $frontendPath)) {
    Write-Host "Frontend folder not found. Expected either '$projectRoot\\frontend' or sibling '../frontend'." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $envPath)) {
    Write-Host "Missing .env file at project root. Create it from .env.example first." -ForegroundColor Red
    exit 1
}

function Read-DotEnvValue([string]$Key) {
    if (-not (Test-Path $envPath)) { return "" }
    $line = Get-Content $envPath | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) { return "" }
    $value = ($line -split "=", 2)[1].Trim()
    return $value.Trim('"').Trim("'")
}

$openaiKey = Read-DotEnvValue "OPENAI_API_KEY"
$summaryModel = Read-DotEnvValue "OPENAI_SUMMARY_MODEL"
$transcribeModel = Read-DotEnvValue "OPENAI_TRANSCRIBE_MODEL"
if (-not $summaryModel) { $summaryModel = "llama3" }
if (-not $transcribeModel) { $transcribeModel = "base" }

$whisperModel = Read-DotEnvValue "WHISPER_MODEL"
if (-not $whisperModel) { $whisperModel = "base" }

$ollamaModel = Read-DotEnvValue "OLLAMA_MODEL"
if (-not $ollamaModel) { $ollamaModel = "llama3" }

$ollamaUrl = Read-DotEnvValue "OLLAMA_API_URL"
if (-not $ollamaUrl) { $ollamaUrl = "http://127.0.0.1:11434" }

$ffmpegPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpegPath) {
    $wingetFfmpeg = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    if (Test-Path $wingetFfmpeg) {
        $ffmpegPath = $wingetFfmpeg
    }
}

if (-not $ffmpegPath) {
    Write-Host "ffmpeg is not available. Install it and retry." -ForegroundColor Red
    exit 1
}

foreach ($port in @(3000, 8010)) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $procIds = $conn | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $procIds) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            try {
                & taskkill /PID $procId /T /F *> $null
            } catch {
                # Process may already be gone; continue startup.
            }
        }
    }
}

Write-Host "Starting one-URL local app at http://localhost:3000 ..." -ForegroundColor Cyan

$backendCmd = @"
Set-Location '$backendPath'
if (-not (Test-Path '.venv\Scripts\python.exe')) { powershell -ExecutionPolicy Bypass -File './scripts/setup_backend.ps1' }
`$env:OPENAI_API_KEY='$openaiKey'
`$env:OPENAI_SUMMARY_MODEL='$summaryModel'
`$env:OPENAI_TRANSCRIBE_MODEL='$transcribeModel'
`$env:WHISPER_MODEL='$whisperModel'
`$env:OLLAMA_MODEL='$ollamaModel'
`$env:OLLAMA_API_URL='$ollamaUrl'
`$env:FFMPEG_PATH='$ffmpegPath'
`$ffmpegDir = Split-Path `$env:FFMPEG_PATH -Parent
if (`$ffmpegDir) { `$env:PATH = "`$ffmpegDir;`$env:PATH" }
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"@

$backendCmd = $backendCmd -replace "--port 8000", "--port 8010"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Start-Sleep -Seconds 2

Push-Location $frontendPath
try {
    if (-not (Test-Path 'node_modules')) {
        & $npmPath install --no-fund --no-audit
    }
} finally {
    Pop-Location
}

$frontendCmd = @"
Set-Location '$frontendPath'
`$env:API_PROXY_TARGET='http://127.0.0.1:8010'
& '$npmPath' run dev -- --port 3000
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host "Frontend entry: http://localhost:3000" -ForegroundColor Green
Write-Host "Backend docs:   http://localhost:3000/api/docs (via proxy)" -ForegroundColor Green
Write-Host "Dev backend:    http://localhost:8010" -ForegroundColor DarkGray
