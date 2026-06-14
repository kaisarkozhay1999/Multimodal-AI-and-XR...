$ErrorActionPreference = "Stop"

$PartBRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $PartBRoot
$Python = Join-Path $WorkspaceRoot ".venv-samurai\Scripts\python.exe"
$Server = Join-Path $WorkspaceRoot "samurai_annotation\server.py"
$Url = "http://127.0.0.1:8767"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "SAMURAI environment is missing: $Python"
}

$env:SAMURAI_TRAIN_ROOT = Join-Path $PartBRoot "paper_tracking\sequences"
$env:SAMURAI_CLASSES_PATH = Join-Path $PartBRoot "config\paper_classes.json"
$env:SAMURAI_PROMPT_DATA_DIR = Join-Path $PartBRoot "paper_tracking\prompts"
$env:SAMURAI_REVIEW_DATA_DIR = Join-Path $PartBRoot "paper_tracking\annotations"
$env:SAMURAI_TRACKS_DIR = Join-Path $PartBRoot "paper_tracking\tracks"
$env:SAMURAI_REVIEW_TEMP_DIR = Join-Path $PartBRoot "paper_tracking\temp"

$Existing = Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction SilentlyContinue
if (-not $Existing) {
    Start-Process -FilePath $Python `
        -ArgumentList @($Server, "--port", "8767") `
        -WorkingDirectory $WorkspaceRoot `
        -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

Start-Process $Url
Write-Output "Part B paper annotation opened at $Url"
