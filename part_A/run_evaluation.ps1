$ErrorActionPreference = "Stop"

$PartARoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PSR_DATA_ROOT = Join-Path $PartARoot "data\labels"
$env:MPLBACKEND = "Agg"

$WorkspacePython = Join-Path (Split-Path -Parent $PartARoot) ".venv-samurai\Scripts\python.exe"
if (Test-Path $WorkspacePython) {
    $Python = $WorkspacePython
} else {
    $Python = "python"
}

& $Python (Join-Path $PartARoot "scripts\persistent_state_pipeline.py") `
    --state-models (Join-Path $PartARoot "models\state") `
    --output (Join-Path $PartARoot "results\final_evaluation")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Copy-Item `
    -LiteralPath (Join-Path $PartARoot "results\final_evaluation\timeline.png") `
    -Destination (Join-Path $PartARoot "figures\timeline.png") `
    -Force

& $Python (Join-Path $PartARoot "scripts\build_pipeline_figures.py")

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Part A evaluation complete."
Write-Output (Join-Path $PartARoot "results\final_evaluation\timeline.png")
