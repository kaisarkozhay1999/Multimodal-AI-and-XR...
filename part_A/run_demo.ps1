$ErrorActionPreference = "Stop"

$PartARoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspacePython = Join-Path (Split-Path -Parent $PartARoot) ".venv-samurai\Scripts\python.exe"

if (Test-Path $WorkspacePython) {
    $Python = $WorkspacePython
} else {
    $Python = "python"
}

& $Python (Join-Path $PartARoot "scripts\gradio_demo.py")
exit $LASTEXITCODE

