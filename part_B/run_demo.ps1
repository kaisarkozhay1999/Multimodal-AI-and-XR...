$ErrorActionPreference = "Stop"

$PartBRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspacePython = Join-Path (Split-Path -Parent $PartBRoot) ".venv-samurai\Scripts\python.exe"

if (Test-Path $WorkspacePython) {
    $Python = $WorkspacePython
} else {
    $Python = "python"
}

& $Python (Join-Path $PartBRoot "scripts\gradio_demo.py")
exit $LASTEXITCODE
