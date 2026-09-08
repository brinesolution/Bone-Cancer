param(
    [ValidateSet('verify', 'test', 'lint', 'compile')]
    [string]$Action = 'verify'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project interpreter missing. Follow requirements/README.md.'
}
Push-Location -LiteralPath $projectRoot
try {
    switch ($Action) {
        'verify' { & $projectPython 'scripts/00_verify_environment.py' --require-cuda }
        'test' { & $projectPython -m pytest -q }
        'lint' { & $projectPython -m ruff check . }
        'compile' { & $projectPython -m compileall -q src scripts app tools }
    }
    $resultCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $resultCode
