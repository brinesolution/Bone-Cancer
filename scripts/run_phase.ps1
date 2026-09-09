param(
    [ValidateSet('verify', 'test', 'lint', 'compile', 'phase2')]
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
    if ($Action -eq 'phase2') {
        $resultCode = 0
        foreach ($script in @(
            'scripts/01_index_dataset.py',
            'scripts/02_audit_dataset.py',
            'scripts/03_find_duplicates.py'
        )) {
            & $projectPython $script
            $resultCode = $LASTEXITCODE
            if ($resultCode -ne 0) {
                break
            }
        }
    }
    else {
        switch ($Action) {
            'verify' { & $projectPython 'scripts/00_verify_environment.py' --require-cuda }
            'test' { & $projectPython -m pytest -q }
            'lint' { & $projectPython -m ruff check . }
            'compile' { & $projectPython -m compileall -q src scripts app tools }
        }
        $resultCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
exit $resultCode
