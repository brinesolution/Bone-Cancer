param(
    [ValidateSet('verify', 'test', 'lint', 'compile', 'phase2', 'phase3', 'phase4')]
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
    if ($Action -in @('phase2', 'phase3')) {
        $resultCode = 0
        $phaseScripts = if ($Action -eq 'phase2') {
            @(
                'scripts/01_index_dataset.py',
                'scripts/02_audit_dataset.py',
                'scripts/03_find_duplicates.py'
            )
        }
        else {
            @(
                'scripts/04_build_clean_manifest.py',
                'scripts/06_make_internal_splits.py'
            )
        }
        foreach ($script in $phaseScripts) {
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
            'phase4' { & $projectPython 'scripts/08_validate_annotations.py' }
        }
        $resultCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
exit $resultCode
