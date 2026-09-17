param(
    [ValidateSet(
        'verify', 'test', 'lint', 'compile', 'audit', 'annotate', 'phase2', 'phase3', 'phase4', 'phase5', 'hosp', 'evaluate', 'app',
        'baseline-densenet121', 'baseline-efficientnet-b3',
        'baseline-convnext-tiny', 'baseline-swin-tiny'
    )]
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
            'audit' { & $projectPython 'scripts/02_audit_dataset.py' --config 'configs/data.yaml' }
            'annotate' { & $projectPython 'scripts/07_launch_annotation_tool.py' }
            'test' { & $projectPython -m pytest -q }
            'lint' { & $projectPython -m ruff check . }
            'compile' { & $projectPython -m compileall -q src scripts app tools }
            'phase4' { & $projectPython 'scripts/08_validate_annotations.py' }
            'phase5' { & $projectPython 'scripts/09_train_baseline.py' --validate-artifacts }
            'hosp' { & $projectPython 'scripts/10_train_hosp_bonenet.py' --config 'configs/hosp_bonenet.yaml' }
            'evaluate' { & $projectPython 'scripts/12_evaluate.py' --config 'configs/evaluation.yaml' }
            'app' { & $projectPython -m streamlit run 'app/streamlit_app.py' }
            'baseline-densenet121' {
                & $projectPython 'scripts/09_train_baseline.py' --config 'configs/baseline_densenet121.yaml'
            }
            'baseline-efficientnet-b3' {
                & $projectPython 'scripts/09_train_baseline.py' --config 'configs/baseline_efficientnet_b3.yaml'
            }
            'baseline-convnext-tiny' {
                & $projectPython 'scripts/09_train_baseline.py' --config 'configs/baseline_convnext_tiny.yaml'
            }
            'baseline-swin-tiny' {
                & $projectPython 'scripts/09_train_baseline.py' --config 'configs/baseline_swin_tiny.yaml'
            }
        }
        $resultCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
exit $resultCode
