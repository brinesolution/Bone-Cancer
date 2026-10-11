<#
.SYNOPSIS
Verify the clone, run portable checks, start Streamlit, and open its web interface.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\run.ps1
.EXAMPLE
.\run.ps1 -Tests full -Port 8502
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8501,
    [ValidateSet('quick', 'full', 'none')][string]$Tests = 'quick',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$NoBrowser,
    [switch]$CheckOnly,
    [switch]$AutoSetup,
    [string]$ModelAssetsUrl = $env:BONE_CANCER_ASSETS_URL,
    [string]$ModelAssetsSha256 = ''
)

$script:RunPortWasSpecified = $PSBoundParameters.ContainsKey('Port')

function Get-AvailablePort {
    param([int]$Preferred)
    for ($candidate = $Preferred; $candidate -le [Math]::Min(65535, $Preferred + 50); $candidate++) {
        $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $candidate)
        try { $listener.Start(); return $candidate } catch [Net.Sockets.SocketException] { continue } finally { $listener.Stop() }
    }
    throw 'No free localhost port is available in the requested range.'
}

function Get-WebArguments {
    param([string]$Root, [int]$Port)
    $app = [IO.Path]::GetFullPath((Join-Path $Root 'app/streamlit_app.py'))
    return @('-m', 'streamlit', 'run', ('"' + $app + '"'), '--server.address', '127.0.0.1', '--server.port', [string]$Port, '--server.headless', 'true', '--browser.gatherUsageStats', 'false')
}

function Wait-WebHealth {
    param([string]$Url, [int]$TimeoutSeconds = 60, [Diagnostics.Process]$ServerProcess = $null)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if ($null -ne $ServerProcess -and $ServerProcess.HasExited) { throw 'Streamlit exited before becoming healthy; inspect logs/web_app_stderr.log.' }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            $content = if ($response.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($response.Content) } else { [string]$response.Content }
            if ($response.StatusCode -eq 200 -and $content.Trim() -eq 'ok') { return $true }
        }
        catch { }
        Start-Sleep -Milliseconds 300
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'Web-interface health check timed out.'
}

function Invoke-RunMain {
    $root = [IO.Path]::GetFullPath($PSScriptRoot)
    $requestedDevice = $Device
    $requestedUrl = $ModelAssetsUrl
    $requestedDigest = $ModelAssetsSha256
    $requestedCheckOnly = $CheckOnly
    . (Join-Path $root 'setup.ps1')
    $CheckOnly = $requestedCheckOnly
    $profile = Get-InstallProfile -Requested $requestedDevice -Root $root
    $env:BONE_CANCER_DEVICE = $profile
    $python = Get-ProjectPython -Root $root
    $reportPath = Join-Path $root 'logs/run_readiness.json'
    $setupArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'setup.ps1'), '-Device', $requestedDevice, '-CheckOnly', '-SkipDataset', '-ReportPath', $reportPath)
    $shell = (Get-Process -Id $PID).Path
    & $shell @setupArguments
    if ($LASTEXITCODE -ne 0 -and $AutoSetup) {
        $installArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'setup.ps1'), '-Device', $requestedDevice)
        if ($requestedUrl) { $installArguments += @('-ModelAssetsUrl', $requestedUrl) }
        if ($requestedDigest) { $installArguments += @('-ModelAssetsSha256', $requestedDigest) }
        & $shell @installArguments
        if ($LASTEXITCODE -ne 0) { throw 'Automatic setup could not prepare the trained app; see logs/setup_report.json.' }
        & $shell @setupArguments
        if ($LASTEXITCODE -ne 0) { throw 'Readiness still failed after setup; see logs/run_readiness.json.' }
        $profile = Get-InstallProfile -Requested $requestedDevice -Root $root
        $env:BONE_CANCER_DEVICE = $profile
    }
    elseif ($LASTEXITCODE -ne 0) { throw 'App readiness failed. Run setup.ps1 first, or use run.ps1 -AutoSetup.' }
    if ($Tests -ne 'none') {
        $testArguments = @('-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=logs/pytest_run')
        if ($Tests -eq 'quick') { $testArguments += Get-PortableTestFiles }
        Invoke-ProjectCommand -Executable $python -Arguments $testArguments -Root $root
    }
    if ($CheckOnly) { Write-Host 'Readiness and requested tests passed; server was not started.'; return 0 }
    $statePath = Join-Path $root 'logs/web_app_process.json'
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$state.pid)" -ErrorAction SilentlyContinue
        if ($null -ne $existing -and $existing.ExecutablePath -eq $python -and $existing.CommandLine.Contains((Join-Path $root 'app\streamlit_app.py')) -and (-not $script:RunPortWasSpecified -or ([Uri]$state.url).Port -eq $Port)) {
            if (($state.PSObject.Properties['device'] -and $state.device -ne $profile) -or (-not $state.PSObject.Properties['device'] -and $requestedDevice -eq 'cpu')) {
                throw 'An app is running with a different compute profile. Stop that instance or choose a different -Port.'
            }
            $null = Wait-WebHealth -Url ($state.url + '/_stcore/health') -TimeoutSeconds 5
            Write-Host "Web interface already running: $($state.url)"
            if (-not $NoBrowser) { Start-Process -FilePath $state.url }
            return 0
        }
    }
    $selectedPort = Get-AvailablePort -Preferred $Port
    $webArguments = Get-WebArguments -Root $root -Port $selectedPort
    $null = New-Item -ItemType Directory -Path (Join-Path $root 'logs') -Force
    $process = Start-Process -FilePath $python -ArgumentList $webArguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root 'logs/web_app_stdout.log') -RedirectStandardError (Join-Path $root 'logs/web_app_stderr.log')
    $url = "http://127.0.0.1:$selectedPort"
    try {
        $null = Wait-WebHealth -Url ($url + '/_stcore/health') -ServerProcess $process
        Write-JsonFile -Path $statePath -Value @{ pid = $process.Id; url = $url; project_root = $root; device = $profile; tests = $Tests; started_utc = [DateTime]::UtcNow.ToString('o') }
    }
    catch {
        # This invocation owns this process; existing app instances are untouched.
        if (-not $process.HasExited) { $process.Kill(); $null = $process.WaitForExit(5000) }
        throw
    }
    finally { $process.Dispose() }
    Write-Host "Web interface ready: $url"
    if (-not $NoBrowser) { Start-Process -FilePath $url }
    return 0
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    $env:PYTHONUTF8 = '1'
    $env:NO_ALBUMENTATIONS_UPDATE = '1'
    $env:MLFLOW_DISABLE_AGENT_HINT = '1'
    [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
    try { exit (Invoke-RunMain) } catch { Write-Error $_; exit 1 }
}
