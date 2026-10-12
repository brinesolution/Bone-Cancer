<#
.SYNOPSIS
Prepare this Windows clone, download missing assets, and verify its environment.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.EXAMPLE
.\setup.ps1 -Device cpu -ModelAssetsUrl https://host/bone-cancer-runtime.zip
.EXAMPLE
.\setup.ps1 -CheckOnly
.EXAMPLE
.\setup.ps1 -ExportAssets
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$CheckOnly,
    [switch]$SkipDataset,
    [string]$ModelAssetsUrl = $env:BONE_CANCER_ASSETS_URL,
    [string]$ModelAssetsSha256 = '',
    [string]$MainDatasetUrl = $env:BONE_CANCER_MAIN_DATASET_URL,
    [string]$MainDatasetSha256 = $env:BONE_CANCER_MAIN_DATASET_SHA256,
    [string]$RoboflowApiKey = $env:ROBOFLOW_API_KEY,
    [string]$RoboflowArchiveUrl = $env:BONE_CANCER_ROBOFLOW_ARCHIVE_URL,
    [string]$RoboflowArchiveSha256 = $env:BONE_CANCER_ROBOFLOW_ARCHIVE_SHA256,
    [string]$RoboflowArchivePath = $env:BONE_CANCER_ROBOFLOW_ARCHIVE_PATH,
    [string]$RoboflowSourceRoot = $env:BONE_CANCER_ROBOFLOW_SOURCE_ROOT,
    [switch]$ExportAssets,
    [switch]$RunTests,
    [string]$ReportPath = ''
)

$script:OfficialBtxrdUrl = 'https://ndownloader.figshare.com/files/50653575'
$script:OfficialBtxrdMD5 = '59132a5036d030580ccade8add8f13df'
$script:RuntimeBundleSha256 = '1116d9088a67a7d556e9527db4eae9d82ed1760bc45db902f17317037d0153a3'
$script:SelectedAlias = 'hosp_bonenet.btxrd.subtype.selected'
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-AssetHash {
    param([string]$LiteralPath, [ValidateSet('SHA256', 'MD5')][string]$Algorithm = 'SHA256')
    $hasher = [Security.Cryptography.HashAlgorithm]::Create($Algorithm)
    $stream = [IO.File]::OpenRead($LiteralPath)
    try {
        $bytes = $hasher.ComputeHash($stream)
        return [PSCustomObject]@{ Hash = [BitConverter]::ToString($bytes).Replace('-', '') }
    }
    finally { $stream.Dispose(); $hasher.Dispose() }
}

function Get-ProjectPython {
    param([Parameter(Mandatory = $true)][string]$Root)
    return [IO.Path]::GetFullPath((Join-Path $Root '.venv\Scripts\python.exe'))
}

function Get-SafeProjectPath {
    param([string]$Root, [string]$Relative)
    $base = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ([string]::IsNullOrWhiteSpace($Relative) -or [IO.Path]::IsPathRooted($Relative)) {
        throw 'A project-relative asset path is required.'
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $base $Relative))
    if (-not $candidate.StartsWith($base + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset path escapes the project: $Relative"
    }
    $ancestor = $candidate
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Asset path crosses a project link: $Relative"
            }
        }
        if ($ancestor -eq $base) { break }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $candidate
}

function Get-BtxrdDatasetState {
    param([string]$Root)
    $source = Join-Path $Root 'dataset\BTXRD'
    $images = Join-Path $source 'images'
    $metadata = Test-Path -LiteralPath (Join-Path $source 'dataset.xlsx') -PathType Leaf
    $count = 0
    if (Test-Path -LiteralPath $images -PathType Container) {
        $count = @(Get-ChildItem -LiteralPath $images -File | Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff') }).Count
    }
    return [PSCustomObject]@{ present = [bool]($metadata -and $count -gt 0); metadata_exists = [bool]$metadata; image_count = $count; root = $source }
}

function Get-MainDatasetState {
    param([string]$Root, [string]$Python = (Get-ProjectPython -Root $Root))
    $scriptPath = Get-SafeProjectPath -Root $Root -Relative 'scripts/dataset_restore.py'
    $mainPath = Get-SafeProjectPath -Root $Root -Relative 'dataset/Main'
    $manifest = Get-SafeProjectPath -Root $Root -Relative 'configs/main_dataset_manifest.csv'
    $metadata = Get-SafeProjectPath -Root $Root -Relative 'configs/main_dataset_manifest.json'
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf) -or -not (Test-Path -LiteralPath $manifest -PathType Leaf) -or -not (Test-Path -LiteralPath $metadata -PathType Leaf)) {
        return [PSCustomObject]@{ present = $false; valid = $false; error = 'Portable Main dataset contract is missing.'; total_images = 0 }
    }
    Push-Location -LiteralPath $Root
    try {
        $output = & $Python -X utf8 $scriptPath validate --main $mainPath --manifest $manifest --metadata $metadata 2>&1
        $exitCode = $LASTEXITCODE
        $json = ($output | Where-Object { $_ -is [string] } | Select-Object -Last 1)
        if ($json) {
            try {
                $state = $json | ConvertFrom-Json
                if ($exitCode -eq 0 -and $state.valid) { return [PSCustomObject]@{ present = $true; valid = $true; error = ''; total_images = $state.total_images; counts = $state.counts; source_counts = $state.source_counts } }
                return [PSCustomObject]@{ present = $false; valid = $false; error = [string]$state.error; total_images = 0 }
            }
            catch { }
        }
        return [PSCustomObject]@{ present = $false; valid = $false; error = 'Main dataset validation could not be completed.'; total_images = 0 }
    }
    finally { Pop-Location }
}

function Restore-MainDataset {
    param([string]$Root, [string]$Python, [string[]]$SourceRoots)
    $scriptPath = Get-SafeProjectPath -Root $Root -Relative 'scripts/dataset_restore.py'
    $destination = Get-SafeProjectPath -Root $Root -Relative 'dataset/Main'
    $manifest = Get-SafeProjectPath -Root $Root -Relative 'configs/main_dataset_manifest.csv'
    $metadata = Get-SafeProjectPath -Root $Root -Relative 'configs/main_dataset_manifest.json'
    $arguments = @('-X', 'utf8', $scriptPath, 'restore')
    foreach ($sourceRoot in $SourceRoots) { $arguments += @('--source-root', $sourceRoot) }
    $arguments += @('--destination', $destination, '--manifest', $manifest, '--metadata', $metadata)
    Invoke-ProjectCommand -Executable $Python -Arguments $arguments -Root $Root
}

function Invoke-RoboflowApiRequest {
    param([System.Net.Http.HttpClient]$Client, [string]$Uri)
    return $Client.GetAsync($Uri).GetAwaiter().GetResult()
}

function Invoke-RoboflowArchiveRequest {
    param([System.Net.Http.HttpClient]$Client, [Uri]$Uri)
    return $Client.GetAsync($Uri, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
}

function Get-RoboflowExportArchive {
    param([string]$Root, [string]$ApiKey)
    if ([string]::IsNullOrWhiteSpace($ApiKey)) { throw 'ROBOFLOW_API_KEY is required for an official Roboflow export.' }
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $downloadClient = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds(10)
    $downloadClient.Timeout = [TimeSpan]::FromSeconds(120)
    $apiUri = 'https://api.roboflow.com/normal-bones/bone-cancer-detection-xa7ru/1/folder?api_key=' + [Uri]::EscapeDataString($ApiKey) + '&nocache=true'
    $downloadUri = $null
    try {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            if ($timer.Elapsed.TotalSeconds -ge 120) { break }
            $response = Invoke-RoboflowApiRequest -Client $client -Uri $apiUri
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ([int]$response.StatusCode -eq 202) {
                $response.Dispose()
                if ($timer.Elapsed.TotalSeconds -ge 120) { break }
                Start-Sleep -Seconds 1
                continue
            }
            $response.Dispose()
            if (-not $response.IsSuccessStatusCode) { throw 'Roboflow export request failed.' }
            try { $payload = $body | ConvertFrom-Json } catch { throw 'Roboflow returned an invalid export response.' }
            $downloadUri = [string]$payload.export.link
            if (-not $downloadUri) { throw 'Roboflow export response did not contain an archive link.' }
            break
        }
        if (-not $downloadUri) { throw 'Roboflow export did not finish within the 120-second timeout.' }
        $uri = New-Object Uri($downloadUri, [UriKind]::Absolute)
        if ($uri.Scheme -ne 'https' -or $uri.UserInfo) { throw 'Roboflow returned an unsupported archive URL.' }
        $archive = Get-SafeProjectPath -Root $Root -Relative ('logs/downloads/roboflow-' + [Guid]::NewGuid().ToString('N') + '.zip')
        $partial = $archive + '.partial'
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force
        try {
            $download = Invoke-RoboflowArchiveRequest -Client $downloadClient -Uri $uri
            if (-not $download.IsSuccessStatusCode) { throw 'Roboflow archive download failed.' }
            $input = $download.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $output = [IO.File]::Create($partial)
            try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose(); $download.Dispose() }
            Move-Item -LiteralPath $partial -Destination $archive
            return $archive
        }
        catch {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
            throw 'Roboflow archive download failed; check the export link and network, then retry.'
        }
    }
    catch {
        $message = [string]$_.Exception.Message
        if ($message -match '^Roboflow ') { throw $message }
        throw 'Roboflow export failed; check ROBOFLOW_API_KEY, network access, and project permissions, then retry.'
    }
    finally { $client.Dispose(); $downloadClient.Dispose() }
}

function Get-RuntimeAssetState {
    param([string]$Root)
    $registryPath = Join-Path $Root 'artifacts\checkpoints\registry.json'
    if (-not (Test-Path -LiteralPath $registryPath -PathType Leaf)) {
        return [PSCustomObject]@{ present = $false; checkpoint_valid = $false; missing = @('artifacts/checkpoints/registry.json'); source_run = '' }
    }
    $registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
    $property = $registry.PSObject.Properties[$script:SelectedAlias]
    if ($null -eq $property) { throw 'Selected BTXRD model is absent from the project registry.' }
    $record = $property.Value
    $checkpoint = Get-SafeProjectPath -Root $Root -Relative ([string]$record.checkpoint_path)
    $run = ([string]$record.source_run).Replace('\', '/').TrimEnd('/')
    $null = Get-SafeProjectPath -Root $Root -Relative $run
    $required = @(
        ([string]$record.checkpoint_path).Replace('\', '/'),
        "$run/metrics.json", "$run/status.json", "$run/contract.json",
        'artifacts/metrics/btxrd_subtype_review.json',
        'artifacts/metrics/btxrd_subtype_temperature.json',
        'artifacts/metrics/btxrd_robustness.json',
        'artifacts/metrics/final_model_comparison.csv'
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Get-SafeProjectPath -Root $Root -Relative $_) -PathType Leaf) })
    $valid = $false
    if (Test-Path -LiteralPath $checkpoint -PathType Leaf) {
        $valid = (Get-AssetHash -LiteralPath $checkpoint -Algorithm SHA256).Hash -eq [string]$record.checkpoint_sha256
    }
    return [PSCustomObject]@{ present = [bool]($missing.Count -eq 0 -and $valid); checkpoint_valid = [bool]$valid; missing = @($missing); source_run = $run }
}

function Save-VerifiedDownload {
    param(
        [string]$Url, [string]$Destination, [string]$Checksum,
        [ValidateSet('SHA256', 'MD5')][string]$Algorithm = 'SHA256'
    )
    if ([string]::IsNullOrWhiteSpace($Checksum) -or $Checksum -notmatch '^[a-fA-F0-9]+$') { throw 'Download checksum is required.' }
    $expectedLength = if ($Algorithm -eq 'MD5') { 32 } else { 64 }
    if ($Checksum.Length -ne $expectedLength) { throw 'Download checksum length is invalid.' }
    $uri = New-Object Uri($Url, [UriKind]::Absolute)
    if (($uri.Scheme -ne 'https' -and $uri.Scheme -ne 'file' -and -not ($uri.Scheme -eq 'http' -and $uri.IsLoopback)) -or $uri.UserInfo) {
        throw 'Downloads require HTTPS (or a local file/loopback test server).'
    }
    $destinationPath = [IO.Path]::GetFullPath($Destination)
    if ((Test-Path -LiteralPath $destinationPath -PathType Leaf) -and (Get-AssetHash -LiteralPath $destinationPath -Algorithm $Algorithm).Hash -eq $Checksum) { return $destinationPath }
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destinationPath) -Force
    $partial = $destinationPath + '.partial'
    try {
        if ($uri.Scheme -eq 'file') {
            Copy-Item -LiteralPath $uri.LocalPath -Destination $partial -Force
        }
        else {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $uri -OutFile $partial -UseBasicParsing -TimeoutSec 120 -ErrorAction Stop
        }
        if ((Get-AssetHash -LiteralPath $partial -Algorithm $Algorithm).Hash -ne $Checksum) { throw 'Downloaded asset checksum mismatch; existing data was preserved.' }
        if (Test-Path -LiteralPath $destinationPath) {
            [IO.File]::Replace($partial, $destinationPath, $null)
        }
        else { Move-Item -LiteralPath $partial -Destination $destinationPath }
        return $destinationPath
    }
    finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    }
}

function Expand-SafeArchive {
    param([string]$Archive, [string]$Destination, [string[]]$AllowedPrefixes = @(''))
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $base = [IO.Path]::GetFullPath($Destination).TrimEnd('\', '/')
    $zip = [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($Archive))
    try {
        $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $entries = @()
        $total = [long]0
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace('\', '/')
            $segments = $name.Split('/')
            $type = ($entry.ExternalAttributes -shr 16) -band 0xF000
            $allowed = @($AllowedPrefixes | Where-Object { $name.StartsWith($_, [StringComparison]::Ordinal) }).Count -gt 0
            if (-not $name -or $name.StartsWith('/') -or $name.Contains(':') -or $segments -contains '..' -or $type -eq 0xA000 -or -not $allowed) { throw "Unsafe archive entry: $name" }
            $path = Get-SafeProjectPath -Root $base -Relative $name
            if (-not $names.Add($path)) { throw "Duplicate archive entry: $name" }
            $entries += [PSCustomObject]@{ entry = $entry; path = $path; name = $name }
            $total += $entry.Length
        }
        $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($base))
        if ($drive.AvailableFreeSpace -lt $total + 256MB) { throw 'Not enough free disk space to extract this archive.' }
        # Validate every entry before creating any output file.
        foreach ($item in $entries) {
            if ($item.name.EndsWith('/')) { $null = New-Item -ItemType Directory -Path $item.path -Force; continue }
            $null = New-Item -ItemType Directory -Path (Split-Path -Parent $item.path) -Force
            [IO.Compression.ZipFileExtensions]::ExtractToFile($item.entry, $item.path, $true)
            [IO.File]::SetLastWriteTimeUtc($item.path, $item.entry.LastWriteTime.UtcDateTime)
        }
    }
    finally { $zip.Dispose() }
}

function Invoke-ProjectCommand {
    param([string]$Executable, [string[]]$Arguments, [string]$Root)
    Push-Location -LiteralPath $Root
    try {
        & $Executable @Arguments | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
    }
    finally { Pop-Location }
}

function Write-JsonFile {
    param([string]$Path, $Value)
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30) + "`n", $script:Utf8)
}

function Get-PortableTestFiles {
    return @('tests/test_bootstrap_launchers.py', 'tests/test_dataset_restore.py', 'tests/test_btxrd_inference.py', 'tests/test_btxrd_phase_b.py', 'tests/test_btxrd_upgrades.py', 'tests/test_inference_predictor.py', 'tests/test_streamlit_app.py')
}

function Get-UvExecutable {
    param([string]$Root)
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    $toolRoot = Get-SafeProjectPath -Root $Root -Relative 'logs/bootstrap-tools'
    $existing = Join-Path $toolRoot 'uv.exe'
    if (Test-Path -LiteralPath $existing) { return $existing }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'The pinned PyTorch environment requires 64-bit Windows.' }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/astral-sh/uv/releases/latest' -Headers @{ 'User-Agent' = 'bone-cancer-ai-setup' }
    $name = 'uv-x86_64-pc-windows-msvc.zip'
    $asset = @($release.assets | Where-Object { $_.name -eq $name })
    $digestAsset = @($release.assets | Where-Object { $_.name -eq ($name + '.sha256') })
    if ($asset.Count -ne 1 -or $digestAsset.Count -ne 1) { throw 'Official uv release did not supply the Windows archive and checksum.' }
    $digestResponse = Invoke-WebRequest -Uri $digestAsset[0].browser_download_url -UseBasicParsing
    $digestText = if ($digestResponse.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($digestResponse.Content) } else { [string]$digestResponse.Content }
    $digest = $digestText.Trim().Split(' ')[0]
    $zipPath = Join-Path $toolRoot $name
    $null = Save-VerifiedDownload -Url $asset[0].browser_download_url -Destination $zipPath -Checksum $digest
    Expand-SafeArchive -Archive $zipPath -Destination $toolRoot
    $found = @(Get-ChildItem -LiteralPath $toolRoot -Recurse -File -Filter uv.exe)
    if ($found.Count -ne 1) { throw 'uv.exe was not found in the verified archive.' }
    return $found[0].FullName
}

function Get-InstallProfile {
    param([string]$Requested, [string]$Root = '')
    if ($Requested -ne 'auto') { return $Requested }
    if ($Root) {
        $python = Get-ProjectPython -Root $Root
        if (Test-Path -LiteralPath $python) {
            $probe = & $python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
            if ($LASTEXITCODE -eq 0 -and ($probe -eq 'cpu' -or $probe -eq 'cuda')) { return [string]$probe }
        }
    }
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -ne $smi) {
        $details = (& $smi.Source 2>$null) -join "`n"
        if ($LASTEXITCODE -eq 0 -and $details -match 'CUDA Version:\s*(\d+)\.(\d+)') {
            if ([int]$Matches[1] -ge 13) { return 'cuda' }
        }
    }
    return 'cpu'
}

function Install-ProjectEnvironment {
    param([string]$Root, [string]$Profile)
    $python = Get-ProjectPython -Root $Root
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (Test-Path -LiteralPath (Join-Path $Root '.venv')) { throw 'An incomplete .venv exists. Preserve or rename it before creating a new environment.' }
        $uv = Get-UvExecutable -Root $Root
        Invoke-ProjectCommand -Executable $uv -Arguments @('python', 'install', '3.11') -Root $Root
        Invoke-ProjectCommand -Executable $uv -Arguments @('venv', '--python', '3.11', '--seed', (Join-Path $Root '.venv')) -Root $Root
    }
    Invoke-ProjectCommand -Executable $python -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required"') -Root $Root
    $index = if ($Profile -eq 'cuda') { 'https://download.pytorch.org/whl/cu130' } else { 'https://download.pytorch.org/whl/cpu' }
    $variant = if ($Profile -eq 'cuda') { 'cu130' } else { 'cpu' }
    Invoke-ProjectCommand -Executable $python -Arguments @('-m', 'pip', 'install', "torch==2.14.0+$variant", "torchvision==0.29.0+$variant", '--index-url', $index) -Root $Root
    $packages = @()
    foreach ($relative in @('requirements/base.txt', 'requirements/train.txt', 'requirements/dev.txt')) {
        $packages += @(Get-Content -LiteralPath (Join-Path $Root $relative) | Where-Object { $_.Trim() -and -not $_.StartsWith('#') -and $_ -notmatch '^(torch|torchvision)==' })
    }
    $constraints = Get-SafeProjectPath -Root $Root -Relative 'logs/bootstrap_constraints.txt'
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $constraints) -Force
    $pins = @(Get-Content -LiteralPath (Join-Path $Root 'requirements/lock_windows_gpu.txt') | Where-Object { $_ -notmatch '^(torch|torchvision)==' })
    [IO.File]::WriteAllLines($constraints, $pins, $script:Utf8)
    Invoke-ProjectCommand -Executable $python -Arguments (@('-m', 'pip', 'install', '--constraint', $constraints, 'setuptools>=77', 'wheel') + $packages) -Root $Root
    Invoke-ProjectCommand -Executable $python -Arguments @('-m', 'pip', 'install', '-e', '.', '--no-deps') -Root $Root
    Invoke-ProjectCommand -Executable $python -Arguments @('-m', 'pip', 'check') -Root $Root
    return $python
}

function Install-ArchiveTree {
    param([string]$Source, [string]$Root, [string[]]$AllowedPrefixes, [switch]$PreserveDifferent)
    foreach ($file in Get-ChildItem -LiteralPath $Source -Recurse -File -Force) {
        $relative = $file.FullName.Substring([IO.Path]::GetFullPath($Source).TrimEnd('\').Length + 1).Replace('\', '/')
        if (@($AllowedPrefixes | Where-Object { $relative.StartsWith($_, [StringComparison]::Ordinal) }).Count -eq 0) { throw "Unexpected installed asset: $relative" }
        $destination = Get-SafeProjectPath -Root $Root -Relative $relative
        if (Test-Path -LiteralPath $destination) {
            if ((Get-AssetHash -LiteralPath $destination).Hash -eq (Get-AssetHash -LiteralPath $file.FullName).Hash) { continue }
            if ($PreserveDifferent) { throw "A different source file already exists; it was preserved: $relative" }
        }
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}

function Get-DefaultModelUrl {
    param([string]$Root)
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -eq $git) { return '' }
    $remote = (& $git.Source -C $Root remote get-url origin 2>$null) -join ''
    if ($LASTEXITCODE -eq 0 -and $remote -match '(?:github\.com[:/])([^/]+)/([^/]+?)(?:\.git)?$') {
        return "https://github.com/$($Matches[1])/$($Matches[2])/releases/latest/download/bone-cancer-runtime.zip"
    }
    return ''
}

function Install-RuntimeArchive {
    param([string]$Root, [string]$Url, [string]$Checksum)
    $archive = Get-SafeProjectPath -Root $Root -Relative 'logs/downloads/bone-cancer-runtime.zip'
    $null = Save-VerifiedDownload -Url $Url -Destination $archive -Checksum $Checksum
    $staging = Get-SafeProjectPath -Root $Root -Relative ('logs/downloads/runtime-' + [Guid]::NewGuid().ToString('N'))
    Expand-SafeArchive -Archive $archive -Destination $staging -AllowedPrefixes @('artifacts/', 'experiments/', 'asset-manifest.json')
    $manifestPath = Join-Path $staging 'asset-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Runtime archive has no asset manifest.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.alias -ne $script:SelectedAlias) { throw 'Runtime archive model alias differs from this project.' }
    $registry = Get-Content -LiteralPath (Join-Path $Root 'artifacts/checkpoints/registry.json') -Raw | ConvertFrom-Json
    $selected = $registry.PSObject.Properties[$script:SelectedAlias].Value
    if ($manifest.checkpoint_sha256 -ne $selected.checkpoint_sha256) { throw 'Runtime archive differs from the checkpoint selected in this clone.' }
    foreach ($entry in $manifest.files) {
        if ($entry.path -notmatch '^(artifacts/(checkpoints|metrics|figures|explainability)/|experiments/btxrd_subtype(?:_balanced)?/)') { throw "Unexpected runtime asset path: $($entry.path)" }
        $path = Get-SafeProjectPath -Root $staging -Relative ([string]$entry.path)
        if (-not (Test-Path -LiteralPath $path) -or (Get-AssetHash -LiteralPath $path).Hash -ne $entry.sha256) { throw "Runtime archive file checksum mismatch: $($entry.path)" }
    }
    # Install only declared files, and keep the checked-in registry authoritative.
    foreach ($entry in $manifest.files) {
        $relative = [string]$entry.path
        if ($relative -eq 'artifacts/checkpoints/registry.json') { continue }
        if ($relative -notmatch '^(artifacts/(checkpoints|metrics|figures|explainability)/|experiments/btxrd_subtype(?:_balanced)?/)') { throw "Unexpected runtime asset path: $relative" }
        $source = Get-SafeProjectPath -Root $staging -Relative $relative
        $destination = Get-SafeProjectPath -Root $Root -Relative $relative
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
        Copy-Item -LiteralPath $source -Destination $destination -Force
        [IO.File]::SetLastWriteTimeUtc($destination, [IO.File]::GetLastWriteTimeUtc($source))
    }
}

function Export-RuntimeArchive {
    param([string]$Root)
    $state = Get-RuntimeAssetState -Root $Root
    if (-not $state.present) { throw 'Complete selected model and result assets are required before export.' }
    $registry = Get-Content -LiteralPath (Join-Path $Root 'artifacts/checkpoints/registry.json') -Raw | ConvertFrom-Json
    $record = $registry.PSObject.Properties[$script:SelectedAlias].Value
    $relativePaths = @([string]$record.checkpoint_path)
    foreach ($relative in @('artifacts/metrics', 'artifacts/figures', 'artifacts/explainability', 'experiments/btxrd_subtype', 'experiments/btxrd_subtype_balanced')) {
        $folder = Join-Path $Root $relative
        if (Test-Path -LiteralPath $folder) {
            $relativePaths += @(Get-ChildItem -LiteralPath $folder -Recurse -File | Where-Object { $_.Extension -in @('.json', '.csv', '.png', '.jpg', '.jpeg') } | ForEach-Object { $_.FullName.Substring($Root.TrimEnd('\').Length + 1).Replace('\', '/') })
        }
    }
    $relativePaths = @($relativePaths | Sort-Object -Unique)
    $files = @($relativePaths | ForEach-Object { $path = Get-SafeProjectPath -Root $Root -Relative $_; [PSCustomObject]@{ path = $_; sha256 = (Get-AssetHash -LiteralPath $path).Hash.ToLowerInvariant(); bytes = (Get-Item -LiteralPath $path).Length } })
    $manifest = [PSCustomObject]@{ format_version = 1; alias = $script:SelectedAlias; checkpoint_sha256 = [string]$record.checkpoint_sha256; files = $files }
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $output = Get-SafeProjectPath -Root $Root -Relative 'dist/bone-cancer-runtime.zip'
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $output) -Force
    $partial = $output + '.partial'
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    $zip = [IO.Compression.ZipFile]::Open($partial, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($relative in $relativePaths) {
            $path = Get-SafeProjectPath -Root $Root -Relative $relative
            $null = [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $path, $relative, [IO.Compression.CompressionLevel]::Optimal)
        }
        $entry = $zip.CreateEntry('asset-manifest.json')
        $stream = New-Object IO.StreamWriter($entry.Open(), $script:Utf8)
        try { $stream.Write(($manifest | ConvertTo-Json -Depth 20)) } finally { $stream.Dispose() }
    }
    finally { $zip.Dispose() }
    if (Test-Path -LiteralPath $output) { [IO.File]::Replace($partial, $output, $null) } else { Move-Item -LiteralPath $partial -Destination $output }
    $digest = (Get-AssetHash -LiteralPath $output).Hash.ToLowerInvariant()
    Write-JsonFile -Path ($output + '.json') -Value @{ archive = 'bone-cancer-runtime.zip'; sha256 = $digest; bytes = (Get-Item -LiteralPath $output).Length; alias = $script:SelectedAlias }
    Write-Host "Runtime bundle: $output"
    Write-Host "SHA-256: $digest"
    Write-Host 'Upload this file once as a GitHub Release asset named bone-cancer-runtime.zip.'
    return $digest
}

function Invoke-PythonReadiness {
    param([string]$Root, [string]$Python, [string]$Profile, [switch]$RepairPaths)
    $code = @'
import importlib.metadata as metadata, json, pathlib, sys
root=pathlib.Path(sys.argv[1]).resolve(); profile=sys.argv[2]; repair=sys.argv[3]=='True'
report={'python_ok':sys.version_info[:2]==(3,11),'requirements_ok':False,'dataset_valid':False,'dataset_errors':[],'main_dataset_valid':False,'main_dataset_errors':[],'model_ready':False,'errors':[]}
try:
 from packaging.requirements import Requirement
 issues=[]
 for name in ('base','train','dev'):
  for line in (root/'requirements'/f'{name}.txt').read_text().splitlines():
   line=line.strip()
   if not line or line.startswith('#'): continue
   req=Requirement(line)
   try: version=metadata.version(req.name)
   except metadata.PackageNotFoundError: issues.append(f'{req.name}: missing'); continue
   if req.name in ('torch','torchvision') and profile=='cpu':
    wanted='2.14.0' if req.name=='torch' else '0.29.0'
    if version.split('+')[0]!=wanted: issues.append(f'{req.name}: {version} != {wanted}')
   elif version not in req.specifier: issues.append(f'{req.name}: {version} does not satisfy {req.specifier}')
 report['requirement_issues']=issues; report['requirements_ok']=not issues
 from bone_cancer_ai.utils.environment import environment_report
 environment=environment_report(require_cuda=profile=='cuda',device=profile)
 report['environment']=environment; report['requirements_ok'] &= environment['ok']
 from bone_cancer_ai.utils.config import load_config
 if repair:
  for name in ('data.yaml','btxrd_data.yaml'):
   path=root/'configs'/name; config=load_config(path)
   dataset=root/'dataset/Main'
   if name=='btxrd_data.yaml' and not all((dataset/split/f'_{split}_labels.csv').is_file() for split in ('train','val','test')): dataset=root/'dataset/BTXRD/images'
   if config.get('project_root')==root.as_posix() and config.get('dataset_root')==dataset.as_posix(): continue
   old=path.read_text(encoding='utf-8')
   import re
   updated=re.sub(r'^project_root:.*$', 'project_root: '+json.dumps(root.as_posix()),old,flags=re.MULTILINE)
   updated=re.sub(r'^dataset_root:.*$', 'dataset_root: '+json.dumps(dataset.as_posix()),updated,flags=re.MULTILINE)
   if updated!=old: path.write_text(updated,encoding='utf-8',newline='\n')
 report['dataset_errors']=[]
 try:
  from bone_cancer_ai.data.btxrd_migration import load_btxrd_metadata
  source=root/'dataset/BTXRD'
  report['dataset_source_exists']=source.is_dir()
  if source.is_dir():
   frame=load_btxrd_metadata(source)
   image_root=source/'images'
   missing=[str(x) for x in frame.image_id if not (image_root/str(x)).is_file()]
   report['dataset_valid']=len(frame)==3746 and not missing
   report['dataset_rows']=len(frame);report['missing_dataset_images']=missing[:20]
   if not report['dataset_valid']:
    report['dataset_errors'].append(f'BTXRD dataset is incomplete: {len(frame)} rows and {len(missing)} missing images (expected 3746 rows and no missing images).')
    report['errors'].extend(report['dataset_errors'])
 except Exception as error:
  message=f'Dataset {type(error).__name__}: {error}'
  report['dataset_errors'].append(message);report['errors'].append(message)
 report['main_dataset_errors']=[]
 try:
  from scripts.dataset_restore import validate_main
  main_state=validate_main(root/'dataset/Main',root/'configs/main_dataset_manifest.csv',root/'configs/main_dataset_manifest.json')
  report['main_dataset_valid']=True;report['main_dataset']=main_state
 except Exception as error:
  message=f'Main dataset {type(error).__name__}: {error}'
  report['main_dataset_valid']=False;report['main_dataset_errors'].append(message)
  if not (root/'dataset/Main').exists():
   report['main_dataset_errors'].append('Restore the combined Main dataset with -MainDatasetUrl and -MainDatasetSha256 (or BONE_CANCER_MAIN_DATASET_URL and the matching checksum).')
 from bone_cancer_ai.inference.btxrd import BtxrdSubtypePredictor
 from bone_cancer_ai.evaluation.btxrd_review import review_btxrd_run
 from bone_cancer_ai.evaluation.robustness import build_robustness_report
 from app.streamlit_app import load_model_comparison
 predictor=BtxrdSubtypePredictor.from_project_root(root,device='cuda' if profile=='cuda' else 'cpu')
 import numpy as np
 result=predictor.predict_array(np.full((64,64,3),128,dtype=np.uint8),sample_id='setup_synthetic_probe')
 assert len(result.subtype_probabilities)==9
 assert np.isfinite(result.binary_probability)
 review=review_btxrd_run(root);robustness=build_robustness_report(root)
 for name,value in [('btxrd_subtype_review.json',review),('btxrd_robustness.json',robustness)]:
  path=root/'artifacts/metrics'/name
  if repair: path.write_text(json.dumps(value,indent=2),encoding='utf-8')
  elif json.loads(path.read_text(encoding='utf-8'))!=value: raise ValueError(name+' is stale')
 assert len(load_model_comparison(str(root)))==5
 report['model_ready']=True;report['checkpoint_sha256']=predictor.checkpoint_sha256
except Exception as error:
 report['errors'].append(f'{type(error).__name__}: {error}')
print(json.dumps(report))
'@
    Push-Location -LiteralPath $Root
    try {
        $result = & $Python -X utf8 -c $code $Root $Profile ([string][bool]$RepairPaths)
        if ($LASTEXITCODE -ne 0) { throw 'Python environment/readiness scan failed.' }
        return ($result | ConvertFrom-Json)
    }
    finally { Pop-Location }
}

function Invoke-SetupMain {
    $root = [IO.Path]::GetFullPath($PSScriptRoot)
    if (-not (Test-Path -LiteralPath (Join-Path $root 'pyproject.toml')) -or -not (Test-Path -LiteralPath (Join-Path $root 'app/streamlit_app.py'))) { throw 'Run setup.ps1 from a complete project clone.' }
    if ($ExportAssets) { $null = Export-RuntimeArchive -Root $root; return 0 }
    $profile = Get-InstallProfile -Requested $Device -Root $root
    $python = Get-ProjectPython -Root $root
    $report = [ordered]@{ project_root = $root; platform = 'Windows'; profile = $profile; environment_ready = $false; dataset_ready = $false; model_ready = $false; app_ready = $false; errors = @() }
    $mainState = $null
    try {
        if (-not $CheckOnly) {
            Write-Host "Preparing $root ($profile)"
            $python = Install-ProjectEnvironment -Root $root -Profile $profile
            $dataset = Get-BtxrdDatasetState -Root $root
            if (-not $SkipDataset -and (-not $dataset.present -or $dataset.image_count -ne 3746)) {
                Write-Host 'Downloading the authors'' official BTXRD archive (840 MB).'
                $archive = Get-SafeProjectPath -Root $root -Relative 'logs/downloads/BTXRD.zip'
                $null = Save-VerifiedDownload -Url $script:OfficialBtxrdUrl -Destination $archive -Checksum $script:OfficialBtxrdMD5 -Algorithm MD5
                $stage = Get-SafeProjectPath -Root $root -Relative ('logs/downloads/dataset-' + [Guid]::NewGuid().ToString('N'))
                Expand-SafeArchive -Archive $archive -Destination $stage -AllowedPrefixes @('BTXRD/')
                Install-ArchiveTree -Source $stage -Root (Join-Path $root 'dataset') -AllowedPrefixes @('BTXRD/') -PreserveDifferent
            }
            if (-not $SkipDataset) {
                $mainState = Get-MainDatasetState -Root $root -Python $python
                if (-not $mainState.present) {
                    $sourceRoots = @((Join-Path $root 'dataset'))
                    if ($RoboflowSourceRoot) {
                        $localSource = [IO.Path]::GetFullPath($RoboflowSourceRoot)
                        if (-not (Test-Path -LiteralPath $localSource -PathType Container)) { throw 'RoboflowSourceRoot must be an existing source directory.' }
                        $sourceRoots += $localSource
                    }
                    $archive = ''
                    $archivePrefixes = @('Main/', 'train/', 'valid/', 'val/', 'test/', 'normal/', 'cancer/', 'README.')
                    if ($MainDatasetUrl) {
                        if (-not $MainDatasetSha256) { throw 'MainDatasetSha256 is required for the combined Main archive.' }
                        $archive = Get-SafeProjectPath -Root $root -Relative 'logs/downloads/Main.zip'
                        try { $null = Save-VerifiedDownload -Url $MainDatasetUrl -Destination $archive -Checksum $MainDatasetSha256 }
                        catch { throw 'Combined Main archive download or checksum verification failed.' }
                        $archivePrefixes = @('Main/')
                    }
                    elseif ($RoboflowArchivePath) {
                        $archive = [IO.Path]::GetFullPath($RoboflowArchivePath)
                        if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw 'RoboflowArchivePath must name an existing ZIP file.' }
                        if (-not $RoboflowArchiveSha256 -or (Get-AssetHash -LiteralPath $archive).Hash -ne $RoboflowArchiveSha256) { throw 'RoboflowArchivePath requires its matching SHA-256 checksum.' }
                    }
                    elseif ($RoboflowArchiveUrl) {
                        if (-not $RoboflowArchiveSha256) { throw 'RoboflowArchiveSha256 is required for the Roboflow archive URL.' }
                        $archive = Get-SafeProjectPath -Root $root -Relative 'logs/downloads/Roboflow.zip'
                        try { $null = Save-VerifiedDownload -Url $RoboflowArchiveUrl -Destination $archive -Checksum $RoboflowArchiveSha256 }
                        catch { throw 'Roboflow archive download or checksum verification failed.' }
                    }
                    elseif ($RoboflowApiKey) {
                        $archive = Get-RoboflowExportArchive -Root $root -ApiKey $RoboflowApiKey
                    }
                    $stage = ''
                    if ($archive) {
                        $stage = Get-SafeProjectPath -Root $root -Relative ('logs/downloads/roboflow-stage-' + [Guid]::NewGuid().ToString('N'))
                    }
                    try {
                        if ($stage) {
                            Expand-SafeArchive -Archive $archive -Destination $stage -AllowedPrefixes $archivePrefixes
                            $sourceRoots += $stage
                        }
                        Restore-MainDataset -Root $root -Python $python -SourceRoots $sourceRoots
                        $mainState = Get-MainDatasetState -Root $root -Python $python
                        if (-not $mainState.present) { throw "Main dataset restoration did not validate: $($mainState.error)" }
                    }
                    finally {
                        if ($stage -and (Test-Path -LiteralPath $stage)) {
                            $safeStage = Get-SafeProjectPath -Root $root -Relative $stage.Substring($root.TrimEnd('\', '/').Length + 1)
                            Remove-Item -LiteralPath $safeStage -Recurse -Force
                        }
                    }
                }
            }
            $assets = Get-RuntimeAssetState -Root $root
            if (-not $assets.present) {
                $url = $ModelAssetsUrl
                if (-not $url) { $url = Get-DefaultModelUrl -Root $root }
                $digest = if ($ModelAssetsSha256) { $ModelAssetsSha256 } else { $script:RuntimeBundleSha256 }
                if (-not $url) { throw 'Model assets are missing. Supply -ModelAssetsUrl (and checksum), or publish bone-cancer-runtime.zip on this repository''s GitHub Release.' }
                Install-RuntimeArchive -Root $root -Url $url -Checksum $digest
            }
        }
        # Validate Main independently so a later environment/model import failure
        # cannot leave its readiness fields at their default false values.
        $mainState = Get-MainDatasetState -Root $root -Python $python
        $report['main_dataset_valid'] = [bool]$mainState.valid
        $report['main_dataset_present'] = [bool]$mainState.valid
        $report['main_dataset_ready'] = [bool]$mainState.valid
        $report['main_dataset_errors'] = if ($mainState.valid) { @() } else { @($mainState.error) }
        if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing; run setup.ps1 without -CheckOnly.' }
        $details = Invoke-PythonReadiness -Root $root -Python $python -Profile $profile -RepairPaths:(-not $CheckOnly)
        if (-not $CheckOnly -and $Device -eq 'auto' -and $profile -eq 'cuda' -and $details.environment -and (-not $details.environment.cuda.available -or -not $details.environment.tensor_check.finite_gradient)) {
            Write-Host 'CUDA verification failed; preparing the CPU build instead.'
            $profile = 'cpu'
            $report.profile = $profile
            $python = Install-ProjectEnvironment -Root $root -Profile $profile
            $details = Invoke-PythonReadiness -Root $root -Python $python -Profile $profile -RepairPaths
        }
        $report.environment_ready = [bool]($details.python_ok -and $details.requirements_ok)
        $report.dataset_ready = [bool]$details.dataset_valid
        $report.model_ready = [bool]$details.model_ready
        $report.app_ready = [bool]($report.environment_ready -and $report.model_ready)
        $report['main_dataset_valid'] = [bool]$mainState.valid
        $report['main_dataset_present'] = [bool]$mainState.valid
        $report['main_dataset_ready'] = [bool]$mainState.valid
        $report['details'] = $details
        $report.errors = @($details.errors)
        $report.main_dataset_errors = if ($mainState.valid) { @() } else { @($mainState.error) }
        if (-not $SkipDataset -and -not $mainState.valid) { $report.errors += @($report.main_dataset_errors) }
        if ($RunTests -and $report.app_ready) {
            Invoke-ProjectCommand -Executable $python -Arguments (@('-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=logs/pytest_setup') + (Get-PortableTestFiles)) -Root $root
            $report['portable_tests'] = 'passed'
        }
    }
    catch {
        $report.errors += $_.Exception.Message
        $report.app_ready = $false
        # Keep a failed download/provisioning report accurate about steps that
        # already succeeded, instead of reporting a prepared environment missing.
        if (Test-Path -LiteralPath $python -PathType Leaf) {
            try {
                $details = Invoke-PythonReadiness -Root $root -Python $python -Profile $profile
                $report.environment_ready = [bool]($details.python_ok -and $details.requirements_ok)
                $report.dataset_ready = [bool]$details.dataset_valid
                $report.model_ready = [bool]$details.model_ready
                $report['details'] = $details
                $report.errors += @($details.errors)
                if ($null -ne $mainState) {
                    $report.main_dataset_valid = [bool]$mainState.valid
                    $report.main_dataset_present = [bool]$mainState.valid
                    $report.main_dataset_ready = [bool]$mainState.valid
                    $report.main_dataset_errors = if ($mainState.valid) { @() } else { @($mainState.error) }
                    if (-not $SkipDataset -and -not $mainState.valid) { $report.errors += @($report.main_dataset_errors) }
                }
            }
            catch { $report.errors += $_.Exception.Message }
        }
    }
    $output = if ($ReportPath) { [IO.Path]::GetFullPath($ReportPath) } else { Join-Path $root 'logs/setup_report.json' }
    Write-JsonFile -Path $output -Value $report
    Write-Host "Readiness report: $output"
    Write-Host "Environment: $($report.environment_ready); dataset: $($report.dataset_ready); trained model: $($report.model_ready); web app: $($report.app_ready)"
    foreach ($message in $report.errors) { Write-Host $message -ForegroundColor Red }
    if (-not $report.app_ready -or (-not $SkipDataset -and (-not $report.dataset_ready -or -not $report.main_dataset_ready))) { return 1 }
    return 0
}

if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    $env:PYTHONUTF8 = '1'
    $env:NO_ALBUMENTATIONS_UPDATE = '1'
    $env:MLFLOW_DISABLE_AGENT_HINT = '1'
    [Console]::OutputEncoding = $script:Utf8
    try { exit (Invoke-SetupMain) } catch { Write-Error $_; exit 1 }
}
