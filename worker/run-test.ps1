# Run the extractor on one image and open the results folder.
#   powershell -ExecutionPolicy Bypass -File run-test.ps1 -Image C:\shots\pack.jpg [-Fast] [-Labels "strip,carton,bottle"]
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$Out,
    [switch]$Fast,          # lighter models for quick tuning on CPU
    [string]$Labels         # override DETECT_LABELS
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run setup.cmd first." -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Image)) { Write-Host "Image not found: $Image" -ForegroundColor Red; exit 1 }
$Image = (Resolve-Path $Image).Path
if (-not $Out) { $Out = Join-Path $PSScriptRoot ("out\" + [IO.Path]::GetFileNameWithoutExtension($Image)) }

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
if ($Fast) {
    $env:DETECTOR_MODEL = "IDEA-Research/grounding-dino-tiny"
    $env:SAM_MODEL      = "facebook/sam-vit-base"
    Write-Host "Fast mode: grounding-dino-tiny + sam-vit-base"
}
if ($Labels) { $env:DETECT_LABELS = $Labels }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$log = Join-Path $Out "run.log"
Write-Host "Processing $Image (first run downloads ~3 GB of models) ..."
Write-Host "Full output is also saved to $log"
$sw = [Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference = "Continue"   # Python logs to stderr; don't let PowerShell treat that as fatal
Set-Content -Path $log -Value "" -Encoding UTF8
& $py -u local_test.py $Image $Out 2>&1 | ForEach-Object { $line = "$_"; Write-Host $line; Add-Content -Path $log -Value $line -Encoding UTF8 }
$code = $LASTEXITCODE
Write-Host ("Done in {0:n0}s" -f $sw.Elapsed.TotalSeconds)
if ($code -eq 0) { Invoke-Item $Out } else { Write-Host "Failed (exit $code) - see the error above." -ForegroundColor Red }
exit $code
