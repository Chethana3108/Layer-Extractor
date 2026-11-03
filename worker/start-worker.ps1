# Start the Content Hub object extraction worker
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Virtual environment not found. Please run setup.cmd first." -ForegroundColor Red
    exit 1
}

# Load environment variables from .env if it exists
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and (-not $line.StartsWith("#")) -and ($line -match '^\s*([^=]+)=(.*)$')) {
            $name = $matches[1].Trim()
            $val = $matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($name, $val, "Process")
        }
    }
}

# Defaults for CPU fast models
if (-not $env:DETECTOR_MODEL) { $env:DETECTOR_MODEL = "IDEA-Research/grounding-dino-tiny" }
if (-not $env:SAM_MODEL)      { $env:SAM_MODEL = "facebook/sam-vit-base" }
$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting Content Hub Object Extraction Worker..." -ForegroundColor Cyan
& $py main.py
