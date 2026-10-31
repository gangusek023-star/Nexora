<#
  Start NEXORA AI locally (http://127.0.0.1:8765) and open it in the browser.

      powershell -ExecutionPolicy Bypass -File .\start.ps1
      powershell -ExecutionPolicy Bypass -File .\start.ps1 -Foreground   # keep it in this window
      powershell -ExecutionPolicy Bypass -File .\start.ps1 -NoBrowser
#>
param([switch]$Foreground, [switch]$NoBrowser, [int]$Port = 8765)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$url = "http://127.0.0.1:$Port"

if (-not (Test-Path $venvPy)) {
    Write-Host "NEXORA is not installed yet. Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Red
    exit 1
}

# already running?
try {
    $r = Invoke-RestMethod -Uri "$url/api/status" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "NEXORA is already running at $url" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
} catch { }

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port $Port is used by another program (PID $($busy[0].OwningProcess)). Start with -Port 8766, for example." -ForegroundColor Red
    exit 1
}

$drive = (Get-Item $PSScriptRoot).PSDrive
if ($drive.Free -lt 1GB) {
    Write-Host "Warning: less than 1 GB free disk space. Training will refuse to save checkpoints below the configured minimum." -ForegroundColor Yellow
}

if ($Foreground) {
    & $venvPy -m nexora --port $Port
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Force -Path "runtime\logs" | Out-Null
$out = "runtime\logs\server-console.log"
$err = "runtime\logs\server-errors.log"
$p = Start-Process -FilePath $venvPy -ArgumentList @("-m", "nexora", "--port", $Port) `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
Write-Host "Starting NEXORA (process $($p.Id)) ..." -ForegroundColor Cyan

for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    if ($p.HasExited) {
        Write-Host "NEXORA stopped during start-up. Last errors:" -ForegroundColor Red
        Get-Content $err -Tail 20
        exit 1
    }
    try {
        Invoke-RestMethod -Uri "$url/api/status" -TimeoutSec 2 -ErrorAction Stop | Out-Null
        Write-Host "NEXORA AI is running at $url" -ForegroundColor Green
        Write-Host "Stop it with:  powershell -ExecutionPolicy Bypass -File .\stop.ps1"
        if (-not $NoBrowser) { Start-Process $url }
        exit 0
    } catch { }
}
Write-Host "NEXORA did not answer within 120 seconds. See $err" -ForegroundColor Red
exit 1
