<#
  Stop NEXORA gracefully: the running training job saves a checkpoint first
  and can be resumed after the next start.
      powershell -ExecutionPolicy Bypass -File .\stop.ps1
#>
param([int]$Port = 8765)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$pidFile = "runtime\nexora.pid"

try {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/shutdown" -TimeoutSec 5 -ErrorAction Stop | Out-Null
    Write-Host "Asked NEXORA to save checkpoints and stop ..." -ForegroundColor Cyan
} catch {
    Write-Host "NEXORA is not answering on port $Port." -ForegroundColor Yellow
}

if (Test-Path $pidFile) {
    $procId = [int](Get-Content $pidFile)
    for ($i = 0; $i -lt 30; $i++) {
        if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Seconds 1
    }
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        Write-Host "Still running after 30 s; forcing it to stop (the last checkpoint is kept)." -ForegroundColor Yellow
        Stop-Process -Id $procId -Force
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}
Write-Host "NEXORA stopped." -ForegroundColor Green
