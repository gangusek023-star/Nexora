<#
  Run the automated test suite (installs pytest into .venv if needed).
      powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
#>
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
& $venvPy -c "import pytest, httpx" 2>$null
if ($LASTEXITCODE -ne 0) { & $venvPy -m pip install --no-cache-dir -r requirements-dev.txt --quiet }
& $venvPy -m pytest tests -q
exit $LASTEXITCODE
