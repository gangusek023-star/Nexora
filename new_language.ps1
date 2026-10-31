<#
  Build a language model in another language (downloads public-domain books and
  trains from random weights, ~20-60 min). Memory, chats and images are kept;
  the previous runtime folder is backed up, not deleted.

      powershell -ExecutionPolicy Bypass -File .\new_language.ps1 -Lang pl
      powershell -ExecutionPolicy Bypass -File .\new_language.ps1 -Lang pl -Also en
  Languages: pl (Polski), en, de, es, uk, code (source code: Python standard library
  from this PC, or your own files in data\raw\code\)
#>
param([Parameter(Mandatory = $true)][ValidateSet("pl", "en", "de", "es", "uk", "code")][string]$Lang,
      [ValidateSet("pl", "en", "de", "es", "uk", "code")][string[]]$Also = @())
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
if (Test-Path "runtime\nexora.lock") {
    Write-Host "Stopping NEXORA first ..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "stop.ps1")
}
$argsList = @("scripts\new_language_model.py", "--lang", $Lang)
if ($Also.Count -gt 0) { $argsList += "--also"; $argsList += $Also }
& $venvPy @argsList
exit $LASTEXITCODE
