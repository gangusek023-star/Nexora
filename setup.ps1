<#
  NEXORA AI - one-time setup for Windows 11.
  Creates .venv, installs CPU-only PyTorch and the other packages step by step,
  checks disk space first and verifies the installation.

  Usage (PowerShell, in the NEXORA folder):
      powershell -ExecutionPolicy Bypass -File .\setup.ps1
      powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Dev     # also installs test tools
#>
param([switch]$Dev)

# Native tools report failure through $LASTEXITCODE, which we check after each step.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Info($m) { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[ ok  ] $m" -ForegroundColor Green }
function Fail($m) { Write-Host "[error] $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- disk space
$drive = (Get-Item $PSScriptRoot).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Info "Free space on drive $($drive.Name): $freeGB GB"
if ($drive.Free -lt 3GB) {
    Fail "At least 3 GB of free disk space is needed (PyTorch CPU ~1 GB + packages + training data). Free some space and run setup again."
}

# ---------------------------------------------------------------- python
$python = $null
$candidates = @("py -3.12", "py -3.11", "py -3.13", "py -3.10", "python")
foreach ($cand in $candidates) {
    $c = @($cand -split " ")
    $exe = $c[0]; $args_ = @(); if ($c.Count -gt 1) { $args_ = @($c[1..($c.Count-1)]) }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $v = & $exe @args_ -c "import sys, struct; print('%d.%d-%d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(1[0-3])-64$') { $python = $c; Ok "Found 64-bit Python $v ($($c -join ' '))"; break }
}
if (-not $python) {
    Fail "Python 3.10 - 3.13 (64-bit) was not found. Install Python 3.12 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH'), then run setup again."
}

# ---------------------------------------------------------------- venv
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Info "Creating virtual environment in .venv ..."
    $exe = $python[0]; $args_ = @(); if ($python.Count -gt 1) { $args_ = @($python[1..($python.Count-1)]) }
    & $exe @args_ -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}
Ok "Virtual environment ready"
& $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "Could not upgrade pip. Check your internet connection." }

# ---------------------------------------------------------------- packages (in small steps)
& $venvPy -c "import torch" 2>$null
$hasTorch = ($LASTEXITCODE -eq 0)
if (-not $hasTorch) {
    Info "Installing PyTorch (CPU-only build, about 200 MB download) ..."
    & $venvPy -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        Info "CPU index not reachable, trying the default package index ..."
        & $venvPy -m pip install --no-cache-dir torch
        if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed. Check your internet connection and free disk space." }
    }
}
Ok "PyTorch installed"

Info "Installing the remaining packages ..."
& $venvPy -m pip install --no-cache-dir -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Fail "Package installation failed (see messages above)." }
if ($Dev) {
    & $venvPy -m pip install --no-cache-dir -r requirements-dev.txt --quiet
    if ($LASTEXITCODE -ne 0) { Fail "Installing test tools failed." }
}

# ---------------------------------------------------------------- built-in model engine
& $venvPy -c "import llama_cpp" 2>$null
if ($LASTEXITCODE -ne 0) {
    Info "Installing the built-in engine for ready-made models (llama-cpp-python, CPU build) ..."
    & $venvPy -m pip install --no-cache-dir --only-binary llama-cpp-python llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn ] Could not install llama-cpp-python. Nexo 1.2 and the Assistant will not work; the rest of NEXORA does. Run setup.ps1 again later." -ForegroundColor Yellow
    } else { Ok "Built-in model engine installed" }
}

# ---------------------------------------------------------------- verify
& $venvPy -c "import torch, numpy, PIL, fastapi, uvicorn, psutil; print('torch', torch.__version__, '| CUDA used:', False)"
if ($LASTEXITCODE -ne 0) { Fail "Verification failed: a package could not be imported." }
Ok "All packages import correctly"

# ---------------------------------------------------------------- data
if (-not (Test-Path "runtime\tokenizer\tokenizer.json")) {
    Info "Preparing the base training data (public-domain Tiny Shakespeare) ..."
    & $venvPy scripts\prepare_lm_data.py
    if ($LASTEXITCODE -ne 0) { Fail "Data preparation failed." }
} else {
    Ok "Base data already prepared (runtime folder present)"
}

$freeAfter = [math]::Round((Get-Item $PSScriptRoot).PSDrive.Free / 1GB, 1)
Ok "Setup complete. Free disk space now: $freeAfter GB"
Write-Host ""
Write-Host "Download the ready-made models (Bielik, Qwen Coder):  powershell -ExecutionPolicy Bypass -File .\setup_models.ps1" -ForegroundColor Yellow
Write-Host "  (or later in NEXORA on the 'AI models' page)" -ForegroundColor Yellow
Write-Host "Start NEXORA with:   powershell -ExecutionPolicy Bypass -File .\start.ps1" -ForegroundColor Yellow
