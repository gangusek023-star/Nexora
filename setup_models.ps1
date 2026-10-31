<#
  Download the ready-made models for Nexo 1.2 and the Assistant.
  They run inside NEXORA itself (built-in engine) - no other program is needed.

      powershell -ExecutionPolicy Bypass -File .\setup_models.ps1

  You can also download them later in NEXORA on the "AI models" page.
#>
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
function Info($m) { Write-Host "[modele] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[  ok  ] $m" -ForegroundColor Green }
function Fail($m) { Write-Host "[ blad ] $m" -ForegroundColor Red; exit 1 }

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Fail "Najpierw uruchom setup.ps1 (NEXORA nie jest zainstalowana)." }

& $venvPy -c "import llama_cpp" 2>$null
if ($LASTEXITCODE -ne 0) {
    Info "Instaluje wbudowany silnik modeli (llama-cpp-python) ..."
    & $venvPy -m pip install --no-cache-dir --only-binary llama-cpp-python llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet
    if ($LASTEXITCODE -ne 0) { Fail "Nie udalo sie zainstalowac silnika. Sprawdz internet i uruchom skrypt ponownie." }
}
Ok "Wbudowany silnik modeli gotowy"

$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$freeGB = [math]::Round((Get-Item $PSScriptRoot).PSDrive.Free / 1GB, 1)
Info "RAM: $ramGB GB, wolne miejsce na dysku: $freeGB GB"

Write-Host ""
Write-Host "Model do rozmowy po polsku:" -ForegroundColor White
Write-Host "  1) Bielik 1.5B - 1.7 GB, szybki, zalecany przy 8 GB RAM (licencja Apache-2.0)"
Write-Host "  2) Bielik 4.5B - 5.1 GB, lepsze odpowiedzi, potrzebuje ok. 6 GB wolnego RAM"
Write-Host "  3) pomin"
$c = Read-Host "Wybierz [1/2/3] (domyslnie 1)"
$chat = switch ($c) { "2" { "bielik-4.5b" } "3" { "" } default { "bielik-1.5b" } }

Write-Host ""
Write-Host "Model do kodu (Lua, Python, Roblox Studio, Blender, czytanie plikow):" -ForegroundColor White
Write-Host "  1) Qwen2.5-Coder 3B   - 2.1 GB, lepszy kod (licencja Qwen Research: tylko uzytek niekomercyjny)"
Write-Host "  2) Qwen2.5-Coder 1.5B - 1.1 GB, szybszy, czesciej sie myli (Apache-2.0)"
Write-Host "  3) pomin"
$k = Read-Host "Wybierz [1/2/3] (domyslnie 1)"
$code = switch ($k) { "2" { "qwen-coder-1.5b" } "3" { "" } default { "qwen-coder-3b" } }

Write-Host ""
$w = Read-Host "Zainstalowac uczenie z filmow (rozpoznawanie mowy, ok. 600 MB)? [t/N]"
$whisper = $w -match '^(t|tak|y|yes)$'

$need = 0
if ($chat -eq "bielik-4.5b") { $need += 5.1 } elseif ($chat) { $need += 1.7 }
if ($code -eq "qwen-coder-3b") { $need += 2.1 } elseif ($code) { $need += 1.1 }
if ($whisper) { $need += 0.6 }
if ($freeGB -lt $need + 1.5) { Fail "Za malo miejsca: potrzeba ok. $need GB + 1.5 GB zapasu, wolne $freeGB GB." }

$models = @($chat, $code) | Where-Object { $_ }
if ($models.Count -gt 0) {
    Info "Pobieram modele z Hugging Face (jednorazowo; przerwane pobieranie wznowi sie przy nastepnym uruchomieniu) ..."
    & $venvPy scripts\download_model.py @models
    if ($LASTEXITCODE -ne 0) { Fail "Pobieranie nie powiodlo sie. Uruchom skrypt ponownie - pobieranie wznowi sie od miejsca przerwania." }
}

if ($whisper) {
    Info "Instaluje rozpoznawanie mowy (faster-whisper) ..."
    & $venvPy -m pip install --no-cache-dir faster-whisper --quiet
    if ($LASTEXITCODE -ne 0) { Write-Host "Nie udalo sie zainstalowac faster-whisper (filmy beda niedostepne, reszta dziala)." -ForegroundColor Yellow }
    else { Ok "Rozpoznawanie mowy zainstalowane (model mowy pobierze sie przy pierwszym filmie)" }
}

$settings = @{ assistant = @{ enabled = $true; backend = "builtin" } }
if ($chat) { $settings.assistant.chat_model = $chat }
if ($code) { $settings.assistant.code_model = $code }
$env:NEXORA_MODELS_JSON = ($settings | ConvertTo-Json -Depth 4 -Compress)
& $venvPy -c "import json, os, sys; sys.path.insert(0, '.'); from nexora.config import save_settings; save_settings(json.loads(os.environ['NEXORA_MODELS_JSON']))"
if ($LASTEXITCODE -ne 0) { Fail "Nie udalo sie zapisac ustawien." }
Ok "Ustawienia zapisane"

Write-Host ""
Write-Host "Gotowe. Uruchom NEXORE (start.ps1) i w czacie wybierz model Nexo 1.2." -ForegroundColor Green
Write-Host "Jesli NEXORA juz dziala, zrestartuj ja (stop.ps1, potem start.ps1), zeby wczytala ustawienia." -ForegroundColor Green
Write-Host "Roblox Studio: install_roblox_plugin.ps1   Blender: strona Polaczenia w NEXORA." -ForegroundColor Green
