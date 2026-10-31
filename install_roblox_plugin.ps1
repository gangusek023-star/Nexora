<#
  Installs the NEXORA Bridge plugin into Roblox Studio (for Nexo 1.2).
      powershell -ExecutionPolicy Bypass -File .\install_roblox_plugin.ps1
  Restart Roblox Studio afterwards. The plugin file contains a private key - do not share it.
#>
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "Najpierw uruchom setup.ps1." -ForegroundColor Red; exit 1 }

$pluginDir = Join-Path $env:LOCALAPPDATA "Roblox\Plugins"
New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null
$target = Join-Path $pluginDir "NexoraBridge.lua"

$env:NEXORA_PLUGIN_OUT = $target
& $venvPy -c @"
import os, sys
sys.path.insert(0, '.')
from nexora.config import Paths, load_config
from nexora.bridge import BridgeHub
paths = Paths()
port = str(load_config(paths)['server']['port'])
token = BridgeHub(paths.root / 'bridge_token.txt').token
src = open('integrations/roblox/NexoraBridge.lua', encoding='utf-8').read()
open(os.environ['NEXORA_PLUGIN_OUT'], 'w', encoding='utf-8').write(src.replace('{{PORT}}', port).replace('{{TOKEN}}', token))
print('ok')
"@
if ($LASTEXITCODE -ne 0) { Write-Host "Nie udalo sie przygotowac wtyczki." -ForegroundColor Red; exit 1 }
Write-Host "Wtyczka zainstalowana: $target" -ForegroundColor Green
Write-Host "1) Zrestartuj Roblox Studio i otworz dowolne miejsce (place)."
Write-Host "2) Gdy Studio zapyta o dostep do 127.0.0.1 - kliknij Allow."
Write-Host "3) W NEXORA wybierz model Nexo 1.2 - w czacie zobaczysz 'Roblox Studio: polaczono'."
