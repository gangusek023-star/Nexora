# Lekcja: Lua / Luau i Roblox Studio

## Podstawy języka Luau
Luau to wersja Lua używana w Roblox. Zmienne zawsze deklaruj jako `local`. Komentarz: `--`. Łączenie tekstu: `..`. Długość tablicy: `#tablica`. Indeksy tablic zaczynają się od 1. Brak wartości to `nil`. Nierówność to `~=`.
```lua
local imie = "Ola"
local punkty = 10
punkty += 5                      -- Luau ma operatory +=, -=, *=, /=
print("Gracz " .. imie .. " ma " .. punkty .. " pkt")
print(`Gracz {imie} ma {punkty} pkt`)   -- interpolacja w backtickach
if punkty >= 15 and imie ~= "" then
	print("dużo punktów")
elseif punkty > 0 then
	print("trochę")
else
	print("zero")
end
```

## Pętle i tablice
```lua
for i = 1, 5 do print(i) end                 -- od 1 do 5
for i = 10, 1, -2 do print(i) end            -- odliczanie co 2
local owoce = {"jabłko", "gruszka"}
table.insert(owoce, "śliwka")
for indeks, nazwa in ipairs(owoce) do print(indeks, nazwa) end
local gracz = {imie = "Ola", hp = 100}       -- słownik
for klucz, wartosc in pairs(gracz) do print(klucz, wartosc) end
while punkty > 0 do punkty -= 1 end
```
Luau ma też `continue` w pętlach. Nigdy nie pisz pętli `while true do` bez `task.wait()` w środku – zawiesi Studio.

## Funkcje
```lua
local function dodaj(a: number, b: number): number   -- typy są opcjonalne
	return a + b
end
local function info(...)
	local argumenty = {...}
	return #argumenty
end
local wynik = dodaj(2, 3)
```

## Tworzenie części (Part) w świecie
Wszystko, co widać w grze, to instancje. Tworzysz je przez `Instance.new`, ustawiasz właściwości, a na końcu ustawiasz `Parent` (najlepiej jako ostatni krok).
```lua
local part = Instance.new("Part")
part.Name = "Podloga"
part.Size = Vector3.new(50, 1, 50)          -- szerokość X, wysokość Y, głębokość Z (w studach)
part.Position = Vector3.new(0, 0.5, 0)       -- Y to wysokość
part.Anchored = true                         -- nie spada, nie przesuwa się
part.Color = Color3.fromRGB(80, 160, 70)
part.Material = Enum.Material.Grass
part.Parent = workspace
```
Kształty: `part.Shape = Enum.PartType.Ball` (kula), `Enum.PartType.Cylinder` (walec leży wzdłuż osi X), `Enum.PartType.Block`. Klin: `Instance.new("WedgePart")`. Przezroczystość: `part.Transparency = 0.5`. Kolizje: `part.CanCollide = false`. Materiały: `Wood`, `Brick`, `Concrete`, `Neon`, `Glass`, `Metal`, `Plastic`, `SmoothPlastic`, `Slate`, `Sand`, `Ice`.

## Obrót i CFrame
`CFrame` to pozycja + obrót. Kąty podaje się w radianach (`math.rad`).
```lua
part.CFrame = CFrame.new(0, 5, 0) * CFrame.Angles(0, math.rad(45), 0)   -- obrót o 45° wokół osi Y
part.Orientation = Vector3.new(0, 90, 0)                                  -- albo w stopniach
local przod = part.CFrame.LookVector
```

## Grupowanie w Model
Kilka części łącz w `Model`, nadaj nazwę i ustaw `PrimaryPart`, żeby przesuwać całość.
```lua
local dom = Instance.new("Model")
dom.Name = "Domek"
local sciana = Instance.new("Part")
sciana.Size = Vector3.new(10, 8, 1)
sciana.Position = Vector3.new(0, 4, -5)
sciana.Anchored = true
sciana.Parent = dom
dom.PrimaryPart = sciana
dom.Parent = workspace
dom:PivotTo(CFrame.new(20, 0, 0))          -- przesuwa cały model
```

## Przykład: domek z dachem
```lua
local dom = Instance.new("Model")
dom.Name = "Domek"
local function blok(nazwa, rozmiar, pozycja, kolor, material)
	local p = Instance.new("Part")
	p.Name = nazwa
	p.Size = rozmiar
	p.Position = pozycja
	p.Color = kolor
	p.Material = material or Enum.Material.SmoothPlastic
	p.Anchored = true
	p.Parent = dom
	return p
end
local podloga = blok("Podloga", Vector3.new(12, 1, 12), Vector3.new(0, 0.5, 0), Color3.fromRGB(120, 90, 60), Enum.Material.Wood)
blok("SciankaTyl", Vector3.new(12, 8, 1), Vector3.new(0, 5, -5.5), Color3.fromRGB(230, 220, 200), Enum.Material.Brick)
blok("SciankaLewa", Vector3.new(1, 8, 12), Vector3.new(-5.5, 5, 0), Color3.fromRGB(230, 220, 200), Enum.Material.Brick)
blok("SciankaPrawa", Vector3.new(1, 8, 12), Vector3.new(5.5, 5, 0), Color3.fromRGB(230, 220, 200), Enum.Material.Brick)
local dach = Instance.new("WedgePart")
dach.Name = "Dach"
dach.Size = Vector3.new(12, 4, 12)
dach.CFrame = CFrame.new(0, 11, 0)
dach.Color = Color3.fromRGB(170, 40, 40)
dach.Anchored = true
dach.Parent = dom
dom.PrimaryPart = podloga
dom.Parent = workspace
print("Zbudowano domek z " .. #dom:GetChildren() .. " części")
```

## Serwisy
Serwisy pobiera się przez `game:GetService("Nazwa")`. Najważniejsze: `Players` (gracze), `ReplicatedStorage` (wspólne dla serwera i klienta), `ServerScriptService` (skrypty serwera), `ServerStorage`, `StarterGui` (interfejs), `StarterPlayer`, `TweenService` (animacje), `RunService` (pętla gry), `UserInputService` (klawiatura/mysz, tylko klient), `Lighting`.

## Rodzaje skryptów
- `Script` – działa na serwerze (umieszczaj w ServerScriptService albo w części).
- `LocalScript` – działa u gracza (StarterPlayerScripts, StarterGui, StarterCharacterScripts).
- `ModuleScript` – kod wielokrotnego użytku, zwraca tabelę, ładowany przez `require(modul)`.
Agent, który buduje coś w Studio, może utworzyć skrypt i wpisać mu kod przez `Source`:
```lua
local s = Instance.new("Script")
s.Name = "ZbieraniePunktow"
s.Source = [[
print("Skrypt działa")
]]
s.Parent = game:GetService("ServerScriptService")
```

## Zdarzenia (Events)
Na zdarzenie reagujesz przez `:Connect(funkcja)`.
```lua
local part = workspace:WaitForChild("Moneta")
part.Touched:Connect(function(innaCzesc)
	local humanoid = innaCzesc.Parent:FindFirstChildOfClass("Humanoid")
	if humanoid then
		print("Dotknięto przez " .. innaCzesc.Parent.Name)
	end
end)
```
Inne: `Players.PlayerAdded`, `humanoid.Died`, `ClickDetector.MouseClick`, `ProximityPrompt.Triggered`, `RunService.Heartbeat`.

## Tabela wyników (leaderstats)
Folder o nazwie dokładnie `leaderstats` w graczu pokazuje się jako tabela wyników.
```lua
local Players = game:GetService("Players")
Players.PlayerAdded:Connect(function(player)
	local leaderstats = Instance.new("Folder")
	leaderstats.Name = "leaderstats"
	leaderstats.Parent = player
	local monety = Instance.new("IntValue")
	monety.Name = "Monety"
	monety.Value = 0
	monety.Parent = leaderstats
end)
```

## Moneta do zebrania (Script w ServerScriptService)
```lua
local moneta = workspace:WaitForChild("Moneta")
local zebrana = {}
moneta.Touched:Connect(function(hit)
	local player = game:GetService("Players"):GetPlayerFromCharacter(hit.Parent)
	if player and not zebrana[player] then
		zebrana[player] = true
		player.leaderstats.Monety.Value += 1
		moneta.Transparency = 1
		moneta.CanTouch = false
		task.wait(5)
		moneta.Transparency = 0
		moneta.CanTouch = true
		zebrana[player] = nil
	end
end)
```

## Czekanie i wątki
Używaj `task.wait(sekundy)`, `task.spawn(funkcja)`, `task.delay(sekundy, funkcja)`. Nie używaj starego `wait()` ani `spawn()`.

## Animacja (TweenService)
```lua
local TweenService = game:GetService("TweenService")
local info = TweenInfo.new(2, Enum.EasingStyle.Sine, Enum.EasingDirection.InOut, -1, true)  -- -1 = bez końca, true = w tę i z powrotem
local tween = TweenService:Create(part, info, {Position = part.Position + Vector3.new(0, 5, 0)})
tween:Play()
```

## Komunikacja serwer–klient (RemoteEvent)
RemoteEvent umieść w ReplicatedStorage. Klient: `remote:FireServer(dane)`. Serwer: `remote.OnServerEvent:Connect(function(player, dane) ... end)`. Serwer do klienta: `remote:FireClient(player, dane)`, klient: `remote.OnClientEvent:Connect(...)`. Nigdy nie ufaj danym od klienta – sprawdzaj je na serwerze.

## Szukanie obiektów
`workspace:FindFirstChild("Nazwa")` (zwraca nil, jeśli brak), `parent:WaitForChild("Nazwa")` (czeka – tylko w skryptach gry), `instance:GetChildren()`, `instance:GetDescendants()`, `instance:IsA("BasePart")`, `instance:Destroy()` (usuwa).

## Typowe błędy
- `attempt to index nil with ...` – obiekt nie istnieje; sprawdź nazwę albo użyj `FindFirstChild` i `if obj then`.
- Części spadają – brak `Anchored = true`.
- Nic nie widać – brak `Parent = workspace` albo `Transparency = 1`.
- `Vector3.new` przyjmuje liczby, nie tekst. Kolory: `Color3.fromRGB(0-255)` albo `Color3.new(0-1)`.
- W kodzie wykonywanym przez wtyczkę w trybie edycji nie ma graczy – `Players.LocalPlayer` to nil.
