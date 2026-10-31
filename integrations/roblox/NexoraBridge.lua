--[[
	NEXORA Bridge - Roblox Studio plugin
	Connects Roblox Studio to NEXORA (Nexo 1.2) running on this computer.
	Nexo 1.2 sends Luau code; this plugin runs it in EDIT mode with an undo
	point (Ctrl+Z undoes everything it did) and sends back the output and a
	list of parts so NEXORA can draw what is in the scene.

	Install: put this file in %LOCALAPPDATA%\Roblox\Plugins (install_roblox_plugin.ps1 does it).
	The first time, Studio asks whether the plugin may connect to 127.0.0.1 - click Allow.
]]

local HttpService = game:GetService("HttpService")
local ChangeHistoryService = game:GetService("ChangeHistoryService")
local LogService = game:GetService("LogService")
local RunService = game:GetService("RunService")

local SERVER = "http://127.0.0.1:{{PORT}}"
local TOKEN = "{{TOKEN}}"
local VERSION = "1.0"
local MAX_PARTS = 800

if RunService:IsRunning() then
	return -- do nothing inside play-test sessions
end

local enabled = plugin:GetSetting("NexoraEnabled")
if enabled == nil then
	enabled = true
end

local toolbar = plugin:CreateToolbar("NEXORA")
local button = toolbar:CreateButton("NexoraToggle", "Connect / disconnect Nexo 1.2", "", "NEXORA")
button.ClickableWhenViewportHidden = true
button:SetActive(enabled)

local running = true
local connected = false
local lastError = nil

local function request(method, path, body)
	local ok, res = pcall(function()
		return HttpService:RequestAsync({
			Url = SERVER .. path,
			Method = method,
			Headers = { ["Content-Type"] = "application/json", ["X-Nexora-Token"] = TOKEN },
			Body = body and HttpService:JSONEncode(body) or nil,
		})
	end)
	if not ok then
		return nil, tostring(res)
	end
	if not res.Success then
		return nil, "HTTP " .. tostring(res.StatusCode) .. " " .. tostring(res.Body)
	end
	local decoded
	ok, decoded = pcall(function()
		return HttpService:JSONDecode(res.Body)
	end)
	if not ok then
		return nil, "bad JSON from NEXORA"
	end
	return decoded
end

local function compile(code)
	local fn, err
	local hasLoadstring = pcall(function()
		fn, err = loadstring(code, "=Nexo")
	end)
	if hasLoadstring and (fn or err) then
		return fn, err
	end
	-- fallback when loadstring is unavailable: run the code as a fresh ModuleScript
	local module = Instance.new("ModuleScript")
	module.Name = "NexoraTemp"
	module.Source = "return function()\n" .. code .. "\nend"
	local okReq, result = pcall(require, module)
	module:Destroy()
	if not okReq then
		return nil, tostring(result)
	end
	return result, nil
end

local function runCode(code)
	local output = {}
	local logConn = LogService.MessageOut:Connect(function(message, messageType)
		if #output < 200 then
			local prefix = ""
			if messageType == Enum.MessageType.MessageWarning then
				prefix = "[warn] "
			elseif messageType == Enum.MessageType.MessageError then
				prefix = "[error] "
			end
			table.insert(output, prefix .. message)
		end
	end)
	local recording = nil
	pcall(function()
		recording = ChangeHistoryService:TryBeginRecording("Nexo 1.2", "Changes made by NEXORA")
	end)
	local fn, compileErr = compile(code)
	local ok, runErr = false, nil
	if fn then
		ok, runErr = xpcall(fn, debug.traceback)
	else
		runErr = "compile error: " .. tostring(compileErr)
	end
	pcall(function()
		if recording then
			local op = Enum.FinishRecordingOperation.Commit
			if not ok then
				op = Enum.FinishRecordingOperation.Cancel -- failed code: roll its changes back
			end
			ChangeHistoryService:FinishRecording(recording, op)
		end
	end)
	task.wait(0.1) -- let pending log messages arrive
	logConn:Disconnect()
	return { ok = ok, output = table.concat(output, "\n"), error = runErr and tostring(runErr) or nil }
end

local function snapshot()
	local parts, total = {}, 0
	for _, inst in ipairs(workspace:GetDescendants()) do
		if inst:IsA("BasePart") and not inst:IsA("Terrain") and inst.Transparency < 1 then
			total = total + 1
			if #parts < MAX_PARTS then
				local x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22 = inst.CFrame:GetComponents()
				local shape = "Block"
				if inst:IsA("Part") then
					shape = inst.Shape.Name
				end
				table.insert(parts, {
					n = inst.Name,
					c = inst.ClassName,
					p = { x, y, z },
					s = { inst.Size.X, inst.Size.Y, inst.Size.Z },
					r = { r00, r01, r02, r10, r11, r12, r20, r21, r22 },
					col = { inst.Color.R, inst.Color.G, inst.Color.B },
					t = inst.Transparency,
					sh = shape,
				})
			end
		end
	end
	local tree = {}
	for _, child in ipairs(workspace:GetChildren()) do
		if not child:IsA("Camera") and not child:IsA("Terrain") then
			table.insert(tree, child.ClassName .. " " .. child.Name .. " (" .. #child:GetDescendants() .. " inside)")
			if #tree >= 80 then
				break
			end
		end
	end
	local scripts = {}
	local services = { "ServerScriptService", "ReplicatedStorage", "StarterGui", "StarterPlayer", "Workspace" }
	for _, name in ipairs(services) do
		local ok, svc = pcall(function()
			return game:GetService(name)
		end)
		if ok and svc then
			for _, d in ipairs(svc:GetDescendants()) do
				if d:IsA("LuaSourceContainer") and #scripts < 60 then
					table.insert(scripts, d.ClassName .. " " .. d:GetFullName())
				end
			end
		end
	end
	return { parts = parts, total_parts = total, tree = tree, scripts = scripts }
end

local function handle(cmd)
	local result = { id = cmd.id }
	if cmd.kind == "run" then
		local r = runCode(cmd.code or "")
		result.ok, result.output, result.error = r.ok, r.output, r.error
	elseif cmd.kind == "snapshot" then
		local ok, snap = pcall(snapshot)
		result.ok = ok
		if ok then
			result.snapshot = snap
		else
			result.error = tostring(snap)
		end
	elseif cmd.kind == "read_script" then
		local ok, src = pcall(function()
			local inst = game
			for part in string.gmatch(cmd.path or "", "[^%.]+") do
				if part ~= "game" then
					inst = inst:FindFirstChild(part)
				end
			end
			return inst.Source
		end)
		result.ok, result.output = ok, ok and src or nil
		if not ok then
			result.error = "script not found: " .. tostring(cmd.path)
		end
	else
		result.ok, result.error = false, "unknown command " .. tostring(cmd.kind)
	end
	request("POST", "/api/bridge/roblox/result", result)
end

button.Click:Connect(function()
	enabled = not enabled
	plugin:SetSetting("NexoraEnabled", enabled)
	button:SetActive(enabled)
	print(enabled and "[NEXORA] Connecting to Nexo 1.2..." or "[NEXORA] Disconnected.")
end)

plugin.Unloading:Connect(function()
	running = false
end)

task.spawn(function()
	while running do
		if enabled then
			local info = HttpService:UrlEncode(game.Name)
			local data, err = request("GET", "/api/bridge/roblox/poll?wait=10&place=" .. info .. "&v=" .. VERSION)
			if data then
				if not connected then
					connected = true
					lastError = nil
					print("[NEXORA] Connected to Nexo 1.2 at " .. SERVER)
				end
				if data.command then
					local ok, e = pcall(handle, data.command)
					if not ok then
						request("POST", "/api/bridge/roblox/result", { id = data.command.id, ok = false, error = tostring(e) })
					end
				end
			else
				if connected or lastError ~= err then
					warn("[NEXORA] Cannot reach NEXORA (" .. tostring(err) .. "). Is it running? Retrying...")
				end
				connected = false
				lastError = err
				task.wait(5)
			end
		else
			connected = false
			task.wait(1)
		end
	end
end)
