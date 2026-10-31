"use strict";
// NEXORA web UI - plain JavaScript, no external libraries (works fully offline).

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 3) => (n === null || n === undefined || Number.isNaN(n) ? "–" : Number(n).toFixed(d));
const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "–");
const tag = (s) => `<span class="tag t-${esc(s)}">${esc(String(s).replace(/_/g, " "))}</span>`;

async function api(path, opts = {}) {
  const o = { headers: { "Content-Type": "application/json" }, ...opts };
  if (o.body && typeof o.body !== "string") o.body = JSON.stringify(o.body);
  const r = await fetch(path, o);
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (r.status === 401) { location.href = "/login"; throw new Error("Sign in required"); }
  if (!r.ok) throw new Error((data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || r.statusText);
  return data;
}
function toast(msg, ms = 4200) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
async function act(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg); return r; } catch (e) { toast(e.message, 7000); throw e; }
}

// Simple SVG line chart. series: [{name, points:[[x,y]], color}]
function chart(series, yLabel = "loss") {
  const pts = series.flatMap((s) => s.points);
  if (pts.length < 2) return `<p class="muted small">Not enough data for a chart yet.</p>`;
  const W = 600, H = 190, L = 44, B = 22, T = 10, R = 10;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 3; i++) {
    const y = y0 + ((y1 - y0) * i) / 3;
    g += `<line x1="${L}" x2="${W - R}" y1="${sy(y)}" y2="${sy(y)}" stroke="var(--line)"/><text x="${L - 6}" y="${sy(y) + 4}" text-anchor="end">${y.toFixed(2)}</text>`;
  }
  g += `<text x="${L}" y="${H - 5}">${x0}</text><text x="${W - R}" y="${H - 5}" text-anchor="end">step ${x1}</text>`;
  const lines = series.map((s) => `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${s.points.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ")}"/>`).join("");
  const legend = series.map((s) => `<span class="small" style="color:${s.color}">■ ${esc(s.name)}</span>`).join(" ");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} chart">${g}${lines}</svg><div class="row">${legend}</div>`;
}

// Light Markdown for answers: headings, **bold**, `code`, bullet and numbered lists
function mdLines(html) {
  return html.split("\n").map((ln) => {
    let l = ln.replace(/`([^`]+)`/g, "<code class=\"inl\">$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    const h = l.match(/^(#{1,4})\s+(.*)$/);
    if (h) return `<span class="md-h md-h${h[1].length}">${h[2]}</span>`;
    const b = l.match(/^\s*[-*•]\s+(.*)$/);
    if (b) return `<span class="md-li">• ${b[1]}</span>`;
    const n = l.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (n) return `<span class="md-li">${n[1]}. ${n[2]}</span>`;
    return l;
  }).join("\n");
}
// the model's reply without its code block (the code is shown separately)
function planOf(text) { return String(text || "").split("```")[0].replace(/^\s*PLAN\s*:\s*/i, "").trim(); }
const TICON = { search: "🔎", think: "💭", look: "👁", code: "⌨", run: "▶", ok: "✅", error: "⚠", retry: "🔁", render: "🖼", wait: "⏳", warn: "⚠" };
function thoughtHtml(t) {
  if (t.type === "step") return `<div class="tline t-${esc(t.icon)}"><span class="ti">${TICON[t.icon] || "•"}</span><span>${esc(t.text)}</span></div>`;
  if (t.type === "think_block") { const plan = planOf(t.text); return plan ? `<pre class="tstream">${esc(plan)}</pre>` : ""; }
  if (t.type === "code") return richText("```" + t.lang + "\n" + t.code + "\n```");
  if (t.type === "image") return `<figure class="tfig"><img src="${esc(t.url)}" alt="${esc(t.caption)}"><figcaption>${esc(t.caption)}</figcaption></figure>`;
  return "";
}

// Assistant answers: ``` code blocks become formatted blocks with a copy button
function richText(t) {
  const parts = String(t ?? "").split(/```/);
  return parts.map((p, i) => {
    if (i % 2 === 0) {
      const trimmed = parts.length > 1 ? p.replace(/^\n+/, i ? "" : "$&").replace(/\n+$/, i < parts.length - 1 ? "" : "$&") : p;
      return mdLines(esc(trimmed));
    }
    const nl = p.indexOf("\n");
    const lang = nl > 0 ? p.slice(0, nl).trim() : "";
    const code = nl > 0 && /^[\w+#.-]*$/.test(lang) ? p.slice(nl + 1) : p;
    return `<div class="codeblock"><div class="codehead"><span>${esc(lang || "code")}</span><button type="button" class="copybtn">${tr("Copy")}</button></div><pre><code>${esc(code.replace(/\n$/, ""))}</code></pre></div>`;
  }).join("");
}
document.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("copybtn")) return;
  const code = e.target.closest(".codeblock").querySelector("code").textContent;
  try { await navigator.clipboard.writeText(code); e.target.textContent = tr("Copied"); } catch { /* clipboard blocked */ }
});

// ---------------------------------------------------------------- status
let STATUS = null;
async function refreshStatus() {
  try { STATUS = await api("/api/status"); } catch { $("#statusbar").innerHTML = `<span class="tag t-failed">server not reachable</span>`; return; }
  const s = STATUS, j = s.job;
  const parts = [];
  parts.push(s.lm.ready ? `Active language model <strong>${esc(s.lm.version)}</strong> (${(s.lm.params / 1e6).toFixed(2)} M parameters)` : `<strong>No trained language model yet</strong>`);
  if (j) {
    const p = j.progress || {};
    parts.push(`Job ${j.id} ${esc(j.kind.replace("_", " "))}: ${tag(j.status)} step ${p.step ?? 0}/${p.max_steps ?? "?"}`);
  }
  if (s.pending_candidates) parts.push(`<a href="#candidates">${s.pending_candidates} candidate(s) waiting for approval</a>`);
  parts.push(`Image model: <strong>${esc(s.image_model || "none")}</strong>`);
  $("#statusbar").innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
  const b = $("#pendingBadge"); b.hidden = !s.pending_candidates; b.textContent = s.pending_candidates;
}

// ---------------------------------------------------------------- pages
const pages = {};
let currentPage = null, pageTimer = null;

pages.chat = async (el) => {
  const convs = await api("/api/conversations");
  const state = pages.chat.state || (pages.chat.state = { cid: null, settings: false });
  const s0 = await api("/api/settings");
  el.innerHTML = `
    <h1>Chat</h1>
    <div class="panel enginepick"><div class="row">
      <label>Model <select id="modelSel">
        <option value="1.2">Nexo 1.2 – agent: conversation, code, Roblox Studio, Blender, knowledge</option>
        <option value="1.1">Nexo 1.1 – the previous version</option></select></label>
      <label id="engineWrap">Who answers <select id="engineSel">
        <option value="assistant" ${s0.assistant.enabled ? "" : "disabled"}>Assistant – ready-made model (Bielik / Qwen Coder), built in</option>
        <option value="nexora">NEXORA – own model trained from scratch</option></select></label>
      <label id="targetWrap">Works in <select id="targetSel">
        <option value="auto">Automatically</option><option value="chat">Conversation only</option>
        <option value="roblox">Roblox Studio</option><option value="blender">Blender</option></select></label>
      <span id="connDots" class="conn small"></span></div>
      <div id="engineNote" class="small muted"></div></div>
    <p class="small muted">Commands: <b>/remember …</b> — <span>stores a fact in memory (the network does not change).</span> <b>/learn …</b> — <span>adds text to the training data (learned after the next approved candidate).</span>
    ${s0.learning.learn_from_prompts ? "<span>Your prompts are also collected as training data (change this in Settings).</span>" : ""}</p>
    <div class="chat">
      <div>
        <div class="convo" id="convo" aria-live="polite"></div>
        <div id="attList" class="attlist"></div>
        <form class="composer" id="chatForm">
          <label class="btn attach" title="Attach a file (code, text, PDF, DOCX, ODT)" for="fileIn">📎</label><input type="file" id="fileIn" multiple hidden>
          <textarea id="chatIn" rows="2" placeholder="Say something… (Enter to send, Shift+Enter for a new line)" aria-label="Message"></textarea>
          <button class="primary" id="sendBtn">Send</button>
          <button type="button" class="danger" id="stopBtn" hidden>Stop</button>
        </form>
        <details class="small" style="margin-top:8px"><summary>Sampling settings (empty = use the personality style)</summary>
          <div class="row" style="margin-top:8px">
            <label>Temperature <input id="g_temp" placeholder="auto" type="number" step="0.05" min="0" max="2" style="width:90px"></label>
            <label>Top-k <input id="g_topk" placeholder="auto" type="number" min="0" max="1000" style="width:90px"></label>
            <label>Top-p <input id="g_topp" placeholder="auto" type="number" step="0.05" min="0.05" max="1" style="width:90px"></label>
            <label>Max tokens <input id="g_max" placeholder="auto" type="number" min="1" max="400" style="width:90px"></label>
            <label>Repetition penalty <input id="g_rep" placeholder="auto" type="number" step="0.05" min="1" max="2" style="width:90px"></label>
            <label>Context turns <input id="g_hist" type="number" min="0" max="8" value="2" style="width:90px"></label>
          </div></details>
      </div>
      <div class="convlist">
        <button id="newChat">New chat</button>
        <h3 style="margin-top:14px">Conversations</h3>
        <div id="convs">${convs.map((c) => `<a href="#chat" data-cid="${c.id}" class="${c.id === state.cid ? "on" : ""}">${esc(c.title || "Chat")} <span class="muted">(${c.n})</span></a>`).join("") || `<p class="muted">None yet.</p>`}</div>
      </div>
    </div>`;
  let engine = state.engine || (s0.assistant.enabled ? (s0.assistant.default_engine || "assistant") : "nexora");
  if (engine === "assistant" && !s0.assistant.enabled) engine = "nexora";
  const engineNote = async () => {
    if (engine === "nexora") { $("#engineNote").textContent = "Your own network: every reply is sampled from weights trained on this computer. Small model — expect odd text, not facts."; return; }
    try { const st = await api("/api/assistant/status");
      $("#engineNote").innerHTML = st.error ? `${esc(st.error)} <a href="#llm">AI models</a>` :
        esc(`Chat: ${st.chat_model} ${st.chat_model_ready ? "✓" : "(not downloaded)"} · Code: ${st.code_model} ${st.code_model_ready ? "✓" : "(not downloaded)"} — runs offline on this computer; not NEXORA's own network.`);
    } catch (e) { $("#engineNote").textContent = e.message; }
  };
  let model = state.model || s0.agent.default_model || "1.2";
  let target = state.target || "auto";
  const modelUi = () => {
    $("#engineWrap").hidden = model !== "1.1";
    $("#targetWrap").hidden = model !== "1.2";
    $("#connDots").hidden = model !== "1.2";
    if (model === "1.2") {
      $("#engineNote").textContent = s0.assistant.enabled
        ? "Nexo 1.2 shows what it is doing at every moment. It reads your lessons (Knowledge page) and can build in Roblox Studio and Blender (Connections page). It uses ready-made models built into NEXORA (AI models page)."
        : "Nexo 1.2 is turned off (Settings → Assistant → enabled).";
      api("/api/assistant/status").then((st) => { if (st.error) $("#engineNote").innerHTML = `${esc(tr(st.error))} <a href="#llm">${tr("AI models")}</a>`; }).catch(() => {});
    } else {
      if (!s0.assistant.enabled) $("#engineNote").textContent = "";
      engineNote();
    }
  };
  const drawConn = async () => {
    try { const st = (await api("/api/integrations")).status;
      $("#connDots").innerHTML = ["roblox", "blender"].map((a) => `<span class="dot ${st[a].online ? "on" : "off"}"></span>${a === "roblox" ? "Roblox Studio" : "Blender"}: ${st[a].online ? "connected" : "not connected"}`).join(" &nbsp; ");
    } catch { /* ignore */ }
  };
  $("#modelSel").value = model;
  $("#targetSel").value = target;
  $("#engineSel").value = engine;
  $("#modelSel").onchange = () => { model = state.model = $("#modelSel").value; modelUi(); };
  $("#targetSel").onchange = () => { target = state.target = $("#targetSel").value; };
  $("#engineSel").onchange = () => { engine = state.engine = $("#engineSel").value; engineNote(); };
  modelUi(); drawConn();
  pageTimer = setInterval(drawConn, 5000);
  const atts = state.atts || (state.atts = []);
  const drawAtts = () => { $("#attList").innerHTML = atts.map((f, i) => `<span class="chip">📄 ${esc(f.name)} <span class="muted">${f.chars.toLocaleString()}</span>
      <button type="button" data-learnfile="${f.id}" title="Add this file to the training data">Learn from this file</button><button type="button" data-rmatt="${i}" aria-label="Remove">✕</button></span>`).join(""); };
  drawAtts();
  $("#fileIn").onchange = async () => {
    for (const file of $("#fileIn").files) {
      const fd = new FormData(); fd.append("file", file); if (state.cid) fd.append("conversation_id", state.cid);
      try { const r = await fetch("/api/files", { method: "POST", body: fd }); const d = await r.json();
        if (!r.ok) throw new Error(d.detail || r.statusText); atts.push(d); }
      catch (err) { toast(`${file.name}: ${err.message}`, 7000); }
    }
    $("#fileIn").value = ""; drawAtts();
  };
  $("#attList").onclick = async (e) => {
    if (e.target.dataset.rmatt) { atts.splice(+e.target.dataset.rmatt, 1); drawAtts(); }
    if (e.target.dataset.learnfile) { e.target.disabled = true;
      const r = await act(() => api(`/api/files/${e.target.dataset.learnfile}/learn`, { method: "POST" }));
      toast(`${r.accepted || 0} of ${r.pieces} parts of ${r.name} added to the training data.`); }
  };

  const convo = $("#convo");
  const render = (msgs) => {
    const shown = !msgs.length && s0.persona.greeting && STATUS && STATUS.lm.ready ? [{ role: "greeting", content: s0.persona.greeting }] : msgs;
    convo.innerHTML = shown.length ? shown.map(msgHtml).join("") + (shown === msgs ? "" : `<p class="small muted">Greeting set on the Personality page (not generated).</p>`) : `<p class="muted">${STATUS && STATUS.lm.ready ? "Start a conversation." : "No model is trained yet — go to <a href='#training'>Training</a> to train one from random weights."}</p>`;
    convo.scrollTop = convo.scrollHeight;
  };
  const msgHtml = (m) => {
    const meta = m.meta ? (typeof m.meta === "string" ? JSON.parse(m.meta || "{}") : m.meta) : {};
    const isAsst = (m.model_version || "").startsWith("assistant:");
    const isAgent = meta.engine === "nexo-1.2";
    const who = m.role === "user" ? "You" : m.role === "assistant" ? (isAgent ? `${s0.persona.name} · Nexo 1.2 (${meta.model})` : isAsst ? `${s0.persona.name} · Nexo 1.1 · Assistant (${m.model_version.slice(10)})` : `${s0.persona.name} · Nexo 1.1 · own model ${m.model_version || ""}`) : m.role === "greeting" ? `${s0.persona.name}` : "Memory & learning system";
    let thoughts = "";
    if (isAgent && meta.thoughts && meta.thoughts.length) {
      const n = meta.thoughts.filter((t) => t.type === "step").length;
      thoughts = `<details class="thoughts"><summary>Thought for ${meta.seconds} s · ${n} steps</summary><div class="tlist">${meta.thoughts.map(thoughtHtml).join("")}</div></details>`;
    }
    let extra = "";
    const web = (meta && meta.web_results) || m.web || [];
    if (web.length) extra += `<div class="recall webres"><span>From the internet (Wikipedia) — not generated by the model:</span><ul>${web.map((w) => `<li><a href="${esc(w.url)}" target="_blank" rel="noopener">${esc(w.title)}</a> <span class="muted">${esc(w.snippet)}</span> <button data-learnurl="${esc(w.url)}">Learn from this article</button></li>`).join("")}</ul></div>`;
    if (m.recall && m.recall.length) extra += `<div class="recall">Found in memory (not generated by the model): ${m.recall.map((f) => `${esc(f.text)} <span class="muted">[${esc(f.kind)}, ${esc(f.source)}]</span>`).join("; ")}</div>`;
    if (m.role === "assistant") extra += `<div class="msgtools" data-mid="${m.id}">
        <button data-fb="1" title="Good answer">Good</button><button data-fb="-1" title="Bad answer">Bad</button>
        <button data-fix="1">Correct it</button><button data-ds="1" title="Queue this output for review as a training example">Propose as example</button>
        ${meta.code ? `<button data-example="1" title="Nexo 1.2 will reuse this working solution for similar tasks">Remember as a good example</button>` : ""}
        ${meta.tokens ? `<span class="muted small">${meta.tokens} tokens</span>` : ""}</div>`;
    const body = m.role === "assistant" ? richText(m.content) : esc(m.content);
    return `<div class="msg ${esc(m.role)} ${isAsst || isAgent ? "asst" : ""}"><div class="who">${esc(who)}</div>${thoughts}<div class="body">${body}</div>${extra}</div>`;
  };
  let msgs = state.cid ? await api(`/api/conversations/${state.cid}`) : [];
  render(msgs);

  const send = async () => {
    const text = $("#chatIn").value.trim(); if (!text) return;
    $("#sendBtn").disabled = true;
    msgs.push({ role: "user", content: text }); render(msgs);
    $("#chatIn").value = "";
    if (model === "1.2") { await sendAgent(text); $("#sendBtn").disabled = false; $("#chatIn").focus(); return; }
    if (engine === "assistant") { await sendAssistant(text); $("#sendBtn").disabled = false; $("#chatIn").focus(); return; }
    if (atts.length) toast("Your own model cannot read files (its memory is only ~300 characters). Switch to the Assistant, or use “Learn from this file”.", 9000);
    try {
      const body = { message: text, conversation_id: state.cid, history_turns: +$("#g_hist").value };
      for (const [id, key] of [["g_temp", "temperature"], ["g_topk", "top_k"], ["g_topp", "top_p"], ["g_max", "max_new_tokens"], ["g_rep", "repetition_penalty"]])
        if ($("#" + id).value !== "") body[key] = +$("#" + id).value;
      const r = await api("/api/chat", { method: "POST", body });
      state.cid = r.conversation_id;
      msgs = await api(`/api/conversations/${state.cid}`);
      if (r.memory_recall) msgs[msgs.length - 1].recall = r.memory_recall;
      if (r.web_error) toast(r.web_error, 7000);
      render(msgs);
    } catch (e) { toast(e.message, 7000); msgs.pop(); render(msgs); }
    $("#sendBtn").disabled = false; $("#chatIn").focus();
  };
  const sendAgent = async (text) => {
    const live = document.createElement("div");
    live.className = "msg assistant asst";
    live.innerHTML = `<div class="who">${esc(s0.persona.name)} · Nexo 1.2</div>
      <details class="thoughts" open><summary>Thinking…</summary><div class="tlist"></div></details><div class="body"></div>`;
    convo.appendChild(live); convo.scrollTop = convo.scrollHeight;
    const tlist = live.querySelector(".tlist"), body = live.querySelector(".body"), summary = live.querySelector("summary");
    const sent = atts.splice(0); drawAtts();
    let acc = "", done = null, stream = null, streamText = "", runId = null, steps = 0;
    const stopBtn = $("#stopBtn"); stopBtn.hidden = false;
    stopBtn.onclick = () => { if (runId) api(`/api/agent/${runId}/stop`, { method: "POST" }).catch(() => {}); };
    const add = (html) => { tlist.insertAdjacentHTML("beforeend", html); tlist.scrollTop = tlist.scrollHeight; convo.scrollTop = convo.scrollHeight; };
    try {
      const r = await fetch("/api/agent/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, conversation_id: state.cid, attachment_ids: sent.map((f) => f.id), target }) });
      if (r.status === 401) { location.href = "/login"; return; }
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || r.statusText); }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done: end } = await reader.read(); if (end) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.type === "start") { state.cid = ev.conversation_id; runId = ev.run_id;
            if (ev.model) live.querySelector(".who").textContent = `${s0.persona.name} · Nexo 1.2 (${ev.model})`; }
          else if (ev.type === "step") { steps++; stream = null; add(thoughtHtml(ev)); }
          else if (ev.type === "think") {
            if (!stream) { add(`<pre class="tstream"></pre>`); stream = tlist.lastElementChild; streamText = ""; }
            streamText += ev.delta; stream.textContent = streamText; stream.scrollTop = stream.scrollHeight; tlist.scrollTop = tlist.scrollHeight; convo.scrollTop = convo.scrollHeight;
          } else if (ev.type === "think_block") {
            if (stream) { const plan = planOf(ev.text); if (plan) stream.textContent = plan; else stream.remove(); }
            stream = null;
          }
          else if (ev.type === "code" || ev.type === "image") { stream = null; add(thoughtHtml(ev)); }
          else if (ev.type === "approve") {
            stream = null;
            add(`<div class="approve"><b>${tr("Run this code?")}</b> ${ev.risky ? `<span class="warnline">${tr("It uses files, the network or other risky functions. Read it before you allow it.")}</span>` : ""}
              <div class="row"><button class="good" data-decide="1" data-run="${esc(ev.run_id)}">${tr("Run")}</button><button data-decide="0" data-run="${esc(ev.run_id)}">${tr("Don't run")}</button></div></div>`);
          } else if (ev.type === "delta") { acc += ev.text; body.innerHTML = richText(acc); convo.scrollTop = convo.scrollHeight; }
          else if (ev.type === "done") done = ev;
        }
      }
      if (done && done.error) toast(done.error, 10000);
    } catch (err) { toast(err.message, 9000); }
    stopBtn.hidden = true;
    if (done && done.note) { msgs = await api(`/api/conversations/${state.cid}`); render(msgs); return; }
    summary.textContent = done ? `${tr("Thought for")} ${done.seconds} s · ${steps} ${tr("steps")}` : tr("Stopped");
    msgs = state.cid ? await api(`/api/conversations/${state.cid}`) : msgs;
    render(msgs);
  };
  const sendAssistant = async (text) => {
    const live = document.createElement("div");
    live.className = "msg assistant asst";
    live.innerHTML = `<div class="who">${esc(s0.persona.name)} · Assistant</div><div class="body"><span class="muted">${tr("Thinking… (on a laptop CPU the first answer can take a minute)")}</span></div>`;
    convo.appendChild(live); convo.scrollTop = convo.scrollHeight;
    const sent = atts.splice(0); drawAtts();
    let acc = "", done = null;
    try {
      const r = await fetch("/api/assistant/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, conversation_id: state.cid, attachment_ids: sent.map((f) => f.id) }) });
      if (r.status === 401) { location.href = "/login"; return; }
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || r.statusText); }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done: end } = await reader.read(); if (end) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.start) { state.cid = ev.conversation_id; live.querySelector(".who").textContent = `${s0.persona.name} · Assistant (${ev.model})`; }
          if (ev.delta) { acc += ev.delta; live.querySelector(".body").innerHTML = richText(acc); convo.scrollTop = convo.scrollHeight; }
          if (ev.done) done = ev;
        }
      }
      if (done && done.error) toast(done.error, 10000);
    } catch (err) { toast(err.message, 9000); }
    msgs = state.cid ? await api(`/api/conversations/${state.cid}`) : msgs;
    render(msgs);
  };
  $("#chatForm").onsubmit = (e) => { e.preventDefault(); send(); };
  $("#chatIn").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  $("#newChat").onclick = () => { state.cid = null; show("chat"); };
  $("#convs").onclick = (e) => { const a = e.target.closest("[data-cid]"); if (a) { e.preventDefault(); state.cid = +a.dataset.cid; show("chat"); } };
  convo.onclick = async (e) => {
    if (e.target.dataset.decide) {
      const card = e.target.closest(".approve");
      card.querySelectorAll("button").forEach((b) => { b.disabled = true; });
      await act(() => api(`/api/agent/${e.target.dataset.run}/decision`, { method: "POST", body: { approve: e.target.dataset.decide === "1" } }));
      card.insertAdjacentHTML("beforeend", `<span class="small muted">${e.target.dataset.decide === "1" ? tr("Approved – running…") : tr("Not run.")}</span>`);
      return;
    }
    if (e.target.dataset.example) {
      const mid = +e.target.closest(".msgtools").dataset.mid; e.target.disabled = true;
      const r = await act(() => api(`/api/agent/example/${mid}`, { method: "POST" }));
      toast(r.duplicate ? tr("Already remembered.") : tr("Saved in the knowledge base. Nexo 1.2 will reuse it for similar tasks."));
      return;
    }
    if (e.target.dataset.learnurl) {
      e.target.disabled = true;
      const r = await act(() => api("/api/web/learn", { method: "POST", body: { url: e.target.dataset.learnurl } }));
      toast(`${r.accepted || 0} paragraphs from “${r.title}” added to the training data.`);
      return;
    }
    const box = e.target.closest(".msgtools"); if (!box) return;
    const mid = +box.dataset.mid;
    if (e.target.dataset.fb) await act(() => api("/api/feedback", { method: "POST", body: { message_id: mid, rating: +e.target.dataset.fb } }), "Rating saved. It informs the next learning cycle; weights are unchanged.");
    if (e.target.dataset.fix) {
      const c = prompt("Write the answer NEXORA should have given. It becomes a training example (dataset), not a memory fact:");
      if (c) await act(() => api("/api/feedback", { method: "POST", body: { message_id: mid, rating: -1, correction: c } }), "Correction saved. It will be added to a candidate dataset at the next learning cycle.");
    }
    if (e.target.dataset.ds) {
      const r = await act(() => api(`/api/examples/from_message/${mid}`, { method: "POST" }));
      toast(r.status === "duplicate" ? "Already proposed." : "Queued for your review in Datasets. Model output is never trained on without your approval.");
    }
  };
};

pages.persona = async (el) => {
  const d = await api("/api/persona");
  const p = d.persona;
  const styleNames = { balanced: "Balanced", precise: "Precise", creative: "Creative", brief: "Brief", talkative: "Talkative" };
  el.innerHTML = `<h1>Personality</h1>
    <p class="lead">Decide how NEXORA behaves and how it speaks. These settings act on the real model: the style changes how text is sampled,
    the description is placed in front of every conversation, and the limits trim what the model wrote. To make a style part of the network itself,
    teach it with example answers below.</p>
    <form id="pform">
      <div class="panel"><h3>Name and greeting</h3><div class="grid2">
        <label>Assistant name <input id="p_name" maxlength="30" value="${esc(p.name)}"></label>
        <label>Greeting shown when a new chat starts (optional) <input id="p_greet" maxlength="300" value="${esc(p.greeting)}"></label></div></div>
      <div class="panel"><h3>Speaking style</h3><div class="stylepick">
        ${Object.entries(d.styles).map(([k, v]) => `<label><input type="radio" name="p_style" value="${k}" ${p.style === k ? "checked" : ""}>${esc(styleNames[k] || k)}<div class="small muted">${esc(v.description)}</div></label>`).join("")}
      </div>
      <label style="margin-top:12px">Maximum sentences per answer (0 = no limit) <input id="p_sent" type="number" min="0" max="20" value="${p.max_sentences}" style="width:120px"></label></div>
      <div class="panel"><h3>Behaviour description</h3>
        <p class="small muted">Written in the language the model was trained on. The model continues the conversation after this text, so describe the scene or write the start of a dialogue in the style you want. Max ${d.max_context_chars} characters; it uses part of the short context window.</p>
        <textarea id="p_ctx" rows="3" maxlength="${d.max_context_chars}">${esc(p.context)}</textarea>
        <label style="margin-top:10px">Blocked words (one per line) – replaced with *** in answers <textarea id="p_block" rows="2">${esc((p.blocked_words || []).join("\n"))}</textarea></label></div>
      <div class="row"><button class="primary">Save personality</button></div>
    </form>
    <h2>Try it before saving</h2>
    <div class="panel"><div class="row"><input id="p_try" style="flex:1" placeholder="Write a test message"><button id="p_tryb">Try</button></div><div id="p_out"></div></div>
    <h2>Teach this personality to the network</h2>
    <div class="panel"><p class="small muted">Write example exchanges, one per line, in the form “message => answer”. They are added to the training data; after the next learning cycle and your approval the network itself answers more like this.</p>
      <textarea id="p_ex" rows="5" placeholder="Hello! => Good day to you, friend. How can I help?"></textarea>
      <button id="p_teach" style="margin-top:8px">Add examples to training data</button></div>`;
  const current = () => ({ name: $("#p_name").value.trim() || "NEXORA", greeting: $("#p_greet").value, style: (document.querySelector("input[name=p_style]:checked") || {}).value || "balanced",
    max_sentences: +$("#p_sent").value || 0, context: $("#p_ctx").value, blocked_words: $("#p_block").value.split("\n").map((x) => x.trim()).filter(Boolean) });
  $("#pform").onsubmit = async (e) => { e.preventDefault(); await act(() => api("/api/settings", { method: "PUT", body: { persona: current() } }), "Personality saved."); };
  $("#p_tryb").onclick = async () => {
    const m = $("#p_try").value.trim(); if (!m) return;
    $("#p_out").innerHTML = `<p class="muted small">Generating…</p>`;
    try { const r = await api("/api/persona/preview", { method: "POST", body: { message: m, persona: current() } });
      $("#p_out").innerHTML = `<div class="msg assistant" style="margin-top:10px"><div class="who">${esc(current().name)}</div><div class="body">${esc(r.text || "…")}</div></div>
        <p class="small muted">temperature ${r.params.temperature}, top-k ${r.params.top_k}, max tokens ${r.params.max_new_tokens}${r.postprocess.length ? " · " + esc(r.postprocess.join(", ")) : ""}</p>`;
    } catch (err) { $("#p_out").innerHTML = `<div class="warn err">${esc(err.message)}</div>`; }
  };
  $("#p_teach").onclick = async () => {
    const ex = $("#p_ex").value.split("\n").map((l) => l.split("=>")).filter((x) => x.length === 2 && x[0].trim() && x[1].trim()).map(([a, b]) => ({ prompt: a.trim(), answer: b.trim() }));
    if (!ex.length) { toast("Use the format: message => answer"); return; }
    const r = await act(() => api("/api/persona/teach", { method: "POST", body: { examples: ex } }));
    toast(`${r.accepted} of ${ex.length} examples added to the training data.`); $("#p_ex").value = "";
  };
};

pages.knowledge = async (el) => {
  const k = await api("/api/knowledge");
  const kinds = { lesson: "lesson", example: "example", video: "video", file: "file", web: "web" };
  el.innerHTML = `<h1>Knowledge (Nexo 1.2)</h1>
    <p class="lead">Lessons, videos and good examples you give Nexo 1.2 are stored here. When you ask something, the most relevant
    fragments are put in front of the model, so it answers using them. The model's weights do not change – it reads its notes every time.</p>
    <div class="panel"><h3>Built-in lessons</h3><div class="row">${k.packs.map((p) => {
      const have = k.docs.some((d) => d.source === "pack:" + p.id);
      return `<button ${have ? "disabled" : ""} data-pack="${esc(p.id)}">${have ? "✓ " : ""}${esc(p.title)}</button>`; }).join("")}</div>
      <p class="small muted">Lua/Luau and the Roblox API; 3D modelling in Blender with Python. Import them once.</p></div>
    <form class="panel" id="lessonForm"><h3>Teach a lesson</h3>
      <div class="grid2"><label>Title <input id="lt" maxlength="200" placeholder="e.g. My game's rules"></label>
      <label>Topic <select id="ltopic"><option value="">general</option><option value="roblox">Roblox / Lua</option><option value="blender">Blender / modelling</option></select></label></div>
      <label style="margin-top:8px">Lesson text (explanations, rules, code examples) <textarea id="ltext" rows="7" required></textarea></label>
      <div class="row" style="margin-top:8px"><button class="primary">Save lesson</button>
      <label class="btn">📄 From a file (txt, md, code, PDF, DOCX) <input type="file" id="lfile" hidden></label></div></form>
    <form class="panel" id="videoForm"><h3>Learn from a video</h3>
      <p class="small muted">A video or audio file, or subtitles (.srt / .vtt). Speech is turned into text on this computer (the first time the speech model, about 500 MB, is downloaded), key frames are saved, and everything becomes a lesson. YouTube downloading is not built in – use your own files, or copy a video's transcript and paste it as a lesson.</p>
      <div class="row"><input type="file" id="vfile" accept="video/*,audio/*,.srt,.vtt" required>
        <label>Topic <select id="vtopic"><option value="">general</option><option value="roblox">Roblox / Lua</option><option value="blender">Blender / modelling</option></select></label>
        <label><span><input type="checkbox" id="vtrain"> also add the transcript to Nexo 1.1 training data</span></label>
        <button class="primary">Learn from video</button></div><div id="vstatus" class="small"></div></form>
    <h2>What Nexo 1.2 knows</h2>
    <div class="panel"><table><thead><tr><th>Title</th><th>Kind</th><th>Topic</th><th class="num">Fragments</th><th>Added</th><th></th></tr></thead><tbody>
    ${k.docs.map((d) => `<tr><td><b>${esc(d.title)}</b><div class="small muted">${esc(d.source || "")}</div></td><td>${tag(kinds[d.kind] || d.kind)}</td><td>${esc(d.topic || "")}</td>
      <td class="num">${d.chunks}</td><td class="small">${when(d.created)}</td>
      <td class="row"><button data-viewdoc="${d.id}">View</button><button class="danger" data-deldoc="${d.id}">Delete</button></td></tr>`).join("") || `<tr><td colspan="6" class="muted">Nothing yet. Import a built-in lesson or teach one.</td></tr>`}
    </tbody></table><div id="docView"></div></div>
    <div class="panel"><h3>Test the search</h3><div class="row"><input id="kq" style="flex:1" placeholder="e.g. how to make a part spin"><button id="kqb">Search</button></div><div id="kres"></div></div>`;
  el.onclick = async (e) => {
    const d = e.target.dataset;
    if (d.pack) { e.target.disabled = true; const r = await act(() => api(`/api/knowledge/packs/${d.pack}`, { method: "POST" })); toast(`${r.chunks} ${tr("fragments saved.")}`); show("knowledge"); }
    if (d.deldoc && confirm(tr("Delete this from the knowledge base?"))) { await act(() => api(`/api/knowledge/${d.deldoc}`, { method: "DELETE" })); show("knowledge"); }
    if (d.viewdoc) { const x = await api(`/api/knowledge/${d.viewdoc}`); $("#docView").innerHTML = `<h3 style="margin-top:12px">${esc(x.title)}</h3><div class="body">${richText(x.text)}</div>`; }
  };
  $("#lessonForm").onsubmit = async (e) => { e.preventDefault();
    const r = await act(() => api("/api/knowledge", { method: "POST", body: { title: $("#lt").value, text: $("#ltext").value, topic: $("#ltopic").value } }));
    toast(r.duplicate ? tr("Already in the knowledge base.") : `${r.chunks} ${tr("fragments saved.")}`); show("knowledge"); };
  $("#lfile").onchange = async () => {
    const f = $("#lfile").files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f); fd.append("topic", $("#ltopic").value);
    const r = await fetch("/api/knowledge/file", { method: "POST", body: fd }); const d = await r.json();
    if (!r.ok) toast(d.detail || r.statusText, 8000); else { toast(`${d.chunks} ${tr("fragments saved.")}`); show("knowledge"); }
  };
  $("#kqb").onclick = async () => { const r = await api(`/api/knowledge-search?q=${encodeURIComponent($("#kq").value)}`);
    $("#kres").innerHTML = r.map((h) => `<div class="pre small">${esc(h.text.slice(0, 500))}</div>`).join("") || `<p class="muted">${tr("No matches.")}</p>`; };
  $("#videoForm").onsubmit = async (e) => {
    e.preventDefault();
    const f = $("#vfile").files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f); fd.append("also_training", $("#vtrain").checked); fd.append("topic", $("#vtopic").value);
    $("#vstatus").textContent = tr("Uploading…");
    const r = await fetch("/api/video", { method: "POST", body: fd }); const d = await r.json();
    if (!r.ok) { $("#vstatus").textContent = d.detail || r.statusText; return; }
    const poll = async () => {
      const j = await api(`/api/jobs/${d.job_id}`); const p = j.progress || {};
      if (["queued", "running", "paused"].includes(j.status)) { $("#vstatus").textContent = `${tr(j.status)}: ${p.message || ""}`; setTimeout(poll, 3000); return; }
      if (j.status === "completed") {
        $("#vstatus").innerHTML = `${tr("Done")}: ${j.result.lines} ${tr("lines of speech")}, ${j.result.frames.length} ${tr("frames")}.<div class="imggrid" style="margin-top:8px">${j.result.frames.map((fr) => `<img src="/generated/${esc(fr)}" alt="">`).join("")}</div>`;
        setTimeout(() => show("knowledge"), 6000);
      } else $("#vstatus").textContent = `${tr(j.status)}: ${j.error || ""}`;
    };
    poll();
  };
};

pages.connections = async (el) => {
  const draw = async () => {
    const st = await api("/api/integrations");
    const card = (id, name, file, steps) => `<div class="panel"><h3><span class="dot ${st.status[id].online ? "on" : "off"}"></span> ${name}: ${st.status[id].online ? "connected" : "not connected"}</h3>
      ${st.status[id].info ? `<p class="small muted">${esc(Object.entries(st.status[id].info).map(([k, v]) => `${k}: ${v}`).join(" · "))}</p>` : ""}
      <a class="btn primary" href="/api/integrations/${id}/download" download="${file}">${tr("Download")} ${file}</a>
      <ol class="small">${steps.map((x) => `<li>${x}</li>`).join("")}</ol></div>`;
    el.innerHTML = `<h1>Connections</h1>
      <p class="lead">Nexo 1.2 can build in Roblox Studio and model in Blender. Install the small connector once; it talks only to NEXORA on this computer and carries a private key, so do not share the downloaded file.</p>
      ${card("roblox", "Roblox Studio", "NexoraBridge.lua", [
        "Easiest: run install_roblox_plugin.ps1 (NEXORA must be running). Or download the file and put it in the folder %LOCALAPPDATA%\\Roblox\\Plugins.",
        "Restart Roblox Studio and open a place. A NEXORA button appears on the Plugins tab.",
        "When Studio asks whether the plugin may connect to 127.0.0.1, click Allow.",
        "Everything Nexo 1.2 does is one undo step – Ctrl+Z removes it. Failed code is rolled back automatically."])}
      ${card("blender", "Blender", "nexora_bridge.py", [
        "Download the file.",
        "In Blender: Edit → Preferences → Add-ons → arrow menu (top right) → Install from Disk… → choose nexora_bridge.py and tick NEXORA Bridge.",
        "Panel: 3D Viewport → press N → NEXORA tab. It connects automatically.",
        "If code fails, everything it created is removed. Ctrl+Z works after successful steps."])}
      <div class="panel"><h3>Safety</h3><p class="small">By default Nexo 1.2 asks before running any code (Settings → Agent → auto run). Code that touches files, the network or downloads is always shown to you first, even with auto run on. Generated code can be wrong – keep a copy of important projects.</p></div>`;
  };
  await draw(); pageTimer = setInterval(draw, 4000);
};

pages.llm = async (el) => {
  const draw = async () => {
    const d = await api("/api/llm/models");
    const roles = { chat: "Conversation (Polish)", code: "Code, Roblox, Blender, files" };
    const card = (m) => {
      const dl = d.downloads[m.id];
      const prog = dl && dl.progress && dl.progress.max_steps ? Math.min(100, (dl.progress.step || 0) / dl.progress.max_steps * 100) : 0;
      return `<div class="panel modelcard"><div class="row" style="justify-content:space-between">
        <h3>${esc(m.name)} ${m.recommended ? `<span class="tag t-ok">${tr("recommended")}</span>` : ""}</h3>
        <span class="small muted">${m.size_gb} GB · ~${m.ram_gb} GB RAM</span></div>
        <p class="small">${esc(m.about)}</p>
        <p class="small muted">${tr("Purpose")}: ${esc(tr(roles[m.role]))} · ${tr("Author")}: ${esc(m.author)} · ${tr("Licence")}: ${esc(tr(m.license))}</p>
        ${m.installed ? `<div class="row"><span class="tag t-active">${tr("downloaded")}</span>
            ${m.role === "chat" ? (m.selected_chat ? `<span class="tag t-ok">${tr("used for conversation")}</span>` : `<button data-use="${m.id}" data-role="chat">${tr("Use for conversation")}</button>`)
                                : (m.selected_code ? `<span class="tag t-ok">${tr("used for code")}</span>` : `<button data-use="${m.id}" data-role="code">${tr("Use for code")}</button>`)}
            <button class="danger" data-delllm="${m.id}">${tr("Delete file")}</button></div>`
          : dl ? `<div class="meter"><i style="width:${prog}%"></i></div><p class="small">${tr(dl.status)} ${esc((dl.progress || {}).message || "")} ${dl.error ? esc(dl.error) : ""}</p>
              <div class="row">${dl.status === "running" ? `<button data-pause="${dl.job_id}">${tr("Pause")}</button>` : `<button data-get="${m.id}">${tr("Resume")}</button>`}<button class="danger" data-cancel="${dl.job_id}">${tr("Cancel")}</button></div>`
          : `<button class="primary" data-get="${m.id}">${tr("Download")} (${m.size_gb} GB)</button>${m.partial_mb ? ` <span class="small muted">${m.partial_mb} MB ${tr("already downloaded")}</span>` : ""}`}
      </div>`;
    };
    el.innerHTML = `<h1>AI models</h1>
      <p class="lead">Ready-made open models that NEXORA runs itself, offline, on this computer – no other program needed. They are downloaded once from Hugging Face. They power Nexo 1.2 and the Assistant in Nexo 1.1; NEXORA's own network (trained from scratch) is separate.</p>
      ${d.engine.installed ? "" : `<div class="warn err">${tr("The built-in model engine (llama-cpp-python) is not installed. Run setup.ps1 again.")}</div>`}
      <p class="small muted">${tr("For 8 GB of RAM: Bielik 1.5B + Qwen2.5-Coder 3B (about 3.8 GB on disk). Only one model is in memory at a time.")}</p>
      <h2>${tr("Conversation")}</h2>${d.models.filter((m) => m.role === "chat").map(card).join("")}
      <h2>${tr("Code")}</h2>${d.models.filter((m) => m.role === "code").map(card).join("")}`;
    return Object.keys(d.downloads).some((k) => ["queued", "running"].includes(d.downloads[k].status));
  };
  el.onclick = async (e) => {
    const t = e.target.dataset;
    if (t.get) { e.target.disabled = true; await act(() => api(`/api/llm/models/${t.get}/download`, { method: "POST" }), tr("Download started.")); }
    if (t.pause) await act(() => api(`/api/jobs/${t.pause}/pause`, { method: "POST" }));
    if (t.cancel && confirm(tr("Cancel this download? The part already downloaded is kept."))) await act(() => api(`/api/jobs/${t.cancel}/cancel`, { method: "POST" }));
    if (t.use) await act(() => api(`/api/llm/models/${t.use}/use?role=${t.role}`, { method: "POST" }), tr("Saved."));
    if (t.delllm && confirm(tr("Delete this model file?"))) await act(() => api(`/api/llm/models/${t.delllm}`, { method: "DELETE" }), tr("Deleted."));
    draw();
  };
  await draw();
  pageTimer = setInterval(draw, 2500);
};

pages.training = async (el) => {
  const [jobs, models] = await Promise.all([api("/api/jobs"), api("/api/models")]);
  const active = models.find((m) => m.status === "active");
  const run = jobs.find((j) => ["running", "paused", "queued"].includes(j.status));
  el.innerHTML = `<h1>Training</h1>
    <p class="lead">Training is the only thing that changes the network's weights. Jobs run one at a time in the background,
    with checkpoints, a time limit and memory/disk guards. Pausing saves a checkpoint; interrupted jobs resume from it.</p>
    <div class="grid2">
      <div class="panel">
        <h3>Base model</h3>
        ${!STATUS.base_prepared ? `<p>Step 1: split the bundled public-domain corpus (Tiny Shakespeare) into training, validation and held-out sets and train the tokenizer.</p><button class="primary" id="prep">Prepare base data</button>` :
          active ? `<p>Active: <b>${esc(active.id)}</b>, ${active.metrics._train ? active.metrics._train.steps + " steps trained" : ""}.</p><p class="muted small">Training a new model from random weights creates a candidate that must be approved before it replaces ${esc(active.id)}.</p>`
                 : `<p>Data is ready. Train a model from randomly initialised weights.</p>`}
        ${STATUS.base_prepared ? `<div class="row"><label>Steps <input id="steps" type="number" min="10" placeholder="config default" style="width:130px"></label>
          <label>Time limit (min) <input id="mins" type="number" min="1" placeholder="config default" style="width:130px"></label>
          <button class="primary" id="trainBase">Train from random weights</button></div>` : ""}
      </div>
      <div class="panel" id="runPanel"></div>
    </div>
    <h2>Loss curves</h2><div class="panel" id="curves"></div>
    <h2>Job history</h2>
    <div class="panel"><table><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th class="num">Step</th><th>Result / error</th><th></th></tr></thead><tbody>
    ${jobs.map((j) => `<tr><td>${j.id}</td><td>${esc(j.kind)}</td><td>${tag(j.status)}</td><td class="num">${j.progress.step ?? ""}/${j.progress.max_steps ?? ""}</td>
      <td class="small">${esc(j.error || (j.result.model_version ? "model " + j.result.model_version : j.result.candidate ? `candidate ${j.result.candidate}: ${j.result.passed ? "passed" : "failed"} guard` : j.result.image_version ? "image model " + j.result.image_version : ""))}</td>
      <td>${["interrupted", "stopped"].includes(j.status) ? `<button data-resume="${j.id}">Resume</button> <button data-cancel="${j.id}">Discard</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="6" class="muted">No jobs yet.</td></tr>`}
    </tbody></table></div>`;
  const prep = $("#prep");
  if (prep) prep.onclick = async () => { prep.disabled = true; prep.textContent = "Preparing (about 10 s)…"; await act(() => api("/api/lm/prepare", { method: "POST" }), "Base data prepared."); await refreshStatus(); show("training"); };
  const tb = $("#trainBase");
  if (tb) tb.onclick = async () => { const b = {}; if ($("#steps").value) b.steps = +$("#steps").value; if ($("#mins").value) b.max_minutes = +$("#mins").value; await act(() => api("/api/lm/train", { method: "POST", body: b }), "Training job queued."); show("training"); };
  el.onclick = async (e) => {
    const t = e.target;
    for (const a of ["resume", "cancel", "pause"]) if (t.dataset[a]) { await act(() => api(`/api/jobs/${t.dataset[a]}/${a}`, { method: "POST" })); show("training"); }
  };
  const draw = async () => {
    const live = run ? await api(`/api/jobs/${run.id}`) : null;
    const p = live ? live.progress : {};
    $("#runPanel").innerHTML = live ? `<h3>Job ${live.id}: ${esc(live.kind.replace("_", " "))} ${tag(live.status)}</h3>
      <div class="meter" aria-label="progress"><i style="width:${Math.min(100, ((p.step || 0) / (p.max_steps || 1)) * 100)}%"></i></div>
      <div class="row" style="margin:10px 0;gap:24px">
        <div class="stat"><span class="v">${p.step ?? 0}</span><span class="k">of ${p.max_steps ?? "?"} steps</span></div>
        <div class="stat"><span class="v">${fmt(p.train_loss)}</span><span class="k">training loss</span></div>
        <div class="stat"><span class="v">${fmt(p.val_loss)}</span><span class="k">validation loss</span></div>
        <div class="stat"><span class="v">${fmt(p.sec_per_step, 2)}</span><span class="k">sec / step</span></div></div>
      <p class="small muted">${esc(p.message || "")}</p>
      <div class="row">${live.status === "running" ? `<button data-pause="${live.id}">Pause</button>` : ""}${live.status === "paused" ? `<button class="primary" data-resume="${live.id}">Resume</button>` : ""}
      <button class="danger" data-cancel="${live.id}">Stop and save</button></div>` : `<h3>No job running</h3><p class="muted">Start one on the left, or let the learning manager queue candidates.</p>`;
    const h = (live && live.progress.history) || null;
    const src = active && active.metrics._train && active.metrics._train.history;
    const hist = live && live.result && live.result.history ? live.result.history : src;
    $("#curves").innerHTML = hist ? `<p class="small muted">${live ? "" : "Training run of the active model " + esc(active.id)}</p>` + chart([
      { name: "training loss", points: hist.train, color: "var(--ink)" }, { name: "validation loss", points: hist.val, color: "var(--wts)" }]) : `<p class="muted">Curves appear after a training run completes.</p>`;
    void h;
  };
  await draw();
  if (run) pageTimer = setInterval(async () => { await draw(); await refreshStatus(); if (!STATUS.job) { clearInterval(pageTimer); show("training"); } }, 3000);
};

pages.learning = async (el) => {
  const [goals, last, s] = await Promise.all([api("/api/goals"), api("/api/learning/last"), api("/api/settings")]);
  el.innerHTML = `<h1>Learning goals</h1>
    <p class="lead">The learning manager looks at your feedback and at evaluation results, proposes goals, gathers data from
    your corrections and approved folders, validates it, and trains a candidate in the background. You only decide on finished candidates.</p>
    <div class="panel"><div class="row"><button class="primary" id="cycle">Run a learning cycle now</button>
      <span class="muted small">Scheduled cycles: ${s.schedule.enabled ? `every ${s.schedule.interval_minutes} min, hours ${esc(s.schedule.allowed_hours)}` : "off (change in Settings)"}</span></div>
      <div id="cycleLog" style="margin-top:10px">${last.last ? cycleHtml(last.last) : `<p class="muted small">No cycle has run since the app started.</p>`}</div></div>
    <h2>Goals</h2>
    <div class="panel"><table><thead><tr><th>Goal</th><th>Kind</th><th>Status</th><th>Keywords</th><th></th></tr></thead><tbody>
    ${goals.map((g) => `<tr><td><b>${esc(g.title)}</b><div class="small muted">${esc(g.description)}</div></td><td>${esc(g.kind)}</td><td>${tag(g.status)}</td>
      <td class="small">${esc(g.keywords.join(", "))}</td><td>${g.status === "active" ? `<button data-dismiss="${g.id}">Dismiss</button>` : g.status === "dismissed" ? `<button data-activate="${g.id}">Reactivate</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">No goals yet. Rate or correct some chat answers, then run a cycle.</td></tr>`}
    </tbody></table></div>
    <h2>Add your own goal</h2>
    <form class="panel" id="goalForm"><div class="row">
      <label style="flex:1">Title <input id="gt" required></label>
      <label style="flex:1">Keywords (comma separated) — used to pick paragraphs from approved folders <input id="gk"></label>
      <button class="primary">Add goal</button></div></form>
    <p class="small muted">Approved folders: ${s.learning.approved_folders.length ? s.learning.approved_folders.map(esc).join(", ") : "none — add them in Settings so the manager can read local .txt/.md files."}</p>`;
  $("#cycle").onclick = async () => { const r = await act(() => api("/api/learning/cycle", { method: "POST" })); $("#cycleLog").innerHTML = cycleHtml(r); refreshStatus(); };
  $("#goalForm").onsubmit = async (e) => { e.preventDefault(); await act(() => api("/api/goals", { method: "POST", body: { title: $("#gt").value, keywords: $("#gk").value.split(",").map((x) => x.trim().toLowerCase()).filter(Boolean) } }), "Goal added."); show("learning"); };
  el.onclick = async (e) => {
    if (e.target.dataset.dismiss) { await act(() => api(`/api/goals/${e.target.dataset.dismiss}/status/dismissed`, { method: "POST" })); show("learning"); }
    if (e.target.dataset.activate) { await act(() => api(`/api/goals/${e.target.dataset.activate}/status/active`, { method: "POST" })); show("learning"); }
  };
};
function cycleHtml(c) {
  return `<p class="small muted">Cycle at ${when(c.started)} (${esc(c.actor)})</p><ol class="small">${c.steps.map((s) => {
    const { msg, ...rest } = s; const kv = Object.entries(rest).map(([k, v]) => `${esc(k)}: ${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(", ");
    return `<li>${esc(msg)}${kv ? ` <span class="muted">(${kv})</span>` : ""}</li>`; }).join("")}</ol>`;
}

pages.candidates = async (el) => {
  const cands = await api("/api/candidates");
  const pending = cands.filter((c) => c.status === "pending_approval");
  const past = cands.filter((c) => c.status !== "pending_approval");
  el.innerHTML = `<h1>Approvals</h1>
    <p class="lead">A candidate reaches this page only after the Learning Guard compared it with the active model on held-out and
    regression sets. Approving makes it the active model (the previous one stays available for rollback). Rejecting leaves everything as it is.</p>
    ${pending.length ? pending.map((c) => reportHtml(c, true)).join("") : `<div class="panel muted">Nothing is waiting for approval.</div>`}
    <h2>Earlier decisions</h2>${past.map((c) => `<details class="panel"><summary>${esc(c.id)} ${tag(c.status)} <span class="muted small">${when(c.created)} — ${esc((c.report.comparison || {}).reasons ? c.report.comparison.reasons.join("; ") : c.note || "")}</span></summary>${reportHtml(c, false)}</details>`).join("") || `<p class="muted">None yet.</p>`}`;
  el.onclick = async (e) => {
    const v = e.target.dataset.approve || e.target.dataset.reject; if (!v) return;
    const approve = !!e.target.dataset.approve;
    if (approve && !confirm(`Make ${v} the active model?`)) return;
    await act(() => api(`/api/candidates/${v}/decide`, { method: "POST", body: { approve, note: ($(`#note-${v}`) || {}).value || "" } }), approve ? `${v} is now active.` : `${v} rejected.`);
    await refreshStatus(); show("candidates");
  };
};
function reportHtml(c, actions) {
  const r = c.report || {};
  const cmp = r.comparison;
  if (!cmp) return `<div class="panel"><h3>${esc(c.id)}</h3><p>Trained from scratch. Metrics: ${esc(JSON.stringify(Object.fromEntries(Object.entries(c.metrics).filter(([k]) => !k.startsWith("_")).map(([k, m]) => [k, fmt(m.loss)]))))}</p>
    ${actions ? `<div class="row"><button class="good" data-approve="${c.id}">Approve and activate</button><button class="danger" data-reject="${c.id}">Reject</button></div>` : ""}</div>`;
  const samples = r.samples ? r.samples.candidate.map((s, i) => `<tr><td class="small">${esc(s.prompt.replace(/\n/g, " ⏎ "))}</td><td><div class="pre small">${esc(r.samples.current[i].output)}</div></td><td><div class="pre small">${esc(s.output)}</div></td></tr>`).join("") : "";
  return `<div class="panel"><h3>${esc(c.id)} — candidate based on ${esc(r.current)} ${cmp.passed ? `<span class="tag t-ok">passed the guard</span>` : `<span class="tag t-rejected">failed the guard</span>`}</h3>
    <p class="small muted">Dataset ${esc(r.dataset || "replay only")} · ${r.training ? r.training.steps : "?"} fine-tuning steps · created ${when(r.created)}</p>
    <ul class="small">${cmp.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>${r.auto_action ? `<div class="warn">${esc(r.auto_action)}</div>` : ""}
    <table><thead><tr><th>Evaluation set</th><th>Role</th><th class="num">Active loss</th><th class="num">Candidate loss</th><th class="num">Change</th><th>Verdict</th></tr></thead><tbody>
    ${cmp.rows.map((x) => `<tr><td>${esc(x.set)}</td><td>${esc(x.role)}</td><td class="num">${fmt(x.current_loss, 4)}</td><td class="num">${fmt(x.candidate_loss, 4)}</td><td class="num">${x.delta_pct > 0 ? "+" : ""}${fmt(x.delta_pct, 2)}%</td><td>${tag(x.verdict)}</td></tr>`).join("")}
    </tbody></table><p class="small muted">Change = loss reduction; positive is better. Losses are measured on data neither model trained on.</p>
    <details><summary>Sample outputs (same prompts, same random seed)</summary><table><thead><tr><th>Prompt</th><th>Active</th><th>Candidate</th></tr></thead><tbody>${samples}</tbody></table></details>
    ${actions ? `<div class="row" style="margin-top:12px"><input id="note-${c.id}" placeholder="Optional note" style="flex:1"><button class="good" data-approve="${c.id}">Approve and activate</button><button class="danger" data-reject="${c.id}">Reject</button></div>` : ""}</div>`;
}

pages.models = async (el) => {
  const [models, evals, backups] = await Promise.all([api("/api/models"), api("/api/evals"), api("/api/backups")]);
  const sets = evals.sets.map((s) => s.name);
  el.innerHTML = `<h1>Evaluation &amp; versions</h1>
    <p class="lead">Every model version is kept as a separate file. Losses below are cross-entropy on immutable evaluation sets
    (lower is better). Activating an older version is an instant rollback.</p>
    <div class="panel" style="overflow-x:auto"><table><thead><tr><th>Version</th><th>Status</th><th>Parent</th><th>Dataset</th>${sets.map((s) => `<th class="num small">${esc(s)}</th>`).join("")}<th></th></tr></thead><tbody>
    ${models.map((m) => `<tr><td><b>${esc(m.id)}</b><div class="small muted">${when(m.created)}</div></td><td>${tag(m.status)}</td><td>${esc(m.parent || "random init")}</td><td>${esc(m.dataset_id || "")}</td>
      ${sets.map((s) => `<td class="num">${m.metrics[s] ? fmt(m.metrics[s].loss) : `<span class="muted">–</span>`}</td>`).join("")}
      <td class="row">${["approved", "pending_approval"].includes(m.status) && m.status !== "pending_approval" ? `<button data-activate="${m.id}">Roll back to this</button>` : ""}
      <button data-eval="${m.id}">Re-evaluate</button>${["rejected", "approved"].includes(m.status) ? `<button class="danger" data-del="${m.id}">Delete</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="9" class="muted">No models yet.</td></tr>`}
    </tbody></table><div id="evalOut"></div></div>
    <h2>Evaluation sets</h2>
    <div class="panel"><table><thead><tr><th>Name</th><th>Role</th><th>Format</th><th class="num">Items / chars</th><th>Integrity</th><th>Description</th></tr></thead><tbody>
    ${evals.sets.map((s) => `<tr><td>${esc(s.name)}</td><td>${s.kind === "regression" ? "regression suite" : "held-out"}</td><td>${esc(s.fmt)}</td><td class="num">${s.n_items}</td><td>${evals.integrity[s.name] === "ok" ? tag("ok") : `<span class="tag t-failed">${esc(evals.integrity[s.name])}</span>`}</td><td class="small">${esc(s.description)}</td></tr>`).join("")}
    </tbody></table><p class="small muted">Sets are hashed when created and refused if the file changes. Data overlapping them is kept out of training.</p></div>
    <h2>Weight backups</h2><div class="panel small">${backups.map((b) => `${esc(b.file)} — ${b.size_mb} MB — ${when(b.created)}`).join("<br>") || "Backups of the active model are written before each candidate training."}</div>`;
  el.onclick = async (e) => {
    const d = e.target.dataset;
    if (d.activate && confirm(`Make ${d.activate} the active model?`)) { await act(() => api(`/api/models/${d.activate}/activate`, { method: "POST" }), `Rolled back to ${d.activate}.`); refreshStatus(); show("models"); }
    if (d.del && confirm(`Delete ${d.del} permanently?`)) { await act(() => api(`/api/models/${d.del}`, { method: "DELETE" }), "Deleted."); show("models"); }
    if (d.eval) { e.target.disabled = true; const r = await act(() => api(`/api/models/${d.eval}/evaluate`, { method: "POST" })); e.target.disabled = false;
      $("#evalOut").innerHTML = `<h3 style="margin-top:12px">${esc(d.eval)} evaluated now</h3>` + Object.entries(r).map(([k, m]) => `${esc(k)}: loss ${fmt(m.loss, 4)}, perplexity ${fmt(m.ppl, 1)}, next-token accuracy ${fmt(m.token_acc * 100, 1)}%`).join("<br>"); }
  };
};

pages.memory = async (el) => {
  const [facts, notes] = await Promise.all([api("/api/memory/facts"), api("/api/memory/notes")]);
  el.innerHTML = `<h1>Memory</h1>
    <p class="lead">Facts and notes are stored in a local SQLite database and looked up when relevant. Remembering a fact does not teach the neural network anything — only approved training does that.</p>
    <form class="panel" id="factForm"><div class="row">
      <label style="flex:3">New fact <input id="ft" required placeholder="e.g. My favourite colour is blue"></label>
      <label>Type <select id="fk"><option value="unverified">Unverified claim</option><option value="verified">Verified</option><option value="preference">My preference</option></select></label>
      <label style="flex:1">Source <input id="fs" value="user"></label><button class="primary">Remember</button></div></form>
    <div class="panel row"><input id="q" placeholder="Search facts and notes" style="flex:1"><button id="qb">Search</button></div><div id="qres"></div>
    <h2>Facts</h2><div class="panel"><table><thead><tr><th>Fact</th><th>Type</th><th>Status</th><th class="num">Confidence</th><th>Source</th><th></th></tr></thead><tbody>
    ${facts.map((f) => `<tr><td>${esc(f.text)}${f.conflicts_with ? `<div class="small" style="color:var(--bad)">contradicts fact(s) ${esc(f.conflicts_with)}</div>` : ""}</td><td>${tag(f.kind)}</td><td>${tag(f.status)}</td><td class="num">${fmt(f.confidence, 2)}</td><td class="small">${esc(f.source)}<div class="muted">${when(f.created)}</div></td>
      <td class="row">${f.kind !== "verified" ? `<button data-verify="${f.id}">Mark verified</button>` : ""}${f.status === "conflict" ? `<button data-resolve="${f.id}">Keep (resolve)</button>` : ""}<button class="danger" data-delfact="${f.id}">Delete</button></td></tr>`).join("") || `<tr><td colspan="6" class="muted">No facts yet. Add one above or type /remember … in Chat.</td></tr>`}
    </tbody></table></div>
    <h2>Notes</h2>
    <form class="panel" id="noteForm"><label>Title <input id="nt"></label><label style="margin-top:6px">Note <textarea id="nb" rows="3" required></textarea></label><button class="primary" style="margin-top:8px">Save note</button></form>
    <div class="panel">${notes.map((n) => `<div style="margin-bottom:10px"><b>${esc(n.title || "Untitled")}</b> <span class="muted small">${when(n.created)} · ${esc(n.source)}</span> <button class="danger" data-delnote="${n.id}">Delete</button><div class="pre small">${esc(n.body)}</div></div>`).join("") || `<p class="muted">No notes.</p>`}</div>
    <h2>Delete everything</h2><div class="panel"><p class="small">Removes all facts, notes, conversations and feedback. Trained models and datasets are not affected.</p><button class="danger" id="wipe">Delete all memory</button></div>`;
  $("#factForm").onsubmit = async (e) => { e.preventDefault(); const r = await act(() => api("/api/memory/facts", { method: "POST", body: { text: $("#ft").value, kind: $("#fk").value, source: $("#fs").value } }));
    toast(r.duplicate ? "Already stored (duplicate)." : r.conflicts.length ? "Stored, but it contradicts an existing fact — marked as a conflict." : "Stored in memory."); show("memory"); };
  $("#noteForm").onsubmit = async (e) => { e.preventDefault(); await act(() => api("/api/memory/notes", { method: "POST", body: { title: $("#nt").value, body: $("#nb").value } }), "Note saved."); show("memory"); };
  const search = async () => { const r = await api(`/api/memory/search?q=${encodeURIComponent($("#q").value)}`);
    $("#qres").innerHTML = `<div class="panel small">${[...r.facts.map((f) => `Fact: ${esc(f.text)} ${tag(f.kind)}`), ...r.notes.map((n) => `Note: <b>${esc(n.title)}</b> ${esc(n.body.slice(0, 160))}`)].join("<br>") || "No matches."}</div>`; };
  $("#qb").onclick = search; $("#q").onkeydown = (e) => { if (e.key === "Enter") search(); };
  $("#wipe").onclick = async () => { if (prompt("Type DELETE to remove all memory") === "DELETE") { await act(() => api("/api/memory/wipe?confirm=DELETE", { method: "POST" }), "All memory deleted."); show("memory"); } };
  el.onclick = async (e) => {
    const d = e.target.dataset;
    if (d.verify) { await act(() => api(`/api/memory/facts/${d.verify}`, { method: "PATCH", body: { kind: "verified", confidence: 0.95 } })); show("memory"); }
    if (d.resolve) { await act(() => api(`/api/memory/facts/${d.resolve}`, { method: "PATCH", body: { status: "active" } })); show("memory"); }
    if (d.delfact) { await act(() => api(`/api/memory/facts/${d.delfact}`, { method: "DELETE" })); show("memory"); }
    if (d.delnote) { await act(() => api(`/api/memory/notes/${d.delnote}`, { method: "DELETE" })); show("memory"); }
  };
};

pages.datasets = async (el) => {
  const [ds, ex] = await Promise.all([api("/api/datasets"), api("/api/examples")]);
  const count = (s) => ex.filter((e) => e.status === s).length;
  el.innerHTML = `<h1>Datasets</h1>
    <p class="lead">Examples wait here until a learning cycle packs them into a new, immutable dataset version. Adding an
    example changes nothing in the network until a candidate trained on it is approved.</p>
    <div class="row" style="gap:28px;margin-bottom:14px">
      <div class="stat"><span class="v" style="color:var(--data)">${count("accepted")}</span><span class="k">accepted, not yet trained</span></div>
      <div class="stat"><span class="v">${count("pending")}</span><span class="k">waiting for your review</span></div>
      <div class="stat"><span class="v">${count("eval")}</span><span class="k">held out for evaluation</span></div>
      <div class="stat"><span class="v">${count("rejected")}</span><span class="k">rejected</span></div></div>
    <form class="panel" id="exForm"><h3>Add a training example</h3><div class="grid2">
      <label>Your message <textarea id="ep" rows="2"></textarea></label><label>The answer NEXORA should give <textarea id="ea" rows="2"></textarea></label></div>
      <button class="primary" style="margin-top:8px">Add example</button> <span class="small muted">Checked for quality, duplicates and overlap with evaluation data.</span></form>
    <form class="panel" id="webForm"><h3>Learn from a web page</h3>
      <div class="row"><input id="wurl" type="url" required placeholder="https://pl.wikipedia.org/wiki/Mikołaj_Kopernik" style="flex:1"><button class="primary">Download text</button></div>
      <p class="small muted">Only sites on the allowed list in Settings → Internet, and only pages whose robots.txt allows it. The text becomes training examples, not memory facts.</p>
      <div id="wpages"></div></form>
    <h2>Example pool</h2><div class="panel" style="max-height:420px;overflow-y:auto"><table><thead><tr><th>Text</th><th>Origin</th><th>Status</th><th>Dataset</th><th></th></tr></thead><tbody>
    ${ex.map((e) => `<tr><td><div class="pre small">${esc(e.text.slice(0, 400))}</div><div class="small muted">${esc(e.source)}${e.reason && e.reason !== "ok" ? " — " + esc(e.reason) : ""}</div></td><td>${tag(e.origin)}</td><td>${tag(e.status)}</td><td>${esc(e.dataset_id || "")}</td>
      <td class="row">${!e.dataset_id && e.status !== "accepted" ? `<button data-acc="${e.id}">Accept</button>` : ""}${!e.dataset_id && e.status !== "rejected" ? `<button data-rej="${e.id}">Reject</button>` : ""}${!e.dataset_id ? `<button class="danger" data-delex="${e.id}">Delete</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">No examples yet.</td></tr>`}
    </tbody></table></div>
    <h2>Dataset versions</h2><div class="panel"><table><thead><tr><th>Version</th><th>Parent</th><th class="num">Documents</th><th class="num">Tokens</th><th>Sources</th><th>SHA-256</th><th></th></tr></thead><tbody>
    ${ds.map((d) => `<tr><td><b>${esc(d.id)}</b><div class="small muted">${when(d.created)}</div></td><td>${esc(d.parent || "")}</td><td class="num">${d.n_examples}</td><td class="num">${d.n_tokens.toLocaleString()}</td><td class="small">${esc(d.sources.slice(0, 4).join(", "))}${d.sources.length > 4 ? "…" : ""}</td><td class="small muted">${esc(d.sha256.slice(0, 12))}</td><td><button data-view="${d.id}">View</button></td></tr>`).join("")}
    </tbody></table><div id="dsView"></div></div>`;
  api("/api/web/pages").then((pg) => { $("#wpages").innerHTML = pg.length ? `<table class="small"><tbody>${pg.map((w) => `<tr><td><a href="${esc(w.url)}" target="_blank" rel="noopener">${esc(w.title || w.url)}</a></td><td class="num">${w.accepted ?? 0}</td><td>${w.error ? `<span style="color:var(--bad)">${esc(w.error)}</span>` : ""}</td><td class="muted">${when(w.fetched)}</td></tr>`).join("")}</tbody></table>` : ""; }).catch(() => {});
  $("#webForm").onsubmit = async (e) => { e.preventDefault(); const b = e.submitter || $("#webForm button"); b.disabled = true;
    try { const r = await api("/api/web/learn", { method: "POST", body: { url: $("#wurl").value } });
      toast(`${r.accepted || 0} paragraphs from “${r.title}” added to the training data.`); show("datasets"); }
    catch (err) { toast(err.message, 8000); b.disabled = false; } };
  $("#exForm").onsubmit = async (e) => { e.preventDefault(); const r = await act(() => api("/api/examples", { method: "POST", body: { prompt: $("#ep").value, answer: $("#ea").value } }));
    toast(r.status === "accepted" ? "Example accepted into the pool." : `Not accepted: ${r.reason}`); show("datasets"); };
  el.onclick = async (e) => {
    const d = e.target.dataset;
    if (d.acc) { await act(() => api(`/api/examples/${d.acc}/review?accept=true`, { method: "POST" })); show("datasets"); }
    if (d.rej) { await act(() => api(`/api/examples/${d.rej}/review?accept=false`, { method: "POST" })); show("datasets"); }
    if (d.delex) { await act(() => api(`/api/examples/${d.delex}`, { method: "DELETE" })); show("datasets"); }
    if (d.view) { const docs = await api(`/api/datasets/${d.view}/documents?limit=8`); $("#dsView").innerHTML = `<h3 style="margin-top:12px">${esc(d.view)} (first ${docs.length} documents)</h3>` + docs.map((x) => `<div class="pre small">${esc(x.text.slice(0, 600))}</div><div class="small muted">${esc(x.source)} · ${esc(x.origin)}</div>`).join(""); }
  };
};

pages.images = async (el) => {
  const [ds, vers] = await Promise.all([api("/api/images/datasets"), api("/api/images/versions")]);
  const active = vers.find((v) => v.status === "active");
  el.innerHTML = `<h1>Images</h1>
    <p class="lead">A text-conditioned diffusion model trained from random weights on your own image–caption data.
    On a laptop CPU it can learn simple images (like the built-in coloured shapes) at 32×32 pixels; photo-like results need far more data and compute.</p>
    <div class="panel"><form id="genForm" class="row">
      <input id="gp" required placeholder="${active ? "e.g. a large red circle on a blue background at the left" : "Train an image model first (below)"}" style="flex:1" ${active ? "" : "disabled"}>
      <label>Images <input id="gn" type="number" value="4" min="1" max="8" style="width:70px"></label>
      <label>Steps <input id="gs" type="number" value="50" min="5" max="400" style="width:80px"></label>
      <label>Guidance <input id="gg" type="number" value="3" step="0.5" min="0" max="15" style="width:80px"></label>
      <button class="primary" ${active ? "" : "disabled"}>Generate</button></form><div id="genOut"></div></div>
    <div class="grid2">
      <div class="panel"><h3>1. Prepare a dataset</h3>
        <form id="dsForm" class="grid2" style="gap:8px">
          <label>Source <select id="src"><option value="synthetic_shapes">Built-in coloured shapes (no copyright)</option><option value="folder">My folder of images + captions</option></select></label>
          <label>Name <input id="dn" value="shapes" pattern="[A-Za-z0-9_-]+"></label>
          <label>Number of images <input id="dc" type="number" value="4000" min="50"></label>
          <label>Folder (for folder source) <input id="df" placeholder="C:\\Users\\me\\captioned_images"></label>
          <button class="primary">Prepare</button></form>
        <p class="small muted">Folder format: each image (photo.jpg) with a text file of the same name (photo.txt) containing its caption, or one captions.csv file with lines “filename,caption”. Use only images you have the right to use.</p>
        <table style="margin-top:8px"><tbody>${ds.map((d) => `<tr><td><b>${esc(d.name)}</b></td><td class="num">${d.images} images</td><td>${d.resolution}px</td><td class="small muted">${esc((d.examples || [])[0] || "")}</td></tr>`).join("") || `<tr><td class="muted">No image datasets yet.</td></tr>`}</tbody></table></div>
      <div class="panel"><h3>2. Train</h3>
        <form id="trForm" class="grid2" style="gap:8px">
          <label>Dataset <select id="td">${ds.map((d) => `<option>${esc(d.name)}</option>`).join("")}</select></label>
          <label>Steps <input id="ts" type="number" placeholder="config default"></label>
          <label>Continue from <select id="tr"><option value="">new model (random weights)</option>${vers.map((v) => `<option>${esc(v.id)}</option>`).join("")}</select></label>
          <label>Time limit (min) <input id="tm" type="number" placeholder="config default"></label>
          <button class="primary" ${ds.length ? "" : "disabled"}>Start image training</button></form>
        <p class="small muted">Progress appears on the Training page. "Continue from" lets you keep training a model later, for example on a faster computer.</p></div>
    </div>
    <h2>Image model versions</h2><div class="panel"><table><thead><tr><th>Version</th><th>Status</th><th class="num">Steps trained</th><th class="num">Validation loss</th><th>Dataset</th><th></th></tr></thead><tbody>
    ${vers.map((v) => `<tr><td>${esc(v.id)}</td><td>${tag(v.status)}</td><td class="num">${v.metrics.total_steps}</td><td class="num">${fmt(v.metrics.val_loss, 4)}</td><td>${esc(v.metrics.dataset)}</td><td>${v.status !== "active" ? `<button data-iact="${v.id}">Use this</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="6" class="muted">No image models yet.</td></tr>`}
    </tbody></table>${active && active.history ? chart([{ name: "training loss", points: active.history.train, color: "var(--ink)" }, { name: "validation loss", points: active.history.val, color: "var(--wts)" }]) : ""}</div>`;
  $("#genForm").onsubmit = async (e) => {
    e.preventDefault(); const out = $("#genOut"); out.innerHTML = `<p class="muted">Denoising… (a few seconds per image on a CPU)</p>`;
    try { const r = await api("/api/images/generate", { method: "POST", body: { prompt: $("#gp").value, n: +$("#gn").value, steps: +$("#gs").value, guidance: +$("#gg").value } });
      out.innerHTML = `${r.warning ? `<div class="warn">${esc(r.warning)}</div>` : ""}${r.unknown_words.length ? `<div class="warn">Words the model never saw in training (ignored): ${esc(r.unknown_words.join(", "))}. Known words include: ${esc(r.vocabulary_sample.join(", "))}</div>` : ""}
        <div class="imggrid" style="margin-top:10px">${r.files.map((f) => `<a href="/generated/${esc(f)}" download><img src="/generated/${esc(f)}" alt="${esc($("#gp").value)}"></a>`).join("")}</div><p class="small muted">${r.files.length} PNG files from ${esc(r.version)} in ${r.seconds}s (click to download).</p>`;
    } catch (err) { out.innerHTML = `<div class="warn err">${esc(err.message)}</div>`; }
  };
  $("#dsForm").onsubmit = async (e) => { e.preventDefault(); toast("Preparing dataset…"); await act(() => api("/api/images/datasets", { method: "POST", body: { source: $("#src").value, name: $("#dn").value, n: +$("#dc").value, folder: $("#df").value || null } }), "Dataset ready."); show("images"); };
  $("#trForm").onsubmit = async (e) => { e.preventDefault(); const b = { dataset: $("#td").value }; if ($("#ts").value) b.steps = +$("#ts").value; if ($("#tr").value) b.resume_from = $("#tr").value; if ($("#tm").value) b.max_minutes = +$("#tm").value;
    await act(() => api("/api/images/train", { method: "POST", body: b }), "Image training queued — follow it on the Training page."); refreshStatus(); };
  el.onclick = async (e) => { if (e.target.dataset.iact) { await act(() => api(`/api/images/versions/${e.target.dataset.iact}/activate`, { method: "POST" })); show("images"); } };
};

pages.hardware = async (el) => {
  const draw = async () => {
    const h = await api("/api/hardware");
    el.innerHTML = `<h1>Hardware</h1><p class="lead">Live usage, refreshed every 3 seconds. Training is limited by the values in Settings → resources.</p>
      <div class="grid2">
        <div class="panel"><h3>CPU ${fmt(h.cpu_percent, 0)}%</h3><div class="meter"><i style="width:${h.cpu_percent}%"></i></div>
          <p class="small muted">${h.cpu_count_physical} cores / ${h.cpu_count_logical} threads; PyTorch uses ${h.torch_threads} threads. ${esc(h.gpu)}.</p>
          ${h.cpu_per_core.map((c, i) => `<div class="small">Thread ${i + 1}<div class="meter"><i style="width:${c}%"></i></div></div>`).join("")}</div>
        <div class="panel"><h3>Memory ${fmt(h.ram_percent, 0)}%</h3><div class="meter"><i style="width:${h.ram_percent}%"></i></div>
          <p class="small">System: ${h.ram_used_mb.toLocaleString()} of ${h.ram_total_mb.toLocaleString()} MB used<br>NEXORA process: <b>${h.process_ram_mb} MB</b> (limit for training ${h.limits.max_process_ram_mb} MB)</p>
          <h3 style="margin-top:14px">Disk</h3><p class="small">${h.disk_free_gb} GB free of ${h.disk_total_gb} GB (training stops below ${h.limits.min_free_disk_mb} MB)</p>
          <table class="small"><tbody>${Object.entries(h.storage_mb).map(([k, v]) => `<tr><td>${esc(k.replace("_", " "))}</td><td class="num">${v} MB</td></tr>`).join("")}</tbody></table></div>
      </div><p class="small muted">${esc(h.platform)} · Python ${esc(h.python)} · PyTorch ${esc(h.torch_version)}</p>`;
  };
  await draw(); pageTimer = setInterval(draw, 3000);
};

pages.settings = async (el) => {
  const s = await api("/api/settings");
  const field = (sec, k, v) => {
    const id = `s_${sec}_${k}`;
    if (typeof v === "boolean") return `<label><span><input type="checkbox" id="${id}" ${v ? "checked" : ""}> ${esc(k.replace(/_/g, " "))}</span></label>`;
    if (Array.isArray(v)) { const lines = textList(sec, k);
      return `<label>${esc(k.replace(/_/g, " "))} ${lines ? "(one per line)" : "(comma separated)"}<textarea id="${id}" rows="2">${esc(lines ? v.join("\n") : v.join(", "))}</textarea></label>`; }
    const type = k.includes("secret") ? "password" : "text";
    return `<label>${esc(k.replace(/_/g, " "))}<input id="${id}" type="${type}" value="${esc(v)}" autocomplete="off"></label>`;
  };
  // settings that only matter for an external model server stay hidden unless one is configured
  const external = s.assistant.backend === "ollama";
  const hiddenKey = (sec, k) => !external && ((sec === "assistant" && ["backend", "ollama_url", "keep_alive", "timeout_seconds"].includes(k)) || (sec === "agent" && k === "vision_model") || (sec === "video" && k === "describe_frames"));
  const textList = (sec, k) => (sec === "learning" && k === "approved_folders") || (sec === "auth" && k === "allowed_emails") || (sec === "internet" && ["allowed_sites", "web_sources"].includes(k));
  const sections = { resources: "Resource limits", lm: "Language model & training", assistant: "Ready-made models (Assistant and Nexo 1.2)", agent: "Agent (Nexo 1.2)", video: "Learning from videos", learning: "Learning manager & guard", internet: "Internet", auth: "Sign in with Google", schedule: "Schedule", generation: "Chat sampling defaults", image: "Image model" };
  const notes = { learning: "learn from prompts: every chat message you write (long enough to be useful) is added to the training data. prompts need review: hold them in Datasets until you accept them. You can always add text explicitly with /learn … in Chat.",
    assistant: "Ready-made open models (not trained by NEXORA) built into NEXORA and running offline: Bielik for Polish conversation, Qwen2.5-Coder for code and files. Download and choose them on the AI models page. While answering they need 2–6 GB of RAM; avoid training NEXORA at the same time. Their answers are never used as training data unless you accept them.",
    agent: "default model: 1.2 or 1.1. auto run: run code in Roblox Studio / Blender without asking (risky code always asks). max attempts: how many times it may fix its own errors. look at result: after a successful step Nexo 1.2 reads back the scene (names, sizes, positions, colours) and fixes it if it does not match your request. render after run: show a preview after every successful step.",
    video: "whisper model: tiny, base, small (recommended for Polish) or medium (slow). language: pl, en… or empty for automatic.",
    internet: "Both switches are off by default. learn from web pages: the learning cycle downloads the pages in “web sources” (and you can add pages on the Datasets page); only allowed sites are used and robots.txt is respected. search in chat: every chat message is also sent to Wikipedia's search, and the results are shown under the answer — the model does not read them. Your messages leave the computer only when this is on.",
    auth: "Create an OAuth client in Google Cloud Console (APIs & Services → Credentials → OAuth client ID, type “Web application”), add the redirect URI shown below to “Authorized redirect URIs”, then paste the client ID and secret here, list the e-mail addresses allowed in, and tick enabled. Open NEXORA at the same address as the redirect URI (127.0.0.1 vs localhost matters). Locked out? Start NEXORA with the environment variable NEXORA_DISABLE_AUTH=1." };
  el.innerHTML = `<h1>Settings</h1><p class="lead">Saved to settings.json in the runtime folder. Model-size settings apply to the next model trained from scratch.</p>
    ${Object.entries(sections).map(([sec, title]) => `<form class="panel" data-sec="${sec}"><h3>${title}</h3>${notes[sec] ? `<p class="small muted">${esc(notes[sec])}</p>` : ""}<div class="grid2" style="gap:10px">${Object.entries(s[sec]).filter(([k]) => !hiddenKey(sec, k)).map(([k, v]) => field(sec, k, v)).join("")}</div><button class="primary" style="margin-top:10px">Save</button></form>`).join("")}
    <p class="small muted">The server address (${esc(s.server.host)}:${s.server.port}) is set in config/default.json; keep 127.0.0.1 so only this computer can reach NEXORA.</p>`;
  el.onsubmit = async (e) => {
    e.preventDefault(); const sec = e.target.dataset.sec; const out = {};
    for (const [k, v] of Object.entries(s[sec])) {
      if (hiddenKey(sec, k)) continue;
      const inp = $(`#s_${sec}_${k}`);
      if (typeof v === "boolean") out[k] = inp.checked;
      else if (Array.isArray(v)) out[k] = textList(sec, k) ? inp.value.split("\n").map((x) => x.trim()).filter(Boolean) : inp.value.split(",").map((x) => Number(x.trim())).filter((x) => !Number.isNaN(x));
      else if (typeof v === "number") out[k] = Number(inp.value);
      else out[k] = inp.value;
    }
    await act(() => api("/api/settings", { method: "PUT", body: { [sec]: out } }), "Settings saved.");
  };
};

pages.audit = async (el) => {
  const rows = await api("/api/audit?limit=300");
  el.innerHTML = `<h1>Audit log</h1><p class="lead">Every training run, evaluation, approval, rollback and deletion is recorded here.</p>
    <div class="panel"><table><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Details</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="small">${when(r.ts)}</td><td>${esc(r.actor)}</td><td>${esc(r.action.replace(/_/g, " "))}</td><td class="small muted">${esc(r.detail.length > 300 ? r.detail.slice(0, 300) + "…" : r.detail)}</td></tr>`).join("")}</tbody></table></div>`;
};

async function show(name) {
  clearInterval(pageTimer);
  if (!pages[name]) name = "chat";
  currentPage = name;
  document.querySelectorAll(".side a").forEach((a) => a.classList.toggle("on", a.dataset.page === name));
  const el = $("#page"); el.onclick = null; el.onsubmit = null;
  if (!STATUS) await refreshStatus();
  try { await pages[name](el); } catch (e) { el.innerHTML = `<div class="warn err">Could not load this page: ${esc(e.message)}</div>`; }
}
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
(function langSelector() {
  const sel = $("#langSel");
  sel.innerHTML = Object.entries(LANGS).map(([k, v]) => `<option value="${k}" ${k === LANG ? "selected" : ""}>${v}</option>`).join("");
  sel.onchange = async () => {
    setLanguage(sel.value);
    try { await api("/api/settings", { method: "PUT", body: { ui: { language: sel.value } } }); } catch { /* still switch locally */ }
    location.reload();
  };
  let stored = null;
  try { stored = localStorage.getItem("nexora_lang"); } catch { /* ignore */ }
  if (!stored) api("/api/settings").then((s) => { if (s.ui && s.ui.language !== LANG) { setLanguage(s.ui.language); location.reload(); } }).catch(() => {});
})();
api("/auth/me").then((me) => { if (me.enabled && me.email) { const a = document.createElement("a"); a.href = "/auth/logout"; a.textContent = `Sign out (${me.email})`; a.className = "small"; $(".legend").before(a); } }).catch(() => {});
refreshStatus().then(() => show(location.hash.slice(1) || "chat"));
setInterval(refreshStatus, 5000);
