# NEXORA AI

> **Szybki start (PL):** zainstaluj 64-bitowy Python 3.12, rozpakuj NEXORA, w PowerShell w tym folderze uruchom `powershell -ExecutionPolicy Bypass -File .\setup.ps1`, a potem `.\start.ps1` (otworzy się przeglądarka na http://127.0.0.1:8765). Interfejs jest domyślnie po polsku (do wyboru też English, Deutsch, Español, Українська – lista w lewym górnym rogu). W czacie: `/remember …` zapisuje fakt w pamięci, `/learn …` (lub `/naucz …`) dodaje tekst do danych treningowych. Zachowanie i sposób mówienia ustawisz na stronie **Osobowość**. Dostęp do internetu (nauka ze stron i wyszukiwanie w Wikipedii) włączysz w Ustawienia → Internet – domyślnie jest wyłączony.
>
> **Nexo 1.1 i Nexo 1.2:** w czacie wybierasz model. **Nexo 1.1** to dotychczasowa wersja (własna sieć albo Asystent). **Nexo 1.2** to agent: rozmawia po polsku, pisze kod, buduje w **Roblox Studio** (`install_roblox_plugin.ps1`) i modeluje w **Blenderze** (dodatek ze strony *Połączenia*), uczy się z lekcji i filmów (strona *Wiedza*), renderuje to, co zbudował, i pokazuje na żywo swoje „myśli”. Przed uruchomieniem kodu pyta o zgodę.
>
> **Normalna rozmowa po polsku i pomoc z kodem:** uruchom `powershell -ExecutionPolicy Bypass -File .\setup_models.ps1` (albo strona *Modele AI* w NEXORA). NEXORA sama pobierze i uruchomi gotowe, darmowe modele działające offline (bez żadnego dodatkowego programu): **Bielik** (polski, do rozmowy) i **Qwen2.5-Coder** (kod w wielu językach, czytanie plików). W czacie wybierasz „Kto odpowiada: Asystent”. Przycisk 📎 dołącza pliki (kod, txt, PDF, DOCX, ODT). To nie jest własna sieć NEXORY – ta działa dalej obok i dalej się uczy.
>
> **Model mówiący po polsku:** dołączony model zna tylko angielski (uczył się na Szekspirze). Aby wytrenować polski model, zatrzymaj NEXORA i uruchom `powershell -ExecutionPolicy Bypass -File .\new_language.ps1 -Lang pl` (lub `-Lang pl -Also en` dla polskiego i angielskiego). Skrypt pobierze z Wolnych Lektur polską klasykę z domeny publicznej (Fredro, Mickiewicz, Prus, Sienkiewicz…), zrobi kopię obecnego modelu i wytrenuje nowy od zera (ok. 20–60 min). Pamięć, czaty i model obrazów zostają.

A small, self-contained AI system you train yourself, on your own Windows PC, from random weights:

- **NEXORA language model** – a decoder-only Transformer (causal self-attention) with its own byte-level BPE tokenizer, trained by next-token prediction. No pretrained weights, no cloud APIs.
- **Learning manager + Learning Guard** – turns your feedback and approved local files into new dataset versions, trains candidate models in the background, compares them with the active model on immutable held-out and regression sets, and asks you before anything replaces the active model.
- **Memory** – SQLite facts, notes, conversations and feedback, with sources, verification status, duplicate and contradiction detection.
- **Image generator** – a text-conditioned diffusion model (UNet + small Transformer text encoder), also trained from random weights on image–caption data you are allowed to use.
- **Learning from your prompts and requests** – `/learn <text>` in chat adds that text to the training data; with *learn from prompts* on (default), every message you type is collected as training data too (optionally held for review).
- **Optional Google sign-in** – OAuth 2.0 with PKCE and an allow-list of e-mail addresses. Off by default.
- **Personality** – name, greeting, speaking style (balanced / precise / creative / brief / talkative), sentence limit, a behaviour description placed before every conversation, blocked words, a live preview, and "teach this personality" example answers that become training data.
- **Optional Assistant** – a ready-made open model (Bielik for Polish, Qwen2.5-Coder for code) built into NEXORA (no other program), running offline, with file reading; clearly separated from NEXORA's own network.
- **Optional internet access** – learn from allowed web pages and show Wikipedia results in chat (both off by default).
- **Five interface languages** – Polski (default), English, Deutsch, Español, Українська.
- **Models in other languages** – `new_language.ps1 -Lang pl|de|es|uk|en` downloads public-domain books and trains a new model from scratch for that language, keeping memory, chats and images.
- **Local web app** – FastAPI on `127.0.0.1:8765` with chat, images, training, goals, approvals, evaluation/rollback, memory, datasets, hardware and settings pages.

Read **Honest expectations** before you start.

---

## Quick start (Windows 11)

1. Install **64-bit Python 3.12** from python.org (tick *Add python.exe to PATH*). 3.10–3.13 work.
2. Unzip NEXORA somewhere with at least **3 GB free** (e.g. `C:\NEXORA`).
3. Open PowerShell in that folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1      # once: venv + CPU PyTorch + packages + model engine (~1.2 GB)
powershell -ExecutionPolicy Bypass -File .\setup_models.ps1  # once: Bielik + Qwen Coder for Nexo 1.2 (~3.8 GB)
powershell -ExecutionPolicy Bypass -File .\start.ps1      # starts the server and opens the browser
powershell -ExecutionPolicy Bypass -File .\stop.ps1       # stops it; running training is checkpointed
```

The package already contains a language model trained from scratch in this build (version `v0001`, see *Measured results*) so chat works immediately. You can retrain everything yourself: delete the `runtime` folder, run `setup.ps1` again (it re-prepares the data), and press **Train from random weights** on the Training page.

Tests: `powershell -ExecutionPolicy Bypass -File .\run_tests.ps1`

## Honest expectations

- The language model has about **2 million parameters** and learns from ~1 MB of public-domain Shakespeare. It produces play-like text with real words, names and dialogue structure, but **it does not understand questions or know facts**. A model that holds a sensible conversation needs hundreds of millions of parameters and gigabytes of text – far beyond an 8 GB laptop CPU.
- The *learning loop* is real: corrections you write become training data, a candidate is fine-tuned, and the guard measures it. With a tiny model, improvements are measurable on the held-out slices but small in conversation quality.
- The image model at 32×32 px on the built-in shapes dataset learns colours, shapes and rough positions in roughly an hour of CPU training. Photographs need much more data, resolution and compute; the same code continues on a faster PC (`--resume-from`).
- Forgetting is **measured and contained, not prevented**: regressions beyond the threshold get the candidate rejected and the active model is never touched until you approve.

## How it fits together

```
 chat ──► feedback / corrections ──┐          approved folders (.txt/.md)
                                   ▼                     │
                      learning cycle (manual or scheduled)◄┘
   analyse ─► goals ─► gather ─► validate + dedupe + leakage check
                                   │
             new immutable dataset version  +  held-out slice (new eval set)
                                   │
   background job: backup active weights ─► fine-tune (replay buffer + L2 anchor)
                                   │
   Learning Guard: active vs candidate on ALL eval sets + sanity checks
          │ fails                              │ passes
   auto-rejected (active unchanged)     waits on the Approvals page
                                               │ you approve        │ you reject
                                   becomes active; old one kept   nothing changes
                                   (one-click rollback)
```

**Memory vs dataset vs weights** – the UI colour-codes them and the code keeps them separate:

| Action | What changes | Network weights |
|---|---|---|
| `/remember …` in chat, Memory page | a row in SQLite, shown as "found in memory" | unchanged |
| "Correct it", Datasets page | an example in the pool, later in a dataset version | unchanged |
| Approving a candidate | the active model file | **changed** |

`/learn …` and collected prompts are *your* text, so they go straight into the example pool (or into review if *prompts need review* is on). They are still only learned when a candidate trained on them passes the guard and you approve it. Model outputs are never trained on automatically: "Propose as example" puts them in a review queue as `generated`, and they need your explicit acceptance.

## Using it

- **Chat** – sampling controls (temperature, top-k, top-p, repetition penalty, max tokens, context turns). Rate answers, correct them, or propose them as examples.
- **Training** – prepare data, train from random weights, watch loss curves live, pause / resume / stop-and-save. Interrupted jobs (app closed, time/memory/disk limit) resume from their checkpoint.
- **Learning goals** – run a cycle now or enable the schedule in Settings (interval, allowed hours like `22-7`, only when CPU is idle). Add your own goals with keywords; paragraphs from approved folders matching them become candidate data.
- **Approvals** – per-set loss table (active vs candidate, % change, verdict), reasons, and side-by-side sample outputs with the same seed.
- **Evaluation & versions** – every version with its scores, re-evaluate, roll back, delete rejected versions; eval-set integrity; weight backups.
- **Memory / Datasets / Hardware / Settings / Audit log** – as named.

Command line equivalents (inside `.venv`): `python scripts\prepare_lm_data.py [files]`, `python scripts\train_lm.py [--steps N]`, `python scripts\prepare_image_data.py shapes`, `python scripts\train_image.py shapes [--resume-from img0001]`.

## Languages

**Interface.** Pick the language in the box under the NEXORA logo; it is remembered per browser and saved in Settings (`ui.language`).

**What the model speaks.** A neural network only speaks the languages in its training text. The bundled `v0001` learned English (Shakespeare). To get another language:

```powershell
powershell -ExecutionPolicy Bypass -File .\stop.ps1
powershell -ExecutionPolicy Bypass -File .\new_language.ps1 -Lang pl            # Polish
powershell -ExecutionPolicy Bypass -File .\new_language.ps1 -Lang pl -Also en   # Polish + English (bigger tokenizer)
```

| Language | Source (downloaded on your PC) |
|---|---|
| pl | Wolne Lektury API – works by Fredro, Mickiewicz, Prus, Sienkiewicz, Orzeszkowa, Żeromski, Słowacki, Reymont, Konopnicka, Krasicki (public domain) |
| de, es, uk | Project Gutenberg books marked "not copyrighted", found through the Gutendex catalogue |
| en | bundled Tiny Shakespeare |

Up to 6 MB of text per language (`--max-mb`). You can also put your own UTF-8 `.txt` files in `data/raw/<lang>/` before running it. The old runtime folder is renamed to `runtime_backup_<date>` (nothing is deleted); memory, conversations, feedback, examples and the image model are carried over. These downloads could not be tested in the build environment (those sites were not reachable from it); the cleaning and rebuild steps are covered by tests.

## Nexo 1.1 and Nexo 1.2

The chat has a **Model** selector:

- **Nexo 1.1** – everything that existed before: NEXORA's own network trained from scratch, or the plain Assistant.
- **Nexo 1.2** – an agent built on the ready-made models (download them with `setup_models.ps1` or on the *AI models* page). It:
  - **shows its thoughts live**: searching its knowledge, looking at the scene, the plan and code as they are written, waiting for approval, running, reading errors, fixing them, rendering, looking at the render;
  - **builds in Roblox Studio** through a plugin (`install_roblox_plugin.ps1`, or download it on the *Connections* page). Code runs in edit mode as one undo step; failed code is cancelled automatically; NEXORA draws an isometric preview of the parts because Studio plugins cannot take screenshots;
  - **models in Blender** through an add-on (*Connections* page → download → Preferences → Add-ons → Install from Disk). Failed attempts remove what they created; after each successful step Blender renders a preview from an automatic 3/4 camera;
  - **checks its own work**: after a successful step it reads the scene back (object names, sizes, positions, colours, materials) and compares it with your request; if something is missing or wrong it writes a fix (Settings → Agent → look at result);
  - **learns from lessons, files, videos and good examples** (*Knowledge* page). Built-in lessons: *Lua/Luau and Roblox Studio* and *3D modelling in Blender (bpy)*. `/learn …` (or `/naucz …`) in Nexo 1.2 adds a lesson. *Remember as a good example* stores a working solution. These are read into the prompt when relevant – the models' weights do not change;
  - **learns from videos**: video/audio files are transcribed on the CPU with faster-whisper (optional, ~600 MB incl. the `small` speech model downloaded on first use), key frames are saved, and the result becomes a lesson (optionally also Nexo 1.1 training data). `.srt`/`.vtt` subtitles work without speech recognition. Downloading from YouTube is deliberately not included.

**Safety.** By default Nexo 1.2 asks before running any code (*Settings → Agent → auto run*). Code that uses files, the network, `InsertService`, `HttpService` or `require(<id>)` always asks, even with auto run on. The plugin/add-on only talks to `127.0.0.1` and carries a private pairing key (`runtime/bridge_token.txt`); requests without it are refused. Keep backups of important places and .blend files – generated code can be wrong.

**What was tested.** The Blender add-on was tested against real Blender 5.2 (`bpy` module): running code, a real Python error coming back and being fixed, cleanup of the failed attempt, Workbench render with material colours. The lesson's Blender code was executed in Blender 5.2, including FBX/OBJ/GLB export. The Roblox plugin was checked for Lua syntax and its protocol was tested with a simulated plugin; it was **not** run inside real Roblox Studio (not available in the build environment). If Studio reports that `loadstring` is unavailable, the plugin falls back to a temporary ModuleScript.

## Ready-made models (built in): normal conversation and code

NEXORA's own network is tiny by design (trained from scratch on a laptop), so it cannot hold a real conversation or write working code. For that NEXORA can run **ready-made open models itself** – no other program is needed. It downloads a model file (GGUF) once from Hugging Face and runs it inside NEXORA with the built-in llama.cpp engine (`llama-cpp-python`, installed by `setup.ps1`). This is a deliberate, opt-in exception to the "no pretrained models" rule; NEXORA's own model, training and Learning Guard are unchanged and keep working side by side.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_models.ps1      # or: the "AI models" page in NEXORA
```

| Purpose | Model (id) | Download | Licence |
|---|---|---|---|
| Polish conversation (recommended for 8 GB RAM) | Bielik 1.5B v3 (`bielik-1.5b`) | 1.7 GB | Apache-2.0 |
| Polish conversation, better but slow (~6 GB free RAM) | Bielik 4.5B v3 (`bielik-4.5b`) | 5.1 GB | Apache-2.0 |
| Code, Roblox Studio, Blender, files (recommended) | Qwen2.5-Coder 3B (`qwen-coder-3b`) | 2.1 GB | Qwen Research – non-commercial use only |
| Code, lighter | Qwen2.5-Coder 1.5B (`qwen-coder-1.5b`) | 1.1 GB | Apache-2.0 |

Downloads can be paused and resumed (also after closing NEXORA), the file is checked to be a GGUF model, and only one model is kept in memory at a time. Command line: `python scripts\download_model.py bielik-1.5b qwen-coder-3b`.

In Nexo 1.1 choose **Who answers → Assistant**; Nexo 1.2 uses these models automatically. Questions about code and attached code files go to the code model. Answers stream in, code appears in blocks with a Copy button, and every answer is labelled with the model that wrote it.

- **Files:** the 📎 button attaches code in ~40 languages, `.txt/.md/.csv`, PDF (text layer), Word `.docx` and `.odt` (up to 10 MB; the first 12,000 characters go to the model). *Learn from this file* puts the file into NEXORA's training data (source code is kept with its indentation and checked for minified code and secrets).
- **Personality** settings (name, style, description, sentence limit, blocked words) become the model's instructions.
- Settings → *Ready-made models*: which models, temperature, context length, answer length, history length, file limits, extra instructions.
- Their answers are never used as training data unless you accept them; your corrections still train NEXORA's own model.
- On an i3-1215U expect roughly a few words per second from a 1.5–3B model; loading a model takes 5–20 s. Do not train NEXORA while using them on 8 GB RAM.

The engine was tested with real llama.cpp on a small test model (generation, streaming, model switching, resumable download); the real Bielik/Qwen files could not be downloaded in the build environment, so their answer quality was not measured.

**Code in NEXORA's own model.** Approved folders now also accept source files (`.py`, `.js`, `.ts`, `.java`, `.c/.cpp`, `.cs`, `.go`, `.rs`, `.php`, … – `node_modules`, `.git` and virtual environments are skipped). `new_language.ps1 -Lang code` builds an own model from Python's standard library already on your PC (or your files in `data/raw/code/`). It will produce code-shaped text, not working programs.

## Internet (optional, off by default)

Settings → **Internet** has two independent switches:

- **learn from web** – the learning cycle downloads the pages listed in *web sources*, and the Datasets page has a *Learn from a web page* box. Only sites in *allowed sites* (default: `wikipedia.org`, `wolnelektury.pl`) are used, `robots.txt` is respected, pages are limited to 3 MB, and each page is re-downloaded at most every `refetch_days`. Wikipedia articles are read through the official API as clean text. Paragraphs go through the usual quality, duplicate and eval-leakage checks into the example pool (origin `web`) – they change the network only after an approved candidate.
- **search in chat** – each chat message is also searched on Wikipedia (official API, no key, language set by `wikipedia_language`). Up to three results are shown under the answer, labelled *not generated by the model*, each with a *Learn from this article* button. The model does not read the results – it is far too small to summarise them.

When either switch is on, the text you send (search queries, page addresses) leaves your computer. These features were tested with simulated responses; the build environment had no access to Wikipedia.

## Personality

The **Personality** page sets how NEXORA behaves and speaks. Everything acts on the real model:

- *style* changes sampling (temperature, top-k, top-p, length); values typed in the chat's sampling panel override it;
- *behaviour description* (max 300 characters) is placed before every conversation; the model continues from it, so write it in the model's language, as a scene or the start of a dialogue;
- *maximum sentences* and *blocked words* trim the generated text (blocked words become `***`);
- *name* and *greeting* change how the assistant is shown (the greeting is your text, not generated);
- *teach this personality*: example answers in the form `message => answer` go into the training data, so a future approved model really answers in that style.

A model of this size follows a description only loosely; examples + training have the stronger effect.

## Sign in with Google (optional)

1. Go to <https://console.cloud.google.com/> → create a project → **APIs & Services → OAuth consent screen** (External, add your own address as a test user).
2. **Credentials → Create credentials → OAuth client ID**, type **Web application**. Under *Authorized redirect URIs* add exactly `http://127.0.0.1:8765/auth/callback`.
3. In NEXORA → Settings → *Sign in with Google*: paste the client ID and client secret, put your Gmail address in *allowed emails*, tick *enabled*, save.
4. Reload: you are sent to the sign-in page. Open NEXORA at `http://127.0.0.1:8765` (not `localhost`), because the address must match the redirect URI.

Only the login itself contacts Google; the models still run offline. Sessions are signed cookies (HttpOnly, 7 days by default) and are dropped as soon as an address is removed from the list. The client secret is stored in `runtime/settings.json` and never sent back to the browser. **Locked out?** Start once with the environment variable `NEXORA_DISABLE_AUTH=1` (`$env:NEXORA_DISABLE_AUTH="1"; .\start.ps1 -Foreground`) and fix the settings.

## Data and licences

- **Language:** `data/raw/tinyshakespeare.txt` (~1.1 MB, from Andrej Karpathy's char-rnn repository) – Shakespeare's plays are public domain. Use your own UTF-8 `.txt` files with `prepare_lm_data.py` (before training; the tokenizer is frozen afterwards).
- **Images:** the built-in *synthetic shapes* set is drawn locally (no copyright). For your own data put images plus same-name `.txt` captions (or `captions.csv`) in a folder. Good legal sources: your own photos, CC0/public-domain images (Wikimedia Commons, Openverse filtered to CC0), or datasets whose licence allows training. You are responsible for the rights.

## Resource controls (config/default.json → Settings page)

| Setting | Default | Purpose |
|---|---|---|
| `resources.torch_threads` | 4 | CPU threads used by PyTorch (i3-1215U has 8) |
| `resources.max_process_ram_mb` | 3500 | training stops gracefully above this |
| `resources.min_free_disk_mb` | 1500 | no checkpoint is written below this |
| `resources.lower_priority` | true | training runs at below-normal priority |
| `lm.max_minutes`, `image.max_minutes` | 60 / 45 | time limit per job run (resumable) |
| `lm.n_layer / n_head / n_embd / block_size / vocab_size` | 4 / 4 / 192 / 128 / 1024 | model size (~2 M params, ~25 MB RAM for weights) |
| `learning.max_regression_pct` | 3.0 | max allowed loss increase on any non-target set |
| `learning.min_improvement_pct` | 0.5 | min loss decrease on the goal's held-out set |
| `learning.replay_fraction`, `anchor_l2` | 0.5, 0.01 | anti-forgetting: old-data replay and L2-SP pull towards approved weights |

Only one training job runs at a time; there is no unbounded loop.

## Project layout

```
nexora/
  lm/        tokenizer.py model.py train.py evaluate.py data.py service.py
  guard/     registry.py (versioned models, datasets, immutable eval sets)  guard.py (comparison)
  learning/  base.py (corpus prep, first training) manager.py (cycle, candidates, approval) scheduler.py
  memory/    db.py (SQLite schema) store.py (facts, notes, chats, feedback)
  jobs/      control.py (pause/cancel/limits) manager.py (background jobs, restart recovery)
  image/     model.py (text encoder, UNet, diffusion) data.py service.py
  agent.py (Nexo 1.2)  bridge.py (Roblox/Blender link)  knowledge.py  video.py  render3d.py  assistant.py  files.py
  api/app.py FastAPI     web/ index.html app.js style.css (no external libraries)
integrations/ roblox/NexoraBridge.lua (Studio plugin)  blender/nexora_bridge.py (Blender add-on)
data/knowledge/ lua_roblox.md  blender_modeling.md (built-in lessons)
scripts/     prepare_lm_data.py train_lm.py prepare_image_data.py train_image.py
tests/       pytest suite
config/default.json    runtime/ (created: db, models, datasets, eval sets, logs, settings.json)
```

## Known limitations

- Single user, no login; the server only listens on 127.0.0.1. Don't expose it to a network.
- The tokenizer is fixed after the first training; switching corpora means starting a new `runtime` folder.
- The contradiction detector is rule-based (`X is Y`, `I live in …`); it catches simple cases only.
- The chat context window is 128 tokens (~300 characters), so "short-term memory" covers only the last exchange or two.
- "Permitted sources" are local approved folders only; there is no web crawling.
- Learning goals of kind *topic* only find data if you have approved folders with matching text.
- The PowerShell scripts were written for Windows 11 but this build was developed and tested on Linux; see *Measured results*.

## Measured results (this build)

Measured in the build environment (Linux, 2 CPU threads, 8 GB RAM, no GPU – roughly comparable to, or slower than, an i3-1215U with 4 threads). Nothing below is estimated.

**Language model `v0001`** – 2,001,024 parameters, trained from random initialisation for 3,000 steps (batch 16 × 128 tokens), ≈0.35 s/step, ≈19 minutes.

| | at random init | after training |
|---|---|---|
| loss (nats/token) | 6.90 at step 10 (chance level is ln 1024 = 6.93) | 3.24 (validation) |
| held-out text loss / perplexity | – | 3.55 / 34.8 |
| held-out next-token accuracy | chance level ≈ 0.1 % (1/1024) | 27.9 % |
| dialogue-continuation regression set loss | – | 3.73 |

Sample (seeded): *YOU: Tell me of love. → NEXORA: I thought, you have been intent; / And by my most we have no passagey; / Therefore put himself that I'll tell thee / I must be sound to sense.* — Shakespeare-like form, made-up words, no real meaning, as expected at this size.

**Learning cycle test** (on a copy, not shipped): 10 corrected answers → 8 trained / 2 held out → 400-step candidate. Guard result: goal held-out loss 3.74 → 2.98 (+20.3 %), base held-out text −2.2 %, dialogue regression set −1.6 % (both inside the 3 % limit) → *passed, pending approval*. Sample outputs showed the candidate had over-fitted: it answered unrelated prompts with memorised correction sentences. That is exactly what the approval screen is for; with so few examples, reject such candidates or add more varied examples. Approve → active; rollback to `v0001` → active again: both verified.

**Image model `img0001`** – 4,000 synthetic 32×32 shapes, 6,000 steps (batch 32), ≈0.35 s/step, ≈40 minutes; validation noise-prediction MSE 0.0102. Generates recognisable circles/squares/triangles in the requested colour pair in ~2 s per batch of 4 (50 DDIM steps), but often swaps foreground and background colours and is unreliable on position. It knows only the 30-odd words of its captions.

**Tests:** 112 automated tests pass (`pytest tests`): tokenizer, model init/causality/forward/backward, loss, training changes weights, resume-from-checkpoint equals uninterrupted training, save/load, generation, evaluation reproducibility, dataset validation and leakage, regression detection, candidate rejection (including a deliberately damaged candidate), rollback, memory persistence, jobs pause/resume/cancel/limits/restart recovery, instance lock, diffusion pipeline, API endpoints, `/learn` and prompt learning, Google login (mocked Google token endpoint: success, wrong account, wrong audience, unverified e-mail, expired token, wrong issuer, forged state/cookie), personality styles/limits/preview/validation, corpus cleaning, rebuilding the model for Polish while keeping memory, Nexo 1.2 (knowledge search with Polish word forms, agent loop with error fixing, approvals, risky-code detection, vision check, bridge token, Roblox plugin Lua syntax, isometric render, Blender add-on with real Blender 5.2, subtitles, video frames and the video-to-lesson job), the built-in model engine with real llama.cpp and a generated test GGUF (generation, streaming, interrupted and resumed download, file check, model API), Assistant/agent logic with scripted model replies (streaming, model routing, persona, errors), file reading (code, cp1250 text, DOCX, PDF), code learning, and the internet features (HTML extraction, allowed sites, robots.txt, Wikipedia search/articles, learning cycle, migration of older databases) with simulated network responses.

**Not verified here:** the PowerShell scripts were not executed on Windows (no Windows machine available in the build environment), and Google sign-in was not tested against Google's real servers (it needs your OAuth client). If something fails, the error text in the console or in `runtime/logs/` tells you which step.

