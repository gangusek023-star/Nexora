"""Learning from videos (Nexo 1.2).

From a video file NEXORA builds a lesson for its knowledge base:
  * speech -> text with timestamps (faster-whisper, runs on the CPU; the
    speech model is downloaded once on first use, ~480 MB for "small")
  * key frames every N seconds -> PNG files shown in the app, and optionally
    a short description of each frame by a vision model (only with an external model server)
Subtitle files (.srt / .vtt) are read directly, no speech model needed.

Downloading from YouTube is not built in (it breaks YouTube's terms). Use your
own recordings, files you are allowed to download, or copy a video's
transcript (YouTube: "...more" > "Show transcript") and paste it as a lesson.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
from pathlib import Path

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".ogg"}
SUB_EXT = {".srt", ".vtt"}


class VideoError(Exception):
    pass


def ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def parse_subtitles(text: str) -> list[tuple[float, str]]:
    """SRT / WebVTT -> [(start_seconds, text)]."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->", block)
        m2 = m or re.search(r"(?:^|\n)(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->", block)
        if not m2:
            continue
        if m:
            start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        else:
            start = int(m2.group(1)) * 60 + int(m2.group(2))
        lines = [ln.strip() for ln in block.split("\n") if ln.strip() and "-->" not in ln and not ln.strip().isdigit()
                 and not ln.startswith("WEBVTT") and not ln.startswith("NOTE")]
        txt = re.sub(r"<[^>]+>", "", " ".join(lines)).strip()
        if txt and (not out or out[-1][1] != txt):
            out.append((float(start), txt))
    return out


def transcribe(path: str, model_size: str = "small", language: str | None = "pl", progress=None):
    """Speech to text with faster-whisper. Yields (start_seconds, text)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise VideoError("Speech recognition is not installed. Run setup_models.ps1 and choose "
                         "'learning from videos' (installs faster-whisper).") from e
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:  # noqa: BLE001
        raise VideoError(f"Could not load the speech model '{model_size}' (the first time it is downloaded "
                         f"from the internet, ~500 MB): {e}") from e
    segments, info = model.transcribe(path, language=language or None, vad_filter=True, beam_size=1)
    for seg in segments:
        if progress:
            progress(seg.end, getattr(info, "duration", 0) or 0)
        yield seg.start, seg.text.strip()


def extract_frames(path: str, out_dir: Path, every: float = 30.0, max_frames: int = 12) -> list[tuple[float, Path]]:
    """Grab one frame every `every` seconds (at most max_frames) as small PNGs."""
    try:
        import av
    except ImportError:
        return []
    out = []
    try:
        with av.open(path) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            duration = float(container.duration / 1_000_000) if container.duration else 0.0
            if duration <= 0 and stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            if duration <= 0:
                return []
            step = max(every, duration / max_frames)
            t = min(step / 2, duration / 2)
            while t < duration and len(out) < max_frames:
                container.seek(int(t * 1_000_000), any_frame=False, backward=True)
                frame = None
                for f in container.decode(video=0):
                    if f.time is None or f.time >= t - 0.5:
                        frame = f
                        break
                if frame is not None:
                    img = frame.to_image()
                    img.thumbnail((480, 270))
                    p = out_dir / f"frame_{int(t):05d}.png"
                    img.save(p)
                    out.append((t, p))
                t += step
    except Exception as e:  # noqa: BLE001
        raise VideoError(f"could not read the video: {e}") from e
    return out


def describe_image(cfg_assistant: dict, model: str, png_path: Path, prompt: str, timeout: float = 300) -> str:
    """Ask a vision model on an external local model server what is in a picture."""
    from .assistant import check_local_url
    base = check_local_url(cfg_assistant.get("ollama_url", ""))
    payload = {"model": model, "stream": False, "keep_alive": cfg_assistant.get("keep_alive", "5m"),
               "options": {"temperature": 0.2},
               "messages": [{"role": "user", "content": prompt,
                             "images": [base64.b64encode(Path(png_path).read_bytes()).decode()]}]}
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.loads(r.read()).get("message") or {}).get("content", "").strip()


def video_job_handler(app):
    """Background job: video/subtitles -> lesson in the knowledge base."""

    def run(job, control):
        from .jobs.control import StopTraining
        p = job["params"]
        vc, ac = app.cfg()["video"], app.cfg()["assistant"]
        src = Path(p["path"])
        name = p.get("name") or src.name
        frames_dir = app.paths.generated / f"video_{job['id']}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        lines, frames = [], []
        try:
            if src.suffix.lower() in SUB_EXT:
                control.report(state="running", message="reading subtitles")
                lines = parse_subtitles(src.read_text(encoding="utf-8", errors="replace"))
            else:
                control.report(state="running", message="extracting key frames")
                frames = extract_frames(str(src), frames_dir, float(vc["frame_every_seconds"]), int(vc["max_frames"]))
                control.report(message="transcribing speech (this takes about as long as the video)")

                def prog(done, total):
                    control.report(step=int(done), max_steps=int(total) or None,
                                   message=f"transcribing: {ts(done)} / {ts(total)}")
                for start, text in transcribe(str(src), vc["whisper_model"], vc.get("language") or None, prog):
                    control.tick()
                    lines.append((start, text))
        except StopTraining as e:
            return {"status": e.reason.split(":")[0], "reason": e.reason}
        captions = []
        vision = app.cfg()["agent"].get("vision_model")
        if frames and vc.get("describe_frames") and vision:
            for i, (t, fp) in enumerate(frames):
                control.report(message=f"looking at frame {i + 1}/{len(frames)}")
                try:
                    captions.append((t, describe_image(ac, vision, fp, "Opisz krótko po polsku, co widać na tym "
                                                                       "kadrze z filmu instruktażowego (1-2 zdania).")))
                except Exception as e:  # noqa: BLE001
                    captions.append((t, f"(could not describe: {e})"))
        if not lines and not captions:
            raise RuntimeError("No speech or subtitles were found in this file.")
        parts = [f"# Film: {name}", ""]
        if lines:
            parts.append("## Transkrypcja")
            parts += [f"[{ts(t)}] {txt}" for t, txt in lines]
        if captions:
            parts += ["", "## Co widać na filmie"] + [f"[{ts(t)}] {c}" for t, c in captions]
        text = "\n".join(parts)
        topic = p.get("topic", "")
        r = app.knowledge.add(f"Film: {name}", text, "video", f"video:{name}", topic)
        added = 0
        if p.get("also_training") and lines:
            sh = app.evals.all_shingles()
            joined, buf = [], ""
            for _, txt in lines:
                buf = f"{buf} {txt}".strip()
                if len(buf) > 400:
                    joined.append(buf); buf = ""
            if buf:
                joined.append(buf)
            for para in joined:
                if app.learning.add_example(para, "user_added", f"video: {name}", eval_shingles=sh)["status"] == "accepted":
                    added += 1
        try:
            src.unlink()  # the uploaded copy is not needed any more (frames and text are kept)
        except OSError:
            pass
        return {"status": "completed", "knowledge_doc": r["doc"]["id"], "lines": len(lines),
                "frames": [f"video_{job['id']}/{fp.name}" for _, fp in frames], "captions": len(captions),
                "training_examples": added, "seconds": time.time()}

    return run


def png_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def save_png(b64: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))
    return path

