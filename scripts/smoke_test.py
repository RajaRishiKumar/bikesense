"""BikeSense day-1 smoke test for the Meta Model API.

Checks that your key works and that each model we depend on answers:
  text    muse-spark-1.3 text reply (Responses API)
  tokens  /responses/input_tokens count for the same request
  audio   muse-spark-1.2 on a WAV clip (Chat Completions, input_audio) + its token count
  sam     sam-3.1 image segmentation with a text prompt (streamed Responses API)
  voice   muse-voice-transcribe-1.0 file transcription (only when --voice is given)

Every call is appended to runs/smoke/trace.jsonl (model, latency, usage, estimated cost).
The API key is read from .env and is never printed or logged.

Usage (from the repo root, venv active):
  python scripts/smoke_test.py
  python scripts/smoke_test.py --wav idle.wav --image chain.jpg --sam-prompt "motorcycle chain"
  python scripts/smoke_test.py --only text,tokens
  python scripts/smoke_test.py --voice note.m4a
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "smoke"
TRACE_PATH = RUN_DIR / "trace.jsonl"
TIMEOUT_S = 180

# Prices from dev.meta.ai/docs/pricing-rate-limits (read 2026-11-01), USD.
SPARK_IN_PER_M = 1.25
SPARK_OUT_PER_M = 4.25
SAM_PER_IMAGE = 2.50 / 1000
VOICE_PER_HOUR = 0.18

ALL_TESTS = ("text", "tokens", "audio", "sam", "voice")


# --------------------------------------------------------------------------- setup


def load_config() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY", "").strip()
    base = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1").strip().rstrip("/")
    if not key:
        print(
            "MODEL_API_KEY is empty.\n"
            "  1. Copy .env.example to .env (if you haven't).\n"
            "  2. Open .env in an editor and paste your key after MODEL_API_KEY=\n"
            "  3. Re-run this script. (Never paste the key into chat or commit .env.)"
        )
        sys.exit(2)
    return key, base


def headers(key: str, *, json_body: bool = True, sse: bool = False) -> dict[str, str]:
    h = {"Authorization": f"Bearer {key}"}
    if json_body:
        h["Content-Type"] = "application/json"
    if sse:
        h["Accept"] = "text/event-stream"
    return h


def trace(entry: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def short(text: str, n: int = 400) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + " ..."


def spark_cost(usage: dict | None) -> float | None:
    if not usage:
        return None
    tin = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    tout = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return round(tin / 1e6 * SPARK_IN_PER_M + tout / 1e6 * SPARK_OUT_PER_M, 6)


def extract_response_text(body: dict) -> str:
    """Collect output_text parts from a Responses API JSON body."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts: list[str] = []
    for item in body.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts)


def fail(test: str, model: str, endpoint: str, t0: float, resp: requests.Response | None, err: str) -> bool:
    status = resp.status_code if resp is not None else None
    body = short(resp.text, 600) if resp is not None else ""
    print(f"  FAIL  HTTP {status}: {err}\n        {body}")
    trace({"test": test, "model": model, "endpoint": endpoint, "ok": False,
           "http_status": status, "latency_s": round(time.time() - t0, 2), "error": err, "body": body})
    return False


# --------------------------------------------------------------------------- media helpers


def synth_engine_wav(seconds: float = 5.0, rpm: float = 1400.0, sr: int = 24000) -> bytes:
    """A synthetic single-cylinder 'idle' sound: firing pulses at rpm/120 Hz plus harmonics and noise.

    Only used when no real recording is given. It is NOT a real engine and is only meant to test the API path.
    """
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * sr)) / sr
    f_fire = rpm / 120.0
    crank = rpm / 60.0
    sig = np.zeros_like(t)
    for k in range(1, 40):
        sig += (1.0 / k) * np.sin(2 * np.pi * k * crank * t + rng.uniform(0, 2 * np.pi))
    pulses = (np.sin(2 * np.pi * f_fire * t) > 0.97).astype(float)
    burst = np.convolve(pulses, np.exp(-np.arange(int(0.02 * sr)) / (0.004 * sr)), mode="same")
    sig = 0.6 * sig / np.abs(sig).max() + 0.5 * burst * rng.normal(0, 1, t.size) + 0.03 * rng.normal(0, 1, t.size)
    pcm = np.int16(np.clip(sig / np.abs(sig).max(), -1, 1) * 0.8 * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_wav_mono(path: Path, sr: int = 24000) -> bytes:
    """Convert any audio/video file to WAV mono 16-bit PCM at `sr` using ffmpeg (needed by Voice Transcribe)."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH. Install it (Windows: winget install Gyan.FFmpeg) and reopen the terminal.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_DIR / f"{path.stem}_mono{sr // 1000}k.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
           "-c:a", "pcm_s16le", "-map_metadata", "-1", str(out)]
    subprocess.run(cmd, check=True)
    return out.read_bytes()


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def image_data_url(path: Path) -> str:
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# --------------------------------------------------------------------------- tests


def test_text(key: str, base: str) -> bool:
    model, endpoint = "muse-spark-1.3", "/responses"
    print(f"[text] {model}")
    body = {"model": model, "input": "Reply with only the word 'ok' written in Tamil script. Nothing else."}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key), json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("text", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("text", model, endpoint, t0, r, "request rejected")
    data = r.json()
    text, usage = extract_response_text(data), data.get("usage")
    cost = spark_cost(usage)
    print(f"  OK    reply: {short(text, 120)!r}\n        usage: {usage}  est. cost: ${cost}")
    trace({"test": "text", "model": model, "endpoint": endpoint, "ok": True, "latency_s": round(time.time() - t0, 2),
           "usage": usage, "est_cost_usd": cost, "reply": short(text, 200)})
    return True


def count_tokens(key: str, base: str, body: dict, label: str) -> bool:
    endpoint = "/responses/input_tokens"
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key), json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail(label, body["model"], endpoint, t0, None, repr(e))
    if not r.ok:
        return fail(label, body["model"], endpoint, t0, r, "token count rejected")
    n = r.json().get("input_tokens")
    print(f"  OK    input_tokens = {n}")
    trace({"test": label, "model": body["model"], "endpoint": endpoint, "ok": True,
           "latency_s": round(time.time() - t0, 2), "input_tokens": n})
    return True


def test_tokens(key: str, base: str) -> bool:
    print("[tokens] /responses/input_tokens (text)")
    body = {"model": "muse-spark-1.3", "input": "Reply with only the word 'ok' written in Tamil script. Nothing else."}
    return count_tokens(key, base, body, "tokens_text")


def test_audio(key: str, base: str, wav_path: Path | None) -> bool:
    model, endpoint = "muse-spark-1.2", "/chat/completions"
    if wav_path:
        wav = wav_path.read_bytes() if wav_path.suffix.lower() == ".wav" else to_wav_mono(wav_path)
        src = str(wav_path)
    else:
        wav, src = synth_engine_wav(), "synthetic 5 s tone (not a real engine)"
    secs = wav_seconds(wav)
    print(f"[audio] {model} on {src} ({secs:.1f} s)")
    b64 = base64.b64encode(wav).decode()
    prompt = ("This clip is a machine sound with no speech (a motorcycle engine at idle, or a test tone). "
              "Describe it in under 80 words: is the rhythm steady or irregular, are there ticks, knocks or rattles "
              "(give approximate seconds), and roughly how high or low is the dominant pitch?")
    body = {"model": model, "max_tokens": 4000, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
    ]}]}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key), json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("audio", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("audio", model, endpoint, t0, r, "request rejected")
    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message", {})
    text, usage = msg.get("content") or "", data.get("usage")
    cost = spark_cost(usage)
    print(f"  OK    reply: {short(text)}\n        usage: {usage}  est. cost: ${cost}")
    trace({"test": "audio", "model": model, "endpoint": endpoint, "ok": True, "latency_s": round(time.time() - t0, 2),
           "audio_seconds": round(secs, 2), "source": src, "usage": usage, "est_cost_usd": cost, "reply": short(text, 600)})

    print("[audio] token count for the same clip (Responses shape)")
    count_body = {"model": model, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
    ]}]}
    if not count_tokens(key, base, count_body, "tokens_audio"):
        print("        (token counting for audio may not be supported; usage above is the reliable number)")
    return True


def test_sam(key: str, base: str, image: Path, prompt: str) -> bool:
    model, endpoint = "sam-3.1", "/responses"
    print(f"[sam] {model} on {image.name} with prompt {prompt!r}")
    body = {"model": model, "stream": True, "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url(image)},
            ]}]}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key, sse=True), json=body, stream=True, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("sam", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("sam", model, endpoint, t0, r, "request rejected")
    out, usage = "", None
    for raw in r.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        payload = raw[6:].decode("utf-8", "replace")
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        etype = event.get("type")
        if etype == "response.output_text.delta":
            out += event.get("delta", "")
        elif etype == "response.completed":
            usage = (event.get("response") or {}).get("usage")
        elif etype in ("response.failed", "error"):
            return fail("sam", model, endpoint, t0, None, short(json.dumps(event), 600))
    boxes = re.findall(r"<\|box;[^|]*\|>", out)
    masks = len(re.findall(r"<\|mask;", out))
    print(f"  OK    {len(boxes)} object(s), {masks} mask(s)")
    if boxes:
        print(f"        first box: {boxes[0]}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "sam_raw_output.txt").write_text(out, encoding="utf-8")
    print(f"        raw output saved to {RUN_DIR / 'sam_raw_output.txt'}")
    trace({"test": "sam", "model": model, "endpoint": endpoint, "ok": True, "latency_s": round(time.time() - t0, 2),
           "image": image.name, "prompt": prompt, "objects": len(boxes), "masks": masks, "usage": usage,
           "est_cost_usd": SAM_PER_IMAGE})
    return True


def test_voice(key: str, base: str, path: Path) -> bool:
    model, endpoint = "muse-voice-transcribe-1.0", "/asr/transcribe"
    wav = to_wav_mono(path, sr=24000)
    secs = wav_seconds(wav)
    print(f"[voice] {model} on {path.name} ({secs:.1f} s)")
    req = {"mode": "DIARIZATION", "model": model, "audioEncoding": "WAV"}
    files = {"request": (None, json.dumps(req), "application/json"),
             "audio": (path.stem + ".wav", wav, "audio/wav")}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, params={"sessionId": "bikesense-smoke"},
                          headers=headers(key, json_body=False), files=files, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("voice", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("voice", model, endpoint, t0, r, "request rejected")
    text = r.text
    cost = round(int(secs) / 3600 * VOICE_PER_HOUR, 6)
    print(f"  OK    {short(text)}\n        est. cost: ${cost}")
    trace({"test": "voice", "model": model, "endpoint": endpoint, "ok": True, "latency_s": round(time.time() - t0, 2),
           "audio_seconds": round(secs, 2), "est_cost_usd": cost, "reply": short(text, 600)})
    return True


# --------------------------------------------------------------------------- main


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Tamil/Hindi text on Windows consoles
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help=f"comma list from {','.join(ALL_TESTS)} (default: all except voice unless --voice)")
    ap.add_argument("--wav", type=Path, help="real engine recording (wav/m4a/mp3/mp4); default: synthetic tone")
    ap.add_argument("--image", type=Path, default=ROOT / "docs" / "aruco_card.png", help="image for the SAM test")
    ap.add_argument("--sam-prompt", default="black and white square marker", help="SAM text prompt (a short noun phrase)")
    ap.add_argument("--voice", type=Path, help="rider voice note for Voice Transcribe (any audio format)")
    args = ap.parse_args()

    selected = set(args.only.split(",")) if args.only else {"text", "tokens", "audio", "sam"} | ({"voice"} if args.voice else set())
    unknown = selected - set(ALL_TESTS)
    if unknown:
        ap.error(f"unknown test(s): {', '.join(sorted(unknown))}")
    if "voice" in selected and not args.voice:
        ap.error("--only voice needs --voice <file>")
    for p in (args.wav, args.voice, args.image if "sam" in selected else None):
        if p and not p.exists():
            ap.error(f"file not found: {p}")

    key, base = load_config()
    print(f"Meta Model API at {base}  (key loaded from .env, not shown)\n")

    results: dict[str, bool] = {}
    if "text" in selected:
        results["text"] = test_text(key, base)
    if "tokens" in selected:
        results["tokens"] = test_tokens(key, base)
    if "audio" in selected:
        results["audio"] = test_audio(key, base, args.wav)
    if "sam" in selected:
        results["sam"] = test_sam(key, base, args.image, args.sam_prompt)
    if "voice" in selected:
        results["voice"] = test_voice(key, base, args.voice)

    print("\nSummary: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print(f"Trace: {TRACE_PATH}")
    return sum(not v for v in results.values())


if __name__ == "__main__":
    sys.exit(main())
