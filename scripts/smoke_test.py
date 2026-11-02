"""BikeSense day-1 smoke test for the Meta Model API.

Checks that your key works and that each model we depend on answers:
  text         muse-spark-1.3 text reply (Responses API, minimal reasoning)
  tokens       /responses/input_tokens count for the same request
  audio        HEARING CHECK: muse-spark-1.2 must count beeps in a synthetic clip. It FAILS if the model
               can't hear non-speech audio. Known result (Nov 2, 2026): Spark's audio input transcribes speech
               but ignores non-speech sound, so this test fails and BikeSense does NOT use Spark on engine audio.
               Python computes every engine-sound number. --wav adds an informational run on your own recording.
  spectrogram  muse-spark-1.3 reads a rendered spectrogram image (the path BikeSense actually uses for sound)
  sam          sam-3.1 image segmentation with a text prompt (streamed Responses API)
  voice        muse-voice-transcribe-1.0 file transcription (only when --voice is given)

Every call is appended to runs/smoke/trace.jsonl (model, latency, usage incl. reasoning tokens, estimated cost).
The API key is read from .env and is never printed or logged.

Usage (from the repo root, venv active):
  python scripts/smoke_test.py
  python scripts/smoke_test.py --wav idle.m4a --image chain.jpg --sam-prompt "motorcycle chain"
  python scripts/smoke_test.py --only text,tokens,spectrogram
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

ALL_TESTS = ("text", "tokens", "audio", "spectrogram", "sam", "voice")
DEFAULT_TESTS = ("text", "tokens", "audio", "spectrogram", "sam")

# Simple calls don't need deep reasoning. "minimal" is the lowest level Muse Spark accepts ("none" returns 400).
EFFORT_SIMPLE = "minimal"
EFFORT_CHART = "low"

# Replies that mean the model did not actually perceive the audio.
CANT_HEAR = re.compile(
    r"NO_AUDIO|can(?:no|')t\s+(?:directly\s+)?(?:hear|listen|process\s+audio)|unable\s+to\s+(?:hear|listen|process)"
    r"|as\s+a\s+text(?:-based)?\s+(?:ai|model)|no\s+audio\s+(?:was\s+)?(?:attached|provided)|don'?t\s+have\s+(?:the\s+)?ability\s+to\s+(?:hear|listen)",
    re.IGNORECASE,
)


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


def reasoning_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    for k in ("output_tokens_details", "completion_tokens_details"):
        if isinstance(usage.get(k), dict) and "reasoning_tokens" in usage[k]:
            return usage[k]["reasoning_tokens"]
    return None


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
    body = {"model": model, "reasoning": {"effort": EFFORT_SIMPLE},
            "input": "Reply with only the word 'ok' written in Tamil script. Nothing else."}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key), json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("text", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("text", model, endpoint, t0, r, "request rejected")
    data = r.json()
    text, usage = extract_response_text(data), data.get("usage")
    cost, rtok = spark_cost(usage), reasoning_tokens(usage)
    print(f"  OK    reply: {short(text, 120)!r}  (reasoning effort={EFFORT_SIMPLE}, reasoning tokens={rtok})\n"
          f"        usage: {usage}  est. cost: ${cost}")
    trace({"test": "text", "model": model, "endpoint": endpoint, "ok": True, "latency_s": round(time.time() - t0, 2),
           "reasoning_effort": EFFORT_SIMPLE, "reasoning_tokens": rtok, "usage": usage, "est_cost_usd": cost,
           "reply": short(text, 200)})
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


def beeps_wav(n_beeps: int = 4, seconds: float = 6.0, sr: int = 24000) -> bytes:
    """Synthetic clip with a known number of 1 kHz beeps, so 'can it hear?' has a checkable answer."""
    t = np.arange(int(seconds * sr)) / sr
    sig = np.zeros_like(t)
    for s in np.linspace(0.5, seconds - 1.0, n_beeps):
        m = (t >= s) & (t < s + 0.3)
        sig[m] = 0.6 * np.sin(2 * np.pi * 1000 * t[m])
    return pcm_wav(sig, sr)


def pcm_wav(sig: np.ndarray, sr: int) -> bytes:
    pcm = np.int16(np.clip(sig, -1, 1) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def spark_audio_call(key: str, base: str, wav: bytes, prompt: str) -> tuple[requests.Response | None, str, dict | None]:
    body = {"model": "muse-spark-1.2", "max_tokens": 4000, "reasoning_effort": EFFORT_SIMPLE,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
            ]}]}
    r = requests.post(base + "/chat/completions", headers=headers(key), json=body, timeout=TIMEOUT_S)
    if not r.ok:
        return r, "", None
    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message", {})
    return r, (msg.get("content") or "").strip(), data.get("usage")


def test_audio(key: str, base: str, wav_path: Path | None) -> bool:
    """Hearing check. PASS only if Spark counts the beeps correctly; FAIL if it says (or shows) it can't hear."""
    model, endpoint = "muse-spark-1.2", "/chat/completions"
    n_beeps = 4
    wav = beeps_wav(n_beeps)
    print(f"[audio] {model} hearing check: synthetic clip with {n_beeps} beeps ({wav_seconds(wav):.1f} s)")
    prompt = ("Listen to the attached audio clip. How many short beeps do you hear? If no audio is attached or you "
              "cannot hear it, reply exactly NO_AUDIO. Answer with just a number or NO_AUDIO.")
    t0 = time.time()
    try:
        r, text, usage = spark_audio_call(key, base, wav, prompt)
    except requests.RequestException as e:
        return fail("audio", model, endpoint, t0, None, repr(e))
    if r is not None and not r.ok:
        return fail("audio", model, endpoint, t0, r, "request rejected")
    nums = re.findall(r"\d+", text)
    heard = not CANT_HEAR.search(text) and nums == [str(n_beeps)]
    tin = (usage or {}).get("prompt_tokens")
    verdict = "PASS" if heard else "FAIL"
    reason = ("counted the beeps correctly" if heard else
              "model did not perceive the non-speech audio" if CANT_HEAR.search(text) else
              f"wrong answer (expected {n_beeps})")
    print(f"  {verdict}  reply: {short(text, 160)!r}  ({reason})\n"
          f"        prompt_tokens={tin} (audio is attached if this grows with clip length; ~12.5 tokens/s measured)")
    if not heard:
        print("        => Spark's audio input is speech-only. BikeSense does not use Spark on engine audio;\n"
              "           Python computes all sound features and Spark 1.3 reads the spectrogram image instead.")
    trace({"test": "audio", "model": model, "endpoint": endpoint, "ok": heard, "latency_s": round(time.time() - t0, 2),
           "reasoning_effort": EFFORT_SIMPLE, "expected_beeps": n_beeps, "reason": reason, "usage": usage,
           "est_cost_usd": spark_cost(usage), "reply": short(text, 300)})

    if wav_path:  # informational only: what does it say about a real engine recording?
        real = wav_path.read_bytes() if wav_path.suffix.lower() == ".wav" else to_wav_mono(wav_path)
        print(f"[audio] informational run on {wav_path.name} ({wav_seconds(real):.1f} s), not scored")
        t1 = time.time()
        try:
            r2, text2, usage2 = spark_audio_call(key, base, real, "Describe the non-speech sound in this clip in under "
                                                 "60 words. If you cannot hear non-speech sound, reply exactly NO_AUDIO.")
        except requests.RequestException as e:
            print(f"        request failed: {e!r}")
        else:
            flag = "model says it can't hear" if CANT_HEAR.search(text2) else "UNVERIFIED, do not trust"
            print(f"        reply: {short(text2, 200)!r}  [{flag}]  prompt_tokens={(usage2 or {}).get('prompt_tokens')}")
            trace({"test": "audio_real_info", "model": model, "ok": None, "latency_s": round(time.time() - t1, 2),
                   "source": wav_path.name, "usage": usage2, "reply": short(text2, 300), "flag": flag})
    return heard


def spectrogram_png(sr: int = 16000, seconds: float = 6.0, f0: float = 125.0, clicks_from: float = 3.0) -> bytes:
    """Render a labelled spectrogram of a synthetic sound with known structure:
    harmonics every f0 Hz for the whole clip, plus a click train (vertical streaks) starting at clicks_from s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * sr)) / sr
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 25))
    clicks = np.zeros_like(t)
    idx = np.arange(int(clicks_from * sr), t.size, int(sr / 8))
    clicks[idx] = 1.0
    clicks = np.convolve(clicks, rng.normal(0, 1, 80), mode="same")
    sig = 0.5 * sig / np.abs(sig).max() + 0.8 * clicks / np.abs(clicks).max()
    f, tt, s = spectrogram(sig, fs=sr, nperseg=1024, noverlap=896)
    keep = f <= 2000
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    ax.pcolormesh(tt, f[keep], 10 * np.log10(s[keep] + 1e-12), shading="auto", cmap="magma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_yticks(np.arange(0, 2001, 250))
    ax.set_title("Spectrogram (test signal)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def test_spectrogram(key: str, base: str) -> bool:
    """The path BikeSense uses for sound: Python renders a spectrogram, Spark 1.3 reads the image."""
    model, endpoint = "muse-spark-1.3", "/responses"
    f0, clicks_from = 125.0, 3.0
    png = spectrogram_png(f0=f0, clicks_from=clicks_from)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "spectrogram_test.png").write_bytes(png)
    print(f"[spectrogram] {model} reads a rendered spectrogram (truth: harmonics every {f0:.0f} Hz, "
          f"clicks from {clicks_from:.0f} s)")
    prompt = ("This is a spectrogram image. 1) Estimate the spacing in Hz between the evenly spaced horizontal "
              "harmonic lines. 2) At what time in seconds do vertical streaks (broadband clicks) begin? "
              'Reply with JSON only: {"harmonic_spacing_hz": <number>, "clicks_start_s": <number>}')
    body = {"model": model, "reasoning": {"effort": EFFORT_CHART}, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode()},
    ]}]}
    t0 = time.time()
    try:
        r = requests.post(base + endpoint, headers=headers(key), json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return fail("spectrogram", model, endpoint, t0, None, repr(e))
    if not r.ok:
        return fail("spectrogram", model, endpoint, t0, r, "request rejected")
    data = r.json()
    text, usage = extract_response_text(data), data.get("usage")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        got = json.loads(m.group(0)) if m else {}
        spacing, start = float(got["harmonic_spacing_hz"]), float(got["clicks_start_s"])
    except (ValueError, KeyError, TypeError):
        return fail("spectrogram", model, endpoint, t0, None, f"unparseable reply: {short(text, 200)}")
    ok = abs(spacing - f0) <= 0.25 * f0 and abs(start - clicks_from) <= 0.75
    print(f"  {'PASS' if ok else 'FAIL'}  read spacing={spacing:g} Hz (truth {f0:g}), clicks start={start:g} s "
          f"(truth {clicks_from:g})  reasoning tokens={reasoning_tokens(usage)}  est. cost: ${spark_cost(usage)}")
    trace({"test": "spectrogram", "model": model, "endpoint": endpoint, "ok": ok, "latency_s": round(time.time() - t0, 2),
           "reasoning_effort": EFFORT_CHART, "truth": {"spacing_hz": f0, "clicks_start_s": clicks_from},
           "read": {"spacing_hz": spacing, "clicks_start_s": start}, "usage": usage, "est_cost_usd": spark_cost(usage)})
    return ok


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
    ap.add_argument("--wav", type=Path, help="real engine recording (wav/m4a/mp3/mp4) for an informational, unscored audio run")
    ap.add_argument("--image", type=Path, default=ROOT / "docs" / "aruco_card.png", help="image for the SAM test")
    ap.add_argument("--sam-prompt", default="black and white square marker", help="SAM text prompt (a short noun phrase)")
    ap.add_argument("--voice", type=Path, help="rider voice note for Voice Transcribe (any audio format)")
    args = ap.parse_args()

    selected = set(args.only.split(",")) if args.only else set(DEFAULT_TESTS) | ({"voice"} if args.voice else set())
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
    if "spectrogram" in selected:
        results["spectrogram"] = test_spectrogram(key, base)
    if "sam" in selected:
        results["sam"] = test_sam(key, base, args.image, args.sam_prompt)
    if "voice" in selected:
        results["voice"] = test_voice(key, base, args.voice)

    print("\nSummary: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print(f"Trace: {TRACE_PATH}")
    return sum(not v for v in results.values())


if __name__ == "__main__":
    sys.exit(main())
