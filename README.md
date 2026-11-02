# BikeSense

**Your mechanic says "fine". BikeSense scans your bike with your phone before and after a service, measures what actually changed, and shows you the evidence, in Tamil, Hindi or English.**

> Built during the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Work in progress. This README will be completed before submission.

## Why
My mechanic marked my bike "fine". Within 5 rides, an oil problem and a chain problem showed up. Riders can't check a repair, because most riders aren't mechanics. BikeSense makes a repair checkable.

## Where each Meta model is used
| Model | Job in BikeSense |
|---|---|
| SAM 3.1 | Segments and tracks the chain, side mirror and engine, oil stains and the dipstick, in photos and video |
| Muse Spark 1.3 | Reads the owner's manual PDF (specs with page numbers), checks that recordings followed the protocol, makes visual judgments, reads the spectrogram, runs the checker agent, writes the report in EN/TA/HI |
| Muse Voice Transcribe 1.0 | The rider's voice note in Tamil or Hindi |

All numbers (chain sag in mm, vibration frequency, RPM, scores) are computed in Python, not by the models.

**Why engine sound goes through a spectrogram image, not raw audio:** in our tests (Nov 2, 2026), Muse Spark's audio input transcribed speech word for word but did not perceive non-speech sound: it answered NO_AUDIO for a beep clip, and reported "background: NONE" for a loud hum mixed under speech, even though the audio tokens were counted (~12.5 tokens/s). So Python measures the engine sound (firing rate, steadiness, ticks, band energy), and Muse Spark 1.3 reads the rendered spectrogram, which it does accurately (`scripts/smoke_test.py` checks both).

## Setup (day-1 state)
Requirements: Python 3.11, git, ffmpeg on PATH (Windows: `winget install Gyan.FFmpeg`).

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1    Git Bash: source .venv/Scripts/activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/install_hooks.py          # pre-commit guard that blocks API keys
cp .env.example .env                     # then open .env and paste your key after MODEL_API_KEY=
python scripts/smoke_test.py             # Spark 1.3 text, token count, audio hearing check (expected FAIL), spectrogram read, SAM 3.1
```

Get a key at [dev.meta.ai](https://dev.meta.ai) → API Keys. Keep it in `.env` only. `.env` is git-ignored and the pre-commit hook blocks it.

Printable scale cards: `docs/aruco_card.pdf` (print at 100%, then measure the black square and set `measured_marker_mm` in `docs/aruco_card.json`).

## Keyless demo mode
Coming soon: recorded Meta API responses are replayed (`BIKESENSE_MODE=replay`) while all Python measurements run live on the real recordings.

## License
MIT
